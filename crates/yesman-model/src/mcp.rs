//! Model Context Protocol (MCP) client and tool-calling integration for Qwen2.5.
//!
//! Connects to MCP servers (such as `mcp-for-blender` / `ahujasid/mcp-for-blender`) over stdio JSON-RPC,
//! fetches available tool schemas, formats them into Qwen2.5 ChatML `<tools>`,
//! and parses `<tool_call>` output blocks for execution.

use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    seq: AtomicU64,
}

impl McpClient {
    /// Spawn an MCP server process over stdio.
    pub fn spawn(cmd: &str) -> Result<Self> {
        let parts = shlex_split(cmd).ok_or_else(|| anyhow!("invalid command string: `{cmd}`"))?;
        if parts.is_empty() {
            bail!("empty command string");
        }

        let mut child = Command::new(&parts[0])
            .args(&parts[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn MCP server `{}`", parts[0]))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("no stdin on child"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no stdout on child"))?;
        let reader = BufReader::new(stdout);

        let mut client = Self {
            child,
            stdin,
            reader,
            seq: AtomicU64::new(1),
        };

        client.initialize()?;
        Ok(client)
    }

    fn next_id(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id();
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut raw = serde_json::to_string(&req)?;
        raw.push('\n');
        self.stdin.write_all(raw.as_bytes())?;
        self.stdin.flush()?;

        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                bail!("MCP server closed connection unexpectedly");
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let resp: Value = serde_json::from_str(line)
                .with_context(|| format!("parsing MCP JSON-RPC response: {line}"))?;

            // Skip notifications or responses for other IDs
            if resp.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(err) = resp.get("error") {
                    bail!("MCP error: {err}");
                }
                return Ok(resp.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }

    fn send_notification(&mut self, method: &str, params: Option<Value>) -> Result<()> {
        let notif = if let Some(p) = params {
            json!({"jsonrpc": "2.0", "method": method, "params": p})
        } else {
            json!({"jsonrpc": "2.0", "method": method})
        };
        let mut raw = serde_json::to_string(&notif)?;
        raw.push('\n');
        self.stdin.write_all(raw.as_bytes())?;
        self.stdin.flush()?;
        Ok(())
    }

    /// Perform the standard MCP `initialize` handshake.
    pub fn initialize(&mut self) -> Result<Value> {
        let res = self.send_request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "yesman",
                    "version": "0.1.0"
                }
            }),
        )?;

        self.send_notification("notifications/initialized", None)?;
        Ok(res)
    }

    /// Query available tools from the MCP server (`tools/list`).
    pub fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let res = self.send_request("tools/list", json!({}))?;
        let tools_val = res
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let tools: Vec<McpTool> = serde_json::from_value(Value::Array(tools_val))?;
        Ok(tools)
    }

    /// Execute a tool on the MCP server (`tools/call`).
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<String> {
        let mut arguments = normalize_tool_arguments(name, arguments);
        expand_tool_arguments(&mut arguments);
        let res = self.send_request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments,
            }),
        )?;

        // Format content blocks into a compact plain string. Image blocks (e.g.
        // `look`) are summarised instead of dumping base64 into the model's
        // context, which would blow up the KV cache after a verification step.
        if let Some(content) = res.get("content").and_then(Value::as_array) {
            let mut out = String::new();
            for item in content {
                let piece = if let Some(text) = item.get("text").and_then(Value::as_str) {
                    Some(text.to_string())
                } else if item.get("type").and_then(Value::as_str) == Some("image") {
                    let mime = item
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("image");
                    Some(format!("[image: {mime}]"))
                } else {
                    None
                };
                if let Some(piece) = piece {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&piece);
                }
            }
            if !out.is_empty() {
                return Ok(out);
            }
        }

        Ok(res.to_string())
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// Simple shell split without extra dependencies.
fn shlex_split(s: &str) -> Option<Vec<String>> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_quote: Option<char> = None;
    let chars = s.chars();

    for c in chars {
        match c {
            '"' | '\'' => {
                if in_quote == Some(c) {
                    in_quote = None;
                } else if in_quote.is_none() {
                    in_quote = Some(c);
                } else {
                    cur.push(c);
                }
            }
            ' ' | '\t' if in_quote.is_none() => {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    Some(words)
}

/// Format MCP tools into the official Qwen2.5 system prompt `<tools>` section.
pub fn format_tools_system_prompt(base_system: &str, tools: &[McpTool]) -> String {
    if tools.is_empty() {
        return base_system.to_string();
    }

    let mut out = base_system.to_string();
    out.push_str("\n\n# Tools\n\n");
    out.push_str("You may call one or more functions to assist with the user specification.\n\n");
    out.push_str(
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );

    for t in tools {
        let sig = json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
        });
        out.push_str(&sig.to_string());
        out.push('\n');
    }

    out.push_str("</tools>\n\n");
    out.push_str("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
    out.push_str(
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );

    out
}

/// Marker the fine-tuned model emits instead of the fixed bpy helper library.
/// See `training/generate_3d_dataset.py` (`STUDIO_MARKER`).
pub const STUDIO_MARKER: &str = "# YESMAN_STUDIO";
const STUDIO_PREAMBLE_PATH: &str = "training/bpy_preamble.py";

#[derive(Debug, Deserialize)]
struct RecipeEntry {
    key: String,
    keywords: Vec<String>,
    body: String,
}

fn load_recipes() -> Vec<RecipeEntry> {
    std::fs::read_to_string("training/recipes.json")
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<RecipeEntry>>(&text).ok())
        .unwrap_or_default()
}

/// Swap a generated body for a verified recipe body when the model clearly
/// aimed at one of them (it reproduces object names reliably even when it
/// garbles coordinates and sizes). This is what makes the built scene actually
/// look like the requested object rather than a cloud of random primitives.
fn repair_scene_body(body: &str) -> (String, Option<String>) {
    let recipes = load_recipes();
    if recipes.is_empty() {
        return (body.to_string(), None);
    }
    let haystack = body.to_lowercase();
    let mut best_score = 0usize;
    let mut best: Option<&RecipeEntry> = None;
    for recipe in &recipes {
        let score: usize = recipe
            .keywords
            .iter()
            .map(|keyword| haystack.matches(keyword.as_str()).count())
            .sum();
        if score > best_score {
            best_score = score;
            best = Some(recipe);
        }
    }
    match best {
        Some(recipe) if best_score >= 3 => (recipe.body.clone(), Some(recipe.key.clone())),
        _ => (body.to_string(), None),
    }
}

fn isolated_body_runner(body: &str, recipe: Option<&str>) -> String {
    // Execute each top-level statement separately. A single typo in a
    // generated loop must not discard the other 30 objects of the scene.
    let literal = serde_json::to_string(body).unwrap_or_else(|_| "\"\"".to_string());
    let note = match recipe {
        Some(key) => format!("print('yesman: verified recipe: {key}')\n"),
        None => String::new(),
    };
    format!(
        r#"import ast as _ya
_ybody = {literal}
{note}_yns = globals()
try:
    _ynodes = _ya.parse(_ybody).body
except SyntaxError as _ye:
    print('yesman: body syntax error:', _ye)
    try:
        exec(_ybody, _yns)
    except Exception as _ye2:
        print('yesman: body error:', _ye2)
else:
    for _ynode in _ynodes:
        try:
            exec(compile(_ya.Module(body=[_ynode], type_ignores=[]), '<yesman>', 'exec'), _yns)
        except Exception as _ye3:
            print('yesman: skipped statement:', _ye3)
    bpy.context.view_layer.update()
    _ybad = [o for o in list(bpy.context.scene.objects) if (o.type=='MESH' and len(o.data.polygons)==0) or (o.type=='CURVE' and len(o.data.splines)==0)]
    for _yo in _ybad:
        bpy.data.objects.remove(_yo, do_unlink=True)
    if _ybad:
        print('yesman: removed', len(_ybad), 'degenerate objects')
    # Curve bound boxes are unreliable (Bezier handles inflate them), so fit
    # the floor to meshes only.
    _yobjs = [o for o in bpy.context.scene.objects if o.type=='MESH' and o.name!='Ground']
    _ycoords = [o.matrix_world @ Vector(c) for o in _yobjs for c in o.bound_box]
    if _ycoords:
        _ymn = Vector((min(v.x for v in _ycoords), min(v.y for v in _ycoords), min(v.z for v in _ycoords)))
        _ymx = Vector((max(v.x for v in _ycoords), max(v.y for v in _ycoords), max(v.z for v in _ycoords)))
        _ysize = max(2.0, max(_ymx.x-_ymn.x, _ymx.y-_ymn.y)*1.35)
        _yg = bpy.data.objects.get('Ground')
        if _yg is not None:
            bpy.data.objects.remove(_yg, do_unlink=True)
        bpy.ops.mesh.primitive_plane_add(size=_ysize, location=((_ymn.x+_ymx.x)/2, (_ymn.y+_ymx.y)/2, _ymn.z-0.02))
        _yg = bpy.context.active_object; _yg.name='Ground'; _yg.data.materials.append(MAT('GroundMat'))
        print('yesman: ground refit to', round(_ysize, 2), 'm around scene bbox')
    _ytarget = globals().get('_yview_target')
    if _ytarget is not None:
        _ysmall = [o for o in bpy.context.scene.objects if o.type=='MESH' and o.name!='Ground' and max(o.dimensions) < 0.6]
        _yext = max((max(o.dimensions) for o in _ysmall), default=0.4)
        _ydist = max(0.8, min(6.0, _yext*4.0))
        try:
            for _yarea in bpy.context.screen.areas:
                if _yarea.type == 'VIEW_3D':
                    _yr3 = _yarea.spaces.active.region_3d
                    _yr3.view_location = Vector(_ytarget)
                    _yr3.view_distance = _ydist
            print('yesman: viewport framed on hero object at', round(_ydist, 2), 'm')
        except Exception as _ye4:
            print('yesman: viewport frame skipped:', _ye4)
"#
    )
}

fn expand_studio_marker_with(text: &str, preamble: &str) -> String {
    if !text.contains(STUDIO_MARKER) {
        return text.to_string();
    }
    // Direct MCP: raw Python script. Run the body through the isolating
    // runner so one malformed statement cannot abort the whole build.
    if let Some(pos) = text.find(&format!("{STUDIO_MARKER}\n")) {
        let marker_len = STUDIO_MARKER.len() + 1;
        let (head, rest) = text.split_at(pos);
        let body = &rest[marker_len..];
        let (repaired, recipe) = repair_scene_body(body);
        return format!(
            "{head}{preamble}\n{}",
            isolated_body_runner(&repaired, recipe.as_deref())
        );
    }
    // OpenCode Code Mode: the bpy script sits inside a JS/JSON string, so the
    // marker's newline is the two chars `\n`. Inline the preamble there.
    let escaped = serde_json::to_string(preamble).unwrap_or_default();
    let inner = escaped
        .get(1..escaped.len().saturating_sub(1))
        .unwrap_or(preamble);
    text.replace(&format!("{STUDIO_MARKER}\\n"), inner)
        .replace(STUDIO_MARKER, preamble)
}

/// Replace the studio marker with the helper library from
/// `training/bpy_preamble.py`. If the file is absent the text is unchanged.
pub fn expand_studio_marker(text: &str) -> String {
    match std::fs::read_to_string(STUDIO_PREAMBLE_PATH) {
        Ok(preamble) => expand_studio_marker_with(text, &preamble),
        Err(_) => text.to_string(),
    }
}

/// Expand the marker inside a tool call's `code` argument, both for direct
/// `execute_blender_code` calls and for OpenCode Code Mode `execute` calls.
pub fn expand_tool_arguments(arguments: &mut Value) {
    let Some(obj) = arguments.as_object_mut() else {
        return;
    };
    let Some(code) = obj.get("code").and_then(Value::as_str).map(str::to_string) else {
        return;
    };
    let expanded = expand_studio_marker(&code);
    if expanded != code {
        obj.insert("code".to_string(), Value::String(expanded));
    }
}

/// Extract any `<tool_call>\n{...}\n</tool_call>` blocks from model text output.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut remaining = String::new();
    let mut cur = text;

    while let Some(start) = cur.find("<tool_call>") {
        remaining.push_str(&cur[..start]);
        let after_start = &cur[start + "<tool_call>".len()..];
        if let Some(brace_start) = after_start.find('{') {
            // Parse matching JSON object
            let mut depth = 0usize;
            let mut in_str = false;
            let mut escape = false;
            let mut brace_end = None;

            for (idx, ch) in after_start[brace_start..].char_indices() {
                if escape {
                    escape = false;
                    continue;
                }
                match ch {
                    '\\' => escape = true,
                    '"' => in_str = !in_str,
                    '{' if !in_str => depth += 1,
                    '}' if !in_str => {
                        depth -= 1;
                        if depth == 0 {
                            brace_end = Some(brace_start + idx + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(end_idx) = brace_end {
                let json_text = after_start[brace_start..end_idx].trim();
                if let Ok(v) = serde_json::from_str::<Value>(json_text)
                    && let (Some(name), Some(args)) =
                        (v.get("name").and_then(Value::as_str), v.get("arguments"))
                {
                    let mut arguments = args.clone();
                    expand_tool_arguments(&mut arguments);
                    calls.push(ToolCall {
                        name: name.to_string(),
                        arguments,
                    });
                }
                let skip_after = if let Some(tag_end) = after_start[end_idx..].find("</tool_call>")
                {
                    end_idx + tag_end + "</tool_call>".len()
                } else {
                    end_idx
                };
                cur = &after_start[skip_after..];
            } else {
                remaining.push_str(cur);
                return (remaining.trim().to_string(), calls);
            }
        } else {
            // Unclosed tool call tag without valid JSON
            remaining.push_str(cur);
            return (remaining.trim().to_string(), calls);
        }
    }
    remaining.push_str(cur);

    (remaining.trim().to_string(), calls)
}

/// Format tool execution result as a `<tool_response>` message according to Qwen2.5 convention.
pub fn format_tool_response(result_text: &str) -> String {
    format!("<tool_response>\n{}\n</tool_response>", result_text.trim())
}

/// Compact tool schemas matching `training/generate_3d_dataset.py`.
///
/// The live `mcp-for-blender` server advertises 9 tools with very long
/// descriptions (several thousand tokens). Putting those into the system
/// prompt makes prefill slow and, more importantly, makes inference diverge
/// from the prompts the model was fine-tuned on. We therefore show only the
/// four tools it knows; calls are still executed against the live server.
pub fn blender_training_tools() -> Vec<McpTool> {
    let tool = |name: &str, description: &str, schema: Value| McpTool {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: schema,
    };
    vec![
        tool(
            "execute_blender_code",
            "Run Python bpy code in Blender. No return statement.",
            json!({
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python bpy script"}},
                "required": ["code"]
            }),
        ),
        tool(
            "look",
            "See the scene or object. Modes: viewport, camera, angles.",
            json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["viewport", "camera", "angles"]},
                    "target": {"type": "array", "items": {"type": "string"}},
                    "shading": {"type": "string", "enum": ["rendered", "solid", "wireframe", "xray"]}
                }
            }),
        ),
        tool(
            "get_scene_info",
            "Get scene hierarchy, objects and materials summary.",
            json!({"type": "object", "properties": {}}),
        ),
        tool(
            "get_addon_status",
            "Check Blender addon status and version.",
            json!({"type": "object", "properties": {}}),
        ),
    ]
}

/// Normalise argument shapes a 1.5B model tends to get wrong before sending
/// them to the MCP server (which validates with pydantic).
pub fn normalize_tool_arguments(name: &str, mut arguments: Value) -> Value {
    if name != "get_scene_info" {
        return arguments;
    }
    let Some(obj) = arguments.as_object_mut() else {
        return arguments;
    };
    // A string like "location,size" becomes a list; unknown names are dropped
    // because the addon only understands a fixed vocabulary of fields.
    const ALLOWED_FIELDS: &[&str] = &[
        "location",
        "size",
        "children",
        "hidden",
        "materials",
        "modifiers",
        "rotation",
        "scale",
        "parent",
        "ground",
        "details",
        "topology",
        "weights",
        "animation",
        "settings",
        "placement",
        "contents",
        "health",
    ];
    let fields_value = obj.remove("fields").and_then(|fields| {
        let list: Vec<Value> = match fields {
            Value::String(s) => s.split(',').map(|v| json!(v.trim())).collect(),
            Value::Array(items) => items,
            _ => return None,
        };
        let clean: Vec<Value> = list
            .into_iter()
            .filter(|v| v.as_str().is_some_and(|s| ALLOWED_FIELDS.contains(&s)))
            .collect();
        if clean.is_empty() {
            None
        } else {
            Some(Value::Array(clean))
        }
    });
    // Only forward keys the live tool actually accepts; anything the model
    // invented would fail pydantic validation and cause a retry loop.
    let mut clean = serde_json::Map::new();
    for key in ["query", "root", "limit", "user_prompt", "fields"] {
        let value = if key == "fields" {
            fields_value.clone()
        } else {
            obj.remove(key)
        };
        if let Some(value) = value {
            clean.insert(key.to_string(), value);
        }
    }
    Value::Object(clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_call_xml() {
        let model_out = r#"
Sure, I will check the scene now.
<tool_call>
{"name": "get_scene_info", "arguments": {}}
</tool_call>
Let me know if you need anything else!
"#;
        let (text, calls) = parse_tool_calls(model_out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_scene_info");
        assert_eq!(calls[0].arguments, json!({}));
        assert!(text.contains("Sure, I will check the scene now."));
        assert!(text.contains("Let me know if you need anything else!"));
    }

    #[test]
    fn parse_unclosed_tool_call() {
        let unclosed = r#"<tool_call>
{"name": "create_sphere", "arguments": {"radius": 1.0}}"#;
        let (text, calls) = parse_tool_calls(unclosed);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "create_sphere");
        assert_eq!(calls[0].arguments, json!({"radius": 1.0}));
        assert_eq!(text, "");
    }

    #[test]
    fn parse_nested_tool_call_tag() {
        let nested = r#"<tool_call>
<tool_call>
{"name": "execute_blender_code", "arguments": {"code": "import bpy\nbpy.ops.mesh.primitive_uv_sphere_add()"}}
</tool_call>"#;
        let (_, calls) = parse_tool_calls(nested);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "execute_blender_code");
        assert_eq!(
            calls[0].arguments,
            json!({"code": "import bpy\nbpy.ops.mesh.primitive_uv_sphere_add()"})
        );
    }

    #[test]
    fn studio_marker_expands_in_python_and_js() {
        let pre = "import bpy\ndef ST():\n    pass\n";
        let raw = "# YESMAN_STUDIO\nM('Wood',(1,0,0,1))\n";
        assert!(expand_studio_marker_with(raw, pre).starts_with("import bpy\n"));
        let js = r##"await tools.blender.execute_blender_code({ code: "# YESMAN_STUDIO\nM('Wood')\n" })"##;
        let out = expand_studio_marker_with(js, pre);
        assert!(out.contains(r"import bpy\ndef ST():\n    pass\n"));
        assert!(!out.contains("# YESMAN_STUDIO"));
    }

    #[test]
    fn tool_arguments_get_expanded() {
        let mut args = json!({"code": "# YESMAN_STUDIO\nprint('hi')\n"});
        // No preamble file in unit tests: expansion is a no-op, not a panic.
        expand_tool_arguments(&mut args);
        assert!(
            args["code"].as_str().unwrap().contains("# YESMAN_STUDIO")
                || args["code"].as_str().unwrap().contains("import bpy")
        );
    }

    #[test]
    fn compact_blender_tools_match_training() {
        let tools = blender_training_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "execute_blender_code",
                "look",
                "get_scene_info",
                "get_addon_status"
            ]
        );
        assert!(tools[0].input_schema.get("properties").is_some());
    }

    #[test]
    fn normalize_scene_info_args() {
        let args = json!({"fields": "location,size,bogus", "unknown_key": 1});
        let out = normalize_tool_arguments("get_scene_info", args);
        assert_eq!(out, json!({"fields": ["location", "size"]}));
        let out = normalize_tool_arguments("get_scene_info", json!({}));
        assert_eq!(out, json!({}));
        let out = normalize_tool_arguments("look", json!({"fields": "size"}));
        assert_eq!(out, json!({"fields": "size"}));
    }

    #[test]
    fn format_tools_prompt() {
        let tool = McpTool {
            name: "test_tool".into(),
            description: "A test tool".into(),
            input_schema: json!({"type": "object", "properties": {"arg": {"type": "string"}}}),
        };
        let p = format_tools_system_prompt("System prompt.", &[tool]);
        assert!(p.contains("<tools>"));
        assert!(p.contains("\"name\":\"test_tool\""));
        assert!(p.contains("<tool_call>"));
    }
}
