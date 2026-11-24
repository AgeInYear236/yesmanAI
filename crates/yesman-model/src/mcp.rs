//! Model Context Protocol (MCP) client and tool-calling integration for Qwen2.5.
//!
//! Connects to MCP servers (such as `op-blender-mcp`) over stdio JSON-RPC,
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
        let res = self.send_request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments,
            }),
        )?;

        // Format content blocks into plain string
        if let Some(content) = res.get("content").and_then(Value::as_array) {
            let mut out = String::new();
            for item in content {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
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
                    calls.push(ToolCall {
                        name: name.to_string(),
                        arguments: args.clone(),
                    });
                }
                let skip_after = if let Some(tag_end) = after_start[end_idx..].find("</tool_call>") {
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
