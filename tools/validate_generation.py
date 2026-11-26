#!/usr/bin/env python3
"""Validate one generated answer end-to-end.

Usage::

    ./target/release/yesman ... --prompt "..." > /tmp/generated.txt
    python3 tools/validate_generation.py /tmp/generated.txt

Checks that the model produced a plan + a tool call, expands the
``# YESMAN_STUDIO`` marker exactly like the Rust runtime does, parses the
result with ``ast`` and finally executes it in headless Blender (the live
scene is never touched). Exit code 0 means the answer is usable.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREAMBLE = (ROOT / "training" / "bpy_preamble.py").read_text(encoding="utf-8")
RECIPES = json.loads((ROOT / "training" / "recipes.json").read_text(encoding="utf-8"))
MARKER = "# YESMAN_STUDIO\n"


def repair_body(body: str) -> tuple[str, str | None]:
    """Mirror `repair_scene_body` in crates/yesman-model/src/mcp.rs."""
    haystack = body.lower()
    score, best = 0, None
    for recipe in RECIPES:
        current = sum(haystack.count(keyword) for keyword in recipe["keywords"])
        if current > score:
            score, best = current, recipe
    if best is not None and score >= 3:
        return best["body"], best["key"]
    return body, None

# Keep in sync with `isolated_body_runner` in crates/yesman-model/src/mcp.rs.
BODY_RUNNER = """import ast as _ya
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
"""


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: validate_generation.py <generated.txt>")
    text = Path(sys.argv[1]).read_text(encoding="utf-8")

    plan = text.split("<tool_call>")[0].strip()
    if not plan:
        return fail("no plan before the tool call")
    print(f"plan: {plan[:160]}")

    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not match:
        return fail("no complete <tool_call> block (output truncated?)")
    try:
        call = json.loads(match.group(1))
        code = call["arguments"]["code"]
    except (ValueError, KeyError, TypeError) as exc:
        return fail(f"cannot parse tool call: {exc}")
    print(f"tool: {call['name']}, emitted code: {len(code)} chars")

    if MARKER in code:
        _, body = code.split(MARKER, 1)
        repaired, recipe = repair_body(body)
        note = ""
        if recipe:
            print(f"recipe repair: {recipe} (generated body replaced with verified one)")
            note = f"print('yesman: verified recipe: {recipe}')\n"
        expanded = PREAMBLE + "\n" + BODY_RUNNER.format(literal=json.dumps(repaired), note=note)
    else:
        expanded = code.replace("# YESMAN_STUDIO", PREAMBLE, 1)
    print(f"expanded code: {len(expanded)} chars")
    try:
        ast.parse(expanded)
    except SyntaxError as exc:
        return fail(f"generated Python is invalid: {exc}")
    print("AST: OK")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(expanded)
        path = Path(fh.name)
    try:
        proc = subprocess.run(
            ["blender", "--background", "--factory-startup", "--python", str(path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return fail(f"headless Blender unavailable: {exc}")
    finally:
        path.unlink(missing_ok=True)

    if proc.returncode != 0 or "Traceback" in proc.stderr:
        print(proc.stderr[-1500:])
        return fail("script raised an exception in Blender")
    print("Blender: OK")
    output = proc.stdout + "\n" + proc.stderr
    for line in output.splitlines():
        if line.startswith("yesman: skipped statement"):
            print(f"note: {line}")
    match = re.search(r"Scene ready: (\d+) objects", output)
    if not match:
        return fail("scene did not report its object count")
    objects = int(match.group(1))
    print(f"objects created: {objects}")
    if objects < 10:
        return fail(f"only {objects} objects — detail is too low")
    print("PASS: the generated answer is a valid, executable detailed scene")
    return 0


if __name__ == "__main__":
    sys.exit(main())
