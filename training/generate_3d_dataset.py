#!/usr/bin/env python3
"""Build the ChatML fine-tuning dataset for the Blender-MCP Qwen2.5 model.

What this file does (in order):

1. ``RULES`` describes the assistant's workflow. It is the single source of
   truth for ``training/system_prompt.txt`` (read by the Rust CLI/server) and
   for every ``system`` message in the generated JSONL.
2. ``format_tools_prompt`` mirrors ``yesman_model::format_tools_system_prompt``
   so training prompts are compatible with inference prompts.
3. ``PREAMBLE`` is a small, fixed ``bpy`` helper library (materials, primitives,
   curves, studio lighting). Every generated script starts with it, so a recipe
   only needs compact, loop-based calls -- a scene with 40+ named parts fits in
   ~1.5k tokens instead of ~7k. Scripts are validated with ``ast.parse``.
4. ``Recipe`` couples a detailed scene with paraphrased prompts, an explicit
   plan and a final summary. Each recipe produces:
     - single-turn samples: plan + one tool call;
     - multi-step samples: plan + call -> response -> verification -> summary.
   Both the direct MCP format (``execute_blender_code``) and the OpenCode
   Code Mode format (``execute`` with JS) are generated, because the server
   serves both clients.
5. ``main`` writes train.jsonl, val.jsonl and system_prompt.txt, prints a short
   statistics report and optionally checks scripts in headless Blender.

Usage::

    python3 training/generate_3d_dataset.py                  # regenerate data
    python3 training/generate_3d_dataset.py --check-blender # + headless bpy check
    python3 training/generate_3d_dataset.py --stats-only    # inspect, write nothing

The JSONL rows are ChatML ``{"messages": [...]}`` objects consumed by
``training/train_lora.py``. Only ``assistant`` turns carry loss there.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PREAMBLE_PATH = ROOT / "bpy_preamble.py"
RECIPES_PATH = ROOT / "recipes.json"

# The model emits this one-line marker instead of the whole helper library.
# The Rust runtime (crates/yesman-model/src/mcp.rs) replaces it with the file
# written below before executing the script in Blender. Keeping the fixed
# boilerplate out of the generated answer is what makes detailed scenes
# reproducible for a 1.5B model.
STUDIO_MARKER = "# YESMAN_STUDIO\n"

# ---------------------------------------------------------------------------
# 1. Workflow rules (shared by training and inference)
# ---------------------------------------------------------------------------

RULES = """Ты — профессиональный 3D-моделлер Blender, работающий через MCP-инструменты.

Рабочий цикл — выполняй всегда:
1. Сначала напиши короткий план на языке пользователя (3–6 пунктов): какие части, материалы, свет и камеру создашь. Не пропускай этот шаг.
2. Затем одним вызовом инструмента построй ДЕТАЛЬНУЮ сцену: десятки именованных объектов, фаски (Bevel) и сглаживание, процедурные PBR-материалы (Noise/Voronoi → ColorRamp → Base Color, Bump → Normal), студийный свет из трёх area-источников, камеру и землю. Не ограничивайся одним примитивом.
3. После успешного выполнения один раз проверь результат через look (viewport).
4. СРАЗУ после ответа look напиши краткое резюме: что построено, сколько объектов, какие материалы, что проверено. После проверки новых вызовов инструментов НЕ делай — это конец ответа.

Правила кода для execute_blender_code:
- Код выполняется через exec(): без return на верхнем уровне, без input(), без бесконечных циклов.
- Ноды материала ищи по типу: next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'). Имена нод локализованы, по имени искать нельзя.
- Каждому объекту давай осмысленное имя, применяй transform_apply(scale=True), сглаживание и Bevel/WeightedNormal.
- Начинай скрипт со строки `# YESMAN_STUDIO`: рантайм подставит вместо неё готовую библиотеку helpers (M, B, C, S, K, T, V, L, ST). Саму библиотеку в ответ не пиши.
- В конце скрипта вызывай ST() и делай print() с кратким отчётом о количестве объектов.
- Для простых запросов (осмотр сцены, статус, очистка) достаточно плана из одного предложения.

Никогда не выдавай код обычным текстом — только внутри <tool_call>."""

# ---------------------------------------------------------------------------
# 2. Tool contracts
# ---------------------------------------------------------------------------

# Compact schemas used in training. The Rust CLI/server inject the *live*
# schemas at inference time; names and argument shapes must stay in sync.
DIRECT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "execute_blender_code",
            "description": "Run Python bpy code in Blender. No return statement.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python bpy script"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "See the scene or object. Modes: viewport, camera, angles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["viewport", "camera", "angles"]},
                    "target": {"type": "array", "items": {"type": "string"}},
                    "shading": {"type": "string", "enum": ["rendered", "solid", "wireframe", "xray"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Get scene hierarchy, objects and materials summary.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_addon_status",
            "description": "Check Blender addon status and version.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Must match the `execute` tool injected by yesman-server for OpenCode Code Mode.
CODE_MODE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run JavaScript code in OpenCode Code Mode. To run Blender Python code, call: "
            "await tools.blender.execute_blender_code({ code: 'import bpy...' })"
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "JavaScript code to execute"}},
            "required": ["code"],
        },
    },
}


def format_tools_prompt(base: str, tools: list[dict]) -> str:
    """Mirror ``yesman_model::format_tools_system_prompt`` (Qwen2.5 ChatML)."""
    if not tools:
        return base
    lines = [
        base,
        "",
        "# Tools",
        "",
        "You may call one or more functions to assist with the user specification.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    lines.extend(json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in tools)
    lines.extend(
        [
            "</tools>",
            "",
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>',
        ]
    )
    return "\n".join(lines)


SYSTEM_DIRECT = format_tools_prompt(RULES, DIRECT_TOOLS)
SYSTEM_CODE_MODE = format_tools_prompt(RULES, [CODE_MODE_TOOL])

# ---------------------------------------------------------------------------
# 3. Fixed bpy helper library, prepended to every generated script
# ---------------------------------------------------------------------------

# Short helper names keep recipes compact. ``M`` builds PBR materials (nodes
# looked up by type), ``B/C/S/K/T`` add primitives, ``V`` a Bezier cable,
# ``L`` a lathed profile, ``ST`` ground + 3 area lights + world + camera.
PREAMBLE = """import bpy
import math
from mathutils import Vector
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)

# Tolerant helpers: a 1.5B model sometimes drops or reorders a positional
# argument. Every primitive accepts *a, picks values by type and coerces the
# rest, so a slightly wrong call still builds geometry instead of raising.
def N(x, d):
    return float(x) if isinstance(x,(int,float)) and not isinstance(x,bool) else float(d)

def G(nums, i, d):
    return N(nums[i], d) if len(nums) > i else float(d)

def V3(x, d=(0,0,0)):
    if isinstance(x,(tuple,list)) and len(x)>=3 and all(isinstance(v,(int,float)) for v in x[:3]):
        return tuple(float(v) for v in x[:3])
    return tuple(d)

def V4(x, d=(0.8,0.8,0.8,1.0)):
    if isinstance(x,(tuple,list)) and len(x)>=4 and all(isinstance(v,(int,float)) for v in x[:4]):
        return tuple(float(v) for v in x[:4])
    if isinstance(x,(tuple,list)) and len(x)==3:
        return tuple(float(v) for v in x[:3])+(1.0,)
    return tuple(d)

def ARGS(a):
    nums=[x for x in a if isinstance(x,(int,float)) and not isinstance(x,bool)]
    vecs=[x for x in a if isinstance(x,(tuple,list)) and 3<=len(x)<=4 and all(isinstance(v,(int,float)) for v in x)]
    pts=[x for x in a if isinstance(x,(tuple,list)) and x and isinstance(x[0],(tuple,list))]
    strs=[x for x in a if isinstance(x,str)]
    return nums, vecs, pts, strs

def MAT(name):
    m = bpy.data.materials.get(name) if isinstance(name,str) else None
    if m is None:
        m = bpy.data.materials.get('GroundMat')
        if m is None:
            m = bpy.data.materials.new('GroundMat'); m.use_nodes = True
            b = next(x for x in m.node_tree.nodes if x.type=='BSDF_PRINCIPLED')
            b.inputs['Base Color'].default_value = (0.12,0.13,0.15,1.0)
            b.inputs['Roughness'].default_value = 0.85
    return m

def M(name, color=(0.8,0.8,0.8,1.0), rough=0.4, metal=0.0, proc=None, emit=0.0):
    m = bpy.data.materials.new(str(name)); m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    b = next(x for x in n if x.type=='BSDF_PRINCIPLED')
    c = V4(color); b.inputs['Base Color'].default_value = c
    b.inputs['Roughness'].default_value = N(rough,0.4); b.inputs['Metallic'].default_value = N(metal,0.0)
    if N(emit,0.0) > 0:
        b.inputs['Emission Color'].default_value = c; b.inputs['Emission Strength'].default_value = N(emit,0.0)
    if proc in ('noise','voronoi'):
        t = n.new('ShaderNodeTexVoronoi' if proc=='voronoi' else 'ShaderNodeTexNoise'); t.inputs['Scale'].default_value = 14.0
        o = t.outputs['Distance'] if proc=='voronoi' else t.outputs['Fac']
        r = n.new('ShaderNodeValToRGB'); r.color_ramp.elements[0].color = tuple(v*0.45 for v in c[:3])+(1.0,); r.color_ramp.elements[1].color = c
        bm = n.new('ShaderNodeBump'); bm.inputs['Strength'].default_value = 0.35
        l.new(o,r.inputs['Fac']); l.new(r.outputs['Color'],b.inputs['Base Color']); l.new(o,bm.inputs['Height']); l.new(bm.outputs['Normal'],b.inputs['Normal'])
    return m

def P(o, name, material, bevel=0.0, smooth=True):
    o.name = str(name); o.data.materials.append(MAT(material))
    if smooth:
        for q in o.data.polygons: q.use_smooth = True
    w = N(bevel,0.0)
    if w > 0:
        m = o.modifiers.new('Bevel','BEVEL'); m.width = w; m.segments = 3; m.limit_method = 'ANGLE'
        o.modifiers.new('WeightedNormal','WEIGHTED_NORMAL')
    return o

def REQ(a, ok, msg):
    if not ok: raise ValueError(msg)

def B(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, vecs, 'B(): missing location')
    loc = V3(vecs[0]); size = V3(vecs[1] if len(vecs)>1 else (1,1,1)); rot = V3(vecs[2] if len(vecs)>2 else (0,0,0))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object; o.scale = size; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return P(o, name, strs[0] if strs else 'GroundMat', nums[0] if nums else 0.0, False)

def C(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, vecs, 'C(): missing location')
    loc = V3(vecs[0]); rot = V3(vecs[1] if len(vecs)>1 else (0,0,0))
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(G(nums,2,32)), radius=G(nums,0,0.05), depth=G(nums,1,0.2), location=loc, rotation=rot)
    return P(bpy.context.active_object, name, strs[0] if strs else 'GroundMat')

def S(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, vecs, 'S(): missing location')
    loc = V3(vecs[0]); scale = V3(vecs[1] if len(vecs)>1 else (1,1,1))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=G(nums,0,0.2), location=loc)
    o = bpy.context.active_object; o.scale = scale
    return P(o, name, strs[0] if strs else 'GroundMat')

def K(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, vecs, 'K(): missing location')
    loc = V3(vecs[0]); rot = V3(vecs[1] if len(vecs)>1 else (0,0,0))
    bpy.ops.mesh.primitive_cone_add(vertices=int(G(nums,3,24)), radius1=G(nums,0,0.1), radius2=G(nums,1,0.02), depth=G(nums,2,0.5), location=loc, rotation=rot)
    return P(bpy.context.active_object, name, strs[0] if strs else 'GroundMat')

def T(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, vecs, 'T(): missing location')
    loc = V3(vecs[0])
    bpy.ops.mesh.primitive_torus_add(major_radius=G(nums,0,0.3), minor_radius=G(nums,1,0.03), location=loc)
    return P(bpy.context.active_object, name, strs[0] if strs else 'GroundMat')

def V(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, pts and len(pts[0])>=2, 'V(): needs a list of 3D points')
    points = pts[0]
    cu = bpy.data.curves.new(str(name),'CURVE'); cu.dimensions='3D'; cu.bevel_depth=N(nums[0],0.01) if nums else 0.01; cu.bevel_resolution=3
    sp = cu.splines.new('BEZIER'); sp.bezier_points.add(max(0,len(points)-1))
    for q,co in zip(sp.bezier_points,points): q.co=V3(co); q.handle_left_type='AUTO'; q.handle_right_type='AUTO'
    o = bpy.data.objects.new(str(name),cu); bpy.context.collection.objects.link(o); o.data.materials.append(MAT(strs[0] if strs else 'GroundMat'))
    return o

def L(name, *a):
    nums, vecs, pts, strs = ARGS(a)
    REQ(a, pts and len(pts[0])>=2, 'L(): needs a profile of 3D points')
    points = pts[0]
    cu = bpy.data.curves.new(str(name),'CURVE'); cu.dimensions='3D'
    sp = cu.splines.new('BEZIER'); sp.bezier_points.add(max(0,len(points)-1))
    for q,co in zip(sp.bezier_points,points): q.co=V3(co); q.handle_left_type='AUTO'; q.handle_right_type='AUTO'
    o = bpy.data.objects.new(str(name),cu); bpy.context.collection.objects.link(o)
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.context.view_layer.objects.active=o
    bpy.ops.object.convert(target='MESH')
    m = o.modifiers.new('Lathe','SCREW'); m.angle=6.283185307179586; m.steps=int(N(nums[0],48)) if nums else 48
    o.data.materials.append(MAT(strs[0] if strs else 'GroundMat'))
    return o

def ST(*a):
    global _yview_target
    nums, vecs, pts, strs = ARGS(a)
    cam = V3(vecs[0] if vecs else (7,-9,6)); target = V3(vecs[1] if len(vecs)>1 else (0,0,1))
    _yview_target = target
    # Modest plate; the runtime refits it to the real scene bounding box so
    # `look` never frames a 200 m floor with a 30 cm mouse lost in the middle.
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0,0,-0.02))
    P(bpy.context.active_object,'Ground','GroundMat',0.0,False)
    def area(n, l, e, s):
        d = bpy.data.lights.new(n,'AREA'); d.energy=e; d.shape='DISK'; d.size=s
        o = bpy.data.objects.new(n,d); bpy.context.collection.objects.link(o); o.location=l
        o.rotation_euler=(Vector((0,0,0))-o.location).to_track_quat('-Z','Y').to_euler()
    area('KeyLight',(4,-5,7),1100,5); area('FillLight',(-5,2,4),500,4); area('RimLight',(0,6,5),800,3)
    w = bpy.context.scene.world or bpy.data.worlds.new('Studio'); bpy.context.scene.world=w; w.use_nodes=True
    bg = next(x for x in w.node_tree.nodes if x.type=='BACKGROUND'); bg.inputs['Color'].default_value=(0.05,0.06,0.08,1.0); bg.inputs['Strength'].default_value=0.6
    d = bpy.data.cameras.new('StudioCamera'); c = bpy.data.objects.new('StudioCamera',d); bpy.context.collection.objects.link(c); c.location=cam
    c.rotation_euler=(Vector(target)-c.location).to_track_quat('-Z','Y').to_euler(); bpy.context.scene.camera=c
"""

GROUND_MATERIAL = "M('GroundMat',(0.12,0.13,0.15,1.0),0.85)"
FINISH = "print('Scene ready: %d objects, %d materials' % (len(bpy.context.scene.objects), len(bpy.data.materials)))"


# ---------------------------------------------------------------------------
# 4. Scene recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    prompts_ru: tuple[str, ...]
    prompts_en: tuple[str, ...]
    plan_ru: tuple[str, ...]
    plan_en: tuple[str, ...]
    summary_ru: str
    summary_en: str
    materials: tuple[str, ...]  # M(...) lines
    body: str                  # compact loop-based bpy code
    camera: tuple
    target: tuple

    def script(self) -> str:
        """The compact answer the model learns: marker + materials + body.

        ``PREAMBLE`` is prepended at runtime, so a scene with 40 parts is only
        ~1.8k chars of target text instead of ~6.2k.
        """
        code = "\n".join(
            [
                STUDIO_MARKER.rstrip("\n"),
                f"# {self.title}",
                GROUND_MATERIAL,
                *self.materials,
                self.body.strip(),
                f"ST({self.camera!r}, {self.target!r})",
                FINISH,
            ]
        ) + "\n"
        ast.parse(code)  # fail fast: never ship a syntactically broken sample
        return code

    def full_script(self) -> str:
        """Marker-expanded script for headless Blender validation."""
        return PREAMBLE + self.script()[len(STUDIO_MARKER) :]


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        key="mouse",
        title="Детализированная компьютерная мышь на деревянном столе",
        prompts_ru=(
            "сделай компьютерную мышку с проводом на столе",
            "создай детальную мышку с кнопками, колесиком и сплайновым проводом",
            "построй стол с компьютерной мышью и проводом",
            "смоделируй игровую мышь со светодиодом и ковриком",
        ),
        prompts_en=(
            "Create a detailed computer mouse with LMB, RMB, wheel and cable on a wooden desk.",
            "Model a gaming mouse with buttons, scroll wheel, LED and a braided cable.",
        ),
        plan_ru=(
            "1. Сделаю стол с фаской, четырьмя ножками и процедурным деревом.",
            "2. Соберу корпус мыши, раздельные ЛКМ/ПКМ, колесо с насечками, сенсор и светодиод.",
            "3. Добавлю сплайновый провод, разъём и коврик из резины.",
            "4. Поставлю три area-источника, камеру и землю, затем проверю сцену.",
        ),
        plan_en=(
            "1. Build a beveled wooden desk with four metal legs and a procedural wood shader.",
            "2. Model the mouse shell, separate LMB/RMB, a grooved wheel, sensor and LED.",
            "3. Add a Bezier cable, plug and rubber mouse pad.",
            "4. Add three area lights, a camera and ground, then verify the scene.",
        ),
        summary_ru="Готово: собрал мышь (~25 объектов) с раздельными кнопками, насеченным колесом, сплайновым проводом и PBR-материалами; проверил сцену через look.",
        summary_en="Done: built the mouse (~25 objects) with separate buttons, a grooved wheel, a Bezier cable and PBR materials; verified the scene with look.",
        materials=(
            "M('Wood',(0.36,0.20,0.10,1.0),0.45,0.0,'noise')",
            "M('Plastic',(0.05,0.05,0.06,1.0),0.35)",
            "M('Rubber',(0.02,0.02,0.025,1.0),0.8)",
            "M('Metal',(0.75,0.78,0.82,1.0),0.2,0.9)",
            "M('LED',(0.1,0.7,1.0,1.0),0.3,0.0,None,4.0)",
        ),
        body="""
B('DeskTop',(0,0,0.74),(3.2,2.0,0.06),'Wood',0.02)
for i,(x,y) in enumerate(((-1.45,-0.85),(1.45,-0.85),(-1.45,0.85),(1.45,0.85))): C(f'DeskLeg_{i}',0.035,0.72,(x,y,0.36),'Metal')
S('MouseBody',1.0,(0.25,0,0.82),'Plastic',(0.16,0.10,0.055))
for n,y in (('MouseLMB',-0.026),('MouseRMB',0.026)): B(n,(0.30,y,0.872),(0.10,0.042,0.008),'Plastic',0.006)
C('MouseWheel',0.016,0.012,(0.305,0,0.884),'Rubber',24,(1.5708,0,0))
for i in range(12): B(f'WheelGroove_{i}',(0.305,-0.008+i*0.0015,0.891),(0.004,0.001,0.002),'Metal')
B('MouseSensor',(0.25,0,0.768),(0.03,0.02,0.003),'Metal')
S('MouseLED',0.006,(0.36,0,0.865),'LED')
B('CablePlug',(0.41,0,0.80),(0.035,0.018,0.018),'Metal',0.004)
V('MouseCable',[(0.43,0,0.80),(0.62,0.12,0.78),(0.88,-0.08,0.77),(1.25,0.05,0.77)],0.006,'Rubber')
B('MousePad',(0.15,0,0.775),(0.55,0.40,0.004),'Rubber',0.01)
""",
        camera=(1.6, -2.2, 1.7),
        target=(0.2, 0, 0.8),
    ),
    Recipe(
        key="vase",
        title="Керамическая ваза вращением профиля",
        prompts_ru=(
            "создай вазу из сплайна",
            "сделай керамическую вазу вращением кривой",
            "построй изящную вазу с ручками и золотыми полосами",
        ),
        prompts_en=(
            "Create an elegant ceramic vase using a Bezier profile and a screw modifier.",
            "Build a glazed vase with gold rim, handles and decorative bands.",
        ),
        plan_ru=(
            "1. Построю профиль Безье и заверну его модификатором Screw в 48 шагов.",
            "2. Добавлю золотой ободок, поддон, две ручки-сплайна и восемь декоративных колец.",
            "3. Настрою глазурь через Noise → ColorRamp → Base Color и Bump → Normal.",
            "4. Поставлю студийный свет и камеру, затем проверю результат.",
        ),
        plan_en=(
            "1. Revolve a Bezier profile with a 48-step Screw modifier.",
            "2. Add a gold rim, foot, two spline handles and eight decorative bands.",
            "3. Set up a glazed shader: Noise → ColorRamp → Base Color, Bump → Normal.",
            "4. Add studio lights and a camera, then verify the result.",
        ),
        summary_ru="Готово: ваза собрана вращением профиля, дополнена ручками и золотыми кольцами, покрыта процедурной глазурью; проверил через look.",
        summary_en="Done: lathed the vase, added handles and gold bands, applied a procedural glaze; verified with look.",
        materials=(
            "M('Ceramic',(0.88,0.91,0.95,1.0),0.12,0.0,'noise')",
            "M('Gold',(1.0,0.766,0.336,1.0),0.18,1.0)",
        ),
        body="""
L('Vase',[(0.02,0,0.0),(0.42,0,0.22),(0.30,0,0.62),(0.16,0,1.05),(0.26,0,1.32)],'Ceramic')
T('VaseRim',0.26,0.018,(0,0,1.32),'Gold'); T('VaseFoot',0.20,0.02,(0,0,0.03),'Gold')
for s in (-1,1): V(f'VaseHandle_{s}',[(s*0.24,0,1.18),(s*0.52,0,1.05),(s*0.50,0,0.62),(s*0.30,0,0.55)],0.025,'Gold')
for i in range(8): T(f'VaseBand_{i}',0.30+0.01*i,0.004,(0,0,0.30+i*0.09),'Gold')
""",
        camera=(2.4, -3.2, 2.0),
        target=(0, 0, 0.7),
    ),
    Recipe(
        key="crab",
        title="Органический краб с процедурным хитином",
        prompts_ru=("создай краба", "создай краба с панцирем, лапками и клешнями", "сделай краба с детализированными текстурами"),
        prompts_en=("Create an organic crab model with procedural chitin texture.", "Model a crab with claws, legs and eye stalks."),
        plan_ru=(
            "1. Слеплю панцирь и нижнюю часть из сглаженных сфер.",
            "2. Добавлю по четыре пары ног с заострёнными кончиками, клешни и глаза на стебельках с каждой стороны.",
            "3. Соберу хитиновый PBR: Voronoi → ColorRamp → Base Color и Bump → Normal.",
            "4. Выставлю свет, камеру и землю, затем осмотрю сцену.",
        ),
        plan_en=(
            "1. Shape the carapace and underside from smoothed spheres.",
            "2. Add four leg pairs with tapered tips, claws and eye stalks on both sides.",
            "3. Build a chitin PBR: Voronoi → ColorRamp → Base Color and Bump → Normal.",
            "4. Set lights, camera and ground, then inspect the scene.",
        ),
        summary_ru="Готово: краб с 8 лапами, клешнями и глазами, процедурный хитин и студийный свет; проверил через look.",
        summary_en="Done: crab with 8 legs, claws and eyes, procedural chitin and studio lighting; verified with look.",
        materials=(
            "M('Chitin',(0.85,0.24,0.05,1.0),0.28,0.0,'voronoi')",
            "M('Claw',(0.95,0.42,0.08,1.0),0.3)",
            "M('Eye',(0.02,0.02,0.02,1.0),0.05)",
        ),
        body="""
S('Carapace',1.0,(0,0,0.62),'Chitin',(1.35,1.0,0.42)); S('Underside',1.0,(0,0,0.45),'Claw',(1.15,0.85,0.22))
for s in (-1,1):
    for i,y in enumerate((-0.55,-0.25,0.05,0.35)):
        C(f'LegUpper_{s}_{i}',0.055,0.85,(s*1.35,y,0.45),'Chitin',12,(0,s*0.75,0))
        K(f'LegTip_{s}_{i}',0.055,0.008,0.65,(s*1.95,y,0.16),'Chitin',12,(0,s*1.25,0))
    C(f'Arm_{s}',0.11,0.75,(s*1.05,-1.05,0.6),'Chitin',16,(0,0,s*0.7))
    S(f'ClawPalm_{s}',1.0,(s*1.55,-1.6,0.6),'Claw',(0.34,0.24,0.20))
    K(f'ClawFinger_{s}',0.13,0.02,0.48,(s*1.82,-1.95,0.6),'Claw',12,(1.5708,0,s*0.4))
    C(f'EyeStalk_{s}',0.025,0.22,(s*0.35,-0.85,0.92),'Chitin',12)
    S(f'EyeBall_{s}',0.055,(s*0.35,-0.9,1.05),'Eye')
""",
        camera=(4.5, -6.0, 4.0),
        target=(0, 0, 0.6),
    ),
    Recipe(
        key="stepped_pyramid",
        title="Ступенчатая пирамида с лестницей и факелами",
        prompts_ru=(
            "создай в блендере пирамиду ступенчатого типа с детализированными текстурами",
            "построй ступенчатую пирамиду Майя с лестницей",
            "сделай пирамиду с колоннами и храмом на вершине",
        ),
        prompts_en=("Create a stepped pyramid with procedural stone texture and a staircase.", "Build a Mayan pyramid with columns and a summit temple."),
        plan_ru=(
            "1. Сложу пять ярусов из блоков с фасками и процедурным песчаником.",
            "2. Добавлю 14 ступеней, четыре колонны с факелами и храм на вершине.",
            "3. Настрою камень через Noise + Bump и эмиссионный материал для огня.",
            "4. Поставлю свет и камеру, затем проверю сцену.",
        ),
        plan_en=(
            "1. Stack five beveled tiers with procedural sandstone.",
            "2. Add 14 stairs, four torch columns and a summit temple.",
            "3. Set up stone with Noise + Bump and an emissive flame material.",
            "4. Add lights and a camera, then verify the scene.",
        ),
        summary_ru="Готово: пять ярусов, лестница, колонны с факелами и храм, процедурный камень; проверил через look.",
        summary_en="Done: five tiers, staircase, torch columns and temple with procedural stone; verified with look.",
        materials=(
            "M('Sandstone',(0.78,0.65,0.46,1.0),0.9,0.0,'noise')",
            "M('Torch',(1.0,0.55,0.12,1.0),0.4,0.0,None,5.0)",
        ),
        body="""
for i,(size,h) in enumerate(((6.4,0.7),(5.2,1.4),(4.0,2.1),(2.8,2.8),(1.6,3.5))): B(f'Tier_{i}',(0,0,h/2),(size,size,h),'Sandstone',0.06)
for i in range(14): B(f'Stair_{i}',(-3.2+i*0.16,-3.35,0.18+i*0.25),(0.32,0.7,0.22+i*0.25),'Sandstone',0.02)
for x in (-2.4,2.4):
    for y in (-2.4,2.4):
        C(f'Column_{x}_{y}',0.16,1.6,(x,y,4.3),'Sandstone',16)
        S(f'Flame_{x}_{y}',0.14,(x,y,5.25),'Torch')
B('Temple',(0,0,4.15),(1.4,1.4,0.9),'Sandstone',0.05)
""",
        camera=(11, -13, 9),
        target=(0, 0, 2.2),
    ),
    Recipe(
        key="house",
        title="Дом с двускатной крышей и окружением",
        prompts_ru=("сделай домик", "создай домик с крышей, окнами и дверью", "построй дом с забором и дорожкой"),
        prompts_en=("Build a detailed house with roof, door, windows and a chimney.", "Create a cottage with a fence, path and framed windows."),
        plan_ru=(
            "1. Соберу фундамент, стены и двускатную крышу из трёхгранного цилиндра.",
            "2. Добавлю дверь с ручкой, два окна с рамами, дымоход и дорожку из плит.",
            "3. Поставлю забор из столбиков, процедурную штукатурку и стекло.",
            "4. Настрою свет, камеру и землю, затем осмотрю результат.",
        ),
        plan_en=(
            "1. Build the foundation, walls and a gable roof from a 3-vertex cylinder.",
            "2. Add a door with knob, two framed windows, a chimney and a stone path.",
            "3. Add a post fence, procedural plaster and glass materials.",
            "4. Set lights, camera and ground, then inspect the result.",
        ),
        summary_ru="Готово: дом с крышей, рамами окон, дымоходом, забором и дорожкой, процедурные материалы; проверил через look.",
        summary_en="Done: house with roof, window frames, chimney, fence and path using procedural materials; verified with look.",
        materials=(
            "M('Wall',(0.90,0.88,0.83,1.0),0.8,0.0,'noise')",
            "M('Roof',(0.55,0.14,0.10,1.0),0.6)",
            "M('Wood',(0.32,0.18,0.09,1.0),0.55)",
            "M('Glass',(0.45,0.72,0.85,1.0),0.05,0.1)",
            "M('Stone',(0.45,0.45,0.45,1.0),0.9)",
        ),
        body="""
B('Foundation',(0,0,0.15),(4.4,3.4,0.3),'Stone',0.03); B('Walls',(0,0,1.7),(4.0,3.0,3.0),'Wall',0.04)
C('Roof',2.35,4.4,(0,0,3.65),'Roof',3,(1.5708,0,0),False)
B('Door',(0,-1.53,0.95),(0.95,0.06,1.9),'Wood',0.02); S('DoorKnob',0.045,(0.32,-1.60,0.95),'Wood')
for i,x in enumerate((-1.25,1.25)):
    B(f'WindowGlass_{i}',(x,-1.54,1.95),(0.85,0.04,0.85),'Glass')
    for dx in (-0.47,0.47): B(f'WindowFrameV_{i}_{dx}',(x+dx,-1.57,1.95),(0.06,0.06,1.0),'Wood')
    for dz in (-0.47,0.47): B(f'WindowFrameH_{i}_{dz}',(x,-1.57,1.95+dz),(1.0,0.06,0.06),'Wood')
B('Chimney',(1.25,0.6,4.4),(0.55,0.55,1.4),'Stone',0.03)
for i in range(6): B(f'PathStone_{i}',(-1.6+i*0.55,-2.6,0.03),(0.45,0.45,0.06),'Stone',0.02)
for i,x in enumerate((-2.2,-1.5,-0.8,0.8,1.5,2.2)): B(f'FencePost_{i}',(x,-3.0,0.55),(0.08,0.08,1.1),'Wood')
B('FenceRail',(0,-3.0,0.8),(4.6,0.06,0.08),'Wood')
""",
        camera=(8, -10, 6),
        target=(0, 0, 1.8),
    ),
    Recipe(
        key="excavator",
        title="Гусеничный экскаватор",
        prompts_ru=("создай экскаватор", "сделай строительный экскаватор с гусеницами и стрелой"),
        prompts_en=("Create an excavator with chassis, tracks, cabin, boom and bucket.", "Model a tracked digger with hydraulic arm and bucket teeth."),
        plan_ru=(
            "1. Соберу шасси, две гусеницы из девяти траков и катки.",
            "2. Добавлю кабину со стёклами, двухзвенную стрелу, штифт и гидроцилиндр.",
            "3. Сделаю ковш с пятью зубьями и рабочие фары с эмиссией.",
            "4. Настрою жёлтый металл, свет и камеру, затем проверю сцену.",
        ),
        plan_en=(
            "1. Build the chassis, two nine-pad tracks and road wheels.",
            "2. Add a glazed cabin, two-part boom, pin and hydraulic cylinder.",
            "3. Model a bucket with five teeth and emissive work lamps.",
            "4. Set yellow metal, lights and camera, then verify the scene.",
        ),
        summary_ru="Готово: экскаватор с гусеницами, кабиной, стрелой и ковшом (~40 объектов); проверил через look.",
        summary_en="Done: excavator with tracks, cabin, boom and bucket (~40 objects); verified with look.",
        materials=(
            "M('Yellow',(0.95,0.72,0.04,1.0),0.35,0.5)",
            "M('Steel',(0.22,0.24,0.27,1.0),0.4,0.9)",
            "M('Rubber',(0.05,0.05,0.06,1.0),0.9)",
            "M('Glass',(0.35,0.6,0.75,1.0),0.08,0.2)",
            "M('Lamp',(1.0,0.95,0.7,1.0),0.3,0.0,None,3.0)",
        ),
        body="""
B('Chassis',(0,0,0.55),(2.8,1.7,0.5),'Yellow',0.06)
for s in (-1,1):
    B(f'TrackFrame_{s}',(0,s*0.95,0.38),(3.1,0.35,0.55),'Steel',0.05)
    for i in range(9): B(f'TrackPad_{s}_{i}',(-1.4+i*0.35,s*0.95,0.10),(0.30,0.42,0.08),'Rubber')
    for x in (-1.25,1.25): C(f'Wheel_{s}_{x}',0.28,0.36,(x,s*0.95,0.38),'Steel',20,(1.5708,0,0))
B('Cabin',(0.25,0,1.55),(1.7,1.35,1.5),'Yellow',0.08)
B('CabinGlassFront',(1.12,0,1.75),(0.05,1.1,1.0),'Glass'); B('CabinGlassSide',(0.25,-0.69,1.75),(1.4,0.05,1.0),'Glass')
B('BoomLower',(1.7,0,2.15),(2.4,0.32,0.38),'Yellow',0.04,(0,-0.55,0))
B('BoomUpper',(3.25,0,1.65),(2.0,0.26,0.32),'Yellow',0.04,(0,0.75,0))
C('BoomPin',0.14,0.5,(2.75,0,1.95),'Steel',24,(1.5708,0,0))
B('Bucket',(4.15,0,0.75),(0.9,1.0,0.8),'Steel',0.05,(0,0.3,0))
for i in range(5): K(f'BucketTooth_{i}',0.07,0.01,0.28,(4.55,-0.4+i*0.2,0.42),'Steel',8,(0,1.5708,0))
C('Hydraulic',0.07,1.5,(1.9,0,2.5),'Steel',16,(0,-0.5,0))
for i,x in enumerate((1.05,1.25)): S(f'WorkLamp_{i}',0.09,(x,-0.5,2.3),'Lamp')
""",
        camera=(8, -9, 6),
        target=(1.5, 0, 1.3),
    ),
    Recipe(
        key="coffee_cup",
        title="Чашка кофе с блюдцем и паром",
        prompts_ru=("создай чашку кофе", "сделай керамическую чашку с блюдцем, ручкой и ложкой"),
        prompts_en=("Model a ceramic coffee cup with saucer, handle and a spoon.", "Create a detailed cup of coffee with steam rings."),
        plan_ru=(
            "1. Построю чашку вращением профиля и добавлю ободок-тор.",
            "2. Сделаю поверхность кофе, блюдце, сплайновую ручку и серебряную ложку.",
            "3. Добавлю три кольца пара и тонкий фарфор с низкой шероховатостью.",
            "4. Поставлю свет и камеру, затем осмотрю сцену.",
        ),
        plan_en=(
            "1. Lathe the cup profile and add a torus rim.",
            "2. Add the coffee surface, saucer, spline handle and a silver spoon.",
            "3. Add three steam rings and a low-roughness porcelain shader.",
            "4. Set lights and camera, then inspect the scene.",
        ),
        summary_ru="Готово: чашка, блюдце, ручка-сплайн, ложка и пар, тонкий фарфор; проверил через look.",
        summary_en="Done: cup, saucer, spline handle, spoon and steam with fine porcelain; verified with look.",
        materials=(
            "M('Porcelain',(0.95,0.95,0.93,1.0),0.1)",
            "M('Coffee',(0.16,0.08,0.04,1.0),0.15)",
            "M('Silver',(0.85,0.87,0.9,1.0),0.15,1.0)",
        ),
        body="""
L('Cup',[(0.28,0,0.02),(0.32,0,0.10),(0.38,0,0.62),(0.40,0,0.66)],'Porcelain',64)
T('CupRim',0.40,0.012,(0,0,0.66),'Porcelain')
C('CoffeeSurface',0.37,0.01,(0,0,0.58),'Coffee',64,(0,0,0),False)
T('Saucer',0.55,0.035,(0,0,0.01),'Porcelain')
V('CupHandle',[(0.38,0,0.55),(0.62,0,0.52),(0.62,0,0.22),(0.34,0,0.18)],0.035,'Porcelain')
C('SpoonHandle',0.012,0.55,(0.62,0.1,0.06),'Silver',16,(0,1.5708,0))
S('SpoonBowl',1.0,(0.90,0.1,0.06),'Silver',(0.09,0.06,0.02))
for i in range(3): T(f'Steam_{i}',0.10+i*0.04,0.006,(0.02*i,0,0.85+i*0.22),'Porcelain')
""",
        camera=(1.8, -2.4, 1.6),
        target=(0.1, 0, 0.4),
    ),
    Recipe(
        key="desk_lamp",
        title="Настольная лампа с шарнирами",
        prompts_ru=("создай настольную лампу", "сделай лампу с шарнирной ножкой и абажуром"),
        prompts_en=("Build an articulated desk lamp with a brass arm and a fabric shade.", "Model a detailed desk lamp with a glowing bulb and power cable."),
        plan_ru=(
            "1. Сделаю массивное латунное основание с кольцом и выключателем.",
            "2. Соберу два рычага, два шарнира-сферы и абажур-конус.",
            "3. Добавлю эмиссионную лампочку, сплайновый кабель и PBR-металл.",
            "4. Настрою студийный свет и камеру, затем проверю результат.",
        ),
        plan_en=(
            "1. Build a heavy brass base with a ring and switch.",
            "2. Add two arms, two spherical joints and a cone shade.",
            "3. Add an emissive bulb, a Bezier power cable and PBR metal.",
            "4. Set studio lights and camera, then verify the result.",
        ),
        summary_ru="Готово: лампа с шарнирами, абажуром, эмиссионной лампочкой и кабелем; проверил через look.",
        summary_en="Done: lamp with joints, shade, emissive bulb and cable; verified with look.",
        materials=(
            "M('Brass',(0.72,0.52,0.20,1.0),0.25,0.9)",
            "M('Shade',(0.12,0.30,0.45,1.0),0.4,0.3)",
            "M('Bulb',(1.0,0.92,0.72,1.0),0.3,0.0,None,6.0)",
            "M('Cable',(0.05,0.05,0.05,1.0),0.8)",
        ),
        body="""
C('LampBase',0.34,0.06,(0,0,0.03),'Brass',48); T('BaseRing',0.30,0.015,(0,0,0.07),'Brass')
C('LowerArm',0.035,1.2,(0,0,0.65),'Brass',16,(0,0.18,0)); S('JointLower',0.075,(0.11,0,1.24),'Brass')
C('UpperArm',0.03,1.0,(0.42,0,1.55),'Brass',16,(0,0,-0.9)); S('JointUpper',0.07,(0.85,0,1.85),'Brass')
K('LampShade',0.42,0.18,0.45,(1.02,0,1.72),'Shade',48,(0,-0.4,0))
S('Bulb',0.12,(1.12,0,1.55),'Bulb')
C('Switch',0.05,0.03,(0.22,0,0.09),'Brass',24)
V('PowerCable',[(-0.3,0,0.04),(-0.6,0.2,0.02),(-1.0,0.5,0.02),(-1.6,0.6,0.02)],0.008,'Cable')
""",
        camera=(3.0, -4.0, 2.8),
        target=(0.5, 0, 1.1),
    ),
    Recipe(
        key="watch",
        title="Наручные часы с циферблатом",
        prompts_ru=("создай наручные часы", "сделай часы с циферблатом, стрелками и ремешком"),
        prompts_en=("Model a wristwatch with a dial, hour marks, hands and a leather strap.", "Create a detailed analog watch with a metallic case."),
        plan_ru=(
            "1. Соберу металлический корпус, циферблат и 12 часовых меток по кругу.",
            "2. Добавлю часовую и минутную стрелки и заводную головку.",
            "3. Сделаю ремешок из 12 сегментов с процедурной кожей.",
            "4. Поставлю свет, камеру и землю, затем осмотрю сцену.",
        ),
        plan_en=(
            "1. Build a metal case, dial and 12 radial hour marks.",
            "2. Add hour/minute hands and a winding crown.",
            "3. Model a 12-segment strap with procedural leather.",
            "4. Set lights, camera and ground, then inspect the scene.",
        ),
        summary_ru="Готово: часы с 12 метками, стрелками, головкой и сегментированным ремешком; проверил через look.",
        summary_en="Done: watch with 12 marks, hands, crown and a segmented strap; verified with look.",
        materials=(
            "M('Case',(0.80,0.82,0.85,1.0),0.18,1.0)",
            "M('Dial',(0.06,0.07,0.09,1.0),0.5)",
            "M('Mark',(0.95,0.85,0.45,1.0),0.2,0.8)",
            "M('Strap',(0.16,0.10,0.07,1.0),0.7,0.0,'noise')",
        ),
        body="""
C('WatchCase',0.62,0.16,(0,0,0.12),'Case',64,(1.5708,0,0))
C('WatchDial',0.54,0.02,(0,0.09,0.12),'Dial',64,(1.5708,0,0))
for i in range(12):
    a = i*0.5235987755982988
    B(f'HourMark_{i}',(0.44*math.sin(a),0.105,0.12+0.44*math.cos(a)),(0.035,0.01,0.10),'Mark',0.0,(0,-a,0))
B('HourHand',(0.0,0.12,0.22),(0.035,0.01,0.30),'Mark'); B('MinuteHand',(0.16,0.13,0.12),(0.36,0.01,0.025),'Mark')
C('Crown',0.07,0.10,(0.66,0,0.12),'Case',24,(0,1.5708,0))
for s in (-1,1):
    for i in range(6): B(f'Strap_{s}_{i}',(0,0,s*(0.72+i*0.22)),(0.48,0.05,0.20),'Strap',0.02)
""",
        camera=(1.8, -2.4, 2.0),
        target=(0, 0, 0.1),
    ),
    Recipe(
        key="robot",
        title="Детализированный человекоподобный робот",
        prompts_ru=("создай робота", "сделай человекоподобного робота с суставами и визором"),
        prompts_en=("Build a humanoid robot with joints, a visor and a chest panel.", "Model a detailed robot with articulated arms and legs."),
        plan_ru=(
            "1. Соберу торсо с фасками, панель груди и три индикатора.",
            "2. Добавлю голову с визором, глазами и антеннами.",
            "3. Смоделирую руки и ноги через сферические суставы, рюкзак и ступни.",
            "4. Настрою металлические PBR, свет и камеру, затем проверю сцену.",
        ),
        plan_en=(
            "1. Build a beveled torso with a chest panel and three LEDs.",
            "2. Add a head with visor, eyes and antennas.",
            "3. Model arms and legs with spherical joints, a backpack and feet.",
            "4. Set metallic PBR, lights and camera, then verify the scene.",
        ),
        summary_ru="Готово: робот (~35 объектов) с суставами, визором, панелью и рюкзаком; проверил через look.",
        summary_en="Done: robot (~35 objects) with joints, visor, chest panel and backpack; verified with look.",
        materials=(
            "M('Shell',(0.82,0.84,0.88,1.0),0.3,0.6)",
            "M('Joint',(0.15,0.17,0.20,1.0),0.5,0.8)",
            "M('Visor',(0.05,0.35,0.55,1.0),0.08,0.4)",
            "M('Accent',(0.95,0.35,0.08,1.0),0.35,0.4)",
        ),
        body="""
B('Torso',(0,0,1.55),(0.95,0.55,1.0),'Shell',0.10); B('ChestPanel',(0,-0.29,1.65),(0.6,0.03,0.45),'Visor',0.03)
for i in range(3): S(f'ChestLED_{i}',0.035,(-0.18+i*0.18,-0.32,1.82),'Accent')
B('Head',(0,0,2.35),(0.62,0.52,0.5),'Shell',0.10); B('Visor',(0,-0.27,2.38),(0.48,0.03,0.20),'Visor',0.02)
for s in (-1,1):
    S(f'Eye_{s}',0.055,(s*0.14,-0.30,2.40),'Accent')
    C(f'Antenna_{s}',0.012,0.35,(s*0.22,0,2.78),'Joint',12)
    S(f'Shoulder_{s}',0.18,(s*0.66,0,1.9),'Joint')
    C(f'UpperArm_{s}',0.11,0.65,(s*0.72,0,1.55),'Shell',20)
    S(f'Elbow_{s}',0.13,(s*0.75,0,1.2),'Joint')
    C(f'Forearm_{s}',0.10,0.6,(s*0.78,0,0.9),'Shell',20)
    B(f'Hand_{s}',(s*0.80,0,0.55),(0.22,0.16,0.18),'Joint',0.03)
    S(f'Hip_{s}',0.16,(s*0.28,0,1.0),'Joint')
    C(f'Thigh_{s}',0.14,0.55,(s*0.30,0,0.7),'Shell',20)
    S(f'Knee_{s}',0.14,(s*0.30,0,0.42),'Joint')
    C(f'Shin_{s}',0.12,0.4,(s*0.30,0,0.22),'Shell',20)
    B(f'Foot_{s}',(s*0.30,-0.12,0.06),(0.28,0.45,0.12),'Joint',0.03)
B('Backpack',(0,0.35,1.6),(0.7,0.22,0.8),'Joint',0.05)
""",
        camera=(3.6, -5.0, 3.4),
        target=(0, 0, 1.3),
    ),
    Recipe(
        key="car",
        title="Стилизованный автомобиль",
        prompts_ru=("создай машину", "сделай автомобиль с колёсами, фарами и окнами"),
        prompts_en=("Create a stylized car with wheels, headlights and windows.", "Model a detailed low-poly car with rims and a grille."),
        plan_ru=(
            "1. Соберу кузов с фасками, кабину, лобовое и боковые стёкла.",
            "2. Добавлю четыре колеса с дисками и спицами.",
            "3. Сделаю фары с эмиссией, решётку радиатора и шесть ламелей.",
            "4. Настрою металлик-краску, свет и камеру, затем осмотрю результат.",
        ),
        plan_en=(
            "1. Build a beveled body, cabin, windshield and side windows.",
            "2. Add four wheels with rims and spokes.",
            "3. Add emissive headlights, a grille and six slats.",
            "4. Set metallic paint, lights and camera, then inspect the result.",
        ),
        summary_ru="Готово: автомобиль с колёсами, дисками, фарами и решёткой (~30 объектов); проверил через look.",
        summary_en="Done: car with wheels, rims, headlights and grille (~30 objects); verified with look.",
        materials=(
            "M('Paint',(0.65,0.08,0.10,1.0),0.22,0.5)",
            "M('Tire',(0.04,0.04,0.05,1.0),0.9)",
            "M('Rim',(0.80,0.82,0.85,1.0),0.2,1.0)",
            "M('Glass',(0.25,0.40,0.55,1.0),0.06,0.3)",
            "M('Headlight',(1.0,0.96,0.82,1.0),0.2,0.0,None,4.0)",
            "M('Taillight',(0.85,0.05,0.04,1.0),0.3,0.0,None,2.0)",
        ),
        body="""
B('Body',(0,0,0.62),(4.2,1.8,0.65),'Paint',0.16); B('Cabin',(-0.25,0,1.15),(2.1,1.6,0.65),'Paint',0.18)
B('Windshield',(0.82,0,1.22),(0.06,1.45,0.55),'Glass',0.0,(0,-0.35,0))
B('RearWindow',(-1.32,0,1.22),(0.06,1.45,0.55),'Glass',0.0,(0,0.35,0))
for s in (-1,1):
    B(f'SideWindow_{s}',(-0.25,s*0.82,1.22),(1.9,0.04,0.5),'Glass')
    for x in (-1.35,1.35):
        C(f'Wheel_{s}_{x}',0.42,0.28,(x,s*0.92,0.42),'Tire',32,(1.5708,0,0))
        C(f'Hub_{s}_{x}',0.24,0.30,(x,s*0.94,0.42),'Rim',24,(1.5708,0,0))
        for i in range(5): B(f'Spoke_{s}_{x}_{i}',(x,s*0.96,0.42),(0.06,0.02,0.42),'Rim',0.0,(i*0.628,0,0))
    S(f'Headlight_{s}',0.16,(2.12,s*0.58,0.68),'Headlight',(0.4,1.0,0.7))
    B(f'Taillight_{s}',(-2.12,s*0.62,0.68),(0.05,0.35,0.16),'Taillight')
B('Grille',(2.13,0,0.55),(0.05,1.0,0.22),'Rim',0.02)
for i in range(6): B(f'GrilleSlat_{i}',(2.17,-0.42+i*0.17,0.55),(0.02,0.06,0.20),'Rim')
""",
        camera=(7.5, -8.5, 4.5),
        target=(0, 0, 0.8),
    ),
)

# ---------------------------------------------------------------------------
# 5. Sample assembly
# ---------------------------------------------------------------------------


def tool_call(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def wrap_code_mode(name: str, arguments: dict) -> tuple[str, dict]:
    """Translate a direct MCP call into an OpenCode Code Mode ``execute`` call."""
    if name == "execute_blender_code":
        js = f"await tools.blender.execute_blender_code({{ code: {json.dumps(arguments['code'], ensure_ascii=False)} }})"
    elif name == "look":
        js = f"await tools.blender.look({json.dumps(arguments, ensure_ascii=False)})"
    elif name == "get_scene_info":
        js = "await tools.blender.get_scene_info({})"
    elif name == "get_addon_status":
        js = "await tools.blender.get_addon_status({})"
    else:
        raise ValueError(f"no Code Mode mapping for tool {name!r}")
    return "execute", {"code": js}


def call(name: str, arguments: dict, code_mode: bool) -> str:
    if code_mode:
        name, arguments = wrap_code_mode(name, arguments)
    return tool_call(name, arguments)


def message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def chat(system: str, *turns: dict) -> dict:
    return {"messages": [message("system", system), *turns]}


def single_turn_sample(system: str, prompt: str, plan: str, tool_name: str, args: dict, code_mode: bool) -> dict:
    reply = f"{plan}\n\n{call(tool_name, args, code_mode)}"
    return chat(system, message("user", prompt), message("assistant", reply))


def multi_turn_sample(
    system: str,
    prompt: str,
    plan: str,
    code: str,
    summary: str,
    code_mode: bool,
    lang: str,
) -> dict:
    """plan + call -> tool response -> verification call -> response -> summary."""
    verify_text = "Проверяю результат во вьюпорте." if lang == "ru" else "Verifying the result in the viewport."
    verify_name, verify_args = "look", {"mode": "viewport", "shading": "solid"}
    ok_response = "<tool_response>\nCode executed successfully\nScene ready: 32 objects, 6 materials\n</tool_response>"
    look_response = "<tool_response>\n[viewport image: 768x512, shading=solid]\n</tool_response>"
    return chat(
        system,
        message("user", prompt),
        message("assistant", f"{plan}\n\n{call('execute_blender_code', {'code': code}, code_mode)}"),
        message("user", ok_response),
        message("assistant", f"{verify_text}\n\n{call(verify_name, verify_args, code_mode)}"),
        message("user", look_response),
        message("assistant", summary),
    )


def plan_text(recipe: Recipe, lang: str) -> str:
    return "\n".join(recipe.plan_ru if lang == "ru" else recipe.plan_en)


def recipe_samples(recipe: Recipe) -> tuple[list[dict], str]:
    code = recipe.script()
    prompts: list[tuple[str, str]] = [(p, "ru") for p in recipe.prompts_ru] + [
        (p, "en") for p in recipe.prompts_en
    ]
    samples: list[dict] = []
    for i, (prompt, lang) in enumerate(prompts):
        # Alternate formats: the model learns to condition on the tools listed
        # in the system prompt (direct MCP vs OpenCode Code Mode).
        code_mode = i % 2 == 1
        system = SYSTEM_CODE_MODE if code_mode else SYSTEM_DIRECT
        samples.append(
            single_turn_sample(system, prompt, plan_text(recipe, lang), "execute_blender_code", {"code": code}, code_mode)
        )

    # One full agentic trajectory per recipe in each output format.
    samples.append(
        multi_turn_sample(SYSTEM_DIRECT, recipe.prompts_ru[0], plan_text(recipe, "ru"),
                          code, recipe.summary_ru, False, "ru")
    )
    samples.append(
        multi_turn_sample(SYSTEM_CODE_MODE, recipe.prompts_en[0], plan_text(recipe, "en"),
                          code, recipe.summary_en, True, "en")
    )
    return samples, code


UTILITY_CALLS: tuple[tuple[str, dict, tuple[str, ...], str, str], ...] = (
    (
        "get_scene_info",
        {},
        ("Какая информация о сцене?", "Что сейчас находится в сцене?", "What objects are in the scene?"),
        "Сначала запрошу текстовый отчёт по иерархии сцены.",
        "First I will request a text report of the scene hierarchy.",
    ),
    (
        "look",
        {"mode": "viewport", "shading": "solid"},
        ("Посмотри на сцену во вьюпорте.", "Покажи сцену.", "Look at the viewport."),
        "Открою вьюпорт, чтобы увидеть сцену целиком.",
        "I will open the viewport to see the whole scene.",
    ),
    (
        "look",
        {"mode": "camera", "shading": "rendered"},
        ("Покажи сцену через камеру.", "Show the camera view."),
        "Сделаю вид из камеры в режиме рендера.",
        "I will take a rendered view through the scene camera.",
    ),
    (
        "get_addon_status",
        {},
        ("Проверь статус аддона mcp-for-blender.", "Check blender addon status."),
        "Проверю версию аддона и доступные библиотеки.",
        "I will check the addon version and available libraries.",
    ),
)


def inspection_trajectory(name: str, args: dict, plan: str, response: str, summary: str, code_mode: bool) -> dict:
    """A read-only request: plan -> one call -> tool response -> final summary.

    Without these the model keeps calling tools after a successful `look`
    instead of finishing the turn (observed in the live smoke test).
    """
    system = SYSTEM_CODE_MODE if code_mode else SYSTEM_DIRECT
    return chat(
        system,
        message("user", "Посмотри на сцену во вьюпорте." if name == "look" else "Что в сцене?"),
        message("assistant", f"{plan}\n\n{call(name, args, code_mode)}"),
        message("user", response),
        message("assistant", summary),
    )


def utility_samples() -> list[dict]:
    samples: list[dict] = []
    for name, args, prompts, plan_ru, plan_en in UTILITY_CALLS:
        for prompt in prompts:
            lang = "en" if prompt.isascii() else "ru"
            plan = plan_en if lang == "en" else plan_ru
            for code_mode in (False, True):
                system = SYSTEM_CODE_MODE if code_mode else SYSTEM_DIRECT
                samples.append(single_turn_sample(system, prompt, plan, name, args, code_mode))
    # Agentic trajectories that END with a summary and no further tool calls.
    look_response = "<tool_response>\n[image: image/png, viewport 768x512]\n</tool_response>"
    info_response = "<tool_response>\nScene: 3 objects, 2 materials, camera: StudioCamera\n</tool_response>"
    for code_mode in (False, True):
        samples.append(
            inspection_trajectory(
                "look",
                {"mode": "viewport"},
                "Открою вьюпорт, чтобы увидеть сцену целиком.",
                look_response,
                "Готово: посмотрел сцену во вьюпорте, вижу 3 объекта и студийную камеру. Новых вызовов не делаю.",
                code_mode,
            )
        )
        samples.append(
            inspection_trajectory(
                "get_scene_info",
                {},
                "Сначала запрошу текстовую сводку по объектам.",
                info_response,
                "В сцене 3 объекта и 2 материала, активна студийная камера. На этом завершаю.",
                code_mode,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------


def check_in_blender(scripts: dict[str, str], limit: int) -> None:
    """Run generated scripts in headless Blender (the live scene is untouched)."""
    for key, code in list(scripts.items())[:limit]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(code)
            path = Path(fh.name)
        try:
            proc = subprocess.run(
                ["blender", "--background", "--factory-startup", "--python", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            ok = proc.returncode == 0 and "Traceback" not in proc.stderr
            print(f"  blender[{key}]: {'OK' if ok else 'FAILED'}")
            if not ok:
                print(proc.stderr[-1200:])
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"  blender[{key}]: skipped ({exc})")
        finally:
            path.unlink(missing_ok=True)


def quoted_identifiers(body: str) -> set[str]:
    """Object/material names used by a recipe — the model reproduces those
    even when it garbles coordinates, so they are reliable match keys."""
    return set(re.findall(r"['\"]([A-Za-z][A-Za-z0-9_]+)['\"]", body))


def write_recipe_library() -> None:
    """Verified recipe bodies + distinctive names for runtime body repair.

    A 1.5B model cannot reliably emit correct coordinates for arbitrary
    geometry. Instead the Rust runtime matches the names appearing in the
    generated body against these keys and, on a confident match, swaps in the
    verified body. See ``crates/yesman-model/src/mcp.rs``.
    """
    bodies = {r.key: r.script()[len(STUDIO_MARKER) :] for r in RECIPES}
    identifiers = {k: quoted_identifiers(body) for k, body in bodies.items()}
    frequency = Counter(name for names in identifiers.values() for name in names)
    payload = []
    for recipe in RECIPES:
        distinctive = sorted(
            name.lower()
            for name in identifiers[recipe.key]
            if frequency[name] <= 2 and len(name) >= 4
        )
        payload.append(
            {
                "key": recipe.key,
                "title": recipe.title,
                "keywords": distinctive,
                "body": bodies[recipe.key],
            }
        )
    RECIPES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {RECIPES_PATH} ({len(payload)} verified recipe bodies)")


def report(samples: list[dict], scripts: dict[str, str]) -> None:
    assistant_turns = [m["content"] for s in samples for m in s["messages"] if m["role"] == "assistant"]
    code_chars = [len(c) for c in scripts.values()]
    print(f"  recipes            : {len(RECIPES)}")
    print(f"  unique bpy scripts : {len(scripts)}")
    print(f"  full script (chars) : min={min(code_chars)} max={max(code_chars)} avg={sum(code_chars)//len(code_chars)}")
    emitted = [len(recipe.script()) for recipe in RECIPES]
    print(f"  model-emitted (chars): min={min(emitted)} max={max(emitted)} avg={sum(emitted)//len(emitted)} (marker + body)")
    print(f"  samples            : {len(samples)}")
    print(f"  assistant turns    : {len(assistant_turns)} (avg {sum(map(len, assistant_turns))//len(assistant_turns)} chars)")
    print(f"  multi-step samples : {sum(1 for s in samples if len(s['messages']) > 3)}")
    print(f"  code-mode samples  : {sum(1 for s in samples if s['messages'][0]['content'] == SYSTEM_CODE_MODE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--check-blender", action="store_true", help="validate scripts in headless Blender")
    parser.add_argument("--blender-limit", type=int, default=3)
    parser.add_argument("--stats-only", action="store_true", help="print stats without writing files")
    args = parser.parse_args()

    random.seed(args.seed)
    scripts: dict[str, str] = {}
    samples: list[dict] = []
    for recipe in RECIPES:
        recipe_rows, _ = recipe_samples(recipe)
        scripts[recipe.key] = recipe.full_script()  # expanded, for headless validation
        samples.extend(recipe_rows)
    samples.extend(utility_samples())

    print("Dataset summary")
    report(samples, scripts)

    if args.check_blender:
        print("Headless Blender validation")
        check_in_blender(scripts, args.blender_limit)

    if args.stats_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(samples)
    val_count = max(8, int(len(samples) * args.val_ratio))
    val_rows, train_rows = samples[:val_count], samples[val_count:]

    for filename, rows in (("train.jsonl", train_rows), ("val.jsonl", val_rows)):
        path = args.out_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(rows)} rows)")

    # Rules only: the Rust CLI/server append the live tool schemas themselves.
    prompt_path = ROOT / "system_prompt.txt"
    prompt_path.write_text(RULES + "\n", encoding="utf-8")
    print(f"  wrote {prompt_path} (workflow rules, {len(RULES)} chars)")

    # Single source of truth for the runtime helper library.
    PREAMBLE_PATH.write_text(PREAMBLE, encoding="utf-8")
    print(f"  wrote {PREAMBLE_PATH} (studio helpers, {len(PREAMBLE)} chars)")
    write_recipe_library()


if __name__ == "__main__":
    main()
