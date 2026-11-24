#!/usr/bin/env python3
"""Dataset preparation script for fine-tuning Qwen2.5 on Blender MCP workflows.

Fetches open 3D/Blender datasets from Hugging Face and synthesizes
tool-calling pairs for Blender MCP (execute_blender_code, get_scene_info, etc.).
Outputs train.jsonl and val.jsonl in Qwen2.5 ChatML format.
"""

import json
import os
import random
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """Ты — добрый, простой и отзывчивый помощник для работы в Blender. Общайся тепло, дружелюбно и живо, без заумных слов. Используй доступные инструменты Blender для выполнения запросов пользователя.

# Tools

You may call one or more functions to assist with the user specification.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "execute_blender_code", "description": "Execute arbitrary Python code in Blender via the bpy API", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Executable Python code using bpy"}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "get_scene_info", "description": "Get detailed information about objects, cameras, lights, and materials in the current Blender scene", "parameters": {"type": "object", "properties": {}}}}
{"type": "function", "function": {"name": "get_object_info", "description": "Get detailed information about a specific object in the scene", "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}}, "required": ["object_name"]}}}
{"type": "function", "function": {"name": "get_viewport_screenshot", "description": "Capture a screenshot of the 3D viewport", "parameters": {"type": "object", "properties": {"max_size": {"type": "integer"}}}}}
{"type": "function", "function": {"name": "set_texture", "description": "Apply a texture to an object", "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}, "texture_id": {"type": "string"}}, "required": ["object_name", "texture_id"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# Curated synthetic examples for core Blender operations
SYNTHETIC_TASKS = [
    {
        "user_ru": "Создай красную обезьянку Сюзанну в центре сцены с гладким шейдингом.",
        "user_en": "Create a red Suzanne monkey head at the center with smooth shading.",
        "code": (
            "import bpy\n\n"
            "# Создаем обезьянку Сюзанну\n"
            "bpy.ops.mesh.primitive_monkey_add(size=2, location=(0, 0, 0))\n"
            "obj = bpy.context.active_object\n"
            "obj.name = 'Suzanne'\n"
            "bpy.ops.object.shade_smooth()\n\n"
            "# Назначаем красный материал\n"
            "mat = bpy.data.materials.new(name='RedMaterial')\n"
            "mat.use_nodes = True\n"
            "bsdf = mat.node_tree.nodes.get('Principled BSDF')\n"
            "if bsdf:\n"
            "    bsdf.inputs['Base Color'].default_value = (1.0, 0.05, 0.05, 1.0)\n"
            "    bsdf.inputs['Roughness'].default_value = 0.3\n\n"
            "if not obj.data.materials:\n"
            "    obj.data.materials.append(mat)\n"
            "else:\n"
            "    obj.data.materials[0] = mat\n"
        ),
    },
    {
        "user_ru": "Посмотри, какие объекты сейчас находятся в сцене Blender.",
        "user_en": "Check what objects are currently in the Blender scene.",
        "tool_call": {"name": "get_scene_info", "arguments": {}},
    },
    {
        "user_ru": "Очисти сцену от всех мешей и добавь золотой торус (бублик).",
        "user_en": "Clear all meshes from the scene and add a golden torus.",
        "code": (
            "import bpy\n\n"
            "# Удаляем все меш-объекты\n"
            "bpy.ops.object.select_all(action='DESELECT')\n"
            "for obj in bpy.context.scene.objects:\n"
            "    if obj.type == 'MESH':\n"
            "        obj.select_set(True)\n"
            "bpy.ops.object.delete()\n\n"
            "# Создаем торус\n"
            "bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.5, location=(0, 0, 1))\n"
            "torus = bpy.context.active_object\n"
            "bpy.ops.object.shade_smooth()\n\n"
            "# Создаем материал золота\n"
            "gold_mat = bpy.data.materials.new(name='Gold')\n"
            "gold_mat.use_nodes = True\n"
            "bsdf = gold_mat.node_tree.nodes.get('Principled BSDF')\n"
            "if bsdf:\n"
            "    bsdf.inputs['Base Color'].default_value = (1.0, 0.766, 0.336, 1.0)\n"
            "    bsdf.inputs['Metallic'].default_value = 1.0\n"
            "    bsdf.inputs['Roughness'].default_value = 0.15\n\n"
            "torus.data.materials.append(gold_mat)\n"
        ),
    },
    {
        "user_ru": "Добавь источник света Солнце силой 5 и направь камеру на объект в центре.",
        "user_en": "Add a Sun light with strength 5 and point the camera at the center.",
        "code": (
            "import bpy\nimport math\n\n"
            "# Создаем солнце\n"
            "light_data = bpy.data.lights.new(name='SunLight', type='SUN')\n"
            "light_data.energy = 5.0\n"
            "light_data.color = (1.0, 0.95, 0.9)\n"
            "light_obj = bpy.data.objects.new(name='SunLight', object_data=light_data)\n"
            "bpy.context.collection.objects.link(light_obj)\n"
            "light_obj.location = (5, -5, 10)\n"
            "light_obj.rotation_euler = (math.radians(45), math.radians(20), math.radians(45))\n"
        ),
    },
    {
        "user_ru": "Сделай снимок вьюпорта (скриншот).",
        "user_en": "Take a screenshot of the 3D viewport.",
        "tool_call": {"name": "get_viewport_screenshot", "arguments": {"max_size": 800}},
    },
    {
        "user_ru": "Создай низкополигональное дерево: ствол из коричневого цилиндра и крону из трех зеленых икосфер.",
        "user_en": "Create a low-poly tree: a brown cylinder trunk and 3 green icospheres for foliage.",
        "code": (
            "import bpy\n\n"
            "# Ствол\n"
            "bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=2, location=(0, 0, 1))\n"
            "trunk = bpy.context.active_object\n"
            "trunk.name = 'TreeTrunk'\n"
            "mat_trunk = bpy.data.materials.new('WoodBrown')\n"
            "mat_trunk.use_nodes = True\n"
            "mat_trunk.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.2, 0.1, 0.05, 1.0)\n"
            "trunk.data.materials.append(mat_trunk)\n\n"
            "# Крона (3 яруса)\n"
            "mat_leaves = bpy.data.materials.new('FoliageGreen')\n"
            "mat_leaves.use_nodes = True\n"
            "mat_leaves.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.1, 0.6, 0.15, 1.0)\n\n"
            "for i, (z, r) in enumerate([(2.2, 1.3), (3.0, 1.0), (3.6, 0.7)]):\n"
            "    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=r, location=(0, 0, z))\n"
            "    leaf = bpy.context.active_object\n"
            "    leaf.name = f'Foliage_{i}'\n"
            "    leaf.data.materials.append(mat_leaves)\n"
        ),
    },
    {
        "user_ru": "Примени модификатор Bevel (фаска) к кубу с именем Cube с шириной 0.1 и 3 сегментами.",
        "user_en": "Apply a Bevel modifier to object 'Cube' with width 0.1 and 3 segments.",
        "code": (
            "import bpy\n\n"
            "obj = bpy.data.objects.get('Cube')\n"
            "if obj:\n"
            "    mod = obj.modifiers.new(name='Bevel', type='BEVEL')\n"
            "    mod.width = 0.1\n"
            "    mod.segments = 3\n"
            "    bpy.context.view_layer.objects.active = obj\n"
            "    bpy.ops.object.modifier_apply(modifier='Bevel')\n"
        ),
    },
    {
        "user_ru": "Создай неоновую светящуюся вывеску (текст 'YESMAN') бирюзового цвета.",
        "user_en": "Create an emission neon sign with text 'YESMAN' in cyan color.",
        "code": (
            "import bpy\n\n"
            "# Добавляем 3D-текст\n"
            "bpy.ops.object.text_add(location=(0, 0, 1))\n"
            "text_obj = bpy.context.active_object\n"
            "text_obj.data.body = 'YESMAN'\n"
            "text_obj.data.extrude = 0.15\n"
            "text_obj.data.bevel_depth = 0.02\n"
            "text_obj.rotation_euler = (1.5708, 0, 0)\n\n"
            "# Материал свечения (Emission)\n"
            "mat = bpy.data.materials.new(name='NeonCyan')\n"
            "mat.use_nodes = True\n"
            "nodes = mat.node_tree.nodes\n"
            "nodes.clear()\n"
            "emission = nodes.new('ShaderNodeEmission')\n"
            "emission.inputs['Color'].default_value = (0.0, 0.9, 1.0, 1.0)\n"
            "emission.inputs['Strength'].default_value = 10.0\n"
            "output = nodes.new('ShaderNodeOutputMaterial')\n"
            "mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])\n"
            "text_obj.data.materials.append(mat)\n"
        ),
    },
]


def make_chatml_sample(user_msg: str, tool_name: str, tool_args: dict) -> dict:
    call_json = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
    assistant_content = f"<tool_call>\n{call_json}\n</tool_call>"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def fetch_hf_dioramas(limit: int = 500) -> list[dict]:
    """Fetch verified Blender bpy scripts from ShiminZ/blender-lowpoly-dioramas via HF API."""
    print(f"Fetching up to {limit} samples from ShiminZ/blender-lowpoly-dioramas...")
    samples = []
    chunk_size = 100
    offset = 0

    while offset < limit:
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=ShiminZ/blender-lowpoly-dioramas&config=default&split=train&offset={offset}&limit={chunk_size}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "yesman-ai"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r.get("row", {})
                instr = row.get("instruction", "").strip()
                script = row.get("script", "").strip()
                if instr and script:
                    samples.append((instr, script))
            offset += len(rows)
            print(f"  downloaded {len(samples)} samples...")
        except Exception as e:
            print(f"  warning: failed to fetch chunk at offset {offset}: {e}")
            break

    print(f"Total HF diorama samples fetched: {len(samples)}")
    return samples


def main():
    out_dir = Path("training/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 1. Add synthetic tasks (both Russian and English prompts)
    for task in SYNTHETIC_TASKS:
        if "code" in task:
            tool_name = "execute_blender_code"
            args = {"code": task["code"]}
        else:
            tool_name = task["tool_call"]["name"]
            args = task["tool_call"]["arguments"]

        all_samples.append(make_chatml_sample(task["user_ru"], tool_name, args))
        all_samples.append(make_chatml_sample(task["user_en"], tool_name, args))

    # 2. Add samples from Hugging Face dataset
    hf_samples = fetch_hf_dioramas(limit=400)
    for instr, script in hf_samples:
        all_samples.append(make_chatml_sample(instr, "execute_blender_code", {"code": script}))

    random.seed(42)
    random.shuffle(all_samples)

    val_count = max(5, int(len(all_samples) * 0.1))
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDataset prepared:")
    print(f"  Train: {len(train_samples)} samples in {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Val:   {len(val_samples)} samples in {val_path} ({val_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
