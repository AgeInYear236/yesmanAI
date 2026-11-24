#!/usr/bin/env python3
"""Dataset preparation script for fine-tuning Qwen2.5 on Blender MCP workflows.

Produces a balanced dataset of atomic Blender commands (cubes, spheres, materials,
modifiers, scene inspection) + complex 3D scenes from open sources,
formatted in Qwen2.5 ChatML with MCP tool-calling schemas.
"""

import json
import os
import random
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """Ты — добрый, простой и отзывчивый помощник для работы в Blender. Общайся тепло, дружелюбно и живо, без заумных слов. Используй доступные инструменты Blender для выполнения запросов пользователя.

Правила для инструмента execute_blender_code:
- Код выполняется через Python exec(). Никогда не используй оператор 'return' вне функций (на верхнем уровне скрипта это вызовет SyntaxError).
- Для простых объектов создавай лаконичный и понятный код bpy.
- Всегда настраивай цвет материалов через ноды Principled BSDF (node_tree.nodes).

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

ATOMIC_BLENDER_TASKS = [
    # 1. Sphere at center
    {
        "user_ru": "Создай сферу в центре сцены.",
        "user_en": "Create a sphere at the center of the scene.",
        "code": (
            "import bpy\n\n"
            "# Создаем UV-сферу в центре сцены\n"
            "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))\n"
            "sphere = bpy.context.active_object\n"
            "sphere.name = 'CenterSphere'\n"
            "bpy.ops.object.shade_smooth()\n"
        ),
    },
    {
        "user_ru": "Создай синий шар в центре сцены.",
        "user_en": "Create a blue sphere at the center of the scene.",
        "code": (
            "import bpy\n\n"
            "# Создаем сферу\n"
            "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))\n"
            "obj = bpy.context.active_object\n"
            "obj.name = 'BlueSphere'\n"
            "bpy.ops.object.shade_smooth()\n\n"
            "# Назначаем синий материал через Principled BSDF\n"
            "mat = bpy.data.materials.new(name='BlueMat')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            "bsdf.inputs['Base Color'].default_value = (0.1, 0.4, 0.9, 1.0)\n"
            "bsdf.inputs['Roughness'].default_value = 0.2\n"
            "obj.data.materials.append(mat)\n"
        ),
    },
    # 2. Cube
    {
        "user_ru": "Создай куб в центре сцены.",
        "user_en": "Create a cube at the center of the scene.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))\n"
            "cube = bpy.context.active_object\n"
            "cube.name = 'Cube'\n"
        ),
    },
    {
        "user_ru": "Создай красный куб размером 2 метра.",
        "user_en": "Create a red cube of size 2 meters.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))\n"
            "cube = bpy.context.active_object\n"
            "cube.name = 'RedCube'\n\n"
            "mat = bpy.data.materials.new(name='RedMat')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            "bsdf.inputs['Base Color'].default_value = (0.9, 0.05, 0.05, 1.0)\n"
            "cube.data.materials.append(mat)\n"
        ),
    },
    # 3. Suzanne
    {
        "user_ru": "Создай обезьянку Сюзанну с золотым материалом.",
        "user_en": "Create Suzanne monkey head with gold material.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0, 0, 0))\n"
            "obj = bpy.context.active_object\n"
            "bpy.ops.object.shade_smooth()\n\n"
            "mat = bpy.data.materials.new(name='GoldMat')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            "bsdf.inputs['Base Color'].default_value = (1.0, 0.766, 0.336, 1.0)\n"
            "bsdf.inputs['Metallic'].default_value = 1.0\n"
            "bsdf.inputs['Roughness'].default_value = 0.15\n"
            "obj.data.materials.append(mat)\n"
        ),
    },
    # 4. Clear scene
    {
        "user_ru": "Удали все объекты со сцены.",
        "user_en": "Delete all objects from the scene.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.object.select_all(action='SELECT')\n"
            "bpy.ops.object.delete()\n"
        ),
    },
    {
        "user_ru": "Очисти сцену от мешей.",
        "user_en": "Clear all mesh objects from the scene.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.object.select_all(action='DESELECT')\n"
            "for obj in bpy.context.scene.objects:\n"
            "    if obj.type == 'MESH':\n"
            "        obj.select_set(True)\n"
            "bpy.ops.object.delete()\n"
        ),
    },
    # 5. Scene info query
    {
        "user_ru": "Посмотри, какие объекты сейчас находятся в сцене Blender.",
        "user_en": "Check what objects are currently in the Blender scene.",
        "tool_call": {"name": "get_scene_info", "arguments": {}},
    },
    {
        "user_ru": "Что сейчас в сцене?",
        "user_en": "What is currently in the scene?",
        "tool_call": {"name": "get_scene_info", "arguments": {}},
    },
    # 6. Object info query
    {
        "user_ru": "Покажи информацию об объекте с именем Cube.",
        "user_en": "Show information about the object named Cube.",
        "tool_call": {"name": "get_object_info", "arguments": {"object_name": "Cube"}},
    },
    # 7. Viewport screenshot
    {
        "user_ru": "Сделай снимок вьюпорта (скриншот).",
        "user_en": "Take a screenshot of the 3D viewport.",
        "tool_call": {"name": "get_viewport_screenshot", "arguments": {"max_size": 800}},
    },
    # 8. Modifiers: Bevel and Subsurf
    {
        "user_ru": "Добавь фаску (модификатор Bevel) на куб Cube шириной 0.1 и 3 сегмента.",
        "user_en": "Add a Bevel modifier to object Cube with width 0.1 and 3 segments.",
        "code": (
            "import bpy\n\n"
            "obj = bpy.data.objects.get('Cube')\n"
            "if obj:\n"
            "    mod = obj.modifiers.new(name='Bevel', type='BEVEL')\n"
            "    mod.width = 0.1\n"
            "    mod.segments = 3\n"
        ),
    },
    {
        "user_ru": "Сделай сглаживание Subdivision Surface для сферы 2 уровня.",
        "user_en": "Add Subdivision Surface modifier with 2 levels to the active object.",
        "code": (
            "import bpy\n\n"
            "obj = bpy.context.active_object\n"
            "if obj:\n"
            "    mod = obj.modifiers.new(name='Subsurf', type='SUBSURF')\n"
            "    mod.levels = 2\n"
            "    mod.render_levels = 2\n"
            "    bpy.ops.object.shade_smooth()\n"
        ),
    },
    # 9. Lighting & Camera
    {
        "user_ru": "Добавь источник света Солнце силой 5 и желтоватым оттенком.",
        "user_en": "Add a Sun light with strength 5 and slightly yellow color.",
        "code": (
            "import bpy\nimport math\n\n"
            "light_data = bpy.data.lights.new(name='Sun', type='SUN')\n"
            "light_data.energy = 5.0\n"
            "light_data.color = (1.0, 0.95, 0.85)\n"
            "light_obj = bpy.data.objects.new(name='Sun', object_data=light_data)\n"
            "bpy.context.collection.objects.link(light_obj)\n"
            "light_obj.location = (4, -4, 8)\n"
            "light_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(45))\n"
        ),
    },
    {
        "user_ru": "Создай камеру, направленную на центр сцены.",
        "user_en": "Create a camera pointing at the center of the scene.",
        "code": (
            "import bpy\nimport math\n\n"
            "cam_data = bpy.data.cameras.new(name='MainCamera')\n"
            "cam_obj = bpy.data.objects.new(name='MainCamera', object_data=cam_data)\n"
            "bpy.context.collection.objects.link(cam_obj)\n"
            "cam_obj.location = (5, -5, 4)\n"
            "cam_obj.rotation_euler = (math.radians(60), 0, math.radians(45))\n"
            "bpy.context.scene.camera = cam_obj\n"
        ),
    },
    # 10. Torus, Cylinder, Cone, Plane
    {
        "user_ru": "Создай золотой бублик (торус) радиусом 1.5.",
        "user_en": "Create a golden torus with major radius 1.5.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.4, location=(0, 0, 0))\n"
            "obj = bpy.context.active_object\n"
            "bpy.ops.object.shade_smooth()\n\n"
            "mat = bpy.data.materials.new(name='Gold')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            "bsdf.inputs['Base Color'].default_value = (1.0, 0.766, 0.336, 1.0)\n"
            "bsdf.inputs['Metallic'].default_value = 1.0\n"
            "bsdf.inputs['Roughness'].default_value = 0.1\n"
            "obj.data.materials.append(mat)\n"
        ),
    },
    {
        "user_ru": "Создай цилиндр высотой 4 и радиусом 0.5 зеленого цвета.",
        "user_en": "Create a green cylinder with depth 4 and radius 0.5.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=4.0, location=(0, 0, 2))\n"
            "obj = bpy.context.active_object\n\n"
            "mat = bpy.data.materials.new(name='GreenMat')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            "bsdf.inputs['Base Color'].default_value = (0.1, 0.8, 0.2, 1.0)\n"
            "obj.data.materials.append(mat)\n"
        ),
    },
    {
        "user_ru": "Создай плоскость (пол) размером 10 на 10 метров с темным матовым материалом.",
        "user_en": "Create a 10x10 meter floor plane with dark matte material.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))\n"
            "plane = bpy.context.active_object\n"
            "plane.name = 'FloorPlane'\n\n"
            "mat = bpy.data.materials.new(name='DarkFloor')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            "bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1.0)\n"
            "bsdf.inputs['Roughness'].default_value = 0.8\n"
            "plane.data.materials.append(mat)\n"
        ),
    },
    # 11. Neon Emission
    {
        "user_ru": "Создай светящуюся неоновую вывеску с текстом YESMAN бирюзового цвета.",
        "user_en": "Create an emission neon sign with text YESMAN in cyan color.",
        "code": (
            "import bpy\n\n"
            "bpy.ops.object.text_add(location=(0, 0, 1))\n"
            "text_obj = bpy.context.active_object\n"
            "text_obj.data.body = 'YESMAN'\n"
            "text_obj.data.extrude = 0.15\n"
            "text_obj.data.bevel_depth = 0.02\n"
            "text_obj.rotation_euler = (1.5708, 0, 0)\n\n"
            "mat = bpy.data.materials.new(name='NeonCyan')\n"
            "mat.use_nodes = True\n"
            "nodes = mat.node_tree.nodes\n"
            "nodes.clear()\n"
            "emission = nodes.new('ShaderNodeEmission')\n"
            "emission.inputs['Color'].default_value = (0.0, 0.9, 1.0, 1.0)\n"
            "emission.inputs['Strength'].default_value = 8.0\n"
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


def fetch_hf_dioramas(limit: int = 100) -> list[tuple[str, str]]:
    """Fetch verified Blender bpy scripts from ShiminZ/blender-lowpoly-dioramas via HF API."""
    print(f"Fetching up to {limit} samples from ShiminZ/blender-lowpoly-dioramas...")
    samples = []
    chunk_size = 50
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

    # 1. Add atomic Blender tasks (both Russian and English prompts, repeated to give high weight to core commands)
    for _ in range(5): # 5x replication to ensure model prioritizes concise direct actions for simple prompts
        for task in ATOMIC_BLENDER_TASKS:
            if "code" in task:
                tool_name = "execute_blender_code"
                args = {"code": task["code"]}
            else:
                tool_name = task["tool_call"]["name"]
                args = task["tool_call"]["arguments"]

            all_samples.append(make_chatml_sample(task["user_ru"], tool_name, args))
            all_samples.append(make_chatml_sample(task["user_en"], tool_name, args))

    # 2. Add high-level diorama scenes (up to 80 samples for rich complex tasks)
    hf_samples = fetch_hf_dioramas(limit=80)
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
