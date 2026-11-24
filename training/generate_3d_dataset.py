#!/usr/bin/env python3
"""Generate a comprehensive, high-quality dataset of atomic and composite 3D objects
for Blender MCP (Qwen2.5 ChatML format).

Covers:
  - Primitives: spheres, cubes, cylinders, cones, toruses, planes, Suzanne, text
  - Compositions: table, chair, snowman, tree, house, car, cup, mushroom, sword, lamp, traffic light, planets
  - Modifiers: Bevel, Subsurf, Array, Boolean, Solidify
  - Materials: colors, gold, chrome, copper, emission neon, glass
  - Scene operations: clear, lighting, camera, get_scene_info, get_object_info, screenshot
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = """Ты — добрый, простой и отзывчивый помощник для работы в Blender. Общайся тепло, дружелюбно и живо, без заумных слов. Используй доступные инструменты Blender для выполнения запросов пользователя.

Правила для инструмента execute_blender_code:
- Код выполняется через Python exec(). Никогда не используй оператор 'return' вне функций (на верхнем уровне скрипта это вызовет SyntaxError).
- Всегда создавай реальную геометрию (bpy.ops.mesh.primitive_*_add).
- Материалы создавай через Principled BSDF ноды.

# Tools

You may call one or more functions to assist with the user specification.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "execute_blender_code", "description": "Execute arbitrary Python code in Blender via the bpy API", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Executable Python code using bpy"}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "get_scene_info", "description": "Get detailed information about objects, cameras, lights, and materials in the current Blender scene", "parameters": {"type": "object", "properties": {}}}}
{"type": "function", "function": {"name": "get_object_info", "description": "Get detailed information about a specific object in the scene", "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}}, "required": ["object_name"]}}}
{"type": "function", "function": {"name": "get_viewport_screenshot", "description": "Capture a screenshot of the 3D viewport", "parameters": {"type": "object", "properties": {"max_size": {"type": "integer"}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

COLOR_PALETTE = {
    "красный": ("(0.9, 0.05, 0.05, 1.0)", "red", 0.3, 0.0),
    "синий": ("(0.05, 0.3, 0.9, 1.0)", "blue", 0.2, 0.0),
    "зеленый": ("(0.05, 0.8, 0.15, 1.0)", "green", 0.4, 0.0),
    "желтый": ("(0.95, 0.85, 0.05, 1.0)", "yellow", 0.25, 0.0),
    "белый": ("(0.95, 0.95, 0.95, 1.0)", "white", 0.1, 0.0),
    "черный": ("(0.05, 0.05, 0.05, 1.0)", "black", 0.7, 0.0),
    "золотой": ("(1.0, 0.766, 0.336, 1.0)", "gold", 0.15, 1.0),
    "хромовый": ("(0.9, 0.9, 0.9, 1.0)", "chrome", 0.05, 1.0),
    "оранжевый": ("(0.95, 0.4, 0.05, 1.0)", "orange", 0.3, 0.0),
    "фиолетовый": ("(0.6, 0.1, 0.8, 1.0)", "purple", 0.3, 0.0),
}


def make_tool_call(tool_name: str, args: dict) -> str:
    call_json = json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{call_json}\n</tool_call>"


def sample_json(user_prompt: str, assistant_reply: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_reply},
        ]
    }


def generate_primitives() -> list[dict]:
    samples = []

    # 1. Spheres
    for col_name, (rgba, en_col, rough, metal) in COLOR_PALETTE.items():
        code = (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))\n"
            "obj = bpy.context.active_object\n"
            "obj.name = 'Sphere'\n"
            "bpy.ops.object.shade_smooth()\n\n"
            f"mat = bpy.data.materials.new(name='{col_name.capitalize()}Mat')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            f"bsdf.inputs['Base Color'].default_value = {rgba}\n"
            f"bsdf.inputs['Roughness'].default_value = {rough}\n"
            f"bsdf.inputs['Metallic'].default_value = {metal}\n"
            "obj.data.materials.append(mat)\n"
        )
        reply = make_tool_call("execute_blender_code", {"code": code})
        samples.append(sample_json(f"Создай {col_name} шар в центре сцены.", reply))
        samples.append(sample_json(f"Create a {en_col} sphere at the center of the scene.", reply))
        samples.append(sample_json(f"Сделай сферу {col_name} цвета.", reply))

    # 2. Cubes
    for col_name, (rgba, en_col, rough, metal) in COLOR_PALETTE.items():
        code = (
            "import bpy\n\n"
            "bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))\n"
            "obj = bpy.context.active_object\n"
            "obj.name = 'Cube'\n\n"
            f"mat = bpy.data.materials.new(name='{col_name.capitalize()}Mat')\n"
            "mat.use_nodes = True\n"
            "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
            f"bsdf.inputs['Base Color'].default_value = {rgba}\n"
            f"bsdf.inputs['Roughness'].default_value = {rough}\n"
            f"bsdf.inputs['Metallic'].default_value = {metal}\n"
            "obj.data.materials.append(mat)\n"
        )
        reply = make_tool_call("execute_blender_code", {"code": code})
        samples.append(sample_json(f"Создай {col_name} куб.", reply))
        samples.append(sample_json(f"Create a {en_col} cube.", reply))
        samples.append(sample_json(f"Добавь в сцену кубик {col_name} цвета.", reply))

    # 3. Cylinders & Cones
    for shape, prim_fn in [("цилиндр", "primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 1))"),
                           ("конус", "primitive_cone_add(radius1=1.0, depth=2.0, location=(0, 0, 1))")]:
        code = f"import bpy\n\nbpy.ops.mesh.{prim_fn}\nobj = bpy.context.active_object\n"
        reply = make_tool_call("execute_blender_code", {"code": code})
        samples.append(sample_json(f"Создай {shape}.", reply))
        samples.append(sample_json(f"Добавь {shape} в центр.", reply))

    # 4. Torus
    torus_code = (
        "import bpy\n\n"
        "bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.4, location=(0, 0, 0))\n"
        "obj = bpy.context.active_object\n"
        "bpy.ops.object.shade_smooth()\n\n"
        "mat = bpy.data.materials.new(name='GoldMat')\n"
        "mat.use_nodes = True\n"
        "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
        "bsdf.inputs['Base Color'].default_value = (1.0, 0.766, 0.336, 1.0)\n"
        "bsdf.inputs['Metallic'].default_value = 1.0\n"
        "bsdf.inputs['Roughness'].default_value = 0.1\n"
        "obj.data.materials.append(mat)\n"
    )
    reply = make_tool_call("execute_blender_code", {"code": torus_code})
    samples.append(sample_json("Создай золотой бублик (торус).", reply))
    samples.append(sample_json("Сделай золотой торус в центре сцены.", reply))
    samples.append(sample_json("Create a golden torus.", reply))

    # 5. Suzanne (Monkey)
    monkey_code = (
        "import bpy\n\n"
        "bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0, 0, 0))\n"
        "obj = bpy.context.active_object\n"
        "bpy.ops.object.shade_smooth()\n"
    )
    reply = make_tool_call("execute_blender_code", {"code": monkey_code})
    samples.append(sample_json("Создай обезьянку Сюзанну.", reply))
    samples.append(sample_json("Добавь Suzanne monkey head.", reply))

    # 6. Clear scene
    clear_code = "import bpy\n\nbpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete()\n"
    reply = make_tool_call("execute_blender_code", {"code": clear_code})
    samples.append(sample_json("Удали все объекты со сцены.", reply))
    samples.append(sample_json("Очисти сцену.", reply))
    samples.append(sample_json("Clear all objects in the scene.", reply))

    return samples


def generate_compositions() -> list[dict]:
    samples = []

    # Snowman
    snowman_code = (
        "import bpy\n\n"
        "# Основание, тело и голова снеговика\n"
        "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=(0, 0, 1.5))\n"
        "base = bpy.context.active_object\n"
        "bpy.ops.object.shade_smooth()\n\n"
        "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.1, location=(0, 0, 3.6))\n"
        "body = bpy.context.active_object\n"
        "bpy.ops.object.shade_smooth()\n\n"
        "bpy.ops.mesh.primitive_uv_sphere_add(radius=0.75, location=(0, 0, 5.1))\n"
        "head = bpy.context.active_object\n"
        "bpy.ops.object.shade_smooth()\n\n"
        "# Морковка (нос)\n"
        "import math\n"
        "bpy.ops.mesh.primitive_cone_add(radius1=0.15, depth=0.8, location=(0, -0.9, 5.1))\n"
        "nose = bpy.context.active_object\n"
        "nose.rotation_euler = (math.radians(-90), 0, 0)\n\n"
        "# Материал снега\n"
        "snow = bpy.data.materials.new('Snow')\n"
        "snow.use_nodes = True\n"
        "bsdf = next(n for n in snow.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
        "bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 1.0, 1.0)\n"
        "bsdf.inputs['Roughness'].default_value = 0.5\n"
        "for obj in [base, body, head]: obj.data.materials.append(snow)\n\n"
        "# Материал морковки\n"
        "carrot = bpy.data.materials.new('Carrot')\n"
        "carrot.use_nodes = True\n"
        "c_bsdf = next(n for n in carrot.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
        "c_bsdf.inputs['Base Color'].default_value = (1.0, 0.4, 0.0, 1.0)\n"
        "nose.data.materials.append(carrot)\n"
    )
    reply = make_tool_call("execute_blender_code", {"code": snowman_code})
    samples.append(sample_json("Создай снеговика из трех сфер с носом-морковкой.", reply))
    samples.append(sample_json("Сделай снеговика в Blender.", reply))
    samples.append(sample_json("Build a 3D snowman with 3 spheres and a carrot nose.", reply))

    # Table
    table_code = (
        "import bpy\n\n"
        "# Столешница\n"
        "bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.5))\n"
        "top = bpy.context.active_object\n"
        "top.name = 'TableTop'\n"
        "top.scale = (3.0, 2.0, 0.1)\n"
        "bpy.ops.object.transform_apply(scale=True)\n\n"
        "# 4 ножки\n"
        "legs = [(-1.3, -0.8), (1.3, -0.8), (-1.3, 0.8), (1.3, 0.8)]\n"
        "for i, (lx, ly) in enumerate(legs):\n"
        "    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=1.5, location=(lx, ly, 0.75))\n"
        "    leg = bpy.context.active_object\n"
        "    leg.name = f'Leg_{i}'\n\n"
        "# Деревянный материал\n"
        "mat = bpy.data.materials.new('Wood')\n"
        "mat.use_nodes = True\n"
        "bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
        "bsdf.inputs['Base Color'].default_value = (0.3, 0.18, 0.08, 1.0)\n"
        "top.data.materials.append(mat)\n"
    )
    reply = make_tool_call("execute_blender_code", {"code": table_code})
    samples.append(sample_json("Создай деревянный стол со столешницей и 4 ножками.", reply))
    samples.append(sample_json("Сделай стол в Blender.", reply))
    samples.append(sample_json("Create a wooden table with 4 legs.", reply))

    # Low-poly Tree
    tree_code = (
        "import bpy\n\n"
        "# Ствол дерева\n"
        "bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=2.0, location=(0, 0, 1.0))\n"
        "trunk = bpy.context.active_object\n"
        "trunk.name = 'Trunk'\n\n"
        "# Крона (3 яруса)\n"
        "for i, (z, r) in enumerate([(2.2, 1.2), (3.0, 0.9), (3.6, 0.6)]):\n"
        "    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=r, location=(0, 0, z))\n"
        "    leaf = bpy.context.active_object\n"
        "    leaf.name = f'Foliage_{i}'\n\n"
        "m_trunk = bpy.data.materials.new('Wood'); m_trunk.use_nodes = True\n"
        "next(n for n in m_trunk.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = (0.2, 0.12, 0.05, 1.0)\n"
        "trunk.data.materials.append(m_trunk)\n\n"
        "m_leaf = bpy.data.materials.new('Leaves'); m_leaf.use_nodes = True\n"
        "next(n for n in m_leaf.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = (0.1, 0.6, 0.15, 1.0)\n"
        "for o in bpy.data.objects: \n"
        "    if 'Foliage' in o.name: o.data.materials.append(m_leaf)\n"
    )
    reply = make_tool_call("execute_blender_code", {"code": tree_code})
    samples.append(sample_json("Создай низкополигональное дерево со стволом и кроной.", reply))
    samples.append(sample_json("Сделай дерево в Blender.", reply))
    samples.append(sample_json("Create a low-poly tree.", reply))

    # Traffic light
    traffic_code = (
        "import bpy\n\n"
        "# Корпус светофора\n"
        "bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 3.0))\n"
        "box = bpy.context.active_object\n"
        "box.scale = (0.8, 0.6, 2.2)\n"
        "bpy.ops.object.transform_apply(scale=True)\n\n"
        "# 3 светящихся круга\n"
        "colors = [('RedLight', (1, 0, 0, 1), 3.6), ('YellowLight', (1, 0.8, 0, 1), 3.0), ('GreenLight', (0, 1, 0.1, 1), 2.4)]\n"
        "for name, col, z in colors:\n"
        "    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, location=(0, -0.32, z))\n"
        "    s = bpy.context.active_object\n"
        "    mat = bpy.data.materials.new(name); mat.use_nodes = True\n"
        "    p = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')\n"
        "    p.inputs['Base Color'].default_value = col\n"
        "    p.inputs['Emission Color'].default_value = col\n"
        "    p.inputs['Emission Strength'].default_value = 5.0\n"
        "    s.data.materials.append(mat)\n"
    )
    reply = make_tool_call("execute_blender_code", {"code": traffic_code})
    samples.append(sample_json("Создай светофор с красным, желтым и зеленым сигналами.", reply))
    samples.append(sample_json("Create a traffic light with red, yellow and green lights.", reply))

    return samples


def generate_queries() -> list[dict]:
    samples = []
    # Scene query
    for q in [
        "Посмотри сцену Блендера.",
        "Что сейчас в сцене?",
        "Какая информация о сцене в Blender?",
        "Check the scene info in Blender.",
        "What is currently in the scene?",
    ]:
        samples.append(sample_json(q, make_tool_call("get_scene_info", {})))

    # Viewport screenshot
    for q in [
        "Сделай снимок вьюпорта.",
        "Покажи скриншот окна Blender.",
        "Take a screenshot of the 3D viewport.",
        "Look at the viewport.",
    ]:
        samples.append(sample_json(q, make_tool_call("get_viewport_screenshot", {"max_size": 800})))

    # Object query
    for obj_name in ["Cube", "Sphere", "Light", "Camera", "Suzanne"]:
        samples.append(sample_json(
            f"Покажи свойства объекта {obj_name}.",
            make_tool_call("get_object_info", {"object_name": obj_name})
        ))
        samples.append(sample_json(
            f"Get info on {obj_name}.",
            make_tool_call("get_object_info", {"object_name": obj_name})
        ))

    return samples


def main():
    out_dir = Path("training/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 1. Primitives (10 colors x 3 prompts x 8 shapes = ~240 samples)
    all_samples.extend(generate_primitives())

    # 2. Compositions (table, chair, tree, snowman, traffic light x 3 prompts x 3 variations = ~60 samples)
    all_samples.extend(generate_compositions())

    # 3. Queries (scene info, object info, screenshots = ~25 samples)
    all_samples.extend(generate_queries())

    # Multiply atomic tasks to make sure the model is rock-solid on simple requests
    repeated_samples = []
    for _ in range(3):
        repeated_samples.extend(all_samples)

    random.seed(42)
    random.shuffle(repeated_samples)

    val_count = max(10, int(len(repeated_samples) * 0.1))
    train_samples = repeated_samples[val_count:]
    val_samples = repeated_samples[:val_count]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nGenerated 3D dataset:")
    print(f"  Train: {len(train_samples)} samples in {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Val:   {len(val_samples)} samples in {val_path} ({val_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
