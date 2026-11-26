import bpy
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
