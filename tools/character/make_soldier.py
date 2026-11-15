# Builds the player character (a male soldier) with MPFB / MakeHuman in
# Blender and exports it for the game.
#
#   blender -b --python tools/character/make_soldier.py -- <mh_assets_dir> <out_dir>
#
# <mh_assets_dir>: a copy of the MakeHuman system assets (skins, eyes, hair,
# clothes ...; CC0), laid out as in MakeHuman's data folder.
# Writes <out_dir>/soldier.glb (mesh + skeleton, glTF binary) and
# <out_dir>/soldier_preview.png.
import bpy
import math
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ASSETS = os.path.abspath(argv[0])
OUT = os.path.abspath(argv[1])
os.makedirs(OUT, exist_ok=True)

from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from bl_ext.blender_org.mpfb.services.targetservice import TargetService
from bl_ext.blender_org.mpfb.services.objectservice import ObjectService


def log(*a):
    print("[soldier]", *a, flush=True)


def asset(*parts):
    p = os.path.join(ASSETS, *parts)
    if not os.path.exists(p):
        raise IOError("missing asset: " + p)
    return p


# ---- empty scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)

# ---- body: a fit man of about thirty ----
macro = TargetService.get_default_macro_info_dict()
macro.update({"gender": 1.0, "age": 0.52, "muscle": 0.78, "weight": 0.52, "proportions": 0.62, "height": 0.62})
macro["race"] = {"asian": 0.0, "caucasian": 1.0, "african": 0.0}
basemesh = HumanService.create_human(mask_helpers=True, detailed_helpers=False, extra_vertex_groups=True,
                                     feet_on_ground=True, scale=0.1, macro_detail_dict=macro)
log("basemesh", basemesh.name, "verts", len(basemesh.data.vertices))

# ---- skeleton (Mixamo-compatible: full fingers, widely supported for retargeting) ----
rig = HumanService.add_builtin_rig(basemesh, "mixamo", import_weights=True)
log("rig", rig.name if rig else None, "bones", len(rig.data.bones) if rig else 0)

# ---- skin ----
HumanService.set_character_skin(asset("skins", "young_caucasian_male", "young_caucasian_male.mhmat"),
                                basemesh, skin_type="GAMEENGINE", material_instances=False)

# ---- body parts and clothes ----
def add(kind, *parts):
    obj = HumanService.add_mhclo_asset(asset(*parts), basemesh, asset_type=kind, subdiv_levels=0,
                                       material_type="GAMEENGINE")
    log("added", kind, parts[-1], "->", obj.name if obj else None)
    return obj

add("Eyes", "eyes", "low-poly", "low-poly.mhclo")
add("Eyebrows", "eyebrows", "eyebrow001", "eyebrow001.mhclo")
add("Eyelashes", "eyelashes", "eyelashes01", "eyelashes01.mhclo")
add("Teeth", "teeth", "teeth_base", "teeth_base.mhclo")
suit = add("Clothes", "clothes", "male_casualsuit05", "male_casualsuit05.mhclo")   # field jacket, shirt, trousers
boots = add("Clothes", "clothes", "shoes03", "shoes03.mhclo")

import bmesh
import numpy as np
from mathutils import Vector


def base_color_image(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                link = node.inputs["Base Color"].links
                if link and link[0].from_node.type == "TEX_IMAGE":
                    return link[0].from_node.image, mat
    return None, None


def smooth_noise(n, cells, seed):
    """Tileable smooth value noise n x n with `cells` blobs per side."""
    rng = np.random.default_rng(seed)
    g = rng.random((cells, cells)).astype(np.float32)
    idx = np.arange(n, dtype=np.float32) * cells / n
    i0 = np.floor(idx).astype(int) % cells
    i1 = (i0 + 1) % cells
    t = idx - np.floor(idx)
    t = t * t * (3.0 - 2.0 * t)
    rows = g[i0][:, i0] * (1 - t)[None, :] + g[i0][:, i1] * t[None, :]
    rows1 = g[i1][:, i0] * (1 - t)[None, :] + g[i1][:, i1] * t[None, :]
    return rows * (1 - t)[:, None] + rows1 * t[:, None]


def woodland_camo(n, seed):
    """Four-colour woodland pattern (RGB, linear), n x n."""
    a = smooth_noise(n, 26, seed) * 0.6 + smooth_noise(n, 61, seed + 1) * 0.3 + smooth_noise(n, 140, seed + 2) * 0.1
    b = smooth_noise(n, 31, seed + 5) * 0.6 + smooth_noise(n, 73, seed + 6) * 0.3 + smooth_noise(n, 150, seed + 7) * 0.1
    out = np.empty((n, n, 3), np.float32)
    out[:] = (0.105, 0.125, 0.060)                      # olive green base
    out[a > 0.56] = (0.185, 0.150, 0.085)               # khaki / tan
    out[b > 0.60] = (0.060, 0.045, 0.030)               # dark brown
    out[(a < 0.36) & (b < 0.50)] = (0.022, 0.026, 0.020)  # near black
    return out


def camouflage(obj, seed):
    """Replaces the garment's colours with camouflage, keeping the cloth's
    own shading (folds, seams, weave) from the original texture."""
    img, _ = base_color_image(obj)
    if img is None:
        log("no base colour image on", obj.name)
        return
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    lum = px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722
    # Shading relative to the garment's local average brightness.
    k = max(8, w // 64)
    small = lum.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    local = np.kron(small, np.ones((k, k), np.float32))
    detail = np.clip(lum / np.maximum(local, 0.02), 0.45, 1.6)
    camo = woodland_camo(w, seed)
    if camo.shape[0] != h:
        camo = np.resize(camo, (h, w, 3))
    px[..., :3] = np.clip(camo * detail[..., None] * 1.15, 0.0, 1.0)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    log("camouflage applied to", obj.name, img.name, w, h)


def flat_material(name, color, rough=0.8, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def shell_from(src, name, keep, thickness, material):
    """New garment made from the part of `src` where keep(world_co) is true:
    pushed outward and given thickness; keeps the vertex weights, so it
    follows the skeleton like the garment under it."""
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.duplicate()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    mw = obj.matrix_world
    drop = [v for v in bm.verts if not keep(mw @ v.co)]
    bmesh.ops.delete(bm, geom=drop, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    mod = obj.modifiers.new("thick", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    # Solidify before the armature so the shell deforms as one piece.
    while obj.modifiers.find("thick") > 0:
        bpy.ops.object.modifier_move_up(modifier="thick")
    log("shell", name, len(obj.data.vertices), "verts")
    return obj


def rigid_part(name, mesh_fn, bone, material):
    """A rigid accessory (helmet, pouch) following one bone."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    mesh_fn(bm)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = True
    vg = obj.vertex_groups.new(name=bone)
    vg.add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
    obj.parent = rig
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    return obj


def world_bbox(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    hi = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return lo, hi


bones = [b.name for b in rig.data.bones]
log("bones", ", ".join(bones))


def bone_named(*keys):
    for b in bones:
        low = b.lower()
        if all(k in low for k in keys):
            return b
    raise KeyError(keys)


head_bone = bone_named("head")
spine_bone = bone_named("spine2") if any("spine2" in b.lower() for b in bones) else bone_named("spine")
head_pos = rig.matrix_world @ rig.data.bones[head_bone].head_local
head_top = rig.matrix_world @ rig.data.bones[head_bone].tail_local
lo, hi = world_bbox(basemesh)
height = hi.z
log("height", round(height, 3), "head bone", head_bone, tuple(round(c, 3) for c in head_pos))

# Uniform: woodland camouflage on the jacket / shirt / trousers.
camouflage(suit, 7)

mat_web = flat_material("webbing", (0.050, 0.058, 0.038), rough=0.9)
mat_helmet = flat_material("helmet", (0.060, 0.072, 0.045), rough=0.75)
mat_black = flat_material("black_gear", (0.012, 0.012, 0.013), rough=0.6)

# Tactical vest: the torso part of the jacket, thicker and in webbing green.
chest_lo, chest_hi = height * 0.585, height * 0.800
half_w = (hi.x - lo.x) * 0.5
shoulder = height * 0.118                               # half shoulder width: keeps the sleeves out
vest = shell_from(suit, "Human.vest",
                  lambda c: chest_lo < c.z < chest_hi and abs(c.x) < shoulder,
                  0.014, mat_web)

# Gloves: the hands of the body, as a thin black shell.
wrist = rig.matrix_world @ rig.data.bones[bone_named("lefthand")].head_local
hand_x = abs(wrist.x) - 0.012
log("wrist", tuple(round(c, 3) for c in wrist))
gloves = shell_from(basemesh, "Human.gloves",
                    lambda c: abs(c.x) > hand_x and c.z > wrist.z - 0.30,
                    0.004, mat_black)
for m in list(gloves.modifiers):
    if m.type == "MASK":
        gloves.modifiers.remove(m)
if len(gloves.data.vertices) == 0:
    bpy.data.objects.remove(gloves)

# Helmet: an ellipsoid shell over the skull, open below the ears.
skull_lo, skull_hi = world_bbox(basemesh)
head_r = Vector((height * 0.058, height * 0.070, height * 0.068))
# The real top of the skull, from the morphed (evaluated) body mesh.
dg = bpy.context.evaluated_depsgraph_get()
ev = basemesh.evaluated_get(dg)
skull = [ev.matrix_world @ v.co for v in ev.data.vertices]
skull = [c for c in skull if abs(c.x) < 0.11 and c.z > head_pos.z + 0.05 and abs(c.y - head_pos.y) < 0.16]
skull_top = max(c.z for c in skull)
skull_y = 0.5 * (min(c.y for c in skull) + max(c.y for c in skull))
log("skull top", round(skull_top, 3), "centre y", round(skull_y, 3), "body max |x|", round(max(abs((ev.matrix_world @ v.co).x) for v in ev.data.vertices), 3))
head_c = Vector((0.0, skull_y, skull_top - head_r.z * 0.86))


def helmet_mesh(bm):
    bmesh.ops.create_uvsphere(bm, u_segments=28, v_segments=16, radius=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * head_r.x * 1.24, v.co.y * head_r.y * 1.22, v.co.z * head_r.z * 1.14)) + head_c
    # Down to the brow in front, lower over the ears and the back of the head.
    cut = head_c.z - head_r.z * 0.42
    def rim(v):
        front = max(0.0, (head_c.y - v.co.y) / (head_r.y * 1.22))
        return cut + head_r.z * 0.62 * front * front
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < rim(v)], context="VERTS")
    # Brim: flare the lowest ring a little.
    for v in bm.verts:
        if v.co.z < rim(v) + head_r.z * 0.12:
            v.co.x *= 1.03
            v.co.y = head_c.y + (v.co.y - head_c.y) * 1.03


helmet = rigid_part("Human.helmet", helmet_mesh, head_bone, mat_helmet)
hm = helmet.modifiers.new("thick", "SOLIDIFY")
hm.thickness = 0.009
hm.offset = 1.0
bpy.context.view_layer.objects.active = helmet
bpy.ops.object.modifier_move_up(modifier="thick")

# Pouches on the vest front and a belt.
front_y = min((vest.matrix_world @ v.co).y for v in vest.data.vertices) if len(vest.data.vertices) else -0.12


def pouch_mesh_at(cx, cz, w, hgt, d):
    def fn(bm):
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co = Vector((cx + v.co.x * w, front_y - d * 0.35 + v.co.y * d, cz + v.co.z * hgt))
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=min(w, hgt, d) * 0.12, segments=2, affect="EDGES")
    return fn


belly = height * 0.635
for i, cx in enumerate((-0.105, -0.035, 0.035, 0.105)):
    rigid_part("Human.pouch%d" % i, pouch_mesh_at(cx * height / 1.8, belly, 0.060, 0.115, 0.045), spine_bone, mat_web)
rigid_part("Human.radio", pouch_mesh_at(0.085 * height / 1.8, height * 0.745, 0.050, 0.095, 0.035), spine_bone, mat_black)

for o in bpy.data.objects:
    log("object", o.name, o.type, len(o.data.vertices) if o.type == "MESH" else "", [m.type for m in o.modifiers] if o.type == "MESH" else "")

# ---- preview render ----
scene = bpy.context.scene
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
cam.location = (1.75, -2.75, 1.15)
cam.rotation_euler = (math.radians(87.0), 0.0, math.radians(32.5))
cam_data.lens = 50
scene.camera = cam
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50.0), 0.0, math.radians(30.0))
scene.collection.objects.link(sun)
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.6, 0.7, 1.0)
scene.world = world
scene.render.resolution_x = 720
scene.render.resolution_y = 960
scene.render.filepath = os.path.join(OUT, "soldier_preview.png")
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except Exception:
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        scene.render.engine = "BLENDER_WORKBENCH"
log("render engine", scene.render.engine)
try:
    bpy.ops.render.render(write_still=True)
    log("preview written")
except Exception as e:
    log("preview failed:", e)

# ---- export ----
cam.select_set(False)
bpy.ops.object.select_all(action="DESELECT")
for o in bpy.data.objects:
    if o.type in ("MESH", "ARMATURE"):
        o.select_set(True)
glb = os.path.join(OUT, "soldier.glb")
bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_apply=True,
                          export_yup=True, export_skins=True, export_animations=False,
                          export_image_format="AUTO")
log("exported", glb, os.path.getsize(glb), "bytes")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "soldier.blend"))
