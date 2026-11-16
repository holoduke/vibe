# Builds the player character (a male soldier) with MPFB / MakeHuman in
# Blender and exports it for the game.
#
#   blender -b --python tools/character/make_soldier.py -- <mh_assets_dir> <out_dir>
#
# <mh_assets_dir>: the MakeHuman system assets plus the community packs
# gloves01, hats02, shoes01 (CC0) and equipment03 (CC-BY) unpacked into it (skins, eyes, hair,
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
basemesh = HumanService.create_human(mask_helpers=True, detailed_helpers=True, extra_vertex_groups=True,
                                     feet_on_ground=True, scale=0.1, macro_detail_dict=macro)
log("basemesh", basemesh.name, "verts", len(basemesh.data.vertices))

# ---- skeleton: the game-engine rig (53 bones, full fingers). It is fitted to
# the body; MPFB's "mixamo" rig comes in a T-pose that does not line up with
# the arms of the base mesh. ----
# The body shape lives in shape keys; bake it into the mesh first, or the
# rig is fitted to the default body (wrong joint positions, not on the ground).
TargetService.bake_targets(basemesh)
rig = HumanService.add_builtin_rig(basemesh, "game_engine", import_weights=True)
for _n in ("pelvis", "foot_l", "hand_l", "head"):
    log("joint", _n, tuple(round(c, 3) for c in rig.data.bones[_n].head_local))
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
# Real gear from the MakeHuman community asset packs (see Credits in README.md):
# tall leather boots, a MOLLE tactical vest with magazine pouches, short
# gloves and an M1 steel helmet with a net.
boots = add("Clothes", "clothes", "culturalibre_male_boots", "culturalibre_male_boots.mhclo")
vest = add("Clothes", "clothes", "mindfront_tactical_vest_male", "mindfront_tactical_vest_male.mhclo")
gloves = add("Clothes", "clothes", "toigo_gloves_short", "toigo_gloves_short.mhclo")
helmet = add("Clothes", "clothes", "mrgreaterthan_m1_helmet", "mrgreaterthan_m1_helmet.mhclo")

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


def tint(obj, mul, desat=0.0):
    """Darkens / recolours a garment texture in place (keeps its detail)."""
    img, _ = base_color_image(obj)
    if img is None:
        log("no base colour image on", obj.name)
        return
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    lum = (px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722)[..., None]
    rgb = px[..., :3] * (1.0 - desat) + lum * desat
    px[..., :3] = np.clip(rgb * np.array(mul, np.float32), 0.0, 1.0)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    log("tint applied to", obj.name, img.name)


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
spine_bone = "spine_02" if "spine_02" in bones else (bone_named("spine2") if any("spine2" in b.lower() for b in bones) else bone_named("spine"))
head_pos = rig.matrix_world @ rig.data.bones[head_bone].head_local
head_top = rig.matrix_world @ rig.data.bones[head_bone].tail_local
lo, hi = world_bbox(basemesh)
height = hi.z
log("height", round(height, 3), "head bone", head_bone, tuple(round(c, 3) for c in head_pos))

# Uniform: woodland camouflage on the jacket / shirt / trousers.
camouflage(suit, 7)

# Black leather boots and gloves; the vest stays in its own olive webbing.
tint(boots, (0.16, 0.15, 0.15), desat=0.85)
tint(gloves, (0.55, 0.55, 0.55), desat=0.6)


def widen(obj, factor):
    """Scales a garment horizontally about its own vertical axis, so it
    fits over what is worn under it (keeps hard details like pouches
    intact, which pushing along the normals does not)."""
    n = len(obj.data.vertices)
    cx = sum(v.co.x for v in obj.data.vertices) / n
    cy = sum(v.co.y for v in obj.data.vertices) / n
    for v in obj.data.vertices:
        v.co.x = cx + (v.co.x - cx) * factor
        v.co.y = cy + (v.co.y - cy) * factor
    obj.data.update()


def cut_below(obj, z_world):
    """Removes the part of a garment below a height (trousers tucked in)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    mw = obj.matrix_world
    drop = [f for f in bm.faces if all((mw @ v.co).z < z_world for v in f.verts)]
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    log("cut", obj.name, "faces below", round(z_world, 3), len(drop))


# The vest is worn over the jacket; the trousers are tucked into the boots.
vest_lo, vest_hi = world_bbox(vest)
vest_cx, vest_cy = 0.5 * (vest_lo.x + vest_hi.x), 0.5 * (vest_lo.y + vest_hi.y)
vest_hw = 0.5 * (vest_hi.x - vest_lo.x)
widen(vest, 1.15)
# ... and the jacket is pulled in under it (it hangs open and loose, and
# would otherwise cover the vest).
_mw = suit.matrix_world
_inv = _mw.inverted()
for v in suit.data.vertices:
    c = _mw @ v.co
    if abs(c.x - vest_cx) > vest_hw * 0.98:
        continue
    t = min((c.z - vest_lo.z) / 0.05, (vest_hi.z - c.z) / 0.05)
    if t <= 0.0:
        continue
    t = min(t, 1.0)
    k = 1.0 - 0.10 * t * t * (3.0 - 2.0 * t)
    c.x = vest_cx + (c.x - vest_cx) * k
    c.y = vest_cy + (c.y - vest_cy) * k
    v.co = _inv @ c
suit.data.update()
boot_lo, boot_hi = world_bbox(boots)
cut_below(suit, boot_hi.z - 0.06)

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
