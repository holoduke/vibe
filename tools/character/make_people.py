# Builds everybody the game draws as a person - the player's soldier and the variety of people
# the NPC engine shows (docs/ARCHITECTURE.md "People who look like people") - and writes them
# for the game in one go, all on the soldier's skeleton and into one texture atlas.
#
#   blender -b <out_dir>/soldier.blend --python tools/character/make_people.py -- <mh_assets_dir> <game_assets_dir>/character
#
# soldier.blend is what make_soldier.py writes (the soldier, rigged; the canonical skeleton).
# Writes into <game_assets_dir>/character:
#   soldier_fp*.qchar, soldier_head*.qchar   the player's soldier, as before (three levels each)
#   people_body_<outfit>*.qchar             the other outfits, on the soldier's own body
#   people_head_<n>*.qchar                  faces: other builds, ages and origins, no headgear
#   people_hair_<n>*.qchar                  a head's longer hair (cards; only where it has some)
#   people_helmet*.qchar, people_cap*.qchar the headgear, apart
#   soldier_atlas.jpg + soldier_atlas_alpha.png   colour, and the coverage of hair cards / the iris
# (export_fp.py did the first two before 2026-11-27; it is replaced by this file.)
#
# HOW A FILE KNOWS WHAT IT IS. Each vertex's texture coordinate u carries a REGION in its whole
# part: u = region * 4 + u_atlas (cube.frag "People": the shading of each). Regions:
#   0 plain (as the atlas has it)   1 skin   2 camouflage (x the unit's shade)   3 top garment
#   4 bottom garment   5 hair card   6 brow card   7 eye   8 gear (x the unit's shade)   9 shoes
# Skin, garments to be tinted, hair and the iris are stored NORMALISED: a linear value of 0.5 is
# the tile's mean, the colour is the person's (the look block, player_body.cpp people_look).
#
# A FACE OF ANOTHER PERSON ON THE SAME SKELETON. Every MakeHuman base mesh has the same vertices.
# A face is made as a whole other human (MPFB: age, origin, build, face targets), and its head
# is put on the soldier: the vertices that follow the head take the other's places, those of the
# neck keep the soldier's (blended between), so every head fits every body at the collar. The
# weights are the soldier's, vertex for vertex. Eyes, brows and hair are fitted to that face.
import bpy
import bmesh
import math
import numpy as np
import os
import struct
import sys
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ASSETS = os.path.abspath(argv[0])
OUT = os.path.abspath(argv[1])
ONLY = argv[2] if len(argv) > 2 else ""          # (tests: "atlas" = the soldier's files only)
os.makedirs(OUT, exist_ok=True)

if "bl_ext.blender_org.mpfb" not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.mpfb")
from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from bl_ext.blender_org.mpfb.services.targetservice import TargetService


def log(*a):
    print("[people]", *a, flush=True)


def asset(*parts):
    p = os.path.join(ASSETS, *parts)
    if not os.path.exists(p):
        raise IOError("missing asset: " + p)
    return p


def to_game(v):
    return (-v[0], v[2], v[1])


# =====================================================================================
# The canonical skeleton (the soldier's), in the order the game reads it
# =====================================================================================
RIG = next(o for o in bpy.data.objects if o.type == "ARMATURE")
BASE = bpy.data.objects["Human"]
bones = []


def _walk(b):
    bones.append(b)
    for c in b.children:
        _walk(c)


for b in RIG.data.bones:
    if b.parent is None:
        _walk(b)
BONE_INDEX = {b.name: i for i, b in enumerate(bones)}


def game_name(n):
    face = {"jaw": "Jaw", "eye": "Eye", "orbicularis03": "UpperLid", "orbicularis04": "LowerLid", "oculi01": "Brow"}
    fside = "Left" if n.endswith(".L") else "Right" if n.endswith(".R") else ""
    fcore = n[:-2] if fside else n
    if fcore in face:
        return fside + face[fcore]
    side = "Left" if n.endswith("_l") else "Right" if n.endswith("_r") else ""
    core = n[:-2] if side else n
    fixed = {"Root": "Root", "pelvis": "Hips", "spine_01": "Spine", "spine_02": "Spine1", "spine_03": "Spine2",
             "neck_01": "Neck", "head": "Head", "clavicle": "Shoulder", "upperarm": "Arm", "lowerarm": "ForeArm",
             "hand": "Hand", "thigh": "UpLeg", "calf": "Leg", "foot": "Foot", "ball": "ToeBase"}
    if core in fixed:
        return side + fixed[core]
    for finger in ("index", "middle", "ring", "pinky", "thumb"):
        if core.startswith(finger + "_0"):
            return side + "Hand" + finger.capitalize() + core[-1]
    return n.split(":")[-1]


OUT_NAME = [game_name(b.name) for b in bones]
BONE_TABLE = []
for b in bones:
    mw = RIG.matrix_world
    BONE_TABLE.append((OUT_NAME[BONE_INDEX[b.name]], BONE_INDEX[b.parent.name] if b.parent else -1,
                       to_game(mw @ b.head_local), to_game(mw @ b.tail_local)))
HEAD_BONES = {i for i, n in enumerate(OUT_NAME) if n == "Head" or n == "Jaw" or n.endswith(("Eye", "Lid", "Brow"))}
NECK_BONES = {i for i, n in enumerate(OUT_NAME) if n == "Neck"}
HAND_BONES = {i for i, n in enumerate(OUT_NAME) if "Hand" in n}
ARM_BONES = {i for i, n in enumerate(OUT_NAME) if n.endswith(("Arm", "ForeArm", "Shoulder"))} | HAND_BONES
TORSO_BONES = {i for i, n in enumerate(OUT_NAME) if n.startswith("Spine") or n in ("Neck",)} | ARM_BONES
LEG_BONES = {i for i, n in enumerate(OUT_NAME) if n.endswith(("UpLeg", "Leg", "Foot", "ToeBase")) or n in ("Hips", "Root")}
FACE_BONE_NAMES = ("head", "jaw", "eye.L", "eye.R", "orbicularis03.L", "orbicularis03.R", "orbicularis04.L",
                   "orbicularis04.R", "oculi01.L", "oculi01.R")
log("skeleton", len(bones), "bones")

# =====================================================================================
# The atlas (4096 x 4096): a fixed layout. Top half: what the player's own view fills.
# =====================================================================================
ATLAS = 4096
atlas = np.zeros((ATLAS, ATLAS, 3), np.float32)
alpha = np.ones((ATLAS, ATLAS), np.float32)
LAYOUT = {
    "suit": (0, 0, 2048), "vest": (2048, 0, 2048),
    "skin_young": (0, 2048, 1024), "skin_old": (1024, 2048, 1024), "tee04": (2048, 2048, 1024), "work": (3072, 2048, 1024),
    "suit_plain": (0, 3072, 1024),
    "boots": (1024, 3072, 512), "gloves": (1536, 3072, 512), "helmet": (2048, 3072, 512), "eyes": (2560, 3072, 512),
    "shoes": (3072, 3072, 512), "tee06": (3584, 3072, 512),
    "hair_a": (1024, 3584, 512), "hair_b": (1536, 3584, 512), "mask": (2048, 3584, 512),
    "brow_a": (2560, 3584, 256), "brow_b": (2816, 3584, 256), "brow_c": (2560, 3840, 256), "brow_d": (2816, 3840, 256),
}
FLAT0 = (3072, 3584)           # 64 px flat colours from here: 16 a row, 8 rows
flat_tiles = {}
filled = set()


def srgb_encode(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_decode(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def tile_uv(key):
    x, y, s = LAYOUT[key]
    return (x / ATLAS, y / ATLAS, s / ATLAS, s / ATLAS)


def image_pixels(img, size):
    """The image as linear RGBA float, size x size (rows bottom first, as Blender keeps them)."""
    cp = img.copy()
    cp.scale(size, size)
    px = np.empty(size * size * 4, np.float32)
    cp.pixels.foreach_get(px)
    bpy.data.images.remove(cp)
    px = px.reshape(size, size, 4)
    if img.colorspace_settings.name != "sRGB":
        px[..., :3] = srgb_decode(px[..., :3])          # (stored as data: they are display values)
    return px


def load_image(path):
    img = bpy.data.images.load(path, check_existing=True)
    return img


def put_tile(key, img, mode="color", alpha_mode="one"):
    """mode: color (as it is), norm (linear / its mean x 0.5, per channel - skin), grey (luminance / mean x 0.5);
    alpha_mode: one, image (the image's own alpha), iris (where the eye is coloured)."""
    if key in filled:
        return tile_uv(key)
    x, y, s = LAYOUT[key]
    px = image_pixels(img, s)
    rgb = px[..., :3]
    a = np.ones((s, s), np.float32)
    if alpha_mode == "image":
        a = px[..., 3].copy()
    if mode == "norm":
        w = (px[..., 3] > 0.5) if alpha_mode == "image" else np.ones((s, s), bool)
        mean = rgb[w].mean(axis=0) if w.any() else np.array([0.5, 0.5, 0.5])
        log("tile", key, "mean colour (linear)", tuple(round(float(c), 4) for c in mean))
        rgb = rgb / np.maximum(mean, 1e-3) * 0.5
    elif mode == "detail":
        # (only the cloth's own shading - folds, seams, weave - relative to its local average: a printed
        # pattern goes, so the person's colour is the colour)
        lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        k = 16                                   # (divides every tile size)
        local = np.kron(lum.reshape(s // k, k, s // k, k).mean(axis=(1, 3)), np.ones((k, k), np.float32))
        rgb = np.repeat((np.clip(lum / np.maximum(local, 0.02), 0.5, 1.6) * 0.5)[..., None], 3, axis=2)
    elif mode == "grey":
        lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        w = (px[..., 3] > 0.5) if alpha_mode == "image" else np.ones((s, s), bool)
        mean = float(lum[w].mean()) if w.any() else 0.5
        rgb = np.repeat((lum / max(mean, 1e-3) * 0.5)[..., None], 3, axis=2)
    if alpha_mode == "iris":
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        a = np.clip((sat - 0.05) / 0.10, 0.0, 1.0)
        a = np.maximum(a, np.clip((0.06 - lum) / 0.04, 0.0, 1.0) * (sat > 0.0))      # (the pupil is iris too: it stays dark)
        # the iris goes grey (its colour is the person's)
        g = lum / 0.25 * 0.5
        rgb = rgb * (1.0 - a[..., None]) + np.repeat(g[..., None], 3, axis=2) * a[..., None]
    # Blender's rows are bottom first; the atlas is written top first at the end (flipped there).
    atlas[y:y + s, x:x + s] = srgb_encode(rgb)
    alpha[y:y + s, x:x + s] = np.clip(a, 0.0, 1.0)
    filled.add(key)
    log("tile", key, img.name, mode, alpha_mode, s)
    return tile_uv(key)


def flat(color, linear=True):
    key = "%.3f,%.3f,%.3f" % tuple(color)
    if key in flat_tiles:
        return flat_tiles[key]
    i = len(flat_tiles)
    x, y = FLAT0[0] + 64 * (i % 16), FLAT0[1] + 64 * (i // 16)
    c = np.array(color, np.float32)
    atlas[y:y + 64, x:x + 64] = srgb_encode(c) if linear else c
    flat_tiles[key] = ((x + 32) / ATLAS, (y + 32) / ATLAS, 0.0, 0.0)
    return flat_tiles[key]


def base_color_image(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                inp = node.inputs["Base Color"]
                seen, stack = set(), [l.from_node for l in inp.links]
                while stack:
                    n = stack.pop()
                    if n in seen:
                        continue
                    seen.add(n)
                    if n.type == "TEX_IMAGE" and n.image:
                        return n.image
                    for i in n.inputs:
                        for l in i.links:
                            stack.append(l.from_node)
    return None


# Regions (cube.frag "People")
R_PLAIN, R_SKIN, R_CAMO, R_TOP, R_BOTTOM, R_HAIR, R_BROW, R_EYE, R_GEAR, R_SHOE = range(10)
GLOVE = flat((0.016, 0.017, 0.016))
SKIN_FLAT = flat((0.5, 0.5, 0.5))           # a normalised skin tile's mean: the neck's cap

# =====================================================================================
# Writing a file
# =====================================================================================
dg = bpy.context.evaluated_depsgraph_get()


class Part:
    """One object in a file. tile: (u0, v0, du, dv) or a function(material index) -> one;
    region: int or function(dominant bone, all bones of the face) -> int;
    keep: function(dominant bones of the face) -> bool (None: every face);
    weights: None (the object's vertex groups) or function(world position) -> [(bone index, w)]."""
    def __init__(self, obj, tile, region, keep=None, ratio=1.0, weld=0.0, weights=None):
        self.obj, self.tile, self.region, self.keep, self.ratio, self.weld, self.weights = obj, tile, region, keep, ratio, weld, weights
        self.masks = None


def write_file(fname, parts, neck_cap=False):
    out_v, out_i, vmap = [], [], {}
    for part in parts:
        obj = part.obj
        for m in obj.modifiers:
            if m.type == "ARMATURE":
                m.show_viewport = False
            # (the skin under clothes is cut away by MPFB's masks: only those of what this outfit wears)
            if m.type == "MASK" and part.masks is not None and m.name.startswith("Delete."):
                m.show_viewport = m.name[7:] in part.masks
        added = []
        if part.weld > 0.0:
            w = obj.modifiers.new("export_weld", "WELD")
            w.merge_threshold = part.weld
            added.append(w)
        if part.ratio < 0.999:
            d = obj.modifiers.new("export_decimate", "DECIMATE")
            d.decimate_type = "COLLAPSE"
            d.ratio = part.ratio
            d.use_collapse_triangulate = True
            added.append(d)
        dg.update()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
        mw = obj.matrix_world
        nm = mw.to_3x3().inverted().transposed()
        groups = {g.index: BONE_INDEX[g.name] for g in obj.vertex_groups if g.name in BONE_INDEX}
        uvl = me.uv_layers.active.data if me.uv_layers.active else None
        vw = []
        for v in me.vertices:
            if part.weights:
                ws = part.weights(mw @ v.co)
            else:
                ws = sorted(((g.weight, groups[g.group]) for g in v.groups if g.group in groups and g.weight > 1e-4), reverse=True)[:4]
                ws = [(b, w) for w, b in ws]
            if not ws:
                ws = [(BONE_INDEX["head"], 1.0)]
            tot = sum(w for _, w in ws)
            q = [int(round(255.0 * w / tot)) for _, w in ws]
            q[0] += 255 - sum(q)
            bi = [b for b, _ in ws]
            while len(bi) < 4:
                bi.append(0)
                q.append(0)
            vw.append((tuple(bi), tuple(q)))
        kept = 0
        for poly in me.polygons:
            vs = list(poly.vertices)
            dom = [vw[v][0][0] for v in vs]
            if part.keep and not part.keep(dom):
                continue
            tile = part.tile(poly.material_index, dom) if callable(part.tile) else part.tile
            region = part.region(dom) if callable(part.region) else part.region
            for li, v in zip(poly.loop_indices, vs):
                co = mw @ me.vertices[v].co
                n = me.corner_normals[li].vector if hasattr(me, "corner_normals") and len(me.corner_normals) else me.loops[li].normal
                n = (nm @ n).normalized()
                if uvl is not None and tile[2] > 0.0:
                    uv = uvl[li].uv
                    uu = uv[0] % 1.0 if not 0.0 <= uv[0] <= 1.0 else uv[0]
                    vv = uv[1] % 1.0 if not 0.0 <= uv[1] <= 1.0 else uv[1]
                    u, w = tile[0] + min(uu, 0.9995) * tile[2], tile[1] + min(vv, 0.9995) * tile[3]
                else:
                    u, w = tile[0], tile[1]
                key = (obj.name, v, round(u, 5), round(w, 5), region, round(n[0], 3), round(n[1], 3), round(n[2], 3))
                idx = vmap.get(key)
                if idx is None:
                    idx = len(out_v)
                    vmap[key] = idx
                    # (v = 0 at the top of the image in the game; the region rides in u's whole part)
                    out_v.append((to_game(co), to_game(n), (u + 4.0 * region, 1.0 - w), vw[v][0], vw[v][1]))
                out_i.append(idx)
            kept += 1
        ev.to_mesh_clear()
        for m in added:
            obj.modifiers.remove(m)
        log(fname, obj.name, "ratio %.2f" % part.ratio, "tris", kept)
    if neck_cap:
        # (the base of the neck inside the collar, for the player's own view: see export_fp.py's history)
        ni = next(iter(NECK_BONES))
        c = to_game(RIG.matrix_world @ bones[ni].head_local)
        cy = c[1] - 0.012
        rx, rz, segs = 0.082, 0.098, 20
        base = len(out_v)
        uvc = (SKIN_FLAT[0] + 4.0 * R_SKIN, 1.0 - SKIN_FLAT[1])
        w = ((ni, 0, 0, 0), (255, 0, 0, 0))
        out_v.append(((c[0], cy + 0.018, c[2] - 0.004), (0.0, 1.0, 0.0), uvc, w[0], w[1]))
        for k in range(segs):
            a = 2.0 * math.pi * k / segs
            out_v.append(((c[0] + rx * math.cos(a), cy, c[2] - 0.004 + rz * math.sin(a)), (0.0, 1.0, 0.0), uvc, w[0], w[1]))
        for k in range(segs):
            out_i.extend((base, base + 1 + (k + 1) % segs, base + 1 + k))
    path = os.path.join(OUT, fname)
    with open(path, "wb") as f:
        f.write(b"QCH1")
        f.write(struct.pack("<III", len(BONE_TABLE), len(out_v), len(out_i)))
        for name, parent, head, tail in BONE_TABLE:
            f.write(struct.pack("<32s", name.encode("ascii", "replace")[:31]))
            f.write(struct.pack("<i", parent))
            f.write(struct.pack("<3f", *head))
            f.write(struct.pack("<3f", *tail))
        for p, n, uv, bi, bw in out_v:
            f.write(struct.pack("<3f3f2f4B4B", *p, *n, *uv, *bi, *bw))
        f.write(np.asarray(out_i, np.uint32).tobytes())
    log("wrote", fname, "vertices", len(out_v), "triangles", len(out_i) // 3, "bytes", os.path.getsize(path))
    return os.path.getsize(path)


def on_head(dom):
    return any(d in HEAD_BONES for d in dom) or all(d in NECK_BONES for d in dom)


def by_bones(dom):
    """Top or bottom of a garment that is one piece: by the bones its face follows."""
    legs = sum(1 for d in dom if d in LEG_BONES)
    return R_BOTTOM if legs * 2 > len(dom) else R_TOP


SIZES = {}

# =====================================================================================
# Tiles every file shares
# =====================================================================================
SKIN_Y = put_tile("skin_young", load_image(asset("skins", "textures", "young_lightskinned_male_diffuse.png")), "norm")
SKIN_O = put_tile("skin_old", load_image(asset("skins", "textures", "old_lightskinned_male_diffuse.png")), "norm")
EYES = put_tile("eyes", load_image(asset("eyes", "materials", "lightblue_eye.png")), "color", "iris")

objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
SUIT, VEST, BOOTS = objs["Human.male_casualsuit05"], objs["Human.mindfront_tactical_vest_male"], objs["Human.culturalibre_male_boots"]
GLOVES, HELMET, EYES_OBJ, BROWS_OBJ = objs["Human.toigo_gloves_short"], objs["Human.mrgreaterthan_m1_helmet"], objs["Human.low-poly"], objs["Human.eyebrow001"]
T_SUIT = put_tile("suit", base_color_image(SUIT), "color")
T_VEST = put_tile("vest", base_color_image(VEST), "color")
T_BOOTS = put_tile("boots", base_color_image(BOOTS), "color")
T_GLOVES = put_tile("gloves", base_color_image(GLOVES), "color")
T_HELMET = put_tile("helmet", base_color_image(HELMET), "color")
T_BROW = put_tile("brow_a", load_image(asset("eyebrows", "eyebrow001", "eyebrow001.png")), "grey", "image")


def body_skin_part(ratio, gloved, head, masks=()):
    """The base mesh: the body below the head (head=False) or the head (True). gloved: the hands
    are the gloves' colour (the gloves are short; the hand under them is painted)."""
    def keep(dom):
        return on_head(dom) == head

    def tile(mi, dom):
        if gloved and all(d in HAND_BONES for d in dom):
            return GLOVE
        return SKIN_Y

    def region(dom):
        if gloved and all(d in HAND_BONES for d in dom):
            return R_PLAIN
        return R_SKIN
    p = Part(BASE, tile, region, keep, ratio)
    p.masks = set(masks)
    return p


# =====================================================================================
# 1. The player's soldier (soldier_fp*, soldier_head*): as export_fp.py had them, with regions
# =====================================================================================
def soldier_body(ratio_of):
    return [Part(SUIT, T_SUIT, R_CAMO, ratio=ratio_of("suit")), Part(VEST, T_VEST, R_GEAR, ratio=ratio_of("vest")),
            Part(BOOTS, T_BOOTS, R_GEAR, ratio=ratio_of("boots")), Part(GLOVES, T_GLOVES, R_PLAIN, ratio=ratio_of("gloves")),
            body_skin_part(ratio_of("body"), True, False, ("male_casualsuit05", "culturalibre_male_boots", "toigo_gloves_short"))]


SIZES["soldier_fp"] = write_file("soldier_fp.qchar", soldier_body(lambda k: 0.45 if k == "boots" else 0.55 if k == "vest" else 1.0), neck_cap=True)
SIZES["soldier_fp_lod1"] = write_file("soldier_fp_lod1.qchar", soldier_body(lambda k: 0.30), neck_cap=True)
SIZES["soldier_fp_lod2"] = write_file("soldier_fp_lod2.qchar", soldier_body(lambda k: 0.09), neck_cap=True)


def eye_weights(p):
    return [(BONE_INDEX["eye.L" if p.x > 0.0 else "eye.R"], 1.0)]


def head_only(p):
    return [(BONE_INDEX["head"], 1.0)]


def soldier_head(r_skin, r_helmet, weld, brows=True):
    parts = [body_skin_part(r_skin, True, True, ("mrgreaterthan_m1_helmet",) if r_helmet > 0.0 else ()), Part(EYES_OBJ, EYES, R_EYE, ratio=1.0)]
    if brows:
        parts.append(Part(BROWS_OBJ, T_BROW, R_BROW, weights=head_only))
    if r_helmet > 0.0:
        parts.append(Part(HELMET, T_HELMET, R_GEAR, ratio=r_helmet, weld=weld))
    return parts


SIZES["soldier_head"] = write_file("soldier_head.qchar", soldier_head(1.0, 0.30, 0.0))
SIZES["soldier_head_lod1"] = write_file("soldier_head_lod1.qchar", soldier_head(0.35, 0.08, 0.004))
SIZES["soldier_head_lod2"] = write_file("soldier_head_lod2.qchar", soldier_head(0.12, 0.08, 0.004, brows=False))
# the helmet alone, and the soldier's own face without it (people_head_0)
SIZES["people_helmet"] = write_file("people_helmet.qchar", [Part(HELMET, T_HELMET, R_GEAR, ratio=0.30)])
SIZES["people_helmet_lod1"] = write_file("people_helmet_lod1.qchar", [Part(HELMET, T_HELMET, R_GEAR, ratio=0.08, weld=0.004)])
SIZES["people_head_0"] = write_file("people_head_0.qchar", soldier_head(1.0, 0.0, 0.0))
SIZES["people_head_0_lod1"] = write_file("people_head_0_lod1.qchar", soldier_head(0.35, 0.0, 0.0))
SIZES["people_head_0_lod2"] = write_file("people_head_0_lod2.qchar", soldier_head(0.12, 0.0, 0.0, brows=False))


# =====================================================================================
# 2. A cap: a crown and a peak, in the uniform's camouflage, on the head bone
# =====================================================================================
def make_cap():
    hb = RIG.data.bones["head"]
    top = RIG.matrix_world @ hb.tail_local
    hd = RIG.matrix_world @ hb.head_local
    # the skull: the head's highest base-mesh vertex and its widest
    hv = [BASE.matrix_world @ v.co for v in BASE.data.vertices if v.co.z > hd.z + 0.05]
    zt = max(p.z for p in hv)
    cx = sum(p.x for p in hv) / len(hv)
    cy = sum(p.y for p in hv) / len(hv)
    rx = max(abs(p.x - cx) for p in hv if p.z > zt - 0.07) + 0.010
    ry = max(abs(p.y - cy) for p in hv if p.z > zt - 0.07) + 0.010
    mesh = bpy.data.meshes.new("cap")
    bm = bmesh.new()
    rings, segs = 6, 28
    z0 = zt - 0.085
    verts = []
    for r in range(rings + 1):
        t = r / rings
        a = t * math.pi * 0.5
        row = []
        for s in range(segs):
            th = 2.0 * math.pi * s / segs
            k = math.cos(a)
            row.append(bm.verts.new((cx + rx * k * math.sin(th), cy + ry * k * math.cos(th), z0 + (zt + 0.012 - z0) * math.sin(a) * (0.86 if t < 1.0 else 0.92))))
        verts.append(row)
    for r in range(rings):
        for s in range(segs):
            a, b, c, d = verts[r][s], verts[r][(s + 1) % segs], verts[r + 1][(s + 1) % segs], verts[r + 1][s]
            bm.faces.new((a, b, c, d))
    # the peak, out over the brow (forward is -y)
    peak = []
    for s in range(9):
        th = math.pi + (s - 4) / 4.0 * 1.05
        p0 = (cx + rx * math.sin(th), cy + ry * math.cos(th), z0 + 0.004)
        p1 = (cx + (rx + 0.065) * math.sin(th) * 0.95, cy + (ry + 0.075) * math.cos(th), z0 - 0.010)
        q0 = (p0[0], p0[1], p0[2] - 0.003); q1 = (p1[0], p1[1], p1[2] - 0.003)
        peak.append((bm.verts.new(p0), bm.verts.new(p1), bm.verts.new(q0), bm.verts.new(q1)))
    for s in range(8):
        bm.faces.new((peak[s][0], peak[s + 1][0], peak[s + 1][1], peak[s][1]))
        bm.faces.new((peak[s][3], peak[s + 1][3], peak[s + 1][2], peak[s][2]))     # (the underside)
    uvl = bm.loops.layers.uv.new()
    for f in bm.faces:
        for l in f.loops:
            co = l.vert.co
            l[uvl].uv = ((co.x - cx) * 3.0 + 0.5, (co.y - cy) * 3.0 + co.z * 2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("cap", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True
    return obj


CAP = make_cap()
SIZES["people_cap"] = write_file("people_cap.qchar", [Part(CAP, T_SUIT, R_CAMO, weights=head_only)])

if ONLY == "atlas":
    log("only the soldier: stopping before the outfits")


# =====================================================================================
# 3. Outfits on the soldier's body (people_body_<name>)
# =====================================================================================
def add_clothes(*parts):
    obj = HumanService.add_mhclo_asset(asset(*parts), BASE, asset_type="Clothes", subdiv_levels=0, material_type="GAMEENGINE")
    log("added", parts[-1], "->", obj.name if obj else None)
    return obj


def cut_faces(obj, drop):
    """Removes the faces for which drop(list of world positions, dominant bone indices) is true."""
    groups = {g.index: BONE_INDEX.get(g.name, -1) for g in obj.vertex_groups}
    dom = []
    for v in obj.data.vertices:
        best = max(((g.weight, groups.get(g.group, -1)) for g in v.groups), default=(0.0, -1))
        dom.append(best[1])
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    mw = obj.matrix_world
    kill = [f for f in bm.faces if drop([mw @ v.co for v in f.verts], [dom[v.index] for v in f.verts])]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    log("cut", obj.name, len(kill), "faces")


def bone_pos(name, tail=False):
    b = RIG.data.bones[name]
    return RIG.matrix_world @ (b.tail_local if tail else b.head_local)


ELBOW_L, ELBOW_R = bone_pos("lowerarm_l"), bone_pos("lowerarm_r")
WRIST_L, WRIST_R = bone_pos("hand_l"), bone_pos("hand_r")
HIPS = bone_pos("pelvis")
FOREARM = {BONE_INDEX["lowerarm_l"]: (ELBOW_L, WRIST_L), BONE_INDEX["lowerarm_r"]: (ELBOW_R, WRIST_R)}


def rolled_sleeves(obj):
    """Sleeves rolled up to just below the elbow: the rest of each sleeve goes, the rolled cuff is
    a band of the sleeve's own cloth pushed out a little."""
    def along(p, b):
        e, w = FOREARM[b]
        d = (w - e)
        return (p - e).dot(d) / d.length_squared
    def drop(ps, dom):
        if any(d in HAND_BONES for d in dom):
            return True
        fs = [(p, d) for p, d in zip(ps, dom) if d in FOREARM]
        return len(fs) == len(ps) and all(along(p, d) > 0.22 for p, d in fs)
    cut_faces(obj, drop)
    # the cuff: a copy of the band 0.10 .. 0.22 of the forearm, pushed out 7 mm
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    cuff = bpy.context.view_layer.objects.active
    cuff.name = obj.name + ".cuff"
    cut_faces(cuff, lambda ps, dom: not (all(d in FOREARM for d in dom) and all(0.08 < along(p, d) <= 0.23 for p, d in zip(ps, dom))))
    for v in cuff.data.vertices:
        v.co = v.co + v.normal * 0.007
    cuff.data.update()
    return cuff


def tint_none(obj):
    return obj


OUTFITS = []      # (name, [Part factories with a ratio function])


def outfit_files(name, parts_of):
    for lod, rat in ((0, lambda k: 0.45 if k == "boots" else 0.55 if k == "vest" else 1.0), (1, lambda k: 0.30), (2, lambda k: 0.09)):
        fname = "people_body_%s%s.qchar" % (name, "" if lod == 0 else "_lod%d" % lod)
        SIZES[fname[:-6]] = write_file(fname, parts_of(rat), neck_cap=True)


if ONLY != "atlas":
    # -- fatigues: the camouflage jacket, sleeves rolled, no vest, no gloves
    fat = SUIT.copy()
    fat.data = SUIT.data.copy()
    fat.name = "Human.fatigues"
    bpy.context.scene.collection.objects.link(fat)
    cuff = rolled_sleeves(fat)
    outfit_files("fatigues", lambda r: [Part(fat, T_SUIT, R_CAMO, ratio=r("suit")), Part(cuff, T_SUIT, R_CAMO, ratio=r("suit")),
                                        Part(BOOTS, T_BOOTS, R_GEAR, ratio=r("boots")), body_skin_part(r("body"), False, False, ("culturalibre_male_boots",))])

    # -- a T-shirt over the camouflage trousers (the jacket off)
    tee6 = add_clothes("clothes", "male_casualsuit06", "male_casualsuit06.mhclo")
    T_TEE6 = put_tile("tee06", base_color_image(tee6), "grey")
    tee6_top = tee6.copy(); tee6_top.data = tee6.data.copy(); tee6_top.name = "Human.tee6top"
    bpy.context.scene.collection.objects.link(tee6_top)
    cut_faces(tee6_top, lambda ps, dom: max(p.z for p in ps) < HIPS.z - 0.03)          # (a straight hem over the belt)
    trousers = SUIT.copy(); trousers.data = SUIT.data.copy(); trousers.name = "Human.trousers"
    bpy.context.scene.collection.objects.link(trousers)
    cut_faces(trousers, lambda ps, dom: min(p.z for p in ps) > HIPS.z + 0.07 or any(d in ARM_BONES for d in dom))
    outfit_files("tee", lambda r: [Part(tee6_top, T_TEE6, R_TOP, ratio=r("suit")), Part(trousers, T_SUIT, R_CAMO, ratio=r("suit")),
                                   Part(BOOTS, T_BOOTS, R_GEAR, ratio=r("boots")), body_skin_part(r("body"), False, False, ("culturalibre_male_boots", "male_casualsuit06"))])

    # -- civilian: T-shirt and jeans, shoes
    tee4 = add_clothes("clothes", "male_casualsuit04", "male_casualsuit04.mhclo")
    T_TEE4 = put_tile("tee04", base_color_image(tee4), "grey")
    shoes = add_clothes("clothes", "shoes01", "shoes01.mhclo")
    T_SHOES = put_tile("shoes", base_color_image(shoes), "grey")
    outfit_files("civil", lambda r: [Part(tee4, T_TEE4, by_bones, ratio=r("suit")), Part(shoes, T_SHOES, R_SHOE, ratio=r("boots")),
                                     body_skin_part(r("body"), False, False, ("male_casualsuit04", "shoes01"))])

    # -- overalls over a T-shirt (the launch pad's crew, a mechanic)
    work = add_clothes("clothes", "male_worksuit01", "male_worksuit01.mhclo")
    T_WORK = put_tile("work", base_color_image(work), "grey")
    outfit_files("work", lambda r: [Part(work, T_WORK, lambda dom: R_BOTTOM, ratio=r("suit")), Part(BOOTS, T_BOOTS, R_GEAR, ratio=r("boots")),
                                    body_skin_part(r("body"), False, False, ("male_worksuit01", "culturalibre_male_boots"))])

    # -- civilian: the field jacket in its own colours (any colour: the person's), shoes
    T_SUITP = put_tile("suit_plain", load_image(asset("clothes", "male_casualsuit05", "male_casualsuit05_diffuse.png")), "detail")
    outfit_files("jacket", lambda r: [Part(SUIT, T_SUITP, by_bones, ratio=r("suit")), Part(shoes, T_SHOES, R_SHOE, ratio=r("boots")),
                                      body_skin_part(r("body"), False, False, ("male_casualsuit05", "shoes01"))])

# =====================================================================================
# 4. Where hair and beard grow: a mask in the skin's own texture space (people_head_*)
#    R scalp hair, G beard, B moustache - from the soldier's head, made for every face
# =====================================================================================
def build_mask():
    x, y, s = LAYOUT["mask"]
    mask = np.zeros((s, s, 3), np.float32)
    me = BASE.data
    uvl = me.uv_layers.active.data
    mw = BASE.matrix_world
    gname = {g.index: g.name for g in BASE.vertex_groups}
    W = np.zeros(len(me.vertices), np.float32)
    J = np.zeros(len(me.vertices), np.float32)
    for v in me.vertices:
        for g in v.groups:
            n = gname.get(g.group, "")
            if n in FACE_BONE_NAMES:
                W[v.index] += g.weight
            if n == "jaw":
                J[v.index] += g.weight
    P = np.array([tuple(mw @ v.co) for v in me.vertices], np.float32)
    eyeL, eyeR = bone_pos("eye.L"), bone_pos("eye.R")
    eye_z = 0.5 * (eyeL.z + eyeR.z)
    hd = bone_pos("head")
    head = W > 0.5
    cx = float(P[head, 0].mean())
    mid = head & (np.abs(P[:, 0] - cx) < 0.012)
    front_y = float(P[mid, 1].min())                       # the nose's tip (forward is -y)
    nose_z = float(P[mid & (P[:, 1] < front_y + 0.004), 2].mean())
    chin_z = float(P[mid & (J > 0.5), 2].min())
    mouth_z = nose_z - 0.42 * (nose_z - chin_z)
    back_y = float(P[head, 1].max())
    cy = 0.5 * (front_y + back_y)
    log("mask: eyes %.3f nose %.3f mouth %.3f chin %.3f, front %.3f back %.3f" % (eye_z, nose_z, mouth_z, chin_z, front_y, back_y))

    def value(p, w, j):
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2]
        ang = math.atan2(abs(dx), -(dy))                  # 0 in front, pi behind
        f = ang / math.pi
        # SCALP: above a hairline that is high in front, at the temples over the ear, low at the nape
        line = eye_z + 0.060 * (1.0 - f) ** 0.6 * (1.0 if f < 0.55 else 1.0) - 0.04 * max(0.0, f - 0.5) * 2.0
        line = eye_z + (0.058 if f < 0.30 else 0.058 - (f - 0.30) / 0.25 * 0.050 if f < 0.55 else 0.008 - (f - 0.55) / 0.45 * 0.115)
        scalp = min(max((dz - line) / 0.012, 0.0), 1.0) * min(max((w - 0.35) / 0.3, 0.0), 1.0)
        # (not the ear: what stands out at the side below the top of the ear)
        if 0.35 < f < 0.75 and dz < eye_z + 0.03 and abs(dx) > 0.068:
            scalp *= max(0.0, 1.0 - (abs(dx) - 0.068) / 0.006)
        # BEARD: the lower face - below the cheekbones, in front of the ear, down under the jaw
        cheek = nose_z - 0.006 + 0.020 * min(abs(dx) / 0.06, 1.0)
        beard = 0.0
        if f < 0.52 and dz < cheek and dz > chin_z - 0.045:
            beard = min((cheek - dz) / 0.014, 1.0)
            beard *= min(max((0.52 - f) / 0.06, 0.0), 1.0)
            if dz < chin_z - 0.005:                           # under the jaw: fades on the throat
                beard *= max(0.0, 1.0 - (chin_z - 0.005 - dz) / 0.035) if f < 0.3 or j > 0.2 else 0.0
            # the lips are not beard
            if abs(dx) < 0.028 and abs(dz - mouth_z) < 0.010 and f < 0.2:
                beard = 0.0
        # MOUSTACHE: between the nose and the upper lip
        mous = 0.0
        if f < 0.18 and abs(dx) < 0.030 and mouth_z + 0.007 < dz < nose_z - 0.006:
            mous = min((dz - mouth_z - 0.007) / 0.004, 1.0) * min((0.030 - abs(dx)) / 0.008, 1.0)
        return scalp, beard, mous

    vals = np.array([value(P[i], W[i], J[i]) for i in range(len(me.vertices))], np.float32)
    # raster: each face of the head, in the skin's texture space (bottom-first rows, as the tiles)
    for poly in me.polygons:
        vi = list(poly.vertices)
        if max(W[i] for i in vi) < 0.2:
            continue
        uv = [uvl[li].uv for li in poly.loop_indices]
        for k in range(1, len(vi) - 1):
            tri = (0, k, k + 1)
            U = np.array([[uv[t][0] * s, uv[t][1] * s] for t in tri], np.float32)
            V = np.array([vals[vi[t]] for t in tri], np.float32)
            x0, y0 = int(max(0, math.floor(U[:, 0].min()))), int(max(0, math.floor(U[:, 1].min())))
            x1, y1 = int(min(s - 1, math.ceil(U[:, 0].max()))), int(min(s - 1, math.ceil(U[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, ay), (bx, by), (qx, qy) = U
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            if abs(den) < 1e-9:
                continue
            l0 = ((by - qy) * (gx - qx) + (qx - bx) * (gy - qy)) / den
            l1 = ((qy - ay) * (gx - qx) + (ax - qx) * (gy - qy)) / den
            l2 = 1.0 - l0 - l1
            inside = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
            if not inside.any():
                continue
            val = l0[..., None] * V[0] + l1[..., None] * V[1] + l2[..., None] * V[2]
            sub = mask[y0:y1 + 1, x0:x1 + 1]
            sub[inside] = np.maximum(sub[inside], np.clip(val[inside], 0.0, 1.0))
    atlas[y:y + s, x:x + s] = mask          # (data, not colour: read as it is - cube.frag undoes the sRGB)
    filled.add("mask")
    log("mask: scalp %.0f%% beard %.0f%% moustache %.0f%% of the tile" % tuple(100.0 * float((mask[..., c] > 0.5).mean()) for c in range(3)))


build_mask()


# =====================================================================================
# 5. Faces (people_head_<n>): other people's heads on the soldier
# =====================================================================================
TARGETS = os.path.join(os.path.dirname(TargetService.target_full_path("head-square") or ""), "..")
CANON = np.array([tuple(BASE.matrix_world @ v.co) for v in BASE.data.vertices], np.float32)
_gn = {g.index: g.name for g in BASE.vertex_groups}
CANON_GROUPS = [[(_gn[g.group], g.weight) for g in v.groups if _gn[g.group] in BONE_INDEX and g.weight > 1e-4] for v in BASE.data.vertices]
HEADW = np.array([sum(w for n, w in gs if n in FACE_BONE_NAMES) for gs in CANON_GROUPS], np.float32)
NECKW = np.array([sum(w for n, w in gs if n == "neck_01") for gs in CANON_GROUPS], np.float32)
NECK_RING = NECKW > 0.6
log("faces: %d vertices follow the head, %d the neck" % (int((HEADW > 0.5).sum()), int(NECK_RING.sum())))

# (age 0.5 = 25 years, 1.0 = 90; race weights; targets: file under MPFB's targets -> weight)
FACES = [
    # 1: a young man of African origin, broad face
    dict(age=0.47, muscle=0.75, weight=0.55, race=(0.0, 0.05, 0.95), brow="eyebrow006", hair=None,
         targets={"head/head-square": 0.35, "nose/nose-scale-horiz-incr": 0.3, "mouth/mouth-scale-horiz-incr": 0.25}),
    # 2: a man in his forties, heavier, fair
    dict(age=0.70, muscle=0.55, weight=0.72, race=(0.0, 1.0, 0.0), brow="eyebrow001", hair="short02",
         targets={"head/head-round": 0.5, "chin/chin-width-incr": 0.4, "neck/neck-scale-horiz-incr": 0.3}),
    # 3: a young man of East Asian origin, lean
    dict(age=0.44, muscle=0.6, weight=0.42, race=(1.0, 0.0, 0.0), brow="eyebrow009", hair="short04",
         targets={"head/head-oval": 0.5, "chin/chin-prominent-decr": 0.3}),
    # 4: an older man, near sixty, narrow face
    dict(age=0.86, muscle=0.45, weight=0.5, race=(0.0, 0.9, 0.1), brow="eyebrow001", hair=None,
         targets={"head/head-rectangular": 0.5, "nose/nose-scale-vert-incr": 0.4, "cheek/l-cheek-volume-decr": 0.4, "cheek/r-cheek-volume-decr": 0.4}),
    # 5: mixed origin, in his thirties, strong jaw
    dict(age=0.60, muscle=0.8, weight=0.58, race=(0.1, 0.45, 0.45), brow="eyebrow006", hair="short02",
         targets={"chin/chin-width-incr": 0.5, "chin/chin-prominent-incr": 0.4, "head/head-square": 0.4}),
    # 6: a very young man, thin face
    dict(age=0.38, muscle=0.5, weight=0.38, race=(0.0, 1.0, 0.0), brow="eyebrow009", hair=None,
         targets={"head/head-invertedtriangular": 0.5, "nose/nose-scale-horiz-decr": 0.3}),
    # 7: South Asian / Middle Eastern features, forties
    dict(age=0.66, muscle=0.62, weight=0.6, race=(0.25, 0.6, 0.15), brow="eyebrow006", hair="short04",
         targets={"nose/nose-scale-vert-incr": 0.5, "nose/nose-hump-incr": 0.5, "head/head-oval": 0.3}),
]
BROW_TILES = {"eyebrow001": "brow_a", "eyebrow006": "brow_b", "eyebrow009": "brow_c"}
HAIR_TILES = {"short02": "hair_a", "short04": "hair_b"}


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_face(i, spec):
    macro = TargetService.get_default_macro_info_dict()
    macro.update({"gender": 1.0, "age": spec["age"], "muscle": spec["muscle"], "weight": spec["weight"], "proportions": 0.62, "height": 0.62})
    macro["race"] = {"asian": spec["race"][0], "caucasian": spec["race"][1], "african": spec["race"][2]}
    hm = HumanService.create_human(mask_helpers=True, detailed_helpers=True, extra_vertex_groups=True, feet_on_ground=True,
                                   scale=0.1, macro_detail_dict=macro)
    for t, w in spec["targets"].items():
        p = os.path.join(TARGETS, t + ".target.gz")
        if os.path.exists(p):
            TargetService.load_target(hm, p, weight=w)
        else:
            log("face", i, "no target", t)
    TargetService.bake_targets(hm)
    if len(hm.data.vertices) != len(CANON):
        raise RuntimeError("face %d: %d vertices, the soldier has %d" % (i, len(hm.data.vertices), len(CANON)))
    var = np.array([tuple(hm.matrix_world @ v.co) for v in hm.data.vertices], np.float32)
    off = CANON[NECK_RING].mean(axis=0) - var[NECK_RING].mean(axis=0)
    mask = smoothstep(0.30, 0.85, HEADW)
    final = var + off
    final = final * mask[:, None] + CANON * (1.0 - mask[:, None])
    # eyes, brows and hair fitted to the face as it was made, then moved as the head was
    eyes = HumanService.add_mhclo_asset(asset("eyes", "low-poly", "low-poly.mhclo"), hm, asset_type="Eyes", subdiv_levels=0, material_type="GAMEENGINE")
    brow = HumanService.add_mhclo_asset(asset("eyebrows", spec["brow"], spec["brow"] + ".mhclo"), hm, asset_type="Eyebrows", subdiv_levels=0, material_type="GAMEENGINE")
    hair = None
    if spec["hair"]:
        hair = HumanService.add_mhclo_asset(asset("hair", spec["hair"], spec["hair"] + ".mhclo"), hm, asset_type="Hair", subdiv_levels=0, material_type="GAMEENGINE")
    inv = hm.matrix_world.inverted()
    for k, v in enumerate(hm.data.vertices):
        v.co = inv @ Vector(final[k].tolist())
    hm.data.update()
    for o in (eyes, brow, hair):
        if o is None:
            continue
        for m in list(o.modifiers):
            o.modifiers.remove(m)
        oi = o.matrix_world.inverted()
        for v in o.data.vertices:
            v.co = oi @ (o.matrix_world @ v.co + Vector(off.tolist()))
        o.data.update()
    # the soldier's weights, vertex for vertex
    for g in list(hm.vertex_groups):          # (the helpers' groups stay: the masks that hide them read them)
        if g.name in BONE_INDEX:
            hm.vertex_groups.remove(g)
    made = {}
    for k, gs in enumerate(CANON_GROUPS):
        for n, w in gs:
            if n not in made:
                made[n] = hm.vertex_groups.new(name=n)
            made[n].add([k], w, "REPLACE")
    for m in list(hm.modifiers):
        if m.type != "MASK":
            hm.modifiers.remove(m)
    bt = put_tile(BROW_TILES[spec["brow"]], load_image(asset("eyebrows", spec["brow"], spec["brow"] + ".png")), "grey", "image")
    ht = None
    if hair:
        ht = put_tile(HAIR_TILES[spec["hair"]], load_image(asset("hair", spec["hair"], spec["hair"] + "_diffuse.png")), "grey", "image")

    def skin(ratio):
        return Part(hm, SKIN_Y, R_SKIN, lambda dom: on_head(dom), ratio)
    for lod, (rs, brows) in enumerate(((1.0, True), (0.35, True), (0.12, False))):
        parts = [skin(rs), Part(eyes, EYES, R_EYE, weights=eye_weights)]
        if brows:
            parts.append(Part(brow, bt, R_BROW, weights=head_only))
        fname = "people_head_%d%s.qchar" % (i, "" if lod == 0 else "_lod%d" % lod)
        SIZES[fname[:-6]] = write_file(fname, parts)
    if hair:
        for lod, r in enumerate((1.0, 0.4)):
            fname = "people_hair_%d%s.qchar" % (i, "" if lod == 0 else "_lod%d" % lod)
            SIZES[fname[:-6]] = write_file(fname, [Part(hair, ht, R_HAIR, ratio=r, weights=head_only)])
    # (out of the way of the next one)
    for o in (hm, eyes, brow, hair):
        if o is not None:
            o.hide_set(True)
            o.location.x += 3.0 * (i + 1)
    log("face", i, "done: offset to the neck", tuple(round(float(c), 3) for c in off))


if ONLY != "atlas":
    for i, spec in enumerate(FACES):
        make_face(i + 1, spec)

# =====================================================================================
# The atlas: colour as JPEG, coverage (hair, brows, iris) as PNG
# =====================================================================================
img = bpy.data.images.new("people_atlas", ATLAS, ATLAS, alpha=True)
img.colorspace_settings.name = "Non-Color"
rgba = np.concatenate([atlas, np.ones((ATLAS, ATLAS, 1), np.float32)], axis=2)
img.pixels.foreach_set(rgba.ravel())
img.file_format = "JPEG"
img.filepath_raw = os.path.join(OUT, "soldier_atlas.jpg")
bpy.context.scene.render.image_settings.quality = 92
img.save()
log("wrote soldier_atlas.jpg", os.path.getsize(img.filepath_raw))
aimg = bpy.data.images.new("people_alpha", ATLAS, ATLAS, alpha=False)
aimg.colorspace_settings.name = "Non-Color"
a3 = np.repeat(alpha[..., None], 3, axis=2)
aimg.pixels.foreach_set(np.concatenate([a3, np.ones((ATLAS, ATLAS, 1), np.float32)], axis=2).ravel())
aimg.file_format = "PNG"
aimg.filepath_raw = os.path.join(OUT, "soldier_atlas_alpha.png")
bpy.context.scene.render.image_settings.color_mode = "BW"
bpy.context.scene.render.image_settings.compression = 90
aimg.save()
log("wrote soldier_atlas_alpha.png", os.path.getsize(aimg.filepath_raw))
total = sum(SIZES.values()) + os.path.getsize(img.filepath_raw) + os.path.getsize(aimg.filepath_raw)
log("all files: %d, %.1f MB" % (len(SIZES) + 2, total / 1048576.0))
