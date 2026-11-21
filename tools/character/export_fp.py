# Exports the soldier built by make_soldier.py for the game: the body without the head (what the
# player sees of himself), the head with helmet and eyes (drawn on other players and the fallen),
# each at three levels of detail, all materials packed into one texture atlas.
#
#   blender -b <out_dir>/soldier.blend --python tools/character/export_fp.py -- <game_assets_dir>/character
#
# Writes soldier_fp.qchar, soldier_head.qchar, their _lod1 / _lod2 versions (binary, see below)
# and soldier_atlas.jpg. The level 0 body is the mesh as built; everything else is decimated.
#
# Game axes: +Y up, the character faces -Z, its right hand is at +X
# (Blender: +Z up, facing -Y, right hand at -X).
#
# qchar layout (little endian):
#   char[4] "QCH1"; u32 bone_count, vertex_count, index_count
#   bones:    char[32] name; i32 parent; f32[3] head; f32[3] tail      (parents first)
#   vertices: f32[3] pos; f32[3] normal; f32[2] uv; u8[4] bones; u8[4] weights (sum 255)
#   indices:  u32
import bpy
import bmesh
import numpy as np
import os
import struct
import sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(argv[0])
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print("[export_fp]", *a, flush=True)


def to_game(v):
    return (-v[0], v[2], v[1])


rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")

# ---- bones, parents first ----
bones = []


def walk(b):
    bones.append(b)
    for c in b.children:
        walk(c)


for b in rig.data.bones:
    if b.parent is None:
        walk(b)
bone_index = {b.name: i for i, b in enumerate(bones)}


def game_name(n):
    """Rig bone name -> the name the game looks for (Hips, LeftArm, RightHandIndex1 ...)."""
    # (the face, borrowed from MakeHuman's default rig: names end in .L / .R)
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


out_name = [game_name(b.name) for b in bones]
log("bones", len(bones))
for b, n in zip(bones, out_name):
    log("  bone", b.name, "=", n, "<-", b.parent.name if b.parent else "-")

# Never exported: the mouth is shut, brows and lashes are alpha cards.
SKIP = ("teeth", "tongue", "hair", "brow", "lash")
# Only on the head: eyes ("low-poly" is the eye mesh) and the helmet.
HEAD_ONLY = ("low-poly", "helmet")
HEAD_BONES = {i for i, n in enumerate(out_name) if n == "Head" or n == "Jaw" or n.endswith(("Eye", "Lid", "Brow"))}
NECK_BONES = {i for i, n in enumerate(out_name) if n == "Neck"}
HAND_BONES = {i for i, n in enumerate(out_name) if "Hand" in n}          # hand + all finger bones

# ---- atlas ----
ATLAS = 4096
atlas = np.zeros((ATLAS, ATLAS, 4), np.float32)
atlas[..., 3] = 1.0
tiles = {}            # key -> (u0, v0, du, dv)
big_slots = [(0, 0, 2048), (2048, 0, 2048), (0, 2048, 1024), (1024, 2048, 1024), (2048, 2048, 1024), (3072, 2048, 1024)]
# (below the row of flat colours: what only the head needs)
helmet_slots = [(0, 3200, 896)]
eye_slots = [(1024, 3200, 512)]
flat_next = [0]


def base_color(mat):
    """(image or None, colour)"""
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                inp = node.inputs["Base Color"]
                if inp.links:
                    n = inp.links[0].from_node
                    # (MPFB skin materials may route the image through a mix node)
                    seen = set()
                    stack = [n]
                    while stack:
                        n = stack.pop()
                        if n in seen:
                            continue
                        seen.add(n)
                        if n.type == "TEX_IMAGE" and n.image:
                            return n.image, None
                        for i in n.inputs:
                            for l in i.links:
                                stack.append(l.from_node)
                return None, tuple(inp.default_value[:3])
    return None, (0.5, 0.5, 0.5)


def image_tile(img, slots=None):
    key = "img:" + img.name
    if key in tiles:
        return tiles[key]
    x, y, size = (slots if slots else big_slots).pop(0)
    cp = img.copy()
    cp.scale(size, size)
    px = np.empty(size * size * 4, np.float32)
    cp.pixels.foreach_get(px)
    px = px.reshape(size, size, 4)
    # Blender hands out linear values for sRGB images; store display values.
    rgb = np.clip(px[..., :3], 0.0, 1.0)
    if img.colorspace_settings.name == "sRGB":
        rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    atlas[y:y + size, x:x + size, :3] = rgb
    bpy.data.images.remove(cp)
    tiles[key] = (x / ATLAS, y / ATLAS, size / ATLAS, size / ATLAS)
    log("tile", img.name, tiles[key])
    return tiles[key]


def flat_tile(color):
    key = "flat:%.3f,%.3f,%.3f" % color
    if key in tiles:
        return tiles[key]
    i = flat_next[0]
    flat_next[0] += 1
    x, y, size = 64 * (i % 32), 3072 + 64 * (i // 32), 64
    c = np.array(color, np.float32)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    atlas[y:y + size, x:x + size, :3] = c
    # All UVs of a flat material land on the tile centre.
    tiles[key] = ((x + 32) / ATLAS, (y + 32) / ATLAS, 0.0, 0.0)
    return tiles[key]


GLOVE = flat_tile((0.016, 0.017, 0.016))
SKIN = flat_tile((0.42, 0.26, 0.18))          # the base of the neck (linear: about the face's tone)

# ---- meshes ----
dg = bpy.context.evaluated_depsgraph_get()


# The two large atlas tiles go to what fills the first-person view: the
# uniform and the vest (the skin is almost entirely covered).
def _prio(o):
    low = o.name.lower()
    return 0 if ("suit" in low or "vest" in low) else 1


def export(fname, head, ratio, weld=0.0):
    """One file: the body (head=False) or the head, decimated to `ratio` of its triangles.
    ratio is a function of the object's lower-case name. weld: metres within which the helmet's
    vertices are merged first (it is made of many loose pieces, which do not collapse)."""
    out_v = []          # (pos, normal, uv, bones, weights)
    out_i = []
    vmap = {}
    for obj in sorted([o for o in bpy.data.objects if o.type == "MESH"], key=_prio):
        low = obj.name.lower()
        if any(k in low for k in SKIP):
            continue
        head_only = any(k in low for k in HEAD_ONLY)
        if head_only and not head:
            continue
        # Rest pose: every modifier but the armature.
        for m in obj.modifiers:
            if m.type == "ARMATURE":
                m.show_viewport = False
        r = ratio(low)
        dec = None
        wel = None
        if weld > 0.0 and "helmet" in low:
            wel = obj.modifiers.new("export_weld", "WELD")
            wel.merge_threshold = weld
        if r < 0.999:
            dec = obj.modifiers.new("export_decimate", "DECIMATE")
            dec.decimate_type = "COLLAPSE"
            dec.ratio = r
            dec.use_collapse_triangulate = True
        dg.update()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
        if hasattr(me, "calc_normals_split"):
            me.calc_normals_split()
        mw = obj.matrix_world
        nm = mw.to_3x3().inverted().transposed()
        groups = {g.index: bone_index[g.name] for g in obj.vertex_groups if g.name in bone_index}
        uvl = me.uv_layers.active.data if me.uv_layers.active else None
        is_body = obj.name == "Human" or low.endswith("body") or (uvl is not None and "skin" in "".join(
            (s.material.name.lower() if s.material else "") for s in obj.material_slots))
        if head and not (is_body or head_only):
            ev.to_mesh_clear()
            if dec:
                obj.modifiers.remove(dec)
            if wel:
                obj.modifiers.remove(wel)
            continue
        mats = [s.material for s in obj.material_slots]
        mat_tile = []
        for mat in mats:
            img, col = base_color(mat)
            mat_tile.append(image_tile(img, (helmet_slots if "helmet" in low else eye_slots) if head_only else None) if img else flat_tile(col))
        if not mat_tile:
            mat_tile = [flat_tile((0.5, 0.5, 0.5))]

        vw = []
        for v in me.vertices:
            ws = sorted(((g.weight, groups[g.group]) for g in v.groups if g.group in groups and g.weight > 1e-4), reverse=True)[:4]
            if not ws:
                ws = [(1.0, next(iter(HEAD_BONES)) if head_only else 0)]
            tot = sum(w for w, _ in ws)
            q = [int(round(255.0 * w / tot)) for w, _ in ws]
            q[0] += 255 - sum(q)
            b = [bi for _, bi in ws]
            while len(b) < 4:
                b.append(0)
                q.append(0)
            vw.append((tuple(b), tuple(q)))

        kept = 0
        for poly in me.polygons:
            vs = list(poly.vertices)
            dom = [vw[v][0][0] for v in vs]
            # The body ends where the head begins: skin that follows the head, or the neck alone.
            on_head = any(d in HEAD_BONES for d in dom) or all(d in NECK_BONES for d in dom)
            if is_body and on_head != head:
                continue
            glove = is_body and all(d in HAND_BONES for d in dom)
            tile = GLOVE if glove else mat_tile[min(poly.material_index, len(mat_tile) - 1)]
            for li, v in zip(poly.loop_indices, vs):
                co = mw @ me.vertices[v].co
                if hasattr(me, "corner_normals") and len(me.corner_normals):
                    n = me.corner_normals[li].vector
                else:
                    n = me.loops[li].normal
                n = (nm @ n).normalized()
                if uvl is not None and tile[2] > 0.0:
                    uv = uvl[li].uv
                    u, w = tile[0] + (uv[0] % 1.0 if not 0.0 <= uv[0] <= 1.0 else uv[0]) * tile[2], \
                           tile[1] + (uv[1] % 1.0 if not 0.0 <= uv[1] <= 1.0 else uv[1]) * tile[3]
                else:
                    u, w = tile[0], tile[1]
                key = (obj.name, v, round(u, 5), round(w, 5), round(n[0], 3), round(n[1], 3), round(n[2], 3))
                idx = vmap.get(key)
                if idx is None:
                    idx = len(out_v)
                    vmap[key] = idx
                    # The game samples with v = 0 at the top of the image.
                    out_v.append((to_game(co), to_game(n), (u, 1.0 - w), vw[v][0], vw[v][1]))
                out_i.append(idx)
            kept += 1
        log(fname, "mesh", obj.name, "ratio %.2f" % r, "tris", kept)
        ev.to_mesh_clear()
        if dec:
            obj.modifiers.remove(dec)
        if wel:
            obj.modifiers.remove(wel)

    if not head:
        # The body has no head and its clothes are open at the collar: from the player's own
        # eyes, looking down, that is a hole into an empty torso. Closed with the base of the
        # neck: a slightly domed patch of skin inside the collar, carried by the neck joint.
        ni = next(iter(NECK_BONES))
        mwr = rig.matrix_world
        c = to_game(mwr @ bones[ni].head_local)
        cy = c[1] - 0.012
        rx, rz, segs = 0.082, 0.098, 20
        base = len(out_v)
        uvc = (SKIN[0], 1.0 - SKIN[1])
        w = ((ni, 0, 0, 0), (255, 0, 0, 0))
        out_v.append(((c[0], cy + 0.018, c[2] - 0.004), (0.0, 1.0, 0.0), uvc, w[0], w[1]))
        import math
        for k in range(segs):
            a = 2.0 * math.pi * k / segs
            out_v.append(((c[0] + rx * math.cos(a), cy, c[2] - 0.004 + rz * math.sin(a)), (0.0, 1.0, 0.0), uvc, w[0], w[1]))
        for k in range(segs):
            # (wound to face up: counter-clockwise seen from above, +Y)
            out_i.extend((base, base + 1 + (k + 1) % segs, base + 1 + k))
    # Blender winds counter-clockwise seen from outside; the axis change is a
    # rotation, so the winding is kept.
    path = os.path.join(OUT, fname)
    with open(path, "wb") as f:
        f.write(b"QCH1")
        f.write(struct.pack("<III", len(bones), len(out_v), len(out_i)))
        for b in bones:
            mwr = rig.matrix_world
            f.write(struct.pack("<32s", out_name[bone_index[b.name]].encode("ascii", "replace")[:31]))
            f.write(struct.pack("<i", bone_index[b.parent.name] if b.parent else -1))
            f.write(struct.pack("<3f", *to_game(mwr @ b.head_local)))
            f.write(struct.pack("<3f", *to_game(mwr @ b.tail_local)))
        for p, n, uv, bi, bw in out_v:
            f.write(struct.pack("<3f3f2f4B4B", *p, *n, *uv, *bi, *bw))
        f.write(np.asarray(out_i, np.uint32).tobytes())
    log("wrote", path, "vertices", len(out_v), "triangles", len(out_i) // 3, "bytes", os.path.getsize(path))


# Level 0 is what the player sees of himself and what stands right in front of him. The boots
# and the vest are modelled far denser than that needs (30k and 37k triangles of 79k): half of
# them here. The helmet: a third (below that its netting loses its texture coordinates).
export("soldier_fp.qchar", False, lambda low: 0.45 if "boots" in low else 0.55 if "vest" in low else 1.0)
export("soldier_fp_lod1.qchar", False, lambda low: 0.30)
export("soldier_fp_lod2.qchar", False, lambda low: 0.09)
export("soldier_head.qchar", True, lambda low: 0.30 if "helmet" in low else 1.0)
export("soldier_head_lod1.qchar", True, lambda low: 0.08 if "helmet" in low else 0.35, 0.004)
export("soldier_head_lod2.qchar", True, lambda low: 0.08 if "helmet" in low else 0.12, 0.004)

img = bpy.data.images.new("soldier_atlas", ATLAS, ATLAS, alpha=False)
img.colorspace_settings.name = "sRGB"
# (values are already display-referred: write them as they are)
img.colorspace_settings.name = "Non-Color"
img.pixels.foreach_set(atlas.ravel())
img.file_format = "JPEG"
img.filepath_raw = os.path.join(OUT, "soldier_atlas.jpg")
bpy.context.scene.render.image_settings.quality = 92
img.save()
log("wrote", img.filepath_raw, os.path.getsize(img.filepath_raw))
