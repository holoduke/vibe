# Builds the game's boulders in Blender: each rock is sculpted procedurally
# at high resolution (fractured block, cracks, bedding, pitting), reduced to
# four game meshes, and the fine detail is baked into one texture atlas.
#
#   blender -b --python tools/rocks/make_rocks.py -- <game_assets_dir>/rocks [preview.png]
#
# Writes rocks.qrock (meshes) and rocks_bake.png (atlas, 3 x 2 tiles):
#   R,G = object-space normal (octahedral, game axes), B = ambient occlusion,
#   A = curvature (0.5 flat, > 0.5 convex edge, < 0.5 crevice).
#
# qrock layout (little endian): char[4] "QRK1"; u32 rock_count; per rock,
# per LOD (4): u32 vertex_count, index_count; vertices f32 pos[3], normal[3],
# uv[2] (atlas uv, v = 0 at the top of the image); indices u32.
# Game axes: +Y up. A rock is about 2 m across, centred 0.35 m above its
# origin (the origin sits on the ground; the foot is buried).
import bpy
import bmesh
import math
import os
import random
import struct
import sys
import numpy as np
from mathutils import Vector, noise

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(argv[0])
PREVIEW = os.path.abspath(argv[1]) if len(argv) > 1 else None
os.makedirs(OUT, exist_ok=True)

TILE = 1024
COLS, ROWS = 3, 2
LOD_TRIS = (5200, 1500, 420, 110)

# kind: (name, scale xyz, fracture planes, crack scale, layered, lumpiness)
ROCKS = [
    ("angular block",  (1.00, 0.86, 0.74), 15, 2.6, False, 0.9),
    ("layered rock",   (1.05, 0.92, 0.70), 11, 2.1, True,  0.7),
    ("round boulder",  (0.95, 0.90, 0.78),  7, 3.2, False, 1.5),
    ("split boulder",  (1.02, 0.80, 0.82), 12, 1.7, False, 1.1),
    ("tall shard",     (0.78, 0.70, 1.02), 14, 2.9, True,  0.8),
]


def log(*a):
    print("[rocks]", *a, flush=True)


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_modifier(obj, mod):
    activate(obj)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def make_high(k, spec):
    name, scale, planes, crack_s, layered, lumps = spec
    rnd = random.Random(1000 + k * 77)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * scale[0], v.co.y * scale[1], v.co.z * scale[2]))
    # Fracture: cut the block with random planes; each cut is a flat face.
    for i in range(planes):
        n = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1)))
        if i == 0:
            n = Vector((rnd.uniform(-0.15, 0.15), rnd.uniform(-0.15, 0.15), -1.0))     # flat-ish underside
        if n.length < 1e-3:
            continue
        n.normalize()
        d = rnd.uniform(0.58, 0.92) * (0.72 if i == 0 else 1.0)
        co = Vector((n.x * d * scale[0], n.y * d * scale[1], n.z * d * scale[2]))
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=co, plane_no=n, clear_outer=True)
        edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        if edges:
            bmesh.ops.edgeloop_fill(bm, edges=edges)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("rock_high_%d" % k)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("rock_high_%d" % k, me)
    bpy.context.scene.collection.objects.link(obj)

    # Even, dense skin (the cuts leave long thin triangles).
    rm = obj.modifiers.new("remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.0105
    apply_modifier(obj, rm)
    me = obj.data
    log(name, "high-res vertices", len(me.vertices))

    # Sculpt: displace every vertex along its normal.
    bed_n = Vector((0.16 * math.sin(k * 1.7), 0.16 * math.cos(k * 2.3), 1.0)).normalized()
    off = Vector((k * 13.1, k * 7.7, k * 3.3))
    cos = np.empty(len(me.vertices) * 3, np.float32)
    nos = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", cos)
    me.vertices.foreach_get("normal", nos)
    cos = cos.reshape(-1, 3)
    nos = nos.reshape(-1, 3)
    out = cos.copy()
    for i in range(len(cos)):
        p = Vector(cos[i])
        q = p + off
        d = 0.0
        # Weathered surface: ridges and hollows the way erosion leaves them
        # (ridged noise: sharp crests, smooth troughs), on two scales.
        d += (noise.ridged_multi_fractal(q * 1.25, 0.9, 2.1, 4, 1.0, 2.0) - 1.1) * 0.040 * lumps
        d += (noise.ridged_multi_fractal(q * 4.6 + Vector((3.0, 7.0, 1.0)), 1.0, 2.0, 3, 1.0, 2.0) - 1.0) * 0.011
        # Fractures: a few long, narrow, deep cracks - only in the zones a
        # low-frequency mask allows, so most of the rock is unbroken.
        zone = smoothstep(0.02, 0.22, noise.noise(q * 0.75 + Vector((11.0, 0.0, 5.0))))
        vd = noise.voronoi(q * crack_s * 0.55, distance_metric="DISTANCE")[0]
        edge = vd[1] - vd[0]
        d -= 0.050 * zone * (1.0 - smoothstep(0.0, 0.030, edge))
        d -= 0.012 * zone * (1.0 - smoothstep(0.0, 0.110, edge))            # the crack's worn shoulders
        # Hairline cracks.
        vd2 = noise.voronoi(q * crack_s * 2.3 + Vector((5.0, 1.0, 9.0)), distance_metric="DISTANCE")[0]
        hz = smoothstep(0.10, 0.35, noise.noise(q * 1.9 + Vector((2.0, 4.0, 8.0))))
        d -= 0.009 * hz * (1.0 - smoothstep(0.0, 0.035, vd2[1] - vd2[0]))
        # Bedding: layers, the joints between them weathered back.
        if layered:
            bc = p.dot(bed_n) * 4.2 + 0.45 * noise.noise(q * 0.9)
            fr = bc - math.floor(bc)
            joint = 1.0 - smoothstep(0.0, 0.10, fr) * (1.0 - smoothstep(0.86, 1.0, fr))
            d -= 0.045 * joint
            d += 0.012 * math.sin(math.floor(bc) * 12.9898)             # each bed stands a little in or out
        # Pitting and grain.
        d -= 0.010 * smoothstep(0.55, 0.9, noise.noise(q * 9.0) * 0.5 + 0.5)
        d += noise.fractal(q * 17.0, 1.0, 2.0, 2) * 0.004
        out[i] = cos[i] + nos[i] * d
    me.vertices.foreach_set("co", out.ravel())
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    return obj


def decimated_copy(src, name, tris):
    activate(src)
    bpy.ops.object.duplicate()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    n = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = min(1.0, tris / max(n, 1))
    mod.use_collapse_triangulate = True
    apply_modifier(obj, mod)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def bake_material(obj, image):
    mat = bpy.data.materials.new(obj.name + "_bake")
    mat.use_nodes = True
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    nt.nodes.active = tex
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return tex


def pointiness_material(obj):
    mat = bpy.data.materials.new(obj.name + "_curv")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    emit = nt.nodes.new("ShaderNodeEmission")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs["Pointiness"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], outn.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def bake(high, low, kind, image):
    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    kw = dict(use_selected_to_active=True, cage_extrusion=0.05, max_ray_distance=0.12, margin=10)
    if kind == "NORMAL":
        bpy.ops.object.bake(type="NORMAL", normal_space="OBJECT", **kw)
    else:
        bpy.ops.object.bake(type=kind, **kw)
    px = np.empty(TILE * TILE * 4, np.float32)
    image.pixels.foreach_get(px)
    return px.reshape(TILE, TILE, 4).copy()


def oct_encode(n):
    """n: (..., 3) unit vectors -> (..., 2) in 0..1."""
    s = np.abs(n).sum(axis=-1, keepdims=True)
    s[s < 1e-6] = 1.0
    m = n / s
    x, y, z = m[..., 0], m[..., 1], m[..., 2]
    neg = z < 0.0
    xo = np.where(neg, (1.0 - np.abs(y)) * np.where(x >= 0.0, 1.0, -1.0), x)
    yo = np.where(neg, (1.0 - np.abs(x)) * np.where(y >= 0.0, 1.0, -1.0), y)
    return np.stack([xo * 0.5 + 0.5, yo * 0.5 + 0.5], axis=-1)


# ---------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.render.bake.use_clear = True

atlas = np.zeros((ROWS * TILE, COLS * TILE, 4), np.float32)
atlas[..., 0:2] = 0.5
atlas[..., 2] = 1.0
atlas[..., 3] = 0.5
meshes = []        # per rock: [ (verts, idx) x 4 ]
highs = []

for k, spec in enumerate(ROCKS):
    high = make_high(k, spec)
    highs.append(high)
    low = decimated_copy(high, "rock_low_%d" % k, LOD_TRIS[0])
    # UVs for the game mesh.
    activate(low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.012)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Only this rock is in the scene while it bakes (the others sit in the
    # same place), and the game mesh itself must not shade the sculpt.
    for o in bpy.data.objects:
        o.hide_render = o not in (high, low)
    for flag in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_shadow", "visible_volume_scatter"):
        setattr(low, flag, False)
    img = bpy.data.images.new("bake_%d" % k, TILE, TILE, alpha=False, float_buffer=True, is_data=True)
    bake_material(low, img)
    nrm = bake(high, low, "NORMAL", img)
    ao = bake(high, low, "AO", img)
    saved = [m for m in high.data.materials]
    pointiness_material(high)
    curv = bake(high, low, "EMIT", img)
    high.data.materials.clear()
    for m in saved:
        high.data.materials.append(m)

    n = nrm[..., :3] * 2.0 - 1.0
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ln[ln < 1e-4] = 1.0
    n = n / ln
    game_n = np.stack([n[..., 0], n[..., 2], -n[..., 1]], axis=-1)        # Blender (x, y, z) -> game (x, z, -y)
    tile = np.empty((TILE, TILE, 4), np.float32)
    tile[..., 0:2] = oct_encode(game_n)
    tile[..., 2] = np.clip(ao[..., 0], 0.0, 1.0)
    # Pointiness is ~0.5 on flat ground with a small spread: stretch it
    # around its median so edges and crevices use the whole range.
    cv = curv[..., 0]
    inside = ao[..., 0] > 0.0
    med = float(np.median(cv[inside])) if inside.any() else 0.5
    spread = float(np.std(cv[inside])) if inside.any() else 0.05
    tile[..., 3] = np.clip(0.5 + (cv - med) / max(spread * 5.0, 1e-3), 0.0, 1.0)
    tx, ty = k % COLS, k // COLS
    atlas[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE] = tile
    log(spec[0], "baked; ao mean %.2f, curvature median %.3f spread %.3f" % (float(ao[..., 0].mean()), med, spread))

    # The four game meshes (the coarser ones keep the UVs of the finest).
    lods = []
    for l, tris in enumerate(LOD_TRIS):
        obj = low if l == 0 else decimated_copy(low, "rock_lod%d_%d" % (l, k), tris)
        me = obj.data
        uvl = me.uv_layers.active.data
        verts, idx, vmap = [], [], {}
        for poly in me.polygons:
            if len(poly.vertices) != 3:
                continue
            for li, vi in zip(poly.loop_indices, poly.vertices):
                uv = uvl[li].uv
                key = (vi, round(uv[0], 5), round(uv[1], 5))
                j = vmap.get(key)
                if j is None:
                    j = len(verts)
                    vmap[key] = j
                    co = me.vertices[vi].co
                    no = me.vertices[vi].normal
                    ua = (tx + uv[0]) / COLS
                    va = (ty + uv[1]) / ROWS
                    verts.append((co.x, co.z + 0.35, -co.y, no.x, no.z, -no.y, ua, 1.0 - va))
                idx.append(j)
        lods.append((verts, idx))
        log("  lod", l, len(verts), "vertices", len(idx) // 3, "triangles")
    meshes.append(lods)

with open(os.path.join(OUT, "rocks.qrock"), "wb") as f:
    f.write(b"QRK1")
    f.write(struct.pack("<I", len(meshes)))
    for lods in meshes:
        for verts, idx in lods:
            f.write(struct.pack("<II", len(verts), len(idx)))
            f.write(np.asarray(verts, np.float32).tobytes())
            f.write(np.asarray(idx, np.uint32).tobytes())
log("wrote rocks.qrock", os.path.getsize(os.path.join(OUT, "rocks.qrock")))

img = bpy.data.images.new("rocks_bake", COLS * TILE, ROWS * TILE, alpha=True, float_buffer=False, is_data=True)
img.alpha_mode = "CHANNEL_PACKED"
img.pixels.foreach_set(atlas.ravel())
img.file_format = "PNG"
img.filepath_raw = os.path.join(OUT, "rocks_bake.png")
img.save()
log("wrote rocks_bake.png", os.path.getsize(img.filepath_raw))

# ---- preview of the sculpted rocks ----
if PREVIEW:
    for o in bpy.data.objects:
        o.hide_render = False
    for o in list(bpy.data.objects):
        if o not in highs:
            bpy.data.objects.remove(o, do_unlink=True)
    grey = bpy.data.materials.new("grey")
    grey.use_nodes = True
    grey.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.45, 0.43, 0.40, 1.0)
    grey.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.85
    for i, h in enumerate(highs):
        h.location = ((i - (len(highs) - 1) * 0.5) * 2.3, 0.0, 0.0)
        h.data.materials.clear()
        h.data.materials.append(grey)
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0.0, -9.5, 2.6)
    cam.rotation_euler = (math.radians(76.0), 0.0, 0.0)
    cam_data.lens = 42
    scene.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.0
    sun.data.angle = math.radians(3.0)
    sun.rotation_euler = (math.radians(52.0), 0.0, math.radians(38.0))
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.42, 0.55, 1.0)
    scene.world = world
    scene.render.resolution_x = 1800
    scene.render.resolution_y = 560
    scene.cycles.samples = 48
    scene.render.filepath = PREVIEW
    bpy.ops.render.render(write_still=True)
    log("preview", PREVIEW)
