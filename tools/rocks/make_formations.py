# Rock formations for the landscape: a KIT of large rock pieces with real gaps - an arch, a
# cleft wall, an overhang, a stack of slabs, leaning pillars, a boulder pile, a cliff buttress,
# a low outcrop - each made from a seed, at three levels of detail, with a collision grid.
#
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --python tools/rocks/make_formations.py
#
# writes assets/rocks/formations.rockf (read by src/engine/vk_engine/rock_formations.cpp).
# Options after "--":  --only arch,pile   make only these (the file then holds only these)
#                      --quick            a quarter of the dense triangles (a fast look)
#                      --preview DIR      also render each piece to DIR/<name>.png
#                      --out FILE         another output file
# Deterministic: the same script gives the same file (every random number comes from the
# piece's seed; no Blender random operators are used).
#
# HOW A PIECE IS MADE
#   1. Block-out: boxes and chopped lumps, laid through each other (blockout_*).
#   2. Fused by a voxel remesh, then bent out of true by slow noise.
#   3. Holes: boolean cuts - the arch's opening, the cleft, the hollow under the overhang,
#      a chimney, windows.
#   4. Voxel remesh at the dense resolution (1.5 - 2.5 million triangles), then the geology:
#      a. beds - level layers of uneven thickness, each with its own hardness: hard beds stand
#         out as ledges, soft ones are eaten back (and undercut the bed above); a parting line
#         between two beds;
#      b. joints - two sets of upright cracks that cut every bed into blocks; a block sits a
#         little in or out of line, here and there one has fallen out;
#      c. weathering - tops rounded and hollowed, the joints opened into grooves on them;
#      then a second remesh and the fine relief along the normal: lumps, thin laminations
#      on the faces, hair cracks, pits (less of all of it on the weathered tops).
#   5. Three game meshes by edge-collapse decimation of the dense mesh (the holes stay: a
#      collapse does not close a hole). Normals: smooth, split at edges sharper than 42 degrees.
#      No normal-map bake: eight pieces would need eight more texture slots and a UV unwrap of
#      shapes with holes; the fine relief comes from the tiling rock material (colour, normal,
#      height through the engine's screen-space displacement) laid on by world position.
#   6. Per vertex, for the material: u = ambient occlusion (rays against the nearest mesh;
#      dark in clefts and under overhangs), v = curvature (0.5 flat, more = an edge, less = a
#      crease).
#   7. Collision: the piece's solid cells in a coarse grid (0.5 - 1.4 m), so the arch's
#      opening and the big gaps stay open; the engine turns them into boxes where it stands.
#
# FILE (little endian): "RKF1", u32 pieces; per piece: char name[24], u32 levels; per level:
# u32 vertices, u32 indices, f32 bmin[3], f32 bmax[3], vertices (f32 pos[3], normal[3], ao,
# curvature), indices u32; then the grid: f32 origin[3] (its least corner), f32 cell,
# u32 nx, ny, nz, u8 solid[nx * ny * nz] (x fastest, then y, then z).
# Game axes: x = the piece's width, y = up, z = toward its FRONT (its back, z < 0, is what is
# sunk into a slope). y = 0 is where the ground is meant to be; every piece has a root below.
import bpy
import bmesh
import math
import os
import struct
import sys
import time
import numpy as np
from mathutils import Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "assets", "rocks", "formations.rockf")
ONLY = None
QUICK = False
PREVIEW = None
i = 0
while i < len(argv):
    if argv[i] == "--only": ONLY = argv[i + 1].split(","); i += 1
    elif argv[i] == "--quick": QUICK = True
    elif argv[i] == "--preview": PREVIEW = os.path.abspath(argv[i + 1]); i += 1
    elif argv[i] == "--out": OUT = os.path.abspath(argv[i + 1]); i += 1
    i += 1

DENSE_TRIS = 500000 if QUICK else 2000000
T0 = time.time()


def log(*a):
    print("[formations %6.1fs]" % (time.time() - T0), *a, flush=True)


# ---------------------------------------------------------------- noise (numpy) ----------
M32 = np.uint64(0xFFFFFFFF)


def hash3(ix, iy, iz, seed):
    """0..1 from three integer arrays (any shape) and a seed."""
    a = (np.asarray(ix).astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    b = (np.asarray(iy).astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    c = (np.asarray(iz).astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    h = (a * np.uint64(0x9E3779B1)) ^ ((b + np.uint64(0x7F4A7C15)) * np.uint64(0x85EBCA6B)) ^ ((c + np.uint64(0x165667B1)) * np.uint64(0xC2B2AE35))
    h = (h ^ np.uint64((seed * 0x27D4EB2F) & 0xFFFFFFFF)) & M32
    h ^= h >> np.uint64(15); h = (h * np.uint64(0x2C1B3C6D)) & M32
    h ^= h >> np.uint64(12); h = (h * np.uint64(0x297A2D39)) & M32
    h ^= h >> np.uint64(15)
    return h.astype(np.float64) / 4294967296.0


def vnoise(p, seed):
    """Value noise 0..1 at points p (N, 3)."""
    f = np.floor(p)
    t = p - f
    t = t * t * (3.0 - 2.0 * t)
    ix, iy, iz = f[:, 0], f[:, 1], f[:, 2]
    out = np.zeros(len(p))
    for dx in (0, 1):
        wx = t[:, 0] if dx else 1.0 - t[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1.0 - t[:, 1]
            for dz in (0, 1):
                wz = t[:, 2] if dz else 1.0 - t[:, 2]
                out += wx * wy * wz * hash3(ix + dx, iy + dy, iz + dz, seed)
    return out


def fbm(p, seed, octaves, gain=0.5):
    out = np.zeros(len(p)); a = 1.0; tot = 0.0; q = p.copy()
    for o in range(octaves):
        out += a * vnoise(q, seed + o * 17)
        tot += a; a *= gain; q = q * 2.03 + 11.7
    return out / tot


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- Blender helpers --------
def new_obj(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def apply_mod(ob, mod):
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(deps))
    ob.modifiers.remove(mod)
    old = ob.data
    ob.data = me
    bpy.data.meshes.remove(old)


def remesh(ob, voxel):
    m = ob.modifiers.new("fuse", "REMESH")
    m.mode = "VOXEL"; m.voxel_size = voxel; m.adaptivity = 0.0
    apply_mod(ob, m)


def cut(ob, cutter):
    m = ob.modifiers.new("cut", "BOOLEAN")
    m.operation = "DIFFERENCE"; m.object = cutter
    try: m.solver = "MANIFOLD"
    except Exception: m.solver = "EXACT"
    apply_mod(ob, m)
    me = cutter.data
    bpy.data.objects.remove(cutter)
    bpy.data.meshes.remove(me)


def coords(me):
    a = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def normals(me):
    a = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", a)
    return a.reshape(-1, 3)


def set_coords(me, co):
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()


def tri_count(me):
    return sum(len(p.vertices) - 2 for p in me.polygons) if len(me.polygons) < 200000 else int(len(me.loops) - 2 * len(me.polygons))


def area(me):
    a = np.empty(len(me.polygons))
    me.polygons.foreach_get("area", a)
    return float(a.sum())


def rot(rx, ry, rz):
    return Euler((rx, ry, rz), "XYZ").to_matrix().to_4x4()


def box(bm, c, size, r=(0.0, 0.0, 0.0)):
    m = Matrix.Translation(Vector(c)) @ rot(*r) @ Matrix.Diagonal((size[0], size[1], size[2], 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)


def lump(bm, rng, c, radii, chops=11, sub=3, r=None):
    """A chopped ball: an icosphere with flats cut into it (a rock, not a potato)."""
    ret = bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    vs = ret["verts"]
    co = np.array([v.co[:] for v in vs])
    for k in range(chops):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        o = rng.uniform(0.42, 0.85)
        t = co @ d - o
        co -= np.outer(np.maximum(t, 0.0), d)
    if r is None: r = rng.uniform(-3.14, 3.14, size=3)
    m = Matrix.Translation(Vector(c)) @ rot(*r) @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    for v, p in zip(vs, co):
        v.co = m @ Vector(p)


def cutter_box(name, c, size, r=(0.0, 0.0, 0.0)):
    bm = bmesh.new()
    box(bm, c, size, r)
    return new_obj(name, bm)


def cutter_ball(name, rng, c, radii, r=(0.0, 0.0, 0.0), rough=0.12, seed=1):
    """A hollow to cut: a ball with an uneven wall."""
    bm = bmesh.new()
    ret = bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    vs = ret["verts"]
    co = np.array([v.co[:] for v in vs])
    co *= (1.0 + rough * 2.0 * (fbm(co * 1.7 + 3.0, seed, 3) - 0.5))[:, None]
    m = Matrix.Translation(Vector(c)) @ rot(*r) @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    for v, p in zip(vs, co):
        v.co = m @ Vector(p)
    return new_obj(name, bm)


def cutter_tunnel(name, c, radius, length, zscale):
    """A round-topped passage along y."""
    bm = bmesh.new()
    m = Matrix.Translation(Vector(c)) @ Matrix.Diagonal((1.0, 1.0, zscale, 1.0)) @ rot(math.pi / 2, 0.0, 0.0)
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=40, radius1=radius, radius2=radius, depth=length, matrix=m)
    return new_obj(name, bm)


# ---------------------------------------------------------------- block-outs --------------
# Blender axes while building: x = width, y = depth (the BACK is +y), z = up; z = 0 the ground.
# Each returns (bmesh of solids, list of cutters, parameters).
def blockout_arch(rng, seed):
    bm = bmesh.new()
    # two piers, each a pile of blocks that do not line up, and a span that is thickest in the middle
    box(bm, (-6.0, 0.2, 0.2), (5.6, 5.8, 6.6), (0.0, 0.05, 0.22))
    box(bm, (-5.4, -0.2, 5.0), (4.4, 4.6, 5.0), (0.04, -0.06, -0.15))
    box(bm, (5.9, 0.3, -0.2), (5.8, 6.2, 6.0), (0.03, -0.05, -0.28))
    box(bm, (5.3, 0.0, 4.6), (4.6, 4.8, 5.2), (-0.04, 0.06, 0.12))
    box(bm, (0.0, 0.0, 7.6), (15.5, 4.2, 2.6), (0.0, 0.05, 0.03))
    box(bm, (0.4, 0.2, 8.5), (9.0, 3.4, 2.6), (0.05, -0.04, -0.08))
    lump(bm, rng, (-5.6, 0.3, 9.2), (3.6, 2.8, 1.5))
    lump(bm, rng, (4.6, -0.2, 9.3), (3.4, 2.6, 1.3))
    lump(bm, rng, (-8.6, 0.4, 0.9), (2.6, 2.8, 2.8))
    lump(bm, rng, (8.8, -0.3, 0.7), (2.4, 3.0, 2.5))
    lump(bm, rng, (-7.4, -1.6, 4.4), (1.8, 1.9, 2.2))
    lump(bm, rng, (7.6, 1.4, 5.2), (1.9, 1.8, 2.0))
    cuts = [cutter_tunnel("c0", (-0.1, 0.0, 2.6), 3.6, 14.0, 1.22),
            cutter_box("c1", (-0.1, 0.0, -0.6), (7.2, 14.0, 6.4)),
            cutter_ball("c2", rng, (5.7, 0.0, 5.0), (1.0, 4.5, 1.4), seed=seed)]
    return bm, cuts, dict(bed=0.80, joint=2.0, lods=(36000, 7000, 1500))


def blockout_cleft(rng, seed):
    bm = bmesh.new()
    box(bm, (0.0, 0.5, 4.0), (14.0, 4.6, 14.0), (0.0, 0.0, 0.03))
    box(bm, (-8.0, 0.8, 2.0), (3.5, 4.0, 8.0), (0.0, 0.05, 0.1))
    box(bm, (8.2, 0.6, 1.2), (3.6, 3.8, 6.6), (0.03, -0.04, -0.08))
    lump(bm, rng, (-4.5, 0.5, 10.6), (3.4, 2.4, 2.0))
    lump(bm, rng, (3.8, 0.4, 11.2), (3.8, 2.5, 2.4))
    lump(bm, rng, (-3.0, -2.6, 0.4), (2.6, 1.6, 1.6))
    lump(bm, rng, (4.5, -2.5, 0.2), (2.2, 1.5, 1.3))
    cuts = [cutter_box("c0", (0.6, 0.0, 3.1), (0.9, 12.0, 4.2), (0.0, 0.10, 0.0)),
            cutter_box("c1", (1.1, 0.0, 6.8), (1.25, 12.0, 4.8), (0.0, -0.07, 0.04)),
            cutter_box("c2", (0.7, 0.0, 11.4), (1.9, 12.0, 5.8), (0.0, 0.05, -0.03)),
            cutter_ball("c3", rng, (-6.4, -2.2, 5.6), (1.5, 2.2, 1.9), seed=seed)]
    return bm, cuts, dict(bed=0.90, joint=2.2, lods=(30000, 6000, 1400))


def blockout_overhang(rng, seed):
    bm = bmesh.new()
    box(bm, (0.0, 2.6, 2.0), (12.0, 4.4, 10.0), (0.0, 0.0, 0.02))
    box(bm, (0.3, -1.2, 5.6), (13.5, 9.5, 2.6), (0.06, 0.0, 0.03))
    box(bm, (-5.6, -0.8, 1.2), (2.6, 5.5, 8.4), (0.0, 0.03, 0.06))
    lump(bm, rng, (-2.0, 1.5, 7.6), (4.5, 3.2, 1.6))
    lump(bm, rng, (3.5, 1.0, 7.4), (3.6, 3.0, 1.4))
    lump(bm, rng, (5.6, 1.6, 0.8), (2.2, 2.6, 2.4))
    cuts = [cutter_ball("c0", rng, (1.2, -2.4, 1.8), (5.2, 4.5, 2.7), seed=seed)]
    return bm, cuts, dict(bed=0.70, joint=2.0, lods=(30000, 6000, 1400))


def blockout_stack(rng, seed):
    bm = bmesh.new()
    z = -2.6
    box(bm, (0.0, 0.0, 2.0), (3.2, 2.6, 9.0))                         # the core the slabs hang on
    for k in range(8):
        t = rng.uniform(0.60, 1.30)
        shrink = 1.0 - 0.045 * k
        w, d = rng.uniform(6.0, 9.0) * shrink, rng.uniform(4.8, 7.0) * shrink
        gap = (0.0, 0.2, 0.3, 0.0)[int(rng.randint(0, 4))] if k > 1 else 0.0
        z += gap
        box(bm, (rng.uniform(-0.9, 0.9), rng.uniform(-0.8, 0.8), z + 0.5 * t), (w, d, t), (rng.uniform(-0.035, 0.035), rng.uniform(-0.035, 0.035), rng.uniform(-0.3, 0.3)))
        z += t
    lump(bm, rng, (0.4, 0.2, z + 0.2), (2.4, 1.9, 0.8))
    return bm, [], dict(bed=0.36, joint=1.7, missing=0.05, lods=(34000, 7000, 1600))


def blockout_pillars(rng, seed):
    bm = bmesh.new()
    P = [((-4.2, 0.5), (2.2, 1.9, 11.0), (0.06, 0.10)), ((-1.2, -0.6), (1.7, 1.6, 8.5), (-0.12, -0.05)),
         ((1.9, 0.9), (2.5, 2.1, 12.5), (0.03, -0.08)), ((4.8, -0.3), (1.8, 1.7, 7.0), (0.14, 0.12)),
         ((0.6, 3.0), (1.9, 2.0, 9.5), (-0.20, 0.04))]
    for (bx, by), (wx, wy, h), (rx, ry) in P:
        R = rot(rx, ry, rng.uniform(-0.4, 0.4))
        c = Vector((bx, by, -3.0)) + R @ Vector((0.0, 0.0, 0.5 * (h + 3.0)))
        m = Matrix.Translation(c) @ R @ Matrix.Diagonal((wx, wy, h + 3.0, 1.0))
        bmesh.ops.create_cube(bm, size=1.0, matrix=m)
        lump(bm, rng, (bx + rng.uniform(-0.5, 0.5), by + rng.uniform(-0.5, 0.5), 0.1), (1.7, 1.6, 1.1))
    box(bm, (3.3, -2.7, 2.3), (1.5, 1.4, 7.6), (0.5, 1.0, 0.3))        # one that fell and leans on its neighbour
    return bm, [], dict(bed=0.60, joint=1.3, amp=0.7, missing=0.05, lods=(34000, 7000, 1600))


def blockout_pile(rng, seed):
    bm = bmesh.new()
    for k in range(6):
        a = k * math.tau / 6 + rng.uniform(-0.2, 0.2)
        r = rng.uniform(1.35, 1.95)
        d = rng.uniform(2.7, 3.3)
        lump(bm, rng, (d * math.cos(a), d * math.sin(a), 0.42 * r), (r, r * rng.uniform(0.8, 1.0), r * 0.78), chops=8, sub=4)
    for k in range(3):
        a = k * math.tau / 3 + 0.5 + rng.uniform(-0.3, 0.3)
        r = rng.uniform(1.15, 1.5)
        lump(bm, rng, (1.45 * math.cos(a), 1.45 * math.sin(a), 2.25 + rng.uniform(-0.1, 0.2)), (r, r * 0.9, r * 0.8), chops=8, sub=4)
    lump(bm, rng, (0.1, -0.1, 3.75), (1.25, 1.1, 0.95), chops=8, sub=4)
    for k in range(5):
        a = k * math.tau / 5 + 0.2 + rng.uniform(-0.3, 0.3)
        r = rng.uniform(0.6, 0.95)
        lump(bm, rng, (4.7 * math.cos(a), 4.7 * math.sin(a), 0.3 * r), (r, r, r * 0.8), chops=7, sub=3)
    return bm, [], dict(bed=0.50, joint=1.5, amp=0.45, missing=0.0, warp=0.4, lods=(30000, 6000, 1400))


def blockout_buttress(rng, seed):
    bm = bmesh.new()
    box(bm, (0.0, 1.5, 1.0), (22.0, 11.0, 14.0), (0.0, 0.0, 0.02))
    box(bm, (-1.0, 2.5, 11.5), (17.0, 9.0, 11.0), (0.0, 0.02, -0.04))
    box(bm, (1.0, 3.4, 20.0), (12.5, 7.4, 10.0), (0.02, 0.0, 0.05))
    box(bm, (-0.5, 4.2, 27.5), (8.0, 5.6, 9.0), (0.0, -0.03, -0.03))
    box(bm, (-6.5, -3.6, 4.0), (4.2, 5.0, 20.0), (0.05, 0.02, 0.1))
    box(bm, (5.5, -3.0, 2.5), (5.0, 5.5, 17.0), (0.03, -0.03, -0.08))
    box(bm, (0.5, -1.6, 9.0), (3.6, 4.0, 24.0), (0.04, 0.0, 0.03))
    for k in range(7):
        zc = rng.uniform(6.0, 30.0)
        half = 11.0 - 0.28 * zc
        lump(bm, rng, (rng.uniform(-half, half), rng.uniform(-3.5, 1.0) + 0.12 * zc, zc), (rng.uniform(2.0, 3.6), rng.uniform(2.0, 3.0), rng.uniform(1.4, 2.6)))
    cuts = [cutter_box("c0", (-2.6, -4.2, 12.0), (1.7, 6.0, 22.0), (0.0, 0.04, 0.05)),
            cutter_ball("c1", rng, (3.2, -5.0, 1.6), (3.2, 3.2, 2.7), seed=seed),
            cutter_ball("c2", rng, (5.6, -1.2, 12.5), (1.7, 5.0, 2.3), seed=seed + 5)]
    return bm, cuts, dict(bed=1.50, joint=3.4, lods=(44000, 9000, 2000))


def blockout_outcrop(rng, seed):
    bm = bmesh.new()
    for k in range(4):
        box(bm, (rng.uniform(-0.4, 0.4), -1.15 + 0.78 * k, 0.25 + 0.14 * k), (5.0 - 0.55 * k + rng.uniform(-0.3, 0.3), 0.64, 3.6 + rng.uniform(-0.3, 0.3)),
            (0.52 + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.06, 0.06)))
    lump(bm, rng, (0.0, 0.3, -0.35), (2.9, 2.3, 1.0))
    lump(bm, rng, (-1.9, -0.9, 0.2), (0.9, 0.8, 0.6))
    return bm, [], dict(bed=0.22, joint=0.9, amp=0.8, lods=(14000, 3500, 800))


KIT = [("arch", 11, blockout_arch), ("cleft", 23, blockout_cleft), ("overhang", 37, blockout_overhang),
       ("stack", 41, blockout_stack), ("pillars", 53, blockout_pillars), ("pile", 67, blockout_pile),
       ("buttress", 79, blockout_buttress), ("outcrop", 83, blockout_outcrop)]


# ---------------------------------------------------------------- the cliff-face family ---
# Sheer faces to stand side by side along a cliff line (the white cliffs of a chalk coast; in
# the dark rock, an inland crag band). A piece is FACE_W wide and FACE_H high at size 1; its
# two ENDS are the same on every variant - the same block-out there, and the relief of the
# face (face_field) goes over, within 4.5 m of an end, into one relief that every piece
# shares - so any two stood FACE_W apart (the engine lays them a little into each other) meet
# without a step. The headland matches at its left end only and turns the corner at its
# right; the sea stack stands alone.
# What a face is made of: a wall leaning back 3.5 degrees; buttresses and gullies between
# upright joints; faint level beds a metre thick with a parting between them and, every fourth
# bed, a line of flint nodules; blocks a little in or out of line, and scars where a panel of
# them fell; a ragged wave notch at the foot; a rubble ramp and fallen blocks in front (the
# blocks are added after the relief of the face, so they keep their shape); the turf of the
# top hanging over the lip, slumped behind it.
FACE_W, FACE_H = 30.0, 45.0
# ---------------------------------------------------------------- river boulders ---
# Single water-worn stones for the river's banks (rock_formations.cpp place_river_boulders):
# four small (0.6 - 1.1 m across), four medium (1.6 - 2.4 m), four large (3.4 - 5.2 m). A
# boulder is one chopped ball - few chops, so it is round with a flat or two, as a stone the
# water has rolled - squashed a little, its middle at 55 % of its height so a root goes into
# the ground; the large ones lean on a second stone. Little of the bedding and jointing the
# outcrops get (amp 0.12): a boulder is one block, worn smooth.
def boulder(radii, chops, extra=None, lods=(10000, 2600, 650)):
    def fn(rng, seed):
        bm = bmesh.new()
        rx, ry, rz = radii
        lump(bm, rng, (0.0, 0.0, 0.55 * rz), (rx, ry, rz), chops=chops, sub=5 if rx > 1.0 else 4,
             r=(rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), rng.uniform(-3.14, 3.14)))
        if extra:
            ex, ey, er = extra
            lump(bm, rng, (ex, ey, 0.5 * er), (er, er * 0.85, er * 0.75), chops=chops, sub=4)
        return bm, [], dict(bed=max(0.9 * rz, 0.3), joint=max(1.6 * rx, 0.6), amp=0.12, missing=0.0, warp=0.25, lods=lods)
    return fn


_S, _M, _L = (5000, 1400, 350), (10000, 2600, 650), (18000, 4500, 1100)
BOULDERS = [      # (added after the whole kit, cliff faces included, in main(): every older piece keeps its number)
    ("boulder_s1", 201, boulder((0.45, 0.38, 0.30), 4, lods=_S)),
    ("boulder_s2", 203, boulder((0.38, 0.33, 0.30), 3, lods=_S)),
    ("boulder_s3", 205, boulder((0.55, 0.40, 0.28), 5, lods=_S)),
    ("boulder_s4", 207, boulder((0.42, 0.42, 0.36), 4, lods=_S)),
    ("boulder_m1", 211, boulder((0.95, 0.75, 0.60), 4, lods=_M)),
    ("boulder_m2", 213, boulder((1.10, 0.80, 0.55), 5, lods=_M)),
    ("boulder_m3", 215, boulder((0.85, 0.80, 0.70), 3, lods=_M)),
    ("boulder_m4", 217, boulder((1.20, 0.90, 0.75), 4, lods=_M)),
    ("boulder_l1", 221, boulder((1.90, 1.50, 1.20), 5, lods=_L)),
    ("boulder_l2", 223, boulder((2.30, 1.60, 1.10), 4, extra=(1.9, 0.9, 0.9), lods=_L)),
    ("boulder_l3", 225, boulder((1.70, 1.60, 1.40), 4, lods=_L)),
    ("boulder_l4", 227, boulder((2.60, 1.90, 1.30), 5, extra=(-2.2, -0.6, 1.0), lods=_L)),
]




def face_base(bm, x0=-15.2, x1=15.2):
    box(bm, (0.5 * (x0 + x1), 5.0, 0.5 * (FACE_H - 4.0)), (x1 - x0, 14.0, FACE_H + 4.0))
    box(bm, (0.5 * (x0 + x1), -4.6, 0.3), (x1 - x0, 7.5, 2.4), (0.36, 0.0, 0.0))        # the rubble ramp


def talus(rng, n, x0=-13.5, x1=13.5, around=None, bm=None):
    if bm is None: bm = bmesh.new()
    for k in range(n):
        d = abs(rng.normal()) * 2.6 + 0.4
        r = rng.uniform(0.3, 1.0) * (1.9 if rng.uniform() < 0.12 else 1.0)
        zs = max(0.0, 2.6 - 0.36 * d)
        if around is None: c = (rng.uniform(x0, x1), -2.2 - d, zs + 0.3 * r)
        else:
            a = rng.uniform(0.0, math.tau)
            c = (around[0] + (around[2] + d) * math.cos(a), around[1] + (around[3] + d) * math.sin(a), zs + 0.3 * r)
        lump(bm, rng, c, (r, r * rng.uniform(0.7, 1.0), r * rng.uniform(0.6, 0.9)), chops=9, sub=2)
    return bm


def blockout_face_a(rng, seed):
    bm = bmesh.new(); face_base(bm)
    return bm, [], dict(bed=0.5, style="face", ends="both", height=FACE_H, late=talus(rng, 45), lods=(60000, 12000, 2500))


def blockout_face_b(rng, seed):
    bm = bmesh.new(); face_base(bm)
    cuts = [cutter_box("c0", (4.0, 0.4, 25.0), (9.0, 3.2, 17.0), (0.05, 0.0, 0.12))]       # a fresh scar
    late = talus(rng, 45)
    for k in range(16):                                                                     # ... and what fell out of it
        r = rng.uniform(0.6, 1.7)
        lump(late, rng, (4.0 + rng.normal() * 3.0, -3.0 - abs(rng.normal()) * 2.5, 0.5 * r + 0.6), (r, r * 0.8, r * 0.7), chops=9, sub=2)
    return bm, cuts, dict(bed=0.5, style="face", ends="both", height=FACE_H, scars=0.24, late=late, lods=(60000, 12000, 2500))


def blockout_face_c(rng, seed):
    bm = bmesh.new(); face_base(bm)
    box(bm, (-1.0, -2.1, 15.0), (17.0, 3.0, 2.2), (0.0, 0.02, 0.0))                       # a ledge across the face
    box(bm, (-7.5, -2.0, 24.0), (3.2, 1.7, 20.0), (-0.03, 0.02, 0.1))                     # a flake standing against it
    cuts = [cutter_box("c0", (-7.5, -0.75, 30.5), (2.3, 0.8, 9.0), (-0.03, 0.02, 0.1))]   # ... parted from it at the top
    return bm, cuts, dict(bed=0.5, style="face", ends="both", height=FACE_H, late=talus(rng, 40), lods=(60000, 12000, 2500))


def blockout_face_cave(rng, seed):
    bm = bmesh.new(); face_base(bm)
    cuts = [cutter_ball("c0", rng, (2.5, 0.5, 2.0), (4.5, 7.0, 5.2), seed=seed),
            cutter_ball("c1", rng, (-6.5, -0.6, 1.4), (2.2, 4.0, 3.0), seed=seed + 3)]
    return bm, cuts, dict(bed=0.5, style="face", ends="both", height=FACE_H, late=talus(rng, 30), lods=(60000, 12000, 2500))


def blockout_headland(rng, seed):
    bm = bmesh.new(); face_base(bm, -15.2, 7.0)
    box(bm, (7.0, 5.0, 0.5 * (FACE_H - 4.0)), (10.0, 10.0, FACE_H + 4.0), (0.0, 0.0, 0.785))      # the nose
    box(bm, (9.5, 5.0, 0.2), (13.0, 15.0, 2.4), (0.0, -0.3, 0.3))
    late = talus(rng, 35, -13.5, 9.0)
    talus(rng, 30, around=(8.5, 5.0, 6.5, 7.5), bm=late)
    return bm, [], dict(bed=0.5, style="face", ends="left", height=FACE_H, late=late, lods=(60000, 12000, 2500))


def blockout_seastack(rng, seed):
    bm = bmesh.new()
    box(bm, (0.0, 0.0, 9.0), (9.0, 8.0, 26.0), (0.02, -0.03, 0.2))
    box(bm, (0.5, 0.3, 21.0), (6.6, 6.0, 20.0), (-0.03, 0.02, 0.7))
    lump(bm, rng, (-0.5, 0.4, 1.0), (6.5, 6.0, 2.6))
    return bm, [], dict(bed=0.5, style="face", ends="none", height=31.0, lean=0.0, late=talus(rng, 26, around=(0.0, 0.0, 4.5, 4.0)), lods=(30000, 6000, 1400))


KIT += [("face_a", 101, blockout_face_a), ("face_b", 103, blockout_face_b), ("face_c", 107, blockout_face_c),
        ("face_cave", 109, blockout_face_cave), ("headland", 113, blockout_headland), ("seastack", 127, blockout_seastack)]


def face_window(x, ends):
    if ends == "both": return sstep(15.0, 10.5, np.abs(x))
    if ends == "left": return sstep(-15.0, -10.5, x)
    return np.ones(len(x))


def face_field(u, z, seed, P):
    """How far a sheer face is set back (m) at `u` along it and height z."""
    H = P["height"]
    zero = np.zeros(len(u))
    def n2(a, b, s): return vnoise(np.stack([a, b, zero], 1), s)
    # buttresses and gullies: broad swells, and V-shaped gullies where a second noise crosses its middle
    g = (n2(u / 8.0, z / 70.0, seed + 1) - 0.5) * 5.5 + 2.4 * (1.0 - np.abs(2.0 * n2(u / 3.6, z / 45.0, seed + 2) - 1.0)) ** 2 - 0.6
    # upright joints: they wander, and each runs for one or two 7 m stretches of height, then another takes over
    zone = np.floor(z / 7.0 + 0.5 * n2(u / 9.0, zero, seed + 11))
    uj = u / 3.1 + 1.6 * (n2(u * 0.04, z * 0.06, seed + 3) - 0.5) + hash3(zone, 3, 0, seed)
    fj = uj - np.floor(uj)
    joint = 0.45 * (1.0 - sstep(0.0, 0.07, np.minimum(fj, 1.0 - fj) * 3.1)) * (hash3(np.floor(uj), zone, 5, seed) > 0.3)
    bc = z / 0.9 + 0.5 * (n2(u * 0.03, z * 0.2, seed + 4) - 0.5)
    bid = np.floor(bc); bf = bc - bid
    bed = (hash3(bid, 0, 0, seed + 5) - 0.5) * 0.09 + 0.06 * (1.0 - sstep(0.0, 0.12, np.minimum(bf, 1.0 - bf)))
    flint = -0.06 * (np.mod(bid, 4) == 1) * (1.0 - sstep(0.0, 0.12, np.abs(bf - 0.5))) * (n2(u * 2.5, bid, seed + 6) > 0.45)
    blk = (hash3(bid, np.floor(uj), 0, seed + 7) - 0.5) * 0.08
    # scars: patches of no particular shape where a slab has come away, their edge ragged and sharp
    sn = n2(u / 7.5, z / 10.0, seed + 8) + 0.10 * (n2(u / 1.3, z / 1.7, seed + 9) - 0.5)
    scar = sstep(1.0 - 2.2 * P.get("scars", 0.15), 1.0 - 2.2 * P.get("scars", 0.15) + 0.012, sn) * (0.5 + 0.8 * n2(u / 11.0, z / 13.0, seed + 12)) * (z > 5.0) * (z < H - 4.0)
    notch = 1.9 * np.exp(-((z - 1.6) / 1.3) ** 2) * (0.6 + 0.8 * n2(u / 3.0, zero + 1.0, seed + 10))
    lip = -0.7 * sstep(H - 1.6, H - 0.5, z) + 0.5 * sstep(H - 3.0, H - 1.6, z) * (1.0 - sstep(H - 1.6, H - 0.9, z))
    return g + joint + bed + flint + blk + scar + notch + lip


def geology_face(co, no, P, seed):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    nh = np.sqrt(no[:, 0] ** 2 + no[:, 1] ** 2)
    side = sstep(0.3, 0.7, nh)
    u = np.where(np.abs(no[:, 1]) >= np.abs(no[:, 0]), x, y + 100.0)
    win = face_window(x, P["ends"])
    f = win * face_field(u, z, seed, P) + (1.0 - win) * face_field(np.zeros(len(u)), z, 999, dict(P, scars=0.0))
    # (the flat ends that lie inside the neighbour stay as they are)
    if P["ends"] == "both": side = side * (1.0 - (np.abs(x) > 14.6) * (np.abs(no[:, 0]) > 0.7))
    if P["ends"] == "left": side = side * (1.0 - (x < -14.6) * (no[:, 0] < -0.7))
    inv = 1.0 / np.maximum(nh, 1e-6)
    co[:, 0] -= no[:, 0] * inv * side * f
    co[:, 1] -= no[:, 1] * inv * side * f
    top = sstep(0.5, 0.85, no[:, 2])
    zero = np.zeros(len(co))
    co[:, 2] += top * win * 0.9 * (fbm(np.stack([x / 5.0, y / 5.0, zero], 1), seed + 6, 3) - 0.5)


# ---------------------------------------------------------------- geology ----------------
def geology_coarse(co, no, P, seed):
    """Beds, joints, blocks, weathered tops: moves the points level (faces) or upright (tops)."""
    tb = P["bed"]; amp = P.get("amp", 1.0)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    zero = np.zeros(len(co))
    slow = vnoise(np.stack([x * 0.07 / tb, y * 0.07 / tb, zero], 1), seed + 1)
    zw = z + 0.03 * x + 0.9 * tb * (slow - 0.5)
    bc = zw / tb
    bc = bc + 0.40 * np.sin(bc * 1.7 + seed) + 0.9 * (vnoise(np.stack([bc * 0.37, zero, zero], 1), seed + 9) - 0.5)   # beds of uneven thickness
    bid = np.floor(bc); bf = bc - bid
    hard = hash3(bid, 0, 0, seed + 2)
    off = (hard - 0.55) * 2.0 * 0.16 * tb - np.where(hard < 0.25, 0.20 * tb, 0.0)
    parting = -0.07 * tb * (1.0 - sstep(0.0, 0.10, np.minimum(bf, 1.0 - bf)))
    th = P.get("joint_angle", 0.35 + 0.1 * (seed % 7))
    sj = P["joint"]
    wob = 0.6 * (vnoise(co * (0.13 / tb), seed + 5) - 0.5) + 0.2 * (vnoise(co * (0.5 / tb), seed + 8) - 0.5) + P.get("lean", 0.12) * z / sj
    u1 = (x * math.cos(th) + y * math.sin(th)) / sj + hash3(bid, 7, 0, seed) + wob
    u2 = (x * math.cos(th + 1.4) + y * math.sin(th + 1.4)) / (sj * 1.3) + hash3(bid, 9, 0, seed) - wob
    j1 = np.floor(u1); f1 = u1 - j1
    j2 = np.floor(u2); f2 = u2 - j2
    boff = (hash3(bid, j1, j2, seed + 3) - 0.5) * 2.0 * 0.10 * tb
    boff = boff - np.where(hash3(bid, j1, j2, seed + 4) < P.get("missing", 0.10), 0.30 * tb, 0.0)
    wj = 0.07 * tb
    g1 = 1.0 - sstep(0.0, wj, np.minimum(f1, 1.0 - f1) * sj)
    g2 = 1.0 - sstep(0.0, wj, np.minimum(f2, 1.0 - f2) * sj * 1.3)
    groove = -0.14 * tb * np.maximum(g1, g2)
    nh = np.sqrt(no[:, 0] ** 2 + no[:, 1] ** 2)
    side = sstep(0.2, 0.6, nh)
    dh = amp * side * (off + boff + parting + groove)
    inv = 1.0 / np.maximum(nh, 1e-6)
    co[:, 0] += no[:, 0] * inv * dh
    co[:, 1] += no[:, 1] * inv * dh
    top = sstep(0.5, 0.85, no[:, 2])
    hollow = fbm(np.stack([x / (2.2 * tb), y / (2.2 * tb), zero], 1), seed + 6, 3) - 0.5
    under = sstep(0.4, 0.8, -no[:, 2])
    co[:, 2] += amp * (top * (0.8 * groove + 0.45 * tb * hollow) + under * 0.8 * boff)


def geology_fine(co, no, P, seed):
    """The relief along the normal: lumps, laminations on the faces, hair cracks, pits."""
    s = P["bed"]; amp = P.get("amp", 1.0) * 0.5 + 0.5
    q = co / s
    nh = np.sqrt(no[:, 0] ** 2 + no[:, 1] ** 2)
    side = sstep(0.2, 0.6, nh)
    top = sstep(0.5, 0.85, no[:, 2])
    d = (fbm(q * 1.6, seed + 10, 4) - 0.5) * 0.10 * s
    lam = (vnoise(np.stack([q[:, 0] * 1.2, q[:, 1] * 1.2, q[:, 2] * 15.0], 1), seed + 11) - 0.5) * 0.04 * s * side
    r = fbm(q * 1.1 + 5.0, seed + 12, 3)
    crack = -(np.clip(1.0 - np.abs(2.0 * r - 1.0) * 10.0, 0.0, 1.0) ** 2) * 0.05 * s
    pit = -np.clip(vnoise(q * 8.0, seed + 13) - 0.74, 0.0, 1.0) * 0.14 * s
    co += no * ((d + lam + crack + pit) * amp * (1.0 - 0.5 * top))[:, None]


# ---------------------------------------------------------------- game meshes -------------
def decimate(ob_dense, target):
    me = ob_dense.data
    tris = int(len(me.loops) - 2 * len(me.polygons))
    tmp = bpy.data.objects.new("lod", me)
    bpy.context.scene.collection.objects.link(tmp)
    m = tmp.modifiers.new("d", "DECIMATE")
    m.decimate_type = "COLLAPSE"; m.ratio = min(1.0, target / max(tris, 1)); m.use_collapse_triangulate = True
    deps = bpy.context.evaluated_depsgraph_get(); deps.update()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(deps))
    bpy.data.objects.remove(tmp)
    bm = bmesh.new(); bm.from_mesh(out)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(out); bm.free()
    ob = bpy.data.objects.new("lodm", out)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def hemi_dirs(k):
    out = []
    for i in range(k):
        u = (i + 0.5) / k
        r = math.sqrt(u); ph = i * 2.399963
        out.append((r * math.cos(ph), r * math.sin(ph), math.sqrt(1.0 - u)))
    return np.array(out)


def vertex_ao(me, bvh, reach):
    co = coords(me); no = normals(me)
    H = hemi_dirs(20)
    a = np.where(np.abs(no[:, 0:1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t = np.cross(no, a); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    b = np.cross(no, t)
    ao = np.ones(len(co))
    for i in range(len(co)):
        o = Vector(co[i] + no[i] * 0.04)
        occ = 0.0
        for h in H:
            d = t[i] * h[0] + b[i] * h[1] + no[i] * h[2]
            hit = bvh.ray_cast(o, Vector(d), reach)
            if hit[0] is not None:
                f = hit[3] / reach
                occ += 1.0 - f * f
        ao[i] = 1.0 - occ / len(H)
    return ao


def vertex_curvature(me):
    co = coords(me); no = normals(me)
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    d = co[e[:, 0]] - co[e[:, 1]]
    ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    acc = np.zeros(len(co)); cnt = np.zeros(len(co))
    np.add.at(acc, e[:, 0], np.einsum("ij,ij->i", no[e[:, 0]], d) / ln)
    np.add.at(acc, e[:, 1], np.einsum("ij,ij->i", no[e[:, 1]], -d) / ln)
    np.add.at(cnt, e[:, 0], 1.0); np.add.at(cnt, e[:, 1], 1.0)
    return np.clip(0.5 + 1.5 * acc / np.maximum(cnt, 1.0), 0.0, 1.0)


def game_mesh(me, bvh, reach):
    """Vertices (pos, normal, ao, curvature) and indices, in game axes."""
    for p in me.polygons: p.use_smooth = True
    me.set_sharp_from_angle(angle=math.radians(42.0))
    nl = len(me.loops)
    vi = np.empty(nl, dtype=np.int32); me.loops.foreach_get("vertex_index", vi)
    cn = np.empty(nl * 3); me.corner_normals.foreach_get("vector", cn); cn = cn.reshape(-1, 3)
    ls = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("loop_start", ls)
    co = coords(me)
    ao = vertex_ao(me, bvh, reach)
    cv = vertex_curvature(me)
    key = np.concatenate([vi[:, None].astype(np.int64), np.round(cn * 100.0).astype(np.int64)], 1)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    v = vi[first]; n = cn[first]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    p = co[v]
    verts = np.stack([p[:, 0], p[:, 2], -p[:, 1], n[:, 0], n[:, 2], -n[:, 1], ao[v], cv[v]], 1).astype("<f4")
    idx = np.stack([inv[ls], inv[ls + 1], inv[ls + 2]], 1).astype("<u4")
    return verts, idx


def solid_grid(bvh, lo, hi, cell):
    """Which cells of a grid are inside the rock (Blender axes in, game axes out)."""
    lo = np.array([lo[0], lo[1], max(lo[2], -1.0)])
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    occ = np.zeros((n[0], n[1], n[2]), dtype=np.uint8)
    for ix in range(n[0]):
        for iy in range(n[1]):
            for iz in range(n[2]):
                p = Vector((lo[0] + (ix + 0.5) * cell, lo[1] + (iy + 0.5) * cell, lo[2] + (iz + 0.5) * cell))
                loc, nrm, _, dist = bvh.find_nearest(p)
                if loc is not None and (p - loc).dot(nrm) < 0.0: occ[ix, iy, iz] = 1
    # game axes: x = x, y = z, z = -y  ->  cell (i, j, k) = blender (i, ny - 1 - k, j)
    g = occ[:, ::-1, :].transpose(0, 2, 1)
    org = (lo[0], lo[2], -(lo[1] + n[1] * cell))
    return org, g


def preview(ob, name):
    try:
        sc = bpy.context.scene
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.render.resolution_x = 720; sc.render.resolution_y = 540
        sh = sc.display.shading
        sh.light = "STUDIO"; sh.show_cavity = True; sh.cavity_type = "BOTH"; sh.show_shadows = True
        sh.color_type = "SINGLE"; sh.single_color = (0.55, 0.52, 0.48)
        co = coords(ob.data)
        lo, hi = co.min(0), co.max(0)
        c = 0.5 * (lo + hi); c[2] = max(c[2], 0.3 * hi[2])
        r = float(np.linalg.norm(hi - lo))
        for tag, d in (("a", Vector((-0.55, -1.0, 0.38))), ("b", Vector((0.85, 0.75, 0.3)))):
            cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d)
            sc.collection.objects.link(cam)
            cam.location = Vector(c) + d.normalized() * r * 1.25
            cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
            cam_d.clip_end = 1000.0
            sc.camera = cam
            sc.render.filepath = os.path.join(PREVIEW, "%s_%s.png" % (name, tag))
            bpy.ops.render.render(write_still=True)
            bpy.data.objects.remove(cam)
    except Exception as e:
        log("preview failed:", e)


def make(name, seed, blockout):
    rng = np.random.RandomState(seed)
    bm, cuts, P = blockout(rng, seed)
    ob = new_obj(name, bm)
    co = coords(ob.data)
    size = co.max(0) - co.min(0)
    big = float(size.max())
    # 2. fuse, bend
    remesh(ob, big / 170.0)
    co = coords(ob.data)
    w = P.get("warp", 1.0) * 0.055 * big
    f = 1.0 / (0.33 * big)
    face = P.get("style") == "face"
    bend = np.stack([fbm(co * f + 3.1, seed + 20, 3), fbm(co * f + 17.3, seed + 21, 3), fbm(co * f + 41.9, seed + 22, 3)], 1) * w * 2.0 - w
    if face:
        # (a face: bent only away from its ends, and leaning back)
        bend *= 0.3 * face_window(co[:, 0], P["ends"])[:, None]
        co[:, 1] += P.get("lean", 0.06) * np.maximum(co[:, 2], 0.0)
    co += bend
    set_coords(ob.data, co)
    # 3. holes
    for c in cuts: cut(ob, c)
    # 4. dense, geology
    voxel = float(np.clip(math.sqrt(2.0 * area(ob.data) / DENSE_TRIS), 0.012, 0.09))
    remesh(ob, voxel)
    co = coords(ob.data); no = normals(ob.data)
    if face: geology_face(co, no, P, seed)
    else: geology_coarse(co, no, P, seed)
    set_coords(ob.data, co)
    if P.get("late") is not None:
        # (what lies loose in front - fallen blocks - joins now, so the relief of the face leaves it alone)
        late = new_obj("late", P["late"])
        bm2 = bmesh.new(); bm2.from_mesh(ob.data); bm2.from_mesh(late.data)
        bm2.to_mesh(ob.data); bm2.free()
        lm = late.data; bpy.data.objects.remove(late); bpy.data.meshes.remove(lm)
    remesh(ob, voxel)
    co = coords(ob.data); no = normals(ob.data)
    geology_fine(co, no, P, seed)
    set_coords(ob.data, co)
    dense = int(len(ob.data.loops) - 2 * len(ob.data.polygons))
    lo, hi = co.min(0), co.max(0)
    log("%s: dense mesh %d triangles (voxel %.3f m), %.1f x %.1f x %.1f m (root to %.1f m)" % (name, dense, voxel, hi[0] - lo[0], hi[1] - lo[1], hi[2], lo[2]))
    # 5. - 7. game meshes, vertex data, collision grid
    levels = []
    src = ob; bvh = None
    reach = float(np.clip(0.22 * hi[2], 0.8, 5.0))
    made = []
    for target in P["lods"]:
        lod = decimate(src, target)
        made.append(lod)
        if bvh is None:
            me = lod.data
            vs = [v.co.copy() for v in me.vertices]
            ps = [tuple(p.vertices) for p in me.polygons]
            bvh = BVHTree.FromPolygons(vs, ps)
            src = lod                                    # the coarser levels from the first: fast
            if PREVIEW: preview(lod, name)
        verts, idx = game_mesh(lod.data, bvh, reach)
        levels.append((verts, idx))
        log("  level %d: %d triangles, %d vertices" % (len(levels) - 1, len(idx), len(verts)))
    cell = float(np.clip(max(hi[0] - lo[0], hi[1] - lo[1]) / 20.0, 0.5, 1.4))
    org, grid = solid_grid(bvh, lo, hi, cell)
    log("  collision grid %d x %d x %d cells of %.2f m, %d solid" % (grid.shape[0], grid.shape[1], grid.shape[2], cell, int(grid.sum())))
    for o in made + [ob]:
        me = o.data
        bpy.data.objects.remove(o)
        if me.users == 0: bpy.data.meshes.remove(me)
    return levels, (org, cell, grid)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if PREVIEW: os.makedirs(PREVIEW, exist_ok=True)
    kit = [k for k in KIT + BOULDERS if ONLY is None or k[0] in ONLY]
    blob = bytearray()
    blob += b"RKF1" + struct.pack("<I", len(kit))
    for name, seed, fn in kit:
        levels, (org, cell, grid) = make(name, seed, fn)
        blob += struct.pack("<24sI", name.encode("ascii"), len(levels))
        for verts, idx in levels:
            lo = verts[:, 0:3].min(0); hi = verts[:, 0:3].max(0)
            blob += struct.pack("<II6f", len(verts), idx.size, *lo, *hi)
            blob += verts.tobytes() + idx.tobytes()
        blob += struct.pack("<4f3I", org[0], org[1], org[2], cell, grid.shape[0], grid.shape[1], grid.shape[2])
        blob += np.ascontiguousarray(grid.transpose(2, 1, 0)).tobytes()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(blob)
    log("wrote %s (%d pieces, %.1f MB)" % (OUT, len(kit), len(blob) / 1e6))


main()
