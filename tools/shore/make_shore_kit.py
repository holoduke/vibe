# The river-bank kit: water-rounded cobbles in three sizes, driftwood and two shingle slabs,
# made the same way as the rock formations and written in the SAME files the engine already
# loads for them (".rockf" + "_bake.bin", read by the rock-formation loader).
#
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --python tools/shore/make_shore_kit.py
#
# writes assets/shore/shore.rockf and assets/shore/shore_bake.bin.
# Options after "--":  --preview DIR   also render the kit (two rows: stones, wood) to DIR
#                      --out FILE      another output file (the bake goes beside it)
#                      --only a,b      make only these pieces
# Deterministic: every random number comes from a piece's seed.
#
# The helpers (noise, remesh, decimation, the unwrap, the Cycles bake, BC5, the file layout)
# are the rock generator's own: tools/rocks/make_formations.py is loaded as a module (its
# main() is not run) and its functions are called - nothing is copied.
#
# PIECES (Blender axes while made: z up; the files hold game axes, y up, y = 0 the ground)
#   cobble_s1..5  8 - 15 cm    cobble_m1..5  15 - 35 cm    cobble_l1..5  35 - 70 cm
#     a unit sphere pushed toward a pillow (flat faces, round edges), flattened 0.35 - 0.6,
#     one end fuller than the other, tilted a little, slow bulges; then along the normal fine
#     pitting and a faint grain. Sunk to 30 % of their height.
#   drift_log     a bleached log ~3.2 m with a root wad (roots radiating from one end)
#   drift_limb    a branched limb ~2.3 m
#   drift_stick1, drift_stick2   short broken sticks (splintered ends)
#     tapered tubes along a skeleton, fused by a voxel remesh; the relief by each point's
#     nearest bone: grain grooves along it, long drying checks, knots, splinters at the
#     broken ends. No ground on the tops (the bake's "tops" weight is zeroed for wood).
#   shingle_1, shingle_2   flat stones stacked and fused, for bank edges.
#
# THE BAKE FILE has one size for every layer (the loader's format), so it is 512: the
# cobbles and slabs bake at 512, the driftwood at 1024 and is box-filtered to 512 (better
# than a 512 bake: the grain is resolved before it is averaged).
import bpy
import bmesh
import math
import os
import struct
import sys
import time
import types
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ROCKS = os.path.join(ROOT, "tools", "rocks", "make_formations.py")


def load_rocks():
    src = open(ROCKS, "r", encoding="utf-8").read()
    lines = src.rstrip().split("\n")
    assert lines[-1].strip() == "main()", "make_formations.py no longer ends with main()"
    m = types.ModuleType("make_formations")
    m.__file__ = ROCKS
    exec(compile("\n".join(lines[:-1]) + "\n", ROCKS, "exec"), m.__dict__)
    return m


R = load_rocks()
R.PREVIEW = None                       # (its own per-piece pictures: not wanted here)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.join(ROOT, "assets", "shore", "shore.rockf")
PREVIEW = None
ONLY = None
i = 0
while i < len(argv):
    if argv[i] == "--preview": PREVIEW = os.path.abspath(argv[i + 1]); i += 1
    elif argv[i] == "--out": OUT = os.path.abspath(argv[i + 1]); i += 1
    elif argv[i] == "--only": ONLY = argv[i + 1].split(","); i += 1
    i += 1
BAKE_OUT = os.path.splitext(OUT)[0] + "_bake.bin"
FILE_BAKE = 512
T0 = time.time()


def log(*a):
    print("[shore %6.1fs]" % (time.time() - T0), *a, flush=True)


# ---------------------------------------------------------------- stones -----------------
def stone_points(rng, seed, radii, sub):
    """A water-rounded stone as (bmesh, None): pillow-shaped, flattened, egg-ended, tilted."""
    bm = bmesh.new()
    ret = bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    vs = ret["verts"]
    n = np.array([v.co[:] for v in vs]); n /= np.linalg.norm(n, axis=1, keepdims=True)
    e = rng.uniform(0.72, 0.88)                                       # < 1: toward a pillow
    p = np.sign(n) * np.abs(n) ** e
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    bulge = 1.0 + 0.10 * (R.fbm(n * 1.3 + rng.uniform(0, 50), seed + 1, 3) - 0.5)
    p *= bulge[:, None]
    p[:, 0] *= 1.0 + rng.uniform(0.06, 0.18) * n[:, 0]               # one end fuller
    p[:, 1] *= 1.0 + rng.uniform(-0.08, 0.08) * n[:, 0]
    p *= np.array(radii)[None, :]
    p[:, 2] += rng.uniform(-0.08, 0.08) * p[:, 0]                    # a slight tilt
    yaw = rng.uniform(-math.pi, math.pi)
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)
    for v, q in zip(vs, p): v.co = Vector(q)
    return bm


def stone_fine(me, size, seed):
    """Pitting and a faint grain along the normal (sizes relative to the stone)."""
    co = R.coords(me); no = R.normals(me)
    q = co / size
    pit = -np.clip(R.vnoise(q * 38.0, seed + 5) - 0.72, 0.0, 1.0) * 0.035 * size
    pit2 = -np.clip(R.vnoise(q * 90.0 + 7.0, seed + 6) - 0.80, 0.0, 1.0) * 0.015 * size
    grain = (R.fbm(q * 14.0, seed + 7, 3) - 0.5) * 0.004 * size
    co += no * (pit + pit2 + grain)[:, None]
    R.set_coords(me, co)


def cobble(radii, sub, lods):
    def fn(rng, seed):
        bm = stone_points(rng, seed, radii, sub)
        ob = R.new_obj("dense", bm)
        co = R.coords(ob.data); h = co[:, 2].max() - co[:, 2].min(); co[:, 2] += -co[:, 2].min() - 0.30 * h
        R.set_coords(ob.data, co)
        stone_fine(ob.data, 2.0 * max(radii), seed)
        return ob, dict(lods=lods, kind="stone")
    return fn


def shingle(n, lods):
    def fn(rng, seed):
        # each stone is DROPPED onto what lies below: a 1 cm height grid of the stack so far
        # (the ground at z = 0); the stone comes down until its lowest point under the grid
        # touches, then sinks 2 mm so the remesh fuses the contact.
        bm = bmesh.new()
        G = 0.01; ext = 0.8
        ng = int(2 * ext / G)
        top = np.zeros((ng, ng))
        for k in range(n):
            a = rng.uniform(0.16, 0.30); b = a * rng.uniform(0.6, 0.85); c = rng.uniform(0.025, 0.05)
            sb = stone_points(rng, seed + 31 * k, (a, b, c), 4)
            spread = 0.35 * (1.0 - 0.6 * k / max(n - 1, 1))              # the upper stones nearer the middle
            ox, oy = rng.uniform(-spread, spread), rng.uniform(-0.6 * spread, 0.6 * spread)
            p = np.array([v.co[:] for v in sb.verts]); p[:, 0] += ox; p[:, 1] += oy
            gi = np.clip(((p[:, :2] + ext) / G).astype(int), 0, ng - 1)
            dz = float(np.max(top[gi[:, 0], gi[:, 1]] - p[:, 2])) - 0.002
            p[:, 2] += dz
            for v, q in zip(sb.verts, p): v.co = Vector(q)
            np.maximum.at(top, (gi[:, 0], gi[:, 1]), p[:, 2])
            tmp = bpy.data.meshes.new("t"); sb.to_mesh(tmp); sb.free(); bm.from_mesh(tmp); bpy.data.meshes.remove(tmp)
        ob = R.new_obj("dense", bm)
        R.remesh(ob, 0.008)
        ar = R.area(ob.data)
        R.remesh(ob, float(np.clip(math.sqrt(2.0 * ar / 600000), 0.002, 0.02)))
        co = R.coords(ob.data); co[:, 2] -= 0.02; R.set_coords(ob.data, co)
        stone_fine(ob.data, 0.4, seed)
        return ob, dict(lods=lods, kind="stone")
    return fn


# ---------------------------------------------------------------- driftwood --------------
def tube(bm, a, b, r0, r1, seg=20):
    a = Vector(a); b = Vector(b); d = b - a; L = d.length
    q = d.normalized().to_track_quat("Z", "Y")
    m = Matrix.Translation((a + b) * 0.5) @ q.to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r0, radius2=r1, depth=L, matrix=m)
    for p, r in ((a, r0), (b, r1)):
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=r, matrix=Matrix.Translation(p))


def wood_relief(me, bones, seed, broken):
    """bones: [(a, b, r0, r1)]. Each point takes its nearest bone: grain grooves along it,
    drying checks, knots; at broken bone ends (broken: set of (bone, end)) splinters."""
    co = R.coords(me); no = R.normals(me)
    N = len(co)
    best = np.full(N, 1e9); bi = np.zeros(N, dtype=int); bt = np.zeros(N)
    for k, (a, b, r0, r1, ch, s0) in enumerate(bones):
        a = np.array(a); b = np.array(b); d = b - a; L = np.linalg.norm(d); d /= L
        t = np.clip((co - a) @ d / L, 0.0, 1.0)
        rad = r0 + (r1 - r0) * t
        dist = np.linalg.norm(co - (a + np.outer(t * L, d)), axis=1) - rad
        m = dist < best
        best[m] = dist[m]; bi[m] = k; bt[m] = t[m]
    disp = np.zeros(N)
    rch = {}
    for bn in bones: rch[bn[4]] = max(rch.get(bn[4], 0.0), 0.5 * (bn[2] + bn[3]))
    for k, (a, b, r0, r1, ch, s0) in enumerate(bones):
        m = bi == k
        if not m.any(): continue
        a = np.array(a); b = np.array(b); d = b - a; L = np.linalg.norm(d); d /= L
        u = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        p = co[m]; t = bt[m]
        v = p - (a + np.outer(t * L, d))
        vl = np.maximum(np.linalg.norm(v, axis=1), 1e-6)
        cu = (v @ u) / vl; cw = (v @ w) / vl
        s = s0 + t * L + 3.7 * ch               # (one chain = one continuous piece of wood: no ring at its joints)
        r = rch[ch]
        kg = max(r / 0.012, 2.0)                                     # ~1.2 cm between grain ridges
        g = R.fbm(np.stack([cu * kg, cw * kg, s * 1.3], 1), seed + 40 + ch, 3)
        grain = -(np.abs(2.0 * g - 1.0) ** 0.6) * 0.0035 - (R.vnoise(np.stack([cu * kg * 3, cw * kg * 3, s * 4.0], 1), seed + 60 + ch) - 0.5) * 0.0012
        kc = max(r / 0.05, 1.5)
        c = R.vnoise(np.stack([cu * kc, cw * kc, s * 0.35], 1), seed + 80 + ch)
        check = -R.sstep(0.90, 0.975, 1.0 - np.abs(2.0 * c - 1.0)) * min(0.25 * r, 0.03)
        kn = R.vnoise(np.stack([cu * 2.0, cw * 2.0, s * 2.2], 1) + 9.0, seed + 90 + ch)
        knot = R.sstep(0.80, 0.92, kn) * 0.12 * r - R.sstep(0.90, 0.95, kn) * 0.10 * r
        dd = disp[np.where(m)[0]]
        dd += grain + check + knot
        # splinters at a broken end: the cap pushed along the bone by a ragged pattern
        for (bk, end) in broken:
            if bk != k: continue
            near = (t > 0.98) if end == 1 else (t < 0.02)
            cap = np.abs(no[m] @ d) > 0.45
            sel = near & cap
            if sel.any():
                sp = R.fbm(np.stack([cu * 5.0, cw * 5.0, np.zeros(len(cu))], 1) * (1.0 / max(r, 0.02)) * 0.04, seed + 99, 3)
                ragged = (sp - 0.35) * 1.6 * r + (R.vnoise(np.stack([cu * kg, cw * kg, np.zeros(len(cu))], 1), seed + 98) - 0.5) * 0.6 * r
                dd[sel] += np.maximum(ragged[sel], -0.3 * r)
        disp[np.where(m)[0]] = dd
    co += no * disp[:, None]
    R.set_coords(me, co)


def wood(bones_fn, dense, lods):
    def fn(rng, seed):
        bones, broken = bones_fn(rng)
        bm = bmesh.new()
        for bn in bones: tube(bm, *bn[:4])
        ob = R.new_obj("dense", bm)
        rmin = min(min(b[2], b[3]) for b in bones)
        R.remesh(ob, float(np.clip(rmin / 3.0, 0.004, 0.02)))
        ar = R.area(ob.data)
        R.remesh(ob, float(np.clip(math.sqrt(2.0 * ar / dense), 0.0018, 0.02)))
        wood_relief(ob.data, bones, seed, broken)
        return ob, dict(lods=lods, kind="wood")
    return fn


def chain(bones, pts, rads, ch):
    """Bones along a polyline, one continuous piece of wood (its grain runs on along it).
    Returns the index of its last bone."""
    s = 0.0
    for k in range(len(pts) - 1):
        a = tuple(float(x) for x in pts[k]); b = tuple(float(x) for x in pts[k + 1])
        bones.append((a, b, float(rads[k]), float(rads[k + 1]), ch, s))
        s += float(np.linalg.norm(np.array(b) - np.array(a)))
    return len(bones) - 1


def log_bones(rng):
    """A bleached trunk, slightly bent and tapering, and a root wad: 6 - 10 thick roots
    splaying out and down from a flared butt, snapped short, none deeper than 0.3 m."""
    L = rng.uniform(3.0, 3.4); r = 0.21
    x0 = -0.5 * L; zc = 0.78 * r
    bones = []; broken = []
    n = 8
    xs = np.linspace(0.0, 1.0, n)
    bow = rng.uniform(0.06, 0.12) * (1.0 if rng.uniform() < 0.5 else -1.0)
    pts = [(x0 + L * u, bow * math.sin(math.pi * u), zc - 0.05 * u * u) for u in xs]
    rads = [r * (1.0 - 0.38 * u) for u in xs]
    broken.append((chain(bones, pts, rads, 0), 1))                  # the snapped top end
    hub = np.array([x0, 0.0, zc + 0.01])
    chain(bones, [hub + [0.25, 0, 0], hub, hub - [0.10, 0, 0]], [1.0 * r, 1.25 * r, 1.2 * r], 1)     # the flared butt
    nr = int(rng.randint(6, 11))
    for k in range(nr):
        ang = 2.0 * math.pi * (k + rng.uniform(-0.3, 0.3)) / nr
        d = np.array([-rng.uniform(0.15, 0.5), math.cos(ang), math.sin(ang) - 0.45])
        d /= np.linalg.norm(d)
        r0 = rng.uniform(0.085, 0.125); r1 = r0 * rng.uniform(0.55, 0.75)
        p0 = hub + d * 0.7 * r
        l1 = rng.uniform(0.15, 0.25); l2 = rng.uniform(0.10, 0.22)
        p1 = p0 + d * l1
        d2 = d + np.array([0.0, 0.0, -0.4]); d2 /= np.linalg.norm(d2)
        p2 = p1 + d2 * l2
        for p in (p1, p2):
            if p[2] - r1 < -0.3: p[2] = -0.3 + r1                       # within 0.3 m below the ground line
        broken.append((chain(bones, [p0, p1, p2], [r0, 0.5 * (r0 + r1), r1], 2 + k), 1))
    return bones, broken


def limb_bones(rng):
    """A limb with three side branches, each thick at its base with a collar."""
    zc = 0.07
    main = [(-1.15, 0.0, zc), (-0.3, 0.06, zc - 0.005), (0.5, -0.02, zc - 0.015), (1.15, 0.08, zc - 0.03)]
    rr = [0.095, 0.08, 0.062, 0.04]
    bones = []; broken = [(chain(bones, main, rr, 0), 1)]
    for k, (base, ang, l) in enumerate(((1, 0.6, 0.85), (2, -0.7, 0.7), (2, 0.9, 0.45))):
        p0 = np.array(main[base]); a = ang + rng.uniform(-0.15, 0.15)
        d = np.array([math.cos(a), math.sin(a), rng.uniform(0.05, 0.25)]); d /= np.linalg.norm(d)
        rb = rr[base]
        pts = [p0 + d * 0.3 * rb, p0 + d * (rb + 0.05), p0 + d * (rb + 0.12), p0 + d * l]
        chain(bones, pts, [0.95 * rb, 0.82 * rb, 0.62 * rb, 0.02], 1 + k)     # the collar, then the branch
    return bones, broken


def stick1_bones(rng):
    bones = []
    last = chain(bones, [(-0.36, 0.0, 0.036), (0.05, 0.02, 0.034), (0.38, -0.03, 0.03)], [0.045, 0.040, 0.034], 0)
    return bones, [(0, 0), (last, 1)]


def stick2_bones(rng):
    bones = []
    chain(bones, [(-0.25, 0.0, 0.03), (0.25, 0.01, 0.026)], [0.035, 0.028], 0)
    chain(bones, [(0.02, 0.005, 0.03), (0.06, 0.04, 0.035), (0.22, 0.17, 0.05)], [0.026, 0.02, 0.012], 1)
    return bones, [(0, 0), (0, 1)]

_CS, _CM, _CL = (600, 200, 60), (1500, 450, 120), (3000, 900, 250)
COBBLE_R = {"s": (0.04, 0.075), "m": (0.075, 0.175), "l": (0.175, 0.35)}    # half the 8-15 / 15-35 / 35-70 cm lengths


def cobble_list():
    out = []
    for cls, lods, sub, seed0 in (("s", _CS, 7, 301), ("m", _CM, 8, 331), ("l", _CL, 8, 361)):
        lo, hi = COBBLE_R[cls]
        rng = np.random.RandomState(seed0)
        for k in range(5):
            a = lo + (hi - lo) * (k + rng.uniform(0.1, 0.9)) / 5.0
            b = a * rng.uniform(0.62, 0.9); c = a * rng.uniform(0.35, 0.6)
            out.append(("cobble_%s%d" % (cls, k + 1), seed0 + 2 * k + 1, cobble((a, b, c), sub, lods)))
    return out


KIT = cobble_list() + [
    ("drift_log", 401, wood(log_bones, 1000000, (12000, 3500, 900))),
    ("drift_limb", 403, wood(limb_bones, 600000, (6000, 1800, 450))),
    ("drift_stick1", 405, wood(stick1_bones, 250000, (1500, 450, 120))),
    ("drift_stick2", 407, wood(stick2_bones, 250000, (1500, 450, 120))),
    ("shingle_1", 501, shingle(7, (5000, 1500, 400))),
    ("shingle_2", 503, shingle(9, (5000, 1500, 400))),
]


def down2(a):
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def refloor(layers, hr, floor, name):
    """The rock bake keeps a height range of at least 1 cm; a pebble's pitting is a few mm, so
    the range is measured again here (99th percentile, as the rock bake does) with a per-piece
    floor ('hfloor', 3 mm by default) and the height re-encoded - before the 8-bit encode,
    so nothing is lost."""
    H = layers[1][1]
    h = (H - 0.5) * 2.0 * hr
    live = np.abs(h) > 1e-7
    if not live.any(): return layers, hr
    hn = float(np.clip(np.percentile(np.abs(h[live]), 99.0), floor, 0.5))
    if hn >= hr: return layers, hr
    H2 = np.clip(0.5 + 0.5 * h / hn, 0.0, 1.0)
    log("  %s: height range +-%.4f m (the rock bake's floor gave +-%.3f)" % (name, hn, hr))
    layers = list(layers); layers[1] = (layers[1][0], H2)
    return layers, hn


def make(name, seed, fn):
    rng = np.random.RandomState(seed)
    ob, P = fn(rng, seed)
    co = R.coords(ob.data)
    lo, hi = co.min(0), co.max(0)
    dense = int(len(ob.data.loops) - 2 * len(ob.data.polygons))
    log("%s: dense %d triangles, %.2f x %.2f x %.2f m (root to %.3f m)" % (name, dense, hi[0] - lo[0], hi[1] - lo[1], hi[2], lo[2]))
    levels = []; made = []; src = ob; bvh = None; layers = None; hr = 0.0
    for target in P["lods"]:
        lod = R.decimate(src, target); made.append(lod)
        if bvh is None:
            me = lod.data
            bvh = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
            src = lod
            R.BAKE = 1024 if P["kind"] == "wood" else FILE_BAKE
            layers, hr = R.bake_piece(lod, ob, name, P)
            layers, hr = refloor(layers, hr, P.get("hfloor", 0.003), name)
            if R.BAKE != FILE_BAKE:
                layers = [tuple(down2(c) for c in pair) for pair in layers]
            if P["kind"] == "wood":
                layers[2] = (layers[2][0], np.zeros_like(layers[2][1]))     # no ground laid on wood
        verts, idx = R.game_mesh(lod.data, bvh, 1.0)
        levels.append((verts, idx))
    log("  %s triangles per level: %s" % (name, " / ".join(str(len(ix)) for _, ix in levels)))
    cell = float(np.clip(max(hi[0] - lo[0], hi[1] - lo[1]) / 10.0, 0.05, 1.4))
    org, grid = R.solid_grid(bvh, lo, hi, cell)
    for o in made + [ob]:
        me = o.data; bpy.data.objects.remove(o)
        if me.users == 0: bpy.data.meshes.remove(me)
    return levels, (org, cell, grid), layers, hr


def render_rows(pieces):
    """Two rows (stones, wood) laid out along x, Workbench, from the front and above."""
    sc = bpy.context.scene
    for o in list(sc.objects): bpy.data.objects.remove(o)
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = 1600; sc.render.resolution_y = 700
    sh = sc.display.shading
    sh.light = "STUDIO"; sh.show_cavity = True; sh.cavity_type = "BOTH"; sh.show_shadows = True
    sh.color_type = "SINGLE"
    for tag, sel, col in (("stones", lambda n: not n.startswith("drift"), (0.55, 0.53, 0.50)),
                          ("wood", lambda n: n.startswith("drift"), (0.70, 0.66, 0.60))):
        sh.single_color = col
        obs = []; x = 0.0
        for name, verts, idx in pieces:
            if not sel(name): continue
            p = verts[:, 0:3].astype(np.float64)
            b = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)
            w = b[:, 0].max() - b[:, 0].min()
            b[:, 0] += x - b[:, 0].min()
            x += w + 0.06 + 0.15 * w
            me = bpy.data.meshes.new(name)
            me.from_pydata(b.tolist(), [], idx.tolist()); me.update()
            for pl in me.polygons: pl.use_smooth = True
            ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); obs.append(ob)
        if not obs: continue
        span = x
        cam_d = bpy.data.cameras.new("cam"); cam_d.type = "ORTHO"; cam_d.ortho_scale = span * 1.08
        cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam)
        c = Vector((0.5 * span, 0.0, 0.1))
        cam.location = c + Vector((0.0, -1.0, 0.75)).normalized() * (span + 5.0)
        cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam_d.clip_end = 1000.0
        sc.camera = cam
        sc.render.filepath = os.path.join(PREVIEW, "shore_kit_%s.png" % tag)
        bpy.ops.render.render(write_still=True)
        log("preview", sc.render.filepath)
        for o in obs + [cam]: bpy.data.objects.remove(o)


from mathutils.bvhtree import BVHTree


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit = [k for k in KIT if ONLY is None or k[0] in ONLY]
    blob = bytearray(b"RKF2" + struct.pack("<I", len(kit)))
    bake = []; ranges = []; shown = []
    for name, seed, fn in kit:
        levels, (org, cell, grid), layers, hr = make(name, seed, fn)
        bake += layers; ranges.append(hr)
        shown.append((name, levels[0][0], levels[0][1]))
        blob += struct.pack("<24sI", name.encode("ascii"), len(levels))
        for verts, idx in levels:
            lo = verts[:, 0:3].min(0); hi = verts[:, 0:3].max(0)
            blob += struct.pack("<II6f", len(verts), idx.size, *lo, *hi)
            blob += verts.tobytes() + idx.tobytes()
        blob += struct.pack("<4f3I", org[0], org[1], org[2], cell, grid.shape[0], grid.shape[1], grid.shape[2])
        blob += np.ascontiguousarray(grid.transpose(2, 1, 0)).tobytes()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f: f.write(blob)
    log("wrote %s (%d pieces, %.2f MB)" % (OUT, len(kit), len(blob) / 1e6))
    R.BAKE = FILE_BAKE
    bake += R.breaker_pattern(FILE_BAKE)          # (the layout the loader expects: 3 a piece + 2)
    bb = bytearray(b"RKB1" + struct.pack("<4I", FILE_BAKE, len(bake), R.BAKE_MIPS, len(kit)) + struct.pack("<%df" % len(kit), *ranges))
    for a, b in bake: bb += R.bc5_mips(a, b)
    with open(BAKE_OUT, "wb") as f: f.write(bb)
    log("wrote %s (%d layers of %d x %d, %.2f MB)" % (BAKE_OUT, len(bake), FILE_BAKE, FILE_BAKE, len(bb) / 1e6))
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        render_rows(shown)


main()
