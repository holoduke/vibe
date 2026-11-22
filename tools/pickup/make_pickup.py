# Builds the pickup - an original modern mid-size double-cab 4x4 pickup truck - in Blender and
# exports it for the game (same pipeline and file formats as tools/jeep/make_jeep.py and
# tools/sportscar/make_sportscar.py):
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --python tools/pickup/make_pickup.py
# (set PICKUP_PREVIEW=1 to also render assets/pickup/preview_*.png with back-face culling on;
#  PICKUP_VIEWS=front34,side limits the views)
# Output (assets/pickup/):
#   body.mesh      everything except the four road wheels and the steering wheel.
#                  Origin: on the ground under the middle of the wheelbase.
#   wheel.mesh     ONE wheel (tyre, rim, brake disc, caliper), centred on the origin, axle along
#                  the engine's X axis. Its OUTER face looks toward +X: draw it as it is for the
#                  right-hand wheels, turned 180 degrees about Y for the left.
#   steering.mesh  the steering wheel alone. Origin: the centre of the rim's plane (its hub).
#                  In the file its rotation axis is the engine's +Z (0, 0, 1) - the side the
#                  driver is on - and it is NOT tilted. To place it: spin it about its own Z,
#                  then rotate about the engine's +X by -STEER_TILT degrees, then translate to
#                  "steering_hub".
#   car.json       the numbers the engine needs (same keys as assets/jeep/jeep.json, plus
#                  "hip_driver" and "legs").
# File format: "mesh <name> <triangles>", then per vertex: position, normal, uv (uv in metres).
#
# How the body is made: a coarse cage of cross-sections -> Catmull-Clark subdivision -> boolean
# cuts (wheel housings, the cabin as a hollow, the load bed, window openings, light pockets,
# panel gaps ...) -> triangles with normals smoothed below a crease angle. Everything else is
# put together from primitives. No badges, logos or lettering anywhere.
#
# Blender space: X = the vehicle's right, Y = forward, Z up.
# Game space:    x = X, y = Z, z = -Y   (so the front points to -z, +x is the right side).
import bmesh
from mathutils.bvhtree import BVHTree
import bpy, math, os, json
from mathutils import Vector, Matrix
from mathutils.geometry import tessellate_polygon

bpy.ops.wm.read_factory_settings(use_empty=True)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'assets', 'pickup')
os.makedirs(OUT, exist_ok=True)
TAU = 2.0 * math.pi

G = {}                      # material -> [(p, n, uv)] three per triangle
XF = [Matrix.Identity(4)]


class xf:
    def __init__(self, m): self.m = m
    def __enter__(self): XF.append(XF[-1] @ self.m)
    def __exit__(self, *a): XF.pop()


def T(x, y, z): return Matrix.Translation(Vector((x, y, z)))
def RX(a): return Matrix.Rotation(a, 4, 'X')
def RY(a): return Matrix.Rotation(a, 4, 'Y')
def RZ(a): return Matrix.Rotation(a, 4, 'Z')
MIR = Matrix.Scale(-1.0, 4, Vector((1, 0, 0)))
IDN = Matrix.Identity(4)
# local (x, y, z) -> world (z, x, y): a profile drawn in (Y, Z), pushed out / revolved along X
PX = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
# local (x, y, z) -> world (x, z, y): a profile drawn in (X, Z), pushed out along Y
PY = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def planar_uv(p, n):
    if abs(n.z) > 0.95: t, b = Vector((1, 0, 0)), Vector((0, 1, 0))
    else:
        t = Vector((0, 0, 1)).cross(n).normalized()
        b = n.cross(t).normalized()
    return (p.dot(t), -p.dot(b))


def tri(g, pts, nrms=None, uvs=None):
    """One triangle. The winding always follows the normals (as the engine's loader does), so
    mirrored transforms are safe. `g` may be a function (normal, centre) -> group name."""
    m = XF[-1]; nm = m.to_3x3().inverted().transposed()
    L = [Vector(p) for p in pts]
    lf = (L[1] - L[0]).cross(L[2] - L[0])
    if lf.length < 1e-11: return
    P = [m @ p for p in L]
    fn = (P[1] - P[0]).cross(P[2] - P[0])
    if fn.length < 1e-11: return
    fn.normalize()
    if nrms is None:
        ln = (nm @ lf).normalized(); N = [ln, ln, ln]
    else:
        N = []
        for n in nrms:
            v = nm @ Vector(n)
            N.append(v.normalized() if v.length > 1e-8 else None)
        ref = Vector((0, 0, 0))
        for v in N:
            if v is not None: ref += v
        if fn.dot(ref) < 0.0: fn = -fn
        N = [v if v is not None else fn for v in N]
    if uvs is not None: uvs = list(uvs)
    if (P[1] - P[0]).cross(P[2] - P[0]).dot(N[0] + N[1] + N[2]) < 0.0:
        P = [P[0], P[2], P[1]]; N = [N[0], N[2], N[1]]
        if uvs is not None: uvs = [uvs[0], uvs[2], uvs[1]]
    fn = (P[1] - P[0]).cross(P[2] - P[0]).normalized()
    if callable(g): g = g(fn, (P[0] + P[1] + P[2]) / 3.0)
    if uvs is None: uvs = [planar_uv(p, fn) for p in P]
    G.setdefault(g, []).extend(zip(P, N, uvs))


def quad(g, a, b, c, d, nrms=None, uvs=None):
    tri(g, (a, b, c), None if nrms is None else (nrms[0], nrms[1], nrms[2]), None if uvs is None else (uvs[0], uvs[1], uvs[2]))
    tri(g, (a, c, d), None if nrms is None else (nrms[0], nrms[2], nrms[3]), None if uvs is None else (uvs[0], uvs[2], uvs[3]))


def quad2(g, a, b, c, d):                   # a thin sheet seen from both sides
    quad(g, a, b, c, d); quad(g, d, c, b, a)


def quad_out(g, a, b, c, d, ref, uvs=None):
    """A flat quad facing away from the point `ref`."""
    a, b, c, d = Vector(a), Vector(b), Vector(c), Vector(d)
    n = (b - a).cross(c - a)
    if n.length < 1e-10: n = (c - a).cross(d - a)
    if n.length < 1e-10: return
    if n.dot((a + b + c + d) / 4.0 - Vector(ref)) < 0.0: n = -n
    quad(g, a, b, c, d, [n] * 4, uvs)


def box(g, lo, hi, skip='', inv=False):
    x0, y0, z0 = [min(a, b) for a, b in zip(lo, hi)]; x1, y1, z1 = [max(a, b) for a, b in zip(lo, hi)]
    F = {'-x': ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
         '+x': ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1, 0, 0)),
         '-y': ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
         '+y': ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (0, 1, 0)),
         '-z': ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (0, 0, -1)),
         '+z': ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1))}
    for k, (a, b, c, d, n) in F.items():
        if k in skip: continue
        if inv: n = (-n[0], -n[1], -n[2])
        quad(g, a, b, c, d, [n] * 4)


def cbox(g, lo, hi, c=0.015):
    """A box with every edge chamfered by c (44 triangles)."""
    lo, hi = [min(a, b) for a, b in zip(lo, hi)], [max(a, b) for a, b in zip(lo, hi)]
    c = min(c, 0.49 * min(hi[i] - lo[i] for i in range(3)))
    def P(s, ax):                           # the corner s's point on the face of axis ax
        return tuple((hi[i] if s[i] > 0 else lo[i]) if i == ax else ((hi[i] - c) if s[i] > 0 else (lo[i] + c)) for i in range(3))
    for ax in range(3):
        o1, o2 = (ax + 1) % 3, (ax + 2) % 3
        for sg in (-1, 1):
            cs = []
            for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                s = [0, 0, 0]; s[ax] = sg; s[o1] = s1; s[o2] = s2
                cs.append(P(s, ax))
            n = [0, 0, 0]; n[ax] = sg
            quad(g, cs[0], cs[1], cs[2], cs[3], [n] * 4)
    for ax in range(3):                     # edges running along ax
        o1, o2 = (ax + 1) % 3, (ax + 2) % 3
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                sa = [0, 0, 0]; sa[ax] = -1; sa[o1] = s1; sa[o2] = s2
                sb = list(sa); sb[ax] = 1
                n = [0, 0, 0]; n[o1] = s1; n[o2] = s2
                quad(g, P(sa, o1), P(sb, o1), P(sb, o2), P(sa, o2), [n] * 4)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                s = (sx, sy, sz)
                tri(g, (P(s, 0), P(s, 1), P(s, 2)), [s] * 3)


def obox(g, p0, p1, w, h, up=(0, 0, 1), c=0.0):
    """A beam from p0 to p1 with a w x h section (w across, h along `up`)."""
    p0 = Vector(p0); ax = Vector(p1) - p0
    L = ax.length
    if L < 1e-6: return
    ax /= L
    side = ax.cross(Vector(up))
    if side.length < 1e-4: side = ax.cross(Vector((1, 0, 0)))
    side.normalize()
    upv = side.cross(ax).normalized()
    m = Matrix((side, upv, ax)).transposed().to_4x4()
    m.translation = p0
    with xf(m):
        if c > 0.0: cbox(g, (-w / 2, -h / 2, 0), (w / 2, h / 2, L), c)
        else: box(g, (-w / 2, -h / 2, 0), (w / 2, h / 2, L))


def cyl(g, p0, p1, r, seg=12, caps=True, r1=None):
    p0, p1 = Vector(p0), Vector(p1)
    if r1 is None: r1 = r
    ax = (p1 - p0).normalized()
    u = ax.cross(Vector((0, 0, 1)))
    if u.length < 1e-4: u = ax.cross(Vector((1, 0, 0)))
    u.normalize(); v = ax.cross(u)
    if caps is True: caps = (True, True)
    elif caps is False: caps = (False, False)
    ring = [(u * math.cos(TAU * i / seg) + v * math.sin(TAU * i / seg)) for i in range(seg)]
    for i in range(seg):
        a, b = ring[i], ring[(i + 1) % seg]
        quad(g, p0 + a * r, p0 + b * r, p1 + b * r1, p1 + a * r1, [a, b, b, a])
        if caps[0]: tri(g, (p0, p0 + b * r, p0 + a * r), [-ax] * 3)
        if caps[1]: tri(g, (p1, p1 + a * r1, p1 + b * r1), [ax] * 3)


def ball(g, c, r, seg=12, rings=8, sc=(1.0, 1.0, 1.0)):
    c = Vector(c)
    def D(i, j):
        th = math.pi * j / rings; ph = TAU * i / seg
        return Vector((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    def P(d): return c + Vector((d.x * r * sc[0], d.y * r * sc[1], d.z * r * sc[2]))
    def Nn(d): return Vector((d.x / sc[0], d.y / sc[1], d.z / sc[2]))
    for j in range(rings):
        for i in range(seg):
            a, b, c2, d = D(i, j), D(i + 1, j), D(i + 1, j + 1), D(i, j + 1)
            quad(g, P(a), P(b), P(c2), P(d), [Nn(a), Nn(b), Nn(c2), Nn(d)])


def fillet(pts, rad, n=5):
    """Round the inner corners of a polyline."""
    P = [Vector(p) for p in pts]; out = [P[0]]
    for i in range(1, len(P) - 1):
        d1 = (P[i - 1] - P[i]); d2 = (P[i + 1] - P[i])
        l1, l2 = d1.length, d2.length; d1 /= l1; d2 /= l2
        ang = d1.angle(d2)
        t = min(rad / max(math.tan(ang * 0.5), 1e-4), 0.49 * l1, 0.49 * l2)
        a, b = P[i] + d1 * t, P[i] + d2 * t
        for k in range(n + 1):
            s = k / n
            out.append(a * (1 - s) * (1 - s) + P[i] * 2 * s * (1 - s) + b * s * s)
    out.append(P[-1])
    return out


def pipe(g, pts, r, seg=8, caps=True, closed=False):
    """A round tube along a polyline."""
    P = [Vector(p) for p in pts]; n = len(P)
    tang = []
    for i in range(n):
        if closed: a, b = P[(i - 1) % n], P[(i + 1) % n]
        else: a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        tang.append((b - a).normalized())
    u = tang[0].cross(Vector((0, 0, 1)))
    if u.length < 1e-3: u = tang[0].cross(Vector((1, 0, 0)))
    u.normalize()
    rings = []
    for i in range(n):
        t = tang[i]
        u = u - t * u.dot(t); u.normalize(); v = t.cross(u)
        rings.append([(u * math.cos(TAU * k / seg) + v * math.sin(TAU * k / seg)) for k in range(seg)])
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        for k in range(seg):
            k2 = (k + 1) % seg
            a, b, c, d = rings[i][k], rings[i][k2], rings[j][k2], rings[j][k]
            quad(g, P[i] + a * r, P[i] + b * r, P[j] + c * r, P[j] + d * r, [a, b, c, d])
    if caps and not closed:
        for i, sg in ((0, -1.0), (n - 1, 1.0)):
            for k in range(seg):
                a, b = rings[i][k], rings[i][(k + 1) % seg]
                tri(g, (P[i], P[i] + a * r, P[i] + b * r), [tang[i] * sg] * 3)


def lathe(g, prof, seg=32, uref=None, sharp=0.75):
    """Revolve the (r, z) profile about the local Z axis. The surface faces to the RIGHT of the
    profile's direction of travel (going up in z: outward)."""
    n = len(prof)
    sn = []
    for i in range(n - 1):
        dr, dz = prof[i + 1][0] - prof[i][0], prof[i + 1][1] - prof[i][1]
        l = math.hypot(dr, dz) or 1.0
        sn.append((dz / l, -dr / l))
    vl = [0.0]
    for i in range(n - 1): vl.append(vl[-1] + math.hypot(prof[i + 1][0] - prof[i][0], prof[i + 1][1] - prof[i][1]))
    ur = uref if uref is not None else max(p[0] for p in prof)

    def vn(i, s):
        o = s - 1 if i == s else s + 1
        if o < 0 or o >= n - 1: return sn[s]
        if sn[s][0] * sn[o][0] + sn[s][1] * sn[o][1] < sharp: return sn[s]
        x, z = sn[s][0] + sn[o][0], sn[s][1] + sn[o][1]
        l = math.hypot(x, z) or 1.0
        return (x / l, z / l)

    for s in range(n - 1):
        (r0, z0), (r1, z1) = prof[s], prof[s + 1]
        n0, n1 = vn(s, s), vn(s + 1, s)
        for k in range(seg):
            b0 = TAU * k / seg; b1 = TAU * (k + 1) / seg
            c0, s0, c1, s1 = math.cos(b0), math.sin(b0), math.cos(b1), math.sin(b1)
            pa = (r0 * c0, r0 * s0, z0); pb = (r0 * c1, r0 * s1, z0)
            pc = (r1 * c1, r1 * s1, z1); pd = (r1 * c0, r1 * s0, z1)
            na = (n0[0] * c0, n0[0] * s0, n0[1]); nb = (n0[0] * c1, n0[0] * s1, n0[1])
            nc = (n1[0] * c1, n1[0] * s1, n1[1]); nd = (n1[0] * c0, n1[0] * s0, n1[1])
            ua, ub = b0 * ur, b1 * ur
            quad(g, pa, pb, pc, pd, [na, nb, nc, nd], [(ua, vl[s]), (ub, vl[s]), (ub, vl[s + 1]), (ua, vl[s + 1])])


def prism(g, poly, z0, z1, cham=0.0, ends=(True, True), caps=(True, True)):
    """Push the (x, y) outline out along local Z from z0 to z1; `cham` chamfers the end
    faces' edges (ends = which of the two)."""
    pts = [Vector((p[0], p[1])) for p in poly]
    n = len(pts)
    if sum(pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y for i in range(n)) < 0.0: pts.reverse()
    en = []
    for i in range(n):
        d = (pts[(i + 1) % n] - pts[i]).normalized()
        en.append(Vector((d.y, -d.x)))
    def inset(c):
        out = []
        for i in range(n):
            n0, n1 = en[i - 1], en[i]
            out.append(pts[i] - (n0 + n1) * (c / max(0.35, 1.0 + n0.dot(n1))))
        return out
    c0 = cham if ends[0] else 0.0; c1 = cham if ends[1] else 0.0
    za, zb = z0 + c0, z1 - c1
    for i in range(n):
        j = (i + 1) % n; nn = (en[i].x, en[i].y, 0.0)
        quad(g, (pts[i].x, pts[i].y, za), (pts[j].x, pts[j].y, za), (pts[j].x, pts[j].y, zb), (pts[i].x, pts[i].y, zb), [nn] * 4)
    for zc, zs, c, sg, cap in ((z0, za, c0, -1.0, caps[0]), (z1, zb, c1, 1.0, caps[1])):
        ring = inset(c) if c > 0.0 else pts
        if c > 0.0:
            for i in range(n):
                j = (i + 1) % n; nn = (en[i].x, en[i].y, sg)
                quad(g, (pts[i].x, pts[i].y, zs), (pts[j].x, pts[j].y, zs), (ring[j].x, ring[j].y, zc), (ring[i].x, ring[i].y, zc), [nn] * 4)
        if cap:
            for a, b, c2 in tessellate_polygon([[Vector((p.x, p.y, 0.0)) for p in ring]]):
                tri(g, ((ring[a].x, ring[a].y, zc), (ring[b].x, ring[b].y, zc), (ring[c2].x, ring[c2].y, zc)), [(0, 0, sg)] * 3)


def thick(path, t):
    """A polyline given thickness t to the left of its direction of travel -> an outline."""
    P = [Vector(p) for p in path]; n = len(P); off = []
    for i in range(n):
        n0 = n1 = None
        if i > 0:
            d = (P[i] - P[i - 1]).normalized(); n0 = Vector((-d.y, d.x))
        if i < n - 1:
            d = (P[i + 1] - P[i]).normalized(); n1 = Vector((-d.y, d.x))
        if n0 is None: m = n1
        elif n1 is None: m = n0
        else: m = (n0 + n1) / max(0.35, 1.0 + n0.dot(n1))
        off.append(P[i] + m * t)
    return [tuple(p) for p in P] + [tuple(p) for p in reversed(off)]



def take():
    global G
    g = G; G = {}
    return g


def write_mesh(name, groups):
    with open(os.path.join(OUT, name + '.mesh'), 'w') as f:
        for g, v in groups.items():
            f.write('mesh %s %d\n' % (g, len(v) // 3))
            for p, n, uv in v:
                f.write('%.4f %.4f %.4f %.3f %.3f %.3f %.4f %.4f\n' % (p.x, p.z, -p.y, n.x, n.z, -n.y, uv[0], uv[1]))


def bounds(groups):
    lo = Vector((1e9, 1e9, 1e9)); hi = Vector((-1e9, -1e9, -1e9))
    for v in groups.values():
        for p, _, _ in v:
            for i in range(3):
                lo[i] = min(lo[i], p[i]); hi[i] = max(hi[i], p[i])
    return lo, hi


def E(p): return [round(p[0], 4), round(p[2], 4), round(-p[1], 4) + 0.0]      # Blender -> engine


def box_uv(p, n):
    """Box mapping in metres: project along the face normal's dominant axis."""
    ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
    if az >= ax and az >= ay: return (p.x, p.y)
    if ax >= ay: return (p.y, p.z)
    return (p.x, p.z)


class smooth:
    """Everything emitted inside gets its normals averaged over the faces that meet at a point
    and differ by less than `deg` (so creases stay sharp); uv='box' also redoes the uvs."""
    def __init__(self, deg=40.0, uv=None):
        self.cos = math.cos(math.radians(deg)); self.uv = uv
    def __enter__(self):
        self.mark = {g: len(v) for g, v in G.items()}
    def __exit__(self, *a):
        tris = []
        for g, v in G.items():
            for i in range(self.mark.get(g, 0), len(v), 3): tris.append((g, i))
        acc = {}
        info = []
        for g, i in tris:
            v = G[g]
            P = [v[i][0], v[i + 1][0], v[i + 2][0]]
            fn = (P[1] - P[0]).cross(P[2] - P[0])
            if fn.length < 1e-12:
                info.append(None); continue
            fn.normalize()
            keys = []
            for k in range(3):
                e1 = P[(k + 1) % 3] - P[k]; e2 = P[(k + 2) % 3] - P[k]
                w = e1.angle(e2, 0.0) if e1.length > 1e-9 and e2.length > 1e-9 else 0.0
                key = (round(P[k].x, 4), round(P[k].y, 4), round(P[k].z, 4))
                acc.setdefault(key, []).append((fn, w))
                keys.append(key)
            info.append((fn, keys))
        for (g, i), inf in zip(tris, info):
            if inf is None: continue
            fn, keys = inf
            v = G[g]
            for k in range(3):
                n = Vector((0, 0, 0))
                for f2, w in acc[keys[k]]:
                    if f2.dot(fn) >= self.cos: n += f2 * w
                n = n.normalized() if n.length > 1e-9 else fn
                p, _, uv = v[i + k]
                if self.uv == 'box': uv = box_uv(p, fn)
                v[i + k] = (p, n, uv)


def loft(g, rings, caps=(None, None), closed=True):
    """Skin a row of rings (each the same number of points). `g` may be a list: one group per
    segment of the ring. caps = groups of the fan that closes the first / last ring."""
    R = [[Vector(p) for p in r] for r in rings]
    n = len(R[0]); m = n if closed else n - 1
    def gk(k): return g[k] if isinstance(g, (list, tuple)) else g
    cen = [sum(r, Vector((0, 0, 0))) / n for r in R]
    tot = 0.0
    for i in range(len(R) - 1):
        for k in range(m):
            a, b, c = R[i][k], R[i][(k + 1) % n], R[i + 1][(k + 1) % n]
            tot += (b - a).cross(c - a).dot((a + b + c) / 3 - (cen[i] + cen[i + 1]) / 2)
    flip = tot < 0
    for i in range(len(R) - 1):
        for k in range(m):
            a, b, c, d = R[i][k], R[i][(k + 1) % n], R[i + 1][(k + 1) % n], R[i + 1][k]
            if flip: quad(gk(k), d, c, b, a)
            else: quad(gk(k), a, b, c, d)
    for gc, i, j in ((caps[0], 0, 1), (caps[1], len(R) - 1, len(R) - 2)):
        if gc is None: continue
        nrm = cen[i] - cen[j]
        for k in range(m):
            tri(gc, (cen[i], R[i][k], R[i][(k + 1) % n]), [nrm] * 3)


def lerp(a, b, t): return a + (b - a) * t


# ================================================================== the measures
WR, WW = 0.40, 0.27            # wheel radius, tyre width
WB, TRACK = 3.20, 1.62
AXF, AXR = WB / 2, -WB / 2     # axle lines (Y)
WX = TRACK / 2
HALFW = 0.935                  # half the hull's width (the arch mouldings stand 0.025 proud of it)
Y_FRONT, Y_REAR = 2.54, -2.745 # the hull's ends (the rear bumper reaches 0.065 further back)
Z_ROOF = 1.85
ARCH_A, ARCH_H, ARCH_P = 0.475, 0.47, 2.7      # wheel opening: half length, height over the axle, squareness
FLOOR = 0.48                   # cabin floor
BED_FLOOR = 0.90
BED_HW = 0.78                  # half the load bed's inner width
SEAT_X = 0.42
SEAT_TOP = 0.78                # top of the seat cushions
STEER_TILT = 25.0              # the column rises toward the driver at this angle from the horizontal
STEER_HUB = (-SEAT_X, 0.42, 1.14)
HIP = (-SEAT_X, -0.085, SEAT_TOP + 0.10)     # rear third of the cushion, 0.10 above it
EYE_L = (-SEAT_X, HIP[1], HIP[2] + 0.52)     # "seat_driver": the engine puts the hip 0.52 under it
EYE_R = (SEAT_X, HIP[1], HIP[2] + 0.52)
LEGS = 0.2                     # 0 = upright chair, 1 = low seat with the legs stretched forward
PAINT, DARK, CARBON, METAL, RUBBER, GLASS, DASH, SEAT = 's_paint', 's_paint_dark', 's_carbon', 's_metal', 's_rubber', 's_glass', 's_dash', 's_seat'
GROUPS = ['s_paint', 's_paint_dark', 's_carbon', 's_glass', 's_rim', 's_tyre', 's_brake', 's_caliper', 's_seat', 's_dash', 's_metal',
          's_rubber', 's_light', 's_light_red', 's_light_amber', 's_gauge']
GI = {g: i for i, g in enumerate(GROUPS)}
MAT = {g: bpy.data.materials.new(g) for g in GROUPS}
rad = math.radians


def slabmesh(g, grid, D, front, back):
    """A solid slab between grid + D * front (its face) and grid + D * back. `g` may be a
    function (i, j) -> group of the cell."""
    D = Vector(D); nu, nv = len(grid), len(grid[0])
    A = [[Vector(p) + D * front for p in row] for row in grid]
    B = [[Vector(p) + D * back for p in row] for row in grid]
    cen = sum((p for row in A for p in row), Vector((0, 0, 0))) / (nu * nv) + D * (back - front) * 0.5
    def gg(i, j): return g(i, j) if callable(g) else g
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = A[i][j], A[i + 1][j], A[i + 1][j + 1], A[i][j + 1]
            quad_out(gg(i, j), a, b, c, d, (a + c) / 2 + D * (back - front))
            a, b, c, d = B[i][j], B[i + 1][j], B[i + 1][j + 1], B[i][j + 1]
            quad_out(gg(i, j), a, b, c, d, (a + c) / 2 - D * (back - front))
    for i in range(nu - 1):
        for j in (0, nv - 1):
            quad_out(gg(i, min(j, nv - 2)), A[i][j], A[i + 1][j], B[i + 1][j], B[i][j], cen)
    for j in range(nv - 1):
        for i in (0, nu - 1):
            quad_out(gg(min(i, nu - 2), j), A[i][j], A[i][j + 1], B[i][j + 1], B[i][j], cen)


def perimeter(grid):
    nu, nv = len(grid), len(grid[0])
    return ([grid[i][0] for i in range(nu)] + [grid[nu - 1][j] for j in range(1, nv)] +
            [grid[i][nv - 1] for i in range(nu - 2, -1, -1)] + [grid[0][j] for j in range(nv - 2, 0, -1)])


def patch(c00, c10, c11, c01, far, p=5.0):
    """Like bilin, with the corners rounded (the unit square is squeezed toward a superellipse)."""
    def fn(u, v):
        m = max(abs(u), abs(v))
        if p and m > 1e-9:
            r = (abs(u) ** p + abs(v) ** p) ** (1.0 / p)
            u, v = u * m / r, v * m / r
        s, t = (u + 1) / 2, (v + 1) / 2
        a = [lerp(lerp(c00[k], c10[k], s), lerp(c01[k], c11[k], s), t) for k in range(2)]
        a.insert(far[0], far[1])
        return a
    return fn


# ================================================================== Blender-side helpers (cage, cuts)
def new_obj(name, verts, faces, fmat=None, fix=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    for g in GROUPS: me.materials.append(MAT[g])
    if fmat is not None:
        if isinstance(fmat, str): fmat = [fmat] * len(faces)
        for p, g in zip(me.polygons, fmat): p.material_index = GI[g]
    me.update()
    if fix:
        bm = bmesh.new(); bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def evaluated_mesh(ob):
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    return bpy.data.meshes.new_from_object(ob.evaluated_get(dg))


def bake(ob, name):
    me = evaluated_mesh(ob)
    o2 = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o2)
    bpy.data.objects.remove(ob)
    return o2


def bvh_of(me):
    return BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])


CUTS = []                       # the cutter objects of the current stage
BVH = None


def cutter(verts, faces, fmat, mirror=False, bevel=0.0, fix=True):
    obs = [new_obj("cut", verts, faces, fmat, fix)]
    if mirror: obs.append(new_obj("cut", [(-v[0], v[1], v[2]) for v in verts], faces if fix else [tuple(reversed(f)) for f in faces], fmat, fix))
    for ob in obs:
        if bevel > 0.0:
            m = ob.modifiers.new('bev', 'BEVEL'); m.width = bevel; m.segments = 3; m.limit_method = 'NONE'
        ob.hide_render = True
        CUTS.append(ob)


def apply_cuts(body, name):
    if os.environ.get("PICKUP_DEBUG") == "1":
        for k, c in enumerate(CUTS):
            m = body.modifiers.new("cut", "BOOLEAN")
            m.operation = "DIFFERENCE"; m.object = c; m.solver = "EXACT"
            body = bake(body, name)
            xs = [v.co for v in body.data.vertices]
            print("CAR cut %d (%d verts in cutter, first %s): body %d polys, y %.2f..%.2f z %.2f..%.2f" % (k, len(c.data.vertices), tuple(round(a, 2) for a in c.data.vertices[0].co), len(body.data.polygons), min(p.y for p in xs), max(p.y for p in xs), min(p.z for p in xs), max(p.z for p in xs)))
    for c in ([] if os.environ.get("PICKUP_DEBUG") == "1" else CUTS):
        m = body.modifiers.new('cut', 'BOOLEAN')
        m.operation = 'DIFFERENCE'; m.object = c; m.solver = 'EXACT'
        try: m.material_mode = 'INDEX'
        except Exception: pass
    out = bake(body, name)
    for c in CUTS: bpy.data.objects.remove(c)
    CUTS.clear()
    return out


def ray(o, d):
    hit = BVH.ray_cast(Vector(o), Vector(d).normalized())
    assert hit[0] is not None, 'ray from %s along %s misses the body' % (tuple(o), tuple(d))
    return hit[0]


def top_z(x, y): return ray((x, y, 3.0), (0, 0, -1)).z
def side_x(y, z): return ray((3.0, y, z), (-1, 0, 0)).x
def tail_y(x, z): return ray((x, -4.0, z), (0, 1, 0)).y
def nose_y(x, z): return ray((x, 4.0, z), (0, -1, 0)).y


def cast_grid(fn, nu, nv, d):
    """fn(u, v) with u, v in -1..1 -> ray origin; returns the hit points [nu][nv]."""
    return [[ray(fn(-1.0 + 2.0 * i / (nu - 1), -1.0 + 2.0 * j / (nv - 1)), d) for j in range(nv)] for i in range(nu)]


def bilin(c00, c10, c11, c01, far):
    """A four-cornered patch in the plane across the axis `far` = (axis index, value)."""
    def fn(u, v):
        s, t = (u + 1) / 2, (v + 1) / 2
        a = [lerp(lerp(c00[k], c10[k], s), lerp(c01[k], c11[k], s), t) for k in range(2)]
        a.insert(far[0], far[1])
        return a
    return fn


def shell(top, bot, g, gfloor=None, closed=False, mirror=False):
    """A closed cutter between two grids of points [nu][nv] (top = outside the body)."""
    nu, nv = len(top), len(top[0])
    verts = [p for row in top for p in row] + [p for row in bot for p in row]
    def t(i, j): return (i % nu) * nv + j
    def b(i, j): return nu * nv + (i % nu) * nv + j
    faces = []; fm = []
    for i in range(nu if closed else nu - 1):
        for j in range(nv - 1):
            faces.append((t(i, j), t(i + 1, j), t(i + 1, j + 1), t(i, j + 1))); fm.append(g)
            faces.append((b(i, j), b(i, j + 1), b(i + 1, j + 1), b(i + 1, j))); fm.append(gfloor or g)
        faces.append((t(i, 0), b(i, 0), b(i + 1, 0), t(i + 1, 0))); fm.append(g)
        faces.append((t(i, nv - 1), t(i + 1, nv - 1), b(i + 1, nv - 1), b(i, nv - 1))); fm.append(g)
    if not closed:
        for j in range(nv - 1):
            faces.append((t(0, j), t(0, j + 1), b(0, j + 1), b(0, j))); fm.append(g)
            faces.append((t(nu - 1, j + 1), t(nu - 1, j), b(nu - 1, j), b(nu - 1, j + 1))); fm.append(g)
    vol = 0.0
    for f in faces:
        for k in range(1, len(f) - 1): vol += Vector(verts[f[0]]).dot(Vector(verts[f[k]]).cross(Vector(verts[f[k + 1]])))
    if vol < 0.0: faces = [tuple(reversed(f)) for f in faces]
    cutter(verts, faces, fm, mirror, fix=False)


def slab(grid, D, out, inn, g, gfloor=None, mirror=False):
    D = Vector(D)
    shell([[p + D * out for p in row] for row in grid], [[p - D * inn for p in row] for row in grid], g, gfloor, False, mirror)


def groove(path, closed=False, hw=0.0035, depth=0.009, mirror=False, radius=0.05, g=RUBBER):
    """A panel gap: a thin slot cut along a path that is snapped onto the body."""
    P = [Vector(p) for p in path]
    if closed:
        m = (P[-1] + P[0]) / 2
        P = fillet([m] + P + [m], radius, 5)[:-1]
    else:
        P = fillet(P, radius, 5)
    n = len(P); D = []
    for i in range(n if closed else n - 1):
        a, b = P[i], P[(i + 1) % n]
        k = max(1, int(math.ceil((b - a).length / 0.03)))
        for s in range(k): D.append(a.lerp(b, s / k))
    if not closed: D.append(P[-1])
    S = []
    for p in D:
        loc, nrm, _, _ = BVH.find_nearest(p)
        S.append((loc, nrm))
    n = len(S); top = []; bot = []
    for i in range(n):
        if closed: a, b = S[(i - 1) % n][0], S[(i + 1) % n][0]
        else: a, b = S[max(i - 1, 0)][0], S[min(i + 1, n - 1)][0]
        p, nr = S[i]
        ac = (b - a).cross(nr)
        if ac.length < 1e-6: ac = Vector((1, 0, 0))
        ac.normalize()
        top.append([p - ac * hw + nr * 0.03, p + ac * hw + nr * 0.03])
        bot.append([p - ac * hw - nr * depth, p + ac * hw - nr * depth])
    shell(top, bot, g, g, closed, mirror)


# ================================================================== the wheel
def build_wheel():
    """In the frame of wheel.mesh: axle = X, outer face toward +X."""
    with xf(PX):                                        # lathe axis (local z) -> X
        half = [(0.236, -0.108), (0.258, -0.124), (0.290, -0.131), (0.296, -0.135), (0.330, -0.135), (0.336, -0.131), (0.360, -0.128),
                (0.376, -0.118), (0.385, -0.102), (0.388, -0.060), (0.388, 0.0)]
        tyre = half + [(r, -x) for r, x in reversed(half[:-1])]
        lathe('s_tyre', tyre, 48, uref=WR, sharp=0.80)

        def lugs(ST, N, phase, fill):                   # tread blocks: rows of lugs, mirrored and staggered
            pitch = TAU / N; m = len(ST) - 1
            for s in (1, -1):
                for k in range(N):
                    th0 = pitch * (k + phase + (0.5 if s < 0 else 0.0))
                    def P(i, side, top):
                        x, rt, rb, sk = ST[i]
                        th = th0 + (sk + side * fill) * pitch; r = rt if top else rb
                        return Vector((r * math.cos(th), r * math.sin(th), s * x)), (th * WR, s * x)
                    for i in range(m):
                        c8 = [P(a, sd, tp) for a in (i, i + 1) for sd in (-1, 1) for tp in (0, 1)]
                        ref = sum((c[0] for c in c8), Vector((0, 0, 0))) / 8.0
                        def Q(a, b, c, d):
                            quad_out('s_tyre', a[0], b[0], c[0], d[0], ref, [a[1], b[1], c[1], d[1]])
                        Q(P(i, -1, 1), P(i, 1, 1), P(i + 1, 1, 1), P(i + 1, -1, 1))
                        for sd in (-1, 1): Q(P(i, sd, 0), P(i, sd, 1), P(i + 1, sd, 1), P(i + 1, sd, 0))
                        if i == 0: Q(P(0, -1, 0), P(0, 1, 0), P(0, 1, 1), P(0, -1, 1))
                        if i == m - 1: Q(P(m, -1, 0), P(m, 1, 0), P(m, 1, 1), P(m, -1, 1))
        lugs([(0.008, 0.400, 0.386, 0.22), (0.052, 0.400, 0.386, -0.06)], 20, 0.0, 0.30)
        lugs([(0.064, 0.400, 0.386, 0.0), (0.108, 0.400, 0.382, 0.0), (0.126, 0.387, 0.362, 0.05), (0.1345, 0.358, 0.340, 0.08)], 20, 0.5, 0.33)
        # the rim: outer flange, barrel, inner flange
        lathe('s_rim', [(0.236, 0.108), (0.242, 0.120), (0.236, 0.126), (0.224, 0.122), (0.218, 0.104), (0.212, -0.100),
                        (0.224, -0.118), (0.236, -0.122), (0.242, -0.116), (0.236, -0.106)], 40)
        # six spokes
        ST = [(0.045, 0.040, 0.070, 0.040), (0.085, 0.030, 0.076, 0.034), (0.125, 0.026, 0.086, 0.030), (0.165, 0.026, 0.098, 0.028),
              (0.200, 0.032, 0.108, 0.028), (0.222, 0.046, 0.114, 0.030)]
        with smooth(50.0):
            for k in range(6):
                a = TAU * (k + 0.25) / 6
                er = Vector((math.cos(a), math.sin(a), 0)); et = Vector((-math.sin(a), math.cos(a), 0)); ez = Vector((0, 0, 1))
                rings = []
                for r, hw, zf, th in ST:
                    c = er * r
                    rings.append([c - et * hw + ez * (zf - 0.012), c - et * hw * 0.55 + ez * zf, c + et * hw * 0.55 + ez * zf,
                                  c + et * hw + ez * (zf - 0.012), c + et * hw * 0.8 + ez * (zf - th), c - et * hw * 0.8 + ez * (zf - th)])
                loft('s_rim', rings)
        # hub, centre cap (blank), wheel nuts
        lathe('s_rim', [(0.072, 0.040), (0.072, 0.074), (0.062, 0.084), (0.032, 0.088), (0.0, 0.0885)], 30)
        lathe('s_metal', [(0.032, 0.088), (0.029, 0.093), (0.0, 0.094)], 20)
        for k in range(6):
            a = TAU * (k + 0.75) / 6
            c = Vector((0.050 * math.cos(a), 0.050 * math.sin(a), 0.083))
            cyl('s_metal', c, c + Vector((0, 0, 0.012)), 0.0085, 6, (False, True))
        # brake: disc with its bell, caliper on top
        lathe('s_brake', [(0.085, 0.000), (0.172, 0.000), (0.172, 0.030), (0.085, 0.030)], 40)
        lathe('s_metal', [(0.0, -0.085), (0.050, -0.085), (0.050, -0.020), (0.092, -0.020), (0.092, 0.040), (0.072, 0.040)], 24)
        arc = [(r * math.cos(rad(a)), r * math.sin(rad(a))) for r, aa in ((0.198, range(58, 123, 8)), (0.115, range(122, 57, -8))) for a in aa]
        prism('s_caliper', arc, -0.016, 0.046, 0.007)
        for a in (74.0, 106.0):                          # the pistons' bosses
            c = Vector((0.156 * math.cos(rad(a)), 0.156 * math.sin(rad(a)), 0.046))
            cyl('s_caliper', c, c + Vector((0, 0, 0.004)), 0.018, 10, (False, True))


# ================================================================== the steering wheel
def build_steering():
    """In the frame of steering.mesh after RX(90): axis = Blender -Y = engine +Z."""
    with xf(RX(rad(90.0))):                             # local z (toward the driver) -> Blender -Y
        R = 0.185
        with smooth(60.0):
            pipe('s_dash', [(R * math.cos(TAU * i / 40), R * math.sin(TAU * i / 40), 0.0) for i in range(40)], 0.0165, 10, closed=True)
        for a, w in ((5.0, 0.044), (175.0, 0.044), (270.0, 0.036)):
            c, s = math.cos(rad(a)), math.sin(rad(a))
            obox('s_carbon', (0.045 * c, 0.045 * s - 0.006, -0.026), (0.178 * c, 0.178 * s, -0.003), w, 0.014, (0, 0, 1), 0.004)
        lathe('s_dash', [(0.070, -0.055), (0.072, -0.028), (0.064, -0.014), (0.034, -0.007), (0.0, -0.006)], 24)
        lathe('s_carbon', [(0.036, -0.110), (0.042, -0.055), (0.070, -0.055)], 20)
        lathe('s_metal', [(0.022, -0.0085), (0.019, -0.0045), (0.0, -0.004)], 16)          # a plain boss, no badge
        for sx in (-1, 1):                                                                  # thumb switches
            cyl('s_metal', (sx * 0.098, 0.004, -0.012), (sx * 0.098, 0.004, -0.004), 0.009, 10, (False, True))
            cyl('s_metal', (sx * 0.126, 0.006, -0.008), (sx * 0.126, 0.006, -0.0005), 0.007, 10, (False, True))


# ================================================================== the body shell
NR = 22                         # points per cross-section (12 on a half, centre points shared)
K = {}                          # things measured on the hull that the fittings need


def make_shell():
    def R(y, w, zb, zc, zs, ins, U, sw=0.0):
        half = [(0.0, zb), (0.55 * w, zb), (w - 0.07, zb), (w - 0.012, zb + 0.05), (w - 0.004, zc), (w, 0.5 * (zc + zs) + 0.05),
                (w - 0.004, zs - 0.05), (w - ins, zs)] + list(U)
        return [y, sw, w, half]
    def bon(w, ins, zt, c): return [(w - ins - 0.10, zt), (0.60 * w, zt + c), (0.30 * w, zt + 1.6 * c), (0.0, zt + 1.8 * c)]
    def mixr(a, b, t):
        return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t), [(lerp(p[0], q[0], t), lerp(p[1], q[1], t)) for p, q in zip(a[3], b[3])]]
    def bed(w): return [(w - 0.07, 1.34), (0.60, 1.34), (0.30, 1.34), (0.0, 1.34)]
    F0 = R(2.54, 0.80, 0.44, 0.60, 1.06, 0.05, bon(0.80, 0.05, 1.10, 0.006))
    F1 = R(2.46, 0.905, 0.38, 0.58, 1.09, 0.05, bon(0.905, 0.05, 1.13, 0.012))
    F2 = R(2.24, 0.93, 0.35, 0.57, 1.12, 0.05, bon(0.93, 0.05, 1.155, 0.016))
    F3 = R(1.60, 0.935, 0.34, 0.55, 1.165, 0.05, bon(0.935, 0.05, 1.19, 0.02))
    F4 = R(1.14, 0.93, 0.34, 0.52, 1.21, 0.045, bon(0.93, 0.045, 1.225, 0.02))
    C0 = R(1.03, 0.93, 0.34, 0.52, 1.22, 0.04, [(0.855, 1.235), (0.78, 1.24), (0.42, 1.255), (0.0, 1.26)])
    C1 = R(0.33, 0.925, 0.34, 0.52, 1.26, 0.03, [(0.832, 1.50), (0.76, 1.775), (0.42, 1.835), (0.0, 1.845)])
    C2 = R(-0.30, 0.925, 0.34, 0.52, 1.28, 0.03, [(0.832, 1.52), (0.76, 1.79), (0.42, 1.85), (0.0, 1.86)])
    C3 = R(-0.90, 0.925, 0.34, 0.52, 1.30, 0.03, [(0.832, 1.53), (0.76, 1.785), (0.42, 1.84), (0.0, 1.85)])
    C4 = R(-1.00, 0.925, 0.36, 0.53, 1.305, 0.03, [(0.832, 1.53), (0.76, 1.775), (0.42, 1.83), (0.0, 1.84)])
    B0 = R(-1.05, 0.925, 0.40, 0.54, 1.31, 0.025, bed(0.925))
    B1 = R(-1.13, 0.925, 0.46, 0.56, 1.31, 0.025, bed(0.925))
    B2 = R(-1.60, 0.935, 0.47, 0.57, 1.31, 0.025, bed(0.935))
    B3 = R(-2.30, 0.93, 0.52, 0.62, 1.31, 0.025, bed(0.93))
    B4 = R(-2.68, 0.925, 0.58, 0.68, 1.31, 0.025, bed(0.925))
    B5 = R(-2.745, 0.905, 0.60, 0.70, 1.31, 0.025, bed(0.905), 0.012)
    RG = [F0, F1, F2, F3, F4, C0, mixr(C0, C1, 0.1), mixr(C0, C1, 0.9), C1, C2, C3, C4, B0, B1, B2, B3, B4, B5]
    CAB = range(5, 12)
    verts = []; faces = []; fm = []
    for y, sw, w, half in RG:
        ring = [(x, y - sw * (x / w) ** 2, z) for x, z in half] + [(-half[j][0], y - sw * (half[j][0] / w) ** 2, half[j][1]) for j in range(10, 0, -1)]
        assert len(ring) == NR
        verts += ring
    def gm(k):
        kk = k if k < 11 else NR - 1 - k
        return DARK if kk <= 3 else PAINT
    for i in range(len(RG) - 1):
        for k in range(NR):
            a, b = i * NR + k, i * NR + (k + 1) % NR
            faces.append((a, b, b + NR, a + NR)); fm.append(gm(k))
    for base in (0, (len(RG) - 1) * NR):                 # the end faces, as strips from side to side
        for j in range(1, 10):
            faces.append((base + j, base + j + 1, base + NR - (j + 1), base + NR - j)); fm.append(DARK if j <= 3 else PAINT)
        faces.append((base, base + 1, base + NR - 1)); fm.append(DARK)
        faces.append((base + 11, base + 12, base + 10)); fm.append(PAINT)
    ob = new_obj('shell', verts, faces, fm)
    me = ob.data
    try:                                                 # edge creases keep the feature lines from washing out
        att = me.attributes.get('crease_edge') or me.attributes.new('crease_edge', 'FLOAT', 'EDGE')
        last = len(RG) - 1
        for e in me.edges:
            (ia, ka), (ib, kb) = divmod(e.vertices[0], NR), divmod(e.vertices[1], NR)
            cr = 0.0
            if ka == kb:
                k = min(ka, NR - ka)
                if k == 7: cr = 0.45
                elif k == 9 and ia in CAB and ib in CAB: cr = 0.6
                elif k == 4: cr = 0.25
            elif ia == ib:
                if ia == 0: cr = 0.65
                elif ia == 11 and min(ka, NR - ka) >= 8 and min(kb, NR - kb) >= 8: cr = 0.6
                elif ia == last: cr = 0.6
            att.data[e.index].value = cr
    except Exception as ex:
        print('CAR no creases:', ex)
    m = ob.modifiers.new('ss', 'SUBSURF'); m.levels = 2; m.render_levels = 2
    ob = bake(ob, 'shell1')
    me = ob.data
    wmax = max(abs(v.co.x) for v in me.vertices)
    y0 = min(v.co.y for v in me.vertices); y1 = max(v.co.y for v in me.vertices)
    z1 = max(v.co.z for v in me.vertices)
    print('CAR raw shell: half width %.3f, y %.3f..%.3f, top %.3f' % (wmax, y0, y1, z1))
    for v in me.vertices:
        v.co.x *= HALFW / wmax
        v.co.y = Y_REAR + (v.co.y - y0) * (Y_FRONT - Y_REAR) / (y1 - y0)
        v.co.z = 0.34 + (v.co.z - 0.34) * (Z_ROOF - 0.34) / (z1 - 0.34)
    me.update()
    return ob


def tris_of(ob):
    me = ob.data
    me.calc_loop_triangles()
    for lt in me.loop_triangles:
        yield GROUPS[lt.material_index], [me.vertices[i].co.copy() for i in lt.vertices]


def arch_outline(yc, n=40):
    out = []
    for i in range(n + 1):
        th = math.pi * i / n; c, s = math.cos(th), math.sin(th)
        out.append((yc + ARCH_A * math.copysign(abs(c) ** (2.0 / ARCH_P), c), WR + ARCH_H * abs(s) ** (2.0 / ARCH_P)))
    return out


def box_cutter(lo, hi, g, mirror=False, bevel=0.0):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    F = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    cutter(V, F, g, mirror, bevel)


def S(y, z): return Vector((side_x(y, z), y, z))         # a point on the right flank
def Tp(x, y): return Vector((x, y, top_z(x, y)))         # a point on the top


def build_shell():
    """The subdivided hull with every opening cut into it; fills BVH (the uncut surface)."""
    global BVH
    body = make_shell()
    BVH = bvh_of(body.data)
    K['rail'] = top_z(0.84, -1.80)
    K['ty'] = tail_y(0.0, 1.0)
    K['ny'] = nose_y(0.0, 0.80)
    K['cabb'] = ray((0.0, -1.6, 1.55), (0, 1, 0)).y
    print('CAR hull: rail %.3f, tail %.3f, nose %.3f, cab back %.3f, roof %.3f, bonnet %.3f / %.3f, belt half widths %.3f %.3f' %
          (K['rail'], K['ty'], K['ny'], K['cabb'], top_z(0, -0.3), top_z(0, 2.2), top_z(0, 1.2), side_x(0.3, 1.0), side_x(0.3, 1.5)))
    # wheel housings
    for yc in (AXF, AXR):
        out = [(yc + ARCH_A, -0.2)] + arch_outline(yc) + [(yc - ARCH_A, -0.2)]
        m = len(out); V = []
        for x in (0.48, 1.4): V += [(x, y, z) for y, z in out]
        F = [(i, (i + 1) % m, m + (i + 1) % m, m + i) for i in range(m)] + [tuple(range(m)), tuple(range(m, 2 * m))]
        cutter(V, F, DARK, True)
    # the cabin: a hollow that keeps 4..6 cm of skin
    K['cavb'] = K['cabb'] + 0.055
    ys = [0.93, 0.80, 0.66, 0.52, 0.38, 0.22, 0.0, -0.3, -0.6, -0.85, K['cavb']]
    V = []; n = 0
    for y in ys:
        zt = top_z(0.0, y) - 0.03
        half = []
        for z in (0.62, 0.90, 1.16):
            half.append((side_x(y, z) - 0.06, z))
        for t in (0.25, 0.55, 0.80, 0.94):
            z = lerp(1.22, zt, t)
            x = side_x(y, z) - 0.06
            half.append((x, min(z, top_z(x, y) - 0.03)))
        xl = half[-1][0]
        for f in (0.66, 0.33):
            half.append((xl * f, top_z(xl * f, y) - 0.03))
        half = [(half[0][0] - 0.03, FLOOR)] + half
        ring = [(0.0, FLOOR)] + half + [(0.0, zt)] + [(-x, z) for x, z in reversed(half)]
        n = len(ring)
        V += [(x, y, z) for x, z in ring]
    F = []
    for i in range(len(ys) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            F.append((a, b, b + n, a + n))
    F += [tuple(range(n)), tuple(range((len(ys) - 1) * n, len(ys) * n))]
    cutter(V, F, CARBON)
    K['cav_hw'] = side_x(0.75, 1.15) - 0.06
    # the load bed
    K['bed_y0'] = -1.11; K['bed_y1'] = K['ty'] + 0.07
    box_cutter((-BED_HW, K['bed_y1'], BED_FLOOR), (BED_HW, K['bed_y0'], 2.6), DARK)
    # grille, lower intake, headlight and tail light pockets
    g2 = cast_grid(patch((-0.40, 0.66), (0.40, 0.66), (0.47, 1.035), (-0.47, 1.035), (1, 4.0), 7.0), 11, 6, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.06, 0.05, DARK, DARK)
    K['grille'] = g2
    g2 = cast_grid(patch((-0.50, 0.475), (0.50, 0.475), (0.50, 0.565), (-0.50, 0.565), (1, 4.0)), 9, 3, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.06, 0.04, DARK, DARK)
    K['intake'] = g2
    g2 = cast_grid(patch((0.50, 0.895), (0.875, 0.915), (0.875, 1.035), (0.51, 1.035), (1, 4.0)), 9, 5, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.10, 0.07, DARK, DARK, True)
    K['head'] = g2
    g2 = cast_grid(patch((0.775, 0.92), (0.89, 0.92), (0.89, 1.285), (0.775, 1.285), (1, -4.0), 8.0), 4, 9, (0, 1, 0))
    slab(g2, (0, -1, 0), 0.06, 0.03, DARK, DARK, True)
    K['tail'] = g2
    # glazing: windscreen, door windows, rear screen
    g2 = cast_grid(patch((-0.70, 0.955), (0.70, 0.955), (0.675, 0.40), (-0.675, 0.40), (2, 3.0)), 13, 7, (0, 0, -1))
    slab(g2, (0, 0, 1), 0.06, 0.10, RUBBER, RUBBER)
    K['ws'] = g2
    g2 = cast_grid(patch((0.78, 1.315), (-0.07, 1.325), (-0.07, 1.715), (0.26, 1.715), (0, 3.0)), 9, 6, (-1, 0, 0))
    slab(g2, (1, 0, 0), 0.06, 0.10, RUBBER, RUBBER, True)
    K['wf'] = g2
    g2 = cast_grid(patch((-0.17, 1.325), (-0.84, 1.335), (-0.81, 1.715), (-0.17, 1.715), (0, 3.0)), 8, 6, (-1, 0, 0))
    slab(g2, (1, 0, 0), 0.06, 0.10, RUBBER, RUBBER, True)
    K['wr'] = g2
    g2 = cast_grid(patch((-0.63, 1.38), (0.63, 1.38), (0.60, 1.665), (-0.60, 1.665), (1, -1.6), 3.5), 9, 5, (0, 1, 0))
    slab(g2, (0, -1, 0), 0.05, 0.11, RUBBER, RUBBER)
    K['rs'] = g2
    # panel gaps: doors, bonnet, tailgate, cab / bed, fuel flap
    groove([S(0.95, 0.54), S(0.95, 1.235), S(0.34, 1.705), S(0.22, 1.755), S(-0.12, 1.76), S(-0.12, 0.54)], True, mirror=True, radius=0.05)
    groove([S(-0.12, 1.76), S(-0.90, 1.755), S(-0.95, 1.33), S(-0.95, 0.54), S(-0.12, 0.54)], False, mirror=True, radius=0.06)
    groove([Tp(-0.80, 1.12), Tp(-0.80, 2.27), Tp(-0.62, 2.40), Tp(0.62, 2.40), Tp(0.80, 2.27), Tp(0.80, 1.12)], False, radius=0.07)
    ty = K['ty']; rl = K['rail']
    groove([(0.745, ty + 0.05, rl), (0.745, ty, rl), (0.745, ty, 0.715), (-0.745, ty, 0.715), (-0.745, ty, rl), (-0.745, ty + 0.05, rl)], False, radius=0.03)
    groove([S(-1.075, 0.58), S(-1.075, 1.30), (0.86, -1.075, rl), (0.80, -1.075, rl)], False, mirror=True, radius=0.02)
    fp = [S(-1.22, 1.19), S(-1.37, 1.19), S(-1.37, 1.04), S(-1.22, 1.04)]
    groove([(-p.x, p.y, p.z) for p in fp], True, radius=0.035)
    body = apply_cuts(body, 'shell3')
    return body


# ================================================================== the body
def gauge(x, y, z, r, needle_deg):
    with xf(T(x, y, z) @ RX(rad(90.0))):                # local z -> toward the driver (-Y)
        lathe(METAL, [(r + 0.006, 0.0), (r + 0.006, 0.008), (r, 0.010)], 24)
        lathe('s_gauge', [(r, 0.005), (0.0, 0.005)], 24)
        for i in range(11):
            a = rad(225.0 - 270.0 * i / 10)
            c, s = math.cos(a), math.sin(a)
            obox(DASH, (r * 0.76 * c, r * 0.76 * s, 0.0055), (r * 0.94 * c, r * 0.94 * s, 0.0055), r * 0.06, 0.001)
        a = rad(needle_deg)
        obox('s_light_red', (0, 0, 0.0065), (r * 0.82 * math.cos(a), r * 0.82 * math.sin(a), 0.0065), r * 0.06, 0.0015)
        cyl(DASH, (0, 0, 0.005), (0, 0, 0.009), r * 0.14, 10, (False, True))


def vent(x, y, z, r=0.04):
    with xf(T(x, y, z) @ RX(rad(90.0))):
        lathe(METAL, [(r + 0.006, 0.0), (r + 0.006, 0.010), (r, 0.012), (r - 0.004, 0.004)], 20)
        lathe(CARBON, [(r - 0.004, 0.004), (0.0, 0.004)], 20)
        for dz in (-0.5, 0.0, 0.5):
            hw = r * math.sqrt(1.0 - dz * dz) * 0.92
            box(DASH, (-hw, dz * r - 0.003, 0.004), (hw, dz * r + 0.003, 0.009), '-z')


def seat(xc, path, par):
    rings = []
    for i, ((y, z), (W, B, th)) in enumerate(zip(path, par)):
        a = Vector(path[max(i - 1, 0)]); b = Vector(path[min(i + 1, len(path) - 1)])
        t = (b - a).normalized(); n = Vector((t.y, -t.x))
        bw = min(0.07, W * 0.32); gap = min(0.04, W * 0.16)
        hs = [(-W + 0.035, -th), (-W, -th * 0.5), (-W, B * 0.55), (-W + 0.02, B), (-W + bw * 0.65, B), (-W + bw, B * 0.4), (-W + bw + gap, 0.0)]
        sec = hs + [(0.0, 0.0)] + [(-s, h) for s, h in reversed(hs)]
        rings.append([(xc + s, y + n.x * h, z + n.y * h) for s, h in sec])
    gs = [SEAT] * 14 + [CARBON]
    with smooth(62.0):
        loft(gs, rings, (SEAT, SEAT))


def build_body():
    body = build_shell()
    # ---------------------------------------------------------------- the hull
    with smooth(38.0, 'box'):
        for g, P in tris_of(body): tri(g, P)
    rl = K['rail']; ty = K['ty']; ny = K['ny']; cabb = K['cabb']; cavb = K['cavb']
    V3 = Vector

    # ---------------------------------------------------------------- wheel arch mouldings (dark cladding)
    for yc, zlf, zlr in ((AXF, 0.45, 0.37), (AXR, 0.48, 0.53)):
        zlow = min(zlf, zlr)
        O = [(y, max(z, zlf if y > yc else zlr)) for y, z in arch_outline(yc, 30)]
        O = [(O[0][0], zlf)] + O + [(O[-1][0], zlr)]
        pts = [Vector(O[0])]
        for p in O[1:]:
            if (Vector(p) - pts[-1]).length > 0.006: pts.append(Vector(p))
        secs = []
        for i, p in enumerate(pts):
            a = pts[max(i - 1, 0)]; b = pts[min(i + 1, len(pts) - 1)]
            t = (b - a).normalized(); nn = Vector((-t.y, t.x))
            if nn.dot(p - Vector((yc, WR + 0.1))) < 0.0: nn = -nn
            if p.y < WR + 0.02 and abs(nn.y) < 0.9: nn = Vector((math.copysign(1.0, p.x - yc), 0.0))
            q = p + nn * 0.04
            xs = side_x(q.x, max(q.y, zlow + 0.12))
            def P3(dx, dn): return Vector((xs + dx, p.x + nn.x * dn, p.y + nn.y * dn))
            secs.append([P3(-0.035, 0.0), P3(0.024, 0.0), P3(0.024, 0.062), P3(-0.004, 0.086)])
        for m in (IDN, MIR):
            with xf(m):
                with smooth(50.0, 'box'):
                    for i in range(len(secs) - 1):
                        A, B = secs[i], secs[i + 1]
                        ref = (A[0] + A[3] + B[0] + B[3]) / 4.0 + Vector((-0.02, 0, 0))
                        for k in range(3): quad_out(DARK, A[k], A[k + 1], B[k + 1], B[k], ref)
                for sec, sg in ((secs[0], -1.0), (secs[-1], -1.0)):                 # close the ends (they look down)
                    quad(DARK, sec[0], sec[1], sec[2], sec[3], [(0, 0, -1)] * 4)
    hull = bounds(G)

    # ---------------------------------------------------------------- glazing
    def glass(grid, D, mir):
        D = Vector(D)
        for m in ((IDN, MIR) if mir else (IDN,)):
            with xf(m):
                with smooth(60.0):
                    for i in range(len(grid) - 1):
                        for j in range(len(grid[0]) - 1):
                            quad2(GLASS, *[p - D * 0.012 for p in (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])])
                with smooth(70.0):
                    pipe(RUBBER, [p - D * 0.003 for p in perimeter(grid)], 0.008, 5, closed=True)
    glass(K['ws'], (0, 0, 1), False)
    glass(K['wf'], (1, 0, 0), True)
    glass(K['wr'], (1, 0, 0), True)
    glass(K['rs'], (0, -1, 0), False)
    # the dark panel between the door windows
    bp = cast_grid(bilin((-0.072, 1.33), (-0.168, 1.33), (-0.168, 1.712), (-0.072, 1.712), (0, 3.0)), 2, 5, (-1, 0, 0))
    for m in (IDN, MIR):
        with xf(m):
            with smooth(60.0): slabmesh(DARK, bp, (1, 0, 0), 0.003, -0.01)
    # the dark cowl panel between bonnet and screen
    cw = [[Tp(-0.80 + 0.2 * i, y) for y in (1.115, 1.06, 1.0)] for i in range(9)]
    with smooth(60.0): slabmesh(DARK, cw, (0, 0, 1), 0.005, -0.02)
    # wipers, parked on the lower edge of the screen
    GP = K['ws']; NS = len(GP) - 1
    gn = (GP[NS // 2][2] - GP[NS // 2][1]).cross(Vector((1, 0, 0))).normalized()
    if gn.z < 0: gn = -gn
    for i0, i1, ip in ((2, 6, 6), (7, 11, 11)):
        wa = GP[i0][1].lerp(GP[i0][2], 0.1) + gn * 0.016; wb = GP[i1][1].lerp(GP[i1][2], 0.1) + gn * 0.016
        obox(RUBBER, wa, wb, 0.016, 0.014, gn)
        piv = GP[ip][0] + gn * 0.014 + Vector((0.05, 0.035, -0.01))
        cyl(METAL, piv - gn * 0.03, piv + gn * 0.012, 0.013, 10)
        obox(RUBBER, piv + gn * 0.010, (wa + wb) / 2 + gn * 0.010, 0.014, 0.006, gn)

    # ---------------------------------------------------------------- front: grille, lamps, skid plate
    gr = K['grille']; nu, nv = len(gr), len(gr[0])
    back = Vector((0, -0.022, 0))
    with smooth(60.0):
        pipe(METAL, [p + Vector((0, 0.002, 0)) for p in perimeter(gr)], 0.013, 6, closed=True)       # the bright surround
    for j in range(1, nv - 1):                                                                       # horizontal bars
        for i in range(nu - 1):
            a, b = gr[i][j] + back, gr[i + 1][j] + back
            a = Vector((a.x, a.y, gr[0][j].z)); b = Vector((b.x, b.y, gr[0][j].z))
            obox(DARK, a, b, 0.030, 0.034, (0, 1, 0))
    for i in range(1, nu - 1):                                                                       # thin uprights
        obox(DARK, gr[i][0] + back * 1.6, gr[i][nv - 1] + back * 1.6, 0.012, 0.016, (0, 1, 0))
    it = K['intake']
    for i in range(len(it) - 1):
        a, b = it[i][1] + back, it[i + 1][1] + back
        obox(DARK, a, b, 0.016, 0.03, (0, 1, 0))
    hl = K['head']; nu, nv = len(hl), len(hl[0])
    for m in (IDN, MIR):
        with xf(m):
            with smooth(60.0, 'box'):                                        # the clear cover, flush with the nose
                for i in range(nu - 1):
                    for j in range(nv - 1):
                        a, b, c, d = [p - Vector((0, 0.003, 0)) for p in (hl[i][j], hl[i + 1][j], hl[i + 1][j + 1], hl[i][j + 1])]
                        quad_out(GLASS, a, b, c, d, (a + c) / 2 - Vector((0, 1, 0)))
            mid = nv // 2
            for i, r in ((2, 0.036), (5, 0.036)):                            # two projectors
                p = hl[i][mid] - Vector((0, 0.062, 0))
                with xf(T(p.x, min(p.y, hl[4][mid].y - 0.062), p.z) @ RX(rad(-90.0))):
                    lathe(METAL, [(r * 1.30, -0.01), (r * 1.30, 0.014), (r, 0.018)], 18)
                    lathe('s_light', [(r, 0.018), (r * 0.85, 0.018 + r * 0.25), (r * 0.5, 0.018 + r * 0.42), (0.0, 0.018 + r * 0.48)], 18)
            pipe('s_light', [hl[i][nv - 1].lerp(hl[i][nv - 2], 0.7) - Vector((0, 0.022, 0)) for i in range(1, nu - 1)], 0.007, 6)     # running light
            pipe('s_light', [hl[i][0].lerp(hl[i][1], 0.7) - Vector((0, 0.022, 0)) for i in range(1, nu - 3)], 0.005, 6)
            pipe('s_light_amber', [hl[nu - 2][j] - Vector((0, 0.026, 0)) for j in range(1, nv - 1)], 0.011, 6)                       # indicator
            # fog lamp in the bumper
            fz = 0.515; fx = 0.70; fy = nose_y(fx, fz)
            with xf(T(fx, fy - 0.004, fz) @ RX(rad(-90.0))):
                lathe(DARK, [(0.062, -0.03), (0.062, 0.006), (0.048, 0.012)], 20)
                lathe('s_light', [(0.048, 0.012), (0.036, 0.018), (0.0, 0.020)], 20)
    cy = nose_y(0.0, 0.47)
    obox(METAL, (0, cy + 0.012, 0.462), (0, cy - 0.15, 0.378), 0.84, 0.02, (0, 0, 1), 0.006)          # skid plate under the chin
    for x in (-0.25, 0.0, 0.25):
        obox(DARK, (x, cy + 0.016, 0.455), (x, cy - 0.12, 0.388), 0.09, 0.026, (0, 0, 1), 0.004)

    # ---------------------------------------------------------------- rear: lamps, bumper, exhaust
    tl = K['tail']
    def tg(i, j): return 's_light_amber' if j <= 1 else ('s_light' if j == 2 else 's_light_red')
    for m in (IDN, MIR):
        with xf(m):
            with smooth(60.0): slabmesh(tg, tl, (0, -1, 0), 0.006, -0.028)
    hy = ray((0.0, -1.6, 1.765), (0, 1, 0)).y
    cbox('s_light_red', (-0.14, hy - 0.010, 1.752), (0.14, hy + 0.02, 1.778), 0.004)                  # high brake light
    by0, by1 = ty - 0.065, ty + 0.11
    cbox(METAL, (-0.925, by0, 0.47), (0.925, by1, 0.665), 0.03)                                       # bumper
    cbox(DARK, (-0.62, by0 + 0.012, 0.66), (0.62, ty - 0.006, 0.674), 0.004)                          # its step pad
    for i in range(9):
        x = -0.56 + 0.14 * i
        box(DARK, (x - 0.045, by0 + 0.016, 0.674), (x + 0.045, ty - 0.012, 0.678), '-z')
    cbox(DARK, (-0.27, by0 - 0.004, 0.50), (0.27, by0 + 0.01, 0.635), 0.004)                          # blank plate panel
    for sx in (-1, 1):
        cbox('s_light_red', (sx * 0.74 - 0.07, by0 - 0.003, 0.545), (sx * 0.74 + 0.07, by0 + 0.01, 0.585), 0.004)   # reflectors
    cbox(DARK, (-0.055, ty - 0.03, 0.395), (0.055, ty + 0.14, 0.47), 0.008)                           # hitch receiver
    box(DARK, (-0.44, ty + 0.08, 0.40), (0.44, ty + 0.15, 0.53))
    cbox(DARK, (-0.09, ty - 0.014, 1.205), (0.09, ty + 0.004, 1.255), 0.006)                          # tailgate handle
    with xf(T(0.26, by0 + 0.005, 0.425) @ RX(rad(90.0))):                                             # exhaust tip (local z -> rearward)
        lathe(METAL, [(0.042, -0.14), (0.042, 0.0), (0.034, 0.0), (0.034, -0.10)], 18)
        lathe(DARK, [(0.034, -0.10), (0.0, -0.10)], 18)
    with smooth(60.0):
        pipe(DARK, fillet([(0.26, -1.0, 0.40), (0.26, -1.38, 0.42), (0.26, -1.50, 0.53), (0.26, -1.72, 0.53), (0.26, -2.0, 0.44),
                           (0.26, by0 + 0.13, 0.425)], 0.05, 3), 0.028, 8)
    cyl(DARK, (0.26, -2.50, 0.435), (0.26, -2.12, 0.44), 0.075, 14)                                   # silencer

    # ---------------------------------------------------------------- running gear that shows
    for sx in (-1, 1):
        box(DARK, (sx * 0.36, ty + 0.06, 0.39), (sx * 0.44, -1.0, 0.53))                              # frame rails
        cyl(DARK, (sx * 0.47, AXF, WR), (sx * 0.73, AXF, WR), 0.05, 12)
        cyl(METAL, (sx * 0.57, AXF, 0.44), (sx * 0.53, AXF, 0.88), 0.022, 10)                         # front strut
        cyl(DARK, (sx * 0.555, AXF, 0.60), (sx * 0.532, AXF, 0.86), 0.055, 12)
        cyl(METAL, (sx * 0.54, AXR + 0.10, 0.40), (sx * 0.51, AXR + 0.17, 0.80), 0.024, 10)           # rear damper
    cyl(DARK, (-0.73, AXR, WR), (0.73, AXR, WR), 0.045, 12)                                           # rear axle
    ball(DARK, (0.0, AXR, WR), 0.16, 14, 10, (1.0, 1.12, 1.0))
    cyl(DARK, (0.0, AXR + 0.17, WR), (0.0, -0.95, 0.40), 0.035, 10)                                   # propeller shaft

    # ---------------------------------------------------------------- side steps, door handles
    for m in (IDN, MIR):
        with xf(m):
            cbox(DARK, (0.80, -1.0, 0.335), (0.958, 1.0, 0.385), 0.015)
            cbox(METAL, (0.950, -0.97, 0.345), (0.962, 0.97, 0.378), 0.005)
            for y0, y1 in ((-0.92, -0.16), (-0.06, 0.90)):
                for k in range(int((y1 - y0) / 0.06)):
                    box(RUBBER, (0.83, y0 + 0.06 * k, 0.385), (0.94, y0 + 0.06 * k + 0.035, 0.391), '-z')
            for y in (-0.80, 0.0, 0.80):
                box(DARK, (0.70, y - 0.03, 0.345), (0.82, y + 0.03, 0.375))
            for y in (-0.02, -0.84):
                xs = side_x(y, 1.17)
                cbox(DARK, (xs - 0.01, y - 0.088, 1.142), (xs + 0.004, y + 0.088, 1.198), 0.004)
                obox(METAL, (xs + 0.017, y - 0.076, 1.17), (xs + 0.017, y + 0.076, 1.17), 0.028, 0.016, (1, 0, 0), 0.005)
                for dy in (-0.06, 0.06):
                    box(METAL, (xs, y + dy - 0.008, 1.162), (xs + 0.012, y + dy + 0.008, 1.178))

    # ---------------------------------------------------------------- load bed: liner ribs, wheel boxes, rail caps
    y0, y1 = K['bed_y1'], K['bed_y0']
    for i in range(13):
        x = -0.72 + 0.12 * i
        box(DARK, (x - 0.032, y0 + 0.03, BED_FLOOR), (x + 0.032, y1 - 0.03, BED_FLOOR + 0.012), '-z')
    for sx in (-1, 1):
        cbox(DARK, (sx * 0.56, AXR - 0.43, BED_FLOOR - 0.02), (sx * (BED_HW + 0.01), AXR + 0.43, BED_FLOOR + 0.18), 0.05)
        cbox(DARK, (sx * (BED_HW - 0.006), y0 - 0.01, rl - 0.01), (sx * 0.872, y1 + 0.01, rl + 0.012), 0.006)
        for z in (1.03, 1.19):
            cbox(DARK, (sx * (BED_HW - 0.012), y0 + 0.08, z - 0.03), (sx * (BED_HW + 0.01), y1 - 0.08, z + 0.03), 0.008)
        for y in (y0 + 0.12, y1 - 0.12):                                                             # lashing eyes
            cyl(METAL, (sx * (BED_HW - 0.014), y, 1.11), (sx * BED_HW, y, 1.11), 0.022, 10)
    cbox(DARK, (-0.872, y1 - 0.004, rl - 0.01), (0.872, -1.08, rl + 0.012), 0.006)
    cbox(DARK, (-0.735, ty + 0.010, rl - 0.01), (0.735, y0 + 0.006, rl + 0.012), 0.006)
    for z in (1.03, 1.19):
        cbox(DARK, (-0.70, y0 - 0.01, z - 0.035), (0.70, y0 + 0.012, z + 0.035), 0.008)
        cbox(DARK, (-0.70, y1 - 0.012, z - 0.035), (0.70, y1 + 0.01, z + 0.035), 0.008)

    # ---------------------------------------------------------------- sports bar
    by = -1.26
    with smooth(60.0):
        pipe(METAL, fillet([(0.83, by, rl), (0.79, by - 0.06, 1.72), (-0.79, by - 0.06, 1.72), (-0.83, by, rl)], 0.15, 6), 0.034, 10)
        pipe(METAL, [(0.80, by - 0.05, 1.50), (-0.80, by - 0.05, 1.50)], 0.022, 8)
    for sx in (-1, 1):
        with smooth(60.0):
            cyl(METAL, (sx * 0.805, by - 0.06, 1.58), (sx * 0.83, -1.86, rl + 0.01), 0.028, 10)
        cbox(DARK, (sx * 0.78, by - 0.07, rl + 0.01), (sx * 0.875, by + 0.07, rl + 0.024), 0.004)
        cbox(DARK, (sx * 0.78, -1.93, rl + 0.01), (sx * 0.875, -1.80, rl + 0.024), 0.004)

    # ---------------------------------------------------------------- the cabin
    fpath = [(0.285, 0.66), (0.27, 0.70), (0.24, 0.775), (0.10, 0.80), (-0.10, 0.80), (-0.22, 0.79), (-0.285, 0.86), (-0.33, 1.05), (-0.37, 1.25),
             (-0.40, 1.42), (-0.415, 1.50), (-0.42, 1.53)]
    fpar = [(0.21, 0.004, 0.06), (0.24, 0.015, 0.10), (0.25, 0.03, 0.14), (0.25, 0.04, 0.14), (0.25, 0.045, 0.13), (0.25, 0.05, 0.12), (0.25, 0.06, 0.11),
            (0.25, 0.07, 0.10), (0.245, 0.06, 0.09), (0.22, 0.04, 0.08), (0.19, 0.02, 0.07), (0.13, 0.004, 0.04)]
    dz = SEAT_TOP - 0.80                                                   # the path below is drawn for a 0.80 cushion
    fpath = [(y, z + dz) for y, z in fpath]
    for xc in (-SEAT_X, SEAT_X):
        seat(xc, fpath, fpar)
        cbox(CARBON, (xc - 0.20, -0.22, FLOOR), (xc + 0.20, 0.20, 0.69 + dz), 0.02)
        ball(SEAT, (xc, -0.425, 1.60 + dz), 1.0, 16, 10, (0.13, 0.055, 0.095))
        for dx in (-0.06, 0.06): cyl(METAL, (xc + dx, -0.41, 1.46 + dz), (xc + dx, -0.425, 1.58 + dz), 0.007, 6)
    yb = cavb
    bpath = [(yb + 0.44, 0.69), (yb + 0.42, 0.765), (yb + 0.32, 0.79), (yb + 0.14, 0.78), (yb + 0.105, 0.84), (yb + 0.085, 1.05),
             (yb + 0.07, 1.30), (yb + 0.065, 1.40)]
    bpar = [(0.66, 0.01, 0.10), (0.67, 0.02, 0.13), (0.67, 0.025, 0.13), (0.67, 0.03, 0.12), (0.67, 0.03, 0.09), (0.67, 0.03, 0.05),
            (0.66, 0.025, 0.05), (0.64, 0.01, 0.045)]
    seat(0.0, bpath, bpar)
    cbox(CARBON, (-0.64, yb + 0.03, FLOOR), (0.64, yb + 0.40, 0.68), 0.02)
    for x in (-0.42, 0.0, 0.42):
        ball(SEAT, (x, yb + 0.07, 1.49 if x else 1.45), 1.0, 16, 10, (0.12, 0.05, 0.085 if x else 0.06))
    # dashboard
    hw = K['cav_hw']
    rings = []
    for i in range(15):
        x = -hw + 2 * hw * i / 14
        zt = top_z(x, 0.94) - 0.05
        zd = min(zt - 0.01, 1.20)
        prof = [(0.94, zt), (0.74, zd), (0.62, zd - 0.012), (0.565, zd - 0.05), (0.555, zd - 0.20), (0.60, zd - 0.34), (0.72, 0.78), (0.94, 0.76)]
        rings.append([(x, y, z) for y, z in prof])
    with smooth(50.0):
        loft(DASH, rings, (DASH, DASH))
    dx = -SEAT_X
    arch_o = [(dx + 0.21 * math.cos(math.pi * k / 16), 1.09 + 0.18 * math.sin(math.pi * k / 16)) for k in range(17)]
    arch_i = [(dx + 0.196 * math.cos(math.pi * k / 16), 1.09 + 0.166 * math.sin(math.pi * k / 16)) for k in range(17)]
    with smooth(50.0):
        with xf(PY):
            prism(DASH, arch_o + arch_i[::-1], 0.50, 0.72)                 # the instrument hood
            prism(CARBON, arch_i, 0.535, 0.55)                              # the panel the dials sit on
    gauge(dx - 0.085, 0.535, 1.15, 0.046, 100.0); gauge(dx + 0.085, 0.535, 1.15, 0.046, 200.0)
    for x in (-hw + 0.09, hw - 0.09): vent(x, 0.566, 1.13)
    for x in (-0.065, 0.065): vent(x, 0.566, 1.135)
    cbox(CARBON, (-0.12, 0.53, 0.92), (0.12, 0.58, 1.08), 0.006)          # the centre screen (blank)
    for i in range(3): cyl(METAL, (-0.07 + 0.07 * i, 0.60, 0.89), (-0.07 + 0.07 * i, 0.565, 0.89), 0.016, 12)
    cbox(CARBON, (0.20, 0.55, 0.95), (hw - 0.16, 0.58, 1.06), 0.005)      # trim panel in front of the passenger
    # steering column and stalks
    t = rad(STEER_TILT)
    hub = Vector(STEER_HUB); A = Vector((0.0, -math.cos(t), math.sin(t)))   # the column's axis, toward the driver
    cyl(CARBON, hub - A * 0.40, hub - A * 0.10, 0.036, 14)
    with xf(T(*STEER_HUB) @ RX(-t)):
        for sx in (-1, 1):
            obox(CARBON, (sx * 0.03, 0.07, 0.0), (sx * 0.17, 0.065, 0.02), 0.014, 0.014)
    # centre console with the selector lever and an armrest
    with xf(PX):
        prism(DASH, [(0.85, FLOOR), (0.85, 0.95), (0.58, 0.95), (0.50, 0.80), (0.30, 0.76), (-0.05, 0.76), (-0.10, 0.84), (-0.42, 0.84),
                     (-0.44, FLOOR)], -0.11, 0.11, 0.012)
    cbox(SEAT, (-0.10, -0.41, 0.835), (0.10, -0.11, 0.875), 0.012)
    cbox(CARBON, (-0.075, 0.02, 0.757), (0.075, 0.44, 0.766), 0.003)
    cyl(METAL, (0.0, 0.30, 0.76), (0.0, 0.27, 0.86), 0.011, 8)
    ball(DASH, (0.0, 0.265, 0.875), 1.0, 10, 8, (0.026, 0.034, 0.026))
    for x in (-0.04, 0.04): cyl(CARBON, (x, 0.10, 0.75), (x, 0.10, 0.768), 0.034, 14, (False, True))
    # door cards, armrests, pulls
    for y0, y1 in ((0.86, -0.09), (-0.15, -0.92)):
        card = [[Vector((side_x(lerp(y0, y1, i / 4), z) - 0.092, lerp(y0, y1, i / 4), z)) for z in (0.55, 0.75, 0.95, 1.12, 1.27)] for i in range(5)]
        for m in (IDN, MIR):
            with xf(m):
                with smooth(50.0, 'box'): slabmesh(DASH, card, (1, 0, 0), 0.0, 0.05)
                xa = card[2][2].x
                cbox(SEAT, (xa - 0.055, min(y0, y1) + 0.10, 0.99), (xa + 0.01, max(y0, y1) - 0.12, 1.04), 0.012)
                cbox(METAL, (xa - 0.012, max(y0, y1) - 0.20, 1.13), (xa + 0.004, max(y0, y1) - 0.10, 1.155), 0.004)
                cbox(CARBON, (xa - 0.008, min(y0, y1) + 0.10, 0.62), (xa + 0.004, max(y0, y1) - 0.20, 0.80), 0.006)   # map pocket
    # pedals, floor mats, inside mirror
    for x0, x1, z0 in ((-0.50, -0.42, 0.60), (-0.35, -0.29, 0.56)):
        with xf(T((x0 + x1) / 2, 0.86, z0 + 0.06) @ RX(rad(-20.0))):
            cbox(METAL, (-(x1 - x0) / 2, -0.008, -0.055), ((x1 - x0) / 2, 0.004, 0.055), 0.003)
        cyl(METAL, ((x0 + x1) / 2, 0.86, z0 + 0.07), ((x0 + x1) / 2, 0.92, z0 + 0.17), 0.007, 6)
    for sx in (-1, 1):
        box(RUBBER, (sx * 0.14, 0.24, FLOOR), (sx * 0.72, 0.84, FLOOR + 0.006), '-z')
        box(RUBBER, (sx * 0.14, cavb + 0.46, FLOOR), (sx * 0.72, -0.28, FLOOR + 0.006), '-z')
    mz = top_z(0.0, 0.46) - 0.05
    cbox(CARBON, (-0.115, 0.44, mz - 0.075), (0.115, 0.462, mz - 0.005), 0.006)
    box(METAL, (-0.10, 0.4385, mz - 0.068), (0.10, 0.44, mz - 0.012), '+y')
    cyl(CARBON, (0.0, 0.46, mz - 0.03), (0.0, 0.50, mz + 0.012), 0.012, 8)

    hull = bounds(G)
    # ---------------------------------------------------------------- door mirrors (outside the hull's bounds)
    for m in (IDN, MIR):
        with xf(m):
            p0 = S(0.86, 1.275)
            c = p0 + Vector((0.15, -0.02, 0.085))
            with smooth(60.0):
                pipe(DARK, [p0 - Vector((0.012, 0, 0)), p0 + Vector((0.05, -0.005, 0.03)), p0 + Vector((0.09, -0.012, 0.06))], 0.027, 8)
            ball(PAINT, c, 1.0, 16, 10, (0.098, 0.056, 0.074))
            ball(METAL, c + Vector((0, -0.034, 0)), 1.0, 16, 8, (0.082, 0.026, 0.060))      # the mirror glass
            cbox('s_light_amber', c + Vector((0.02, 0.046, -0.012)), c + Vector((0.085, 0.057, 0.008)), 0.003)
    return hull


# ================================================================== build, check, write
build_wheel(); WHEEL = take()
build_steering(); STEER = take()
HULL = build_body(); BODY = take()


def ntri(groups): return sum(len(v) for v in groups.values()) // 3

print('CAR body %d tris %s' % (ntri(BODY), {g: len(v) // 3 for g, v in sorted(BODY.items())}))
print('CAR wheel %d tris %s' % (ntri(WHEEL), {g: len(v) // 3 for g, v in sorted(WHEEL.items())}))
print('CAR steering %d tris %s' % (ntri(STEER), {g: len(v) // 3 for g, v in sorted(STEER.items())}))
wlo, whi = bounds(WHEEL)
for i in range(3): assert abs(wlo[i] + whi[i]) * 0.5 < 0.005, 'wheel not centred on the origin'
wr = max(math.hypot(p.y, p.z) for v in WHEEL.values() for p, _, _ in v)
assert abs(wr - WR) < 0.005, 'wheel radius %.4f' % wr
assert abs((whi.x - wlo.x) - WW) < 0.005, 'tyre width %.4f' % (whi.x - wlo.x)
slo, shi = bounds(STEER)
for i in (0, 2): assert abs(slo[i] + shi[i]) * 0.5 < 0.005, 'steering wheel not centred'
blo, bhi = HULL                                          # everything but the door mirrors
flo, fhi = bounds(BODY)
L_, W_, H_ = bhi.y - blo.y, bhi.x - blo.x, bhi.z
print('CAR headroom over the driver: headliner %.3f, cushion top %.3f' % (top_z(-SEAT_X, HIP[1]) - 0.03, SEAT_TOP))
print('CAR size L %.3f W %.3f H %.3f, clearance %.3f, full bounds %s %s' % (L_, W_, H_, blo.z, tuple(round(a, 3) for a in flo), tuple(round(a, 3) for a in fhi)))
assert abs(L_ - 5.35) < 0.06, 'length %.3f' % L_
assert abs(W_ - 1.92) < 0.03, 'width %.3f' % W_
assert abs(H_ - 1.85) < 0.02, 'height %.3f' % H_
assert 0.22 <= blo.z <= 0.26, 'ground clearance %.3f' % blo.z
NEED = {'body': ['s_paint', 's_paint_dark', 's_carbon', 's_glass', 's_seat', 's_dash', 's_metal', 's_rubber', 's_light', 's_light_red',
                 's_light_amber', 's_gauge'],
        'wheel': ['s_tyre', 's_rim', 's_brake', 's_caliper', 's_metal'], 'steering': ['s_dash', 's_carbon', 's_metal']}
for nm, grp in (('body', BODY), ('wheel', WHEEL), ('steering', STEER)):
    for g in NEED[nm]: assert len(grp.get(g, [])) >= 3, '%s: group %s is empty' % (nm, g)
    for g, v in grp.items(): assert len(v) >= 3 and g in GROUPS
    write_mesh(nm, grp)
assert ntri(BODY) <= 62000, 'body triangles %d' % ntri(BODY)
assert ntri(WHEEL) <= 8000, 'wheel triangles %d' % ntri(WHEEL)
assert ntri(STEER) <= 3000, 'steering triangles %d' % ntri(STEER)

t_ = rad(STEER_TILT)
HEAD = K['head'][4][len(K['head'][0]) // 2]
TAIL = K['tail'][1][5]
RL = K['rail']


def cbx(lo, hi):
    return {'center': E(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)),
            'half_extents': [round((hi[0] - lo[0]) / 2, 4), round((hi[2] - lo[2]) / 2, 4), round((hi[1] - lo[1]) / 2, 4)]}

HW_ = W_ / 2                                             # half width over the wheel arch mouldings
INFO = {
    'comment': 'engine frame: +y up, front = -z, +x = right; metres. body.mesh origin = on the ground under the middle of the wheelbase',
    'wheel_radius': WR, 'wheel_width': WW, 'wheelbase': WB, 'track': TRACK,
    'wheel_mesh_axle_axis': [1, 0, 0],
    'wheel_mesh_outer_side': '+x (draw as is on the right, rotated 180 deg about y on the left)',
    'wheels': {'front_left': E((-WX, AXF, WR)), 'front_right': E((WX, AXF, WR)), 'rear_left': E((-WX, AXR, WR)), 'rear_right': E((WX, AXR, WR))},
    'seat_driver': E(EYE_L), 'seat_passenger': E(EYE_R),
    'hip_driver': E(HIP), 'legs': LEGS,
    'steering_hub': E(STEER_HUB),
    'steering_axis': [0.0, round(math.sin(t_), 5), round(math.cos(t_), 5)],
    'steering_tilt_deg': STEER_TILT,
    'steering_mesh_axis': [0, 0, 1],
    'steering_placement': 'spin about mesh +z, then rotate about +x by -steering_tilt_deg, then translate to steering_hub',
    'headlights': [E((-HEAD.x, HEAD.y + 0.02, HEAD.z)), E((HEAD.x, HEAD.y + 0.02, HEAD.z))],
    'tail_lights': [E((-TAIL.x, TAIL.y - 0.02, TAIL.z)), E((TAIL.x, TAIL.y - 0.02, TAIL.z))],
    'exit_left': E((-(HW_ + 1.0), -0.05, 0.0)), 'exit_right': E((HW_ + 1.0, -0.05, 0.0)),
    'bounds_min': E((flo.x, fhi.y, flo.z)), 'bounds_max': E((fhi.x, flo.y, fhi.z)),
    'bounds_note': 'bounds are of the whole body.mesh, door mirrors included; without the mirrors it is %.3f wide' % W_,
    'collision_boxes': [
        dict(name='bonnet', **cbx((-HW_, 1.00, 0.42), (HW_, bhi.y - 0.02, 1.19))),
        dict(name='cab', **cbx((-HW_, -1.05, 0.36), (HW_, 1.00, 1.28))),
        dict(name='windscreen', **cbx((-0.76, 0.36, 1.28), (0.76, 0.72, 1.50))),
        dict(name='roof', **cbx((-0.80, K['cabb'], 1.28), (0.80, 0.36, bhi.z))),
        dict(name='bed', **cbx((-HW_, blo.y, 0.47), (HW_, -1.05, RL + 0.01))),
    ],
}
with open(os.path.join(OUT, 'car.json'), 'w') as f:
    json.dump(INFO, f, indent=2)
    f.write('\n')
print('CAR written to', OUT)


# ================================================================== preview renders (PICKUP_PREVIEW=1)
def preview():
    COLS = {'s_paint': (0.10, 0.22, 0.42), 's_paint_dark': (0.035, 0.035, 0.04), 's_carbon': (0.10, 0.10, 0.11), 's_glass': (0.55, 0.75, 0.85),
            's_rim': (0.55, 0.56, 0.58), 's_tyre': (0.03, 0.03, 0.03), 's_brake': (0.42, 0.41, 0.40), 's_caliper': (0.75, 0.10, 0.08),
            's_seat': (0.45, 0.38, 0.30), 's_dash': (0.06, 0.06, 0.065), 's_metal': (0.75, 0.75, 0.74), 's_rubber': (0.015, 0.015, 0.015),
            's_light': (1.0, 0.98, 0.88), 's_light_red': (0.85, 0.04, 0.03), 's_light_amber': (1.0, 0.55, 0.05), 's_gauge': (0.92, 0.90, 0.84)}
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    mats = {}
    for g, c in COLS.items():
        m = bpy.data.materials.new('pv_' + g)
        m.diffuse_color = (c[0], c[1], c[2], 0.30 if g == 's_glass' else 1.0)
        m.roughness = 0.25 if g in ('s_paint', 's_glass', 's_rim', 's_metal') else 0.7
        m.metallic = 0.6 if g in ('s_rim', 's_metal') else 0.0
        mats[g] = m
    glass_obs = []

    def add(name, groups, mat):
        for g, v in groups.items():
            me = bpy.data.meshes.new(name + '_' + g)
            me.from_pydata([tuple(p) for p, _, _ in v], [], [(i, i + 1, i + 2) for i in range(0, len(v), 3)])
            me.update()
            for p in me.polygons: p.use_smooth = True
            try: me.normals_split_custom_set_from_vertices([tuple(n) for _, n, _ in v])
            except Exception as e: print('CAR preview: no custom normals', e)
            me.materials.append(mats[g])
            ob = bpy.data.objects.new(name + '_' + g, me)
            ob.matrix_world = mat
            sc.collection.objects.link(ob)
            if g == 's_glass': glass_obs.append(ob)

    add('body', BODY, IDN)
    add('steer', STEER, T(*STEER_HUB) @ RX(-t_))
    for sx, y in ((1, AXF), (1, AXR), (-1, AXF), (-1, AXR)):
        add('wheel', WHEEL, T(sx * WX, y, WR) @ (IDN if sx > 0 else RZ(math.pi)))
    gm = bpy.data.meshes.new('ground')
    gm.from_pydata([(-14, -14, 0), (14, -14, 0), (14, 14, 0), (-14, 14, 0)], [], [(0, 1, 2, 3)])
    gmat = bpy.data.materials.new('ground'); gmat.diffuse_color = (0.55, 0.55, 0.52, 1.0); gm.materials.append(gmat)
    ground = bpy.data.objects.new('ground', gm); sc.collection.objects.link(ground)

    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.image_settings.file_format = 'PNG'
    sh = sc.display.shading
    sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'
    sh.show_backface_culling = True
    sh.show_cavity = True
    sh.show_shadows = False
    sh.show_specular_highlight = True
    sc.world = bpy.data.worlds.new('w'); sc.world.color = (0.75, 0.80, 0.86)
    cam = bpy.data.cameras.new('cam'); cob = bpy.data.objects.new('cam', cam)
    sc.collection.objects.link(cob); sc.camera = cob
    # name, camera, look-at, lens, hide ground, near clip, hide glass
    views = [('front34', (3.9, 6.0, 2.2), (0.0, 0.2, 0.85), 40.0, False, 0.03, False),
             ('rear34', (-4.0, -6.2, 2.5), (0.0, -0.3, 0.85), 40.0, False, 0.03, False),
             ('side', (9.0, 0.0, 1.1), (0.0, -0.1, 0.92), 50.0, False, 0.03, False),
             ('top', (0.0, -0.1, 11.5), (0.0, -0.1, 0.5), 50.0, False, 0.03, False),
             ('interior', (-3.4, -0.25, 1.45), (0.0, -0.05, 1.05), 32.0, False, 2.66, True),      # the left side cut away
             ('front', (0.6, 7.0, 1.1), (0.0, 1.0, 0.85), 50.0, False, 0.03, False),
             ('rear', (-0.6, -7.2, 1.3), (0.0, -1.0, 0.85), 50.0, False, 0.03, False),
             ('wheel', (2.2, 2.9, 0.7), (0.85, 1.6, 0.45), 50.0, False, 0.03, False),
             ('bed', (-1.6, -4.6, 3.0), (0.0, -1.6, 1.0), 35.0, False, 0.03, False),
             ('driver', EYE_L, (-0.25, 3.0, 1.15), 14.0, False, 0.03, True),
             ('cabin', (0.0, 0.2, 3.4), (0.0, -0.05, 0.9), 22.0, False, 1.62, True),             # the roof cut away
             ('close', (2.3, 3.6, 1.55), (0.5, 1.2, 0.95), 35.0, False, 0.03, False),
             ('closerear', (-2.3, -4.3, 1.75), (-0.3, -1.9, 1.0), 35.0, False, 0.03, False),
             ('under', (2.9, 3.2, -2.4), (0.0, 0.0, 0.5), 35.0, True, 0.03, False)]
    only = os.environ.get('PICKUP_VIEWS')
    for name, pos, look, lens, hide_ground, near, no_glass in views:
        if only and name not in only.split(','): continue
        cob.location = Vector(pos)
        d = Vector(look) - Vector(pos)
        cob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        if name == 'top': cob.rotation_euler = (0.0, 0.0, math.radians(90.0))      # front to the right
        cam.lens = lens
        cam.clip_start = near; cam.clip_end = 200.0
        ground.hide_render = hide_ground
        for ob in glass_obs: ob.hide_render = no_glass
        sc.render.filepath = os.path.join(OUT, 'preview_%s.png' % name)
        bpy.ops.render.render(write_still=True)
        print('CAR preview', sc.render.filepath)


if os.environ.get('PICKUP_PREVIEW') == '1':
    preview()
