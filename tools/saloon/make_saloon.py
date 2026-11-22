# Builds the saloon - an original modern four-door executive saloon - in Blender and exports it
# for the game (same pipeline and file formats as tools/jeep/make_jeep.py):
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --python tools/saloon/make_saloon.py
# (set SALOON_PREVIEW=1 to also render assets/saloon/preview_*.png with back-face culling on;
#  SALOON_VIEWS=side,top limits the views)
# Output (assets/saloon/):
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
# cuts (cabin, wheel housings, window openings, light pockets, panel gaps ...) -> triangles with
# normals smoothed below a crease angle. The glass is laid back into the window openings on the
# uncut surface, so it sits flush. Everything else is put together from primitives.
# No badges, logos or lettering anywhere.
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
OUT = os.path.join(ROOT, 'assets', 'saloon')
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
WR, WW = 0.34, 0.24            # wheel radius, tyre width
WB, TRACK = 2.90, 1.60
AXF, AXR = WB / 2, -WB / 2     # axle lines (Y)
WX = TRACK / 2
HALFW = 0.93                   # half the body's width
Y_NOSE, Y_TAIL = 2.35, -2.50   # 4.85 m long
Z_UNDER, Z_ROOF = 0.15, 1.45
ARCH = 0.39                    # wheel housing radius
WALL = 0.045                   # thickness of the body shell round the cabin
FLOOR = Z_UNDER + WALL
SEAT_X = 0.37
SEAT_TOP = 0.38                # top of the seat cushion
HIP = (-SEAT_X, -0.03, SEAT_TOP + 0.10)      # "hip_driver": the seated driver's pelvis joint
EYE_L = (-SEAT_X, HIP[1], HIP[2] + 0.52)     # "seat_driver" (as in the other cars: 0.52 above the hip)
EYE_R = (SEAT_X, HIP[1], HIP[2] + 0.52)
LEGS = 0.7                     # 0 = upright chair, 1 = low seat with the legs stretched forward
STEER_TILT = 24.0              # the column rises toward the driver at this angle from the horizontal
STEER_HUB = (-SEAT_X, 0.40, 0.85)
PAINT, DARK, CARBON, METAL, RUBBER, GLASS, DASH, SEAT = 's_paint', 's_paint_dark', 's_carbon', 's_metal', 's_rubber', 's_glass', 's_dash', 's_seat'
GROUPS = ['s_paint', 's_paint_dark', 's_carbon', 's_glass', 's_rim', 's_tyre', 's_brake', 's_caliper', 's_seat', 's_dash', 's_metal',
          's_rubber', 's_light', 's_light_red', 's_light_amber', 's_gauge']
GI = {g: i for i, g in enumerate(GROUPS)}
MAT = {g: bpy.data.materials.new(g) for g in GROUPS}
rad = math.radians



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
    if os.environ.get("SALOON_DEBUG") == "1":
        for k, c in enumerate(CUTS):
            m = body.modifiers.new("cut", "BOOLEAN")
            m.operation = "DIFFERENCE"; m.object = c; m.solver = "EXACT"
            body = bake(body, name)
            xs = [v.co for v in body.data.vertices]
            print("CAR cut %d (%d verts in cutter, first %s): body %d polys, y %.2f..%.2f z %.2f..%.2f" % (k, len(c.data.vertices), tuple(round(a, 2) for a in c.data.vertices[0].co), len(body.data.polygons), min(p.y for p in xs), max(p.y for p in xs), min(p.z for p in xs), max(p.z for p in xs)))
    for c in ([] if os.environ.get("SALOON_DEBUG") == "1" else CUTS):
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


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


# ================================================================== the wheel
def build_wheel():
    """In the frame of wheel.mesh: axle = X, outer face toward +X."""
    k = WW / 0.28
    with xf(PX @ Matrix.Diagonal((1.0, 1.0, k, 1.0))):  # lathe axis (local z) -> X
        half = [(0.262, -0.128), (0.287, -0.139), (0.312, -0.140), (0.325, -0.134), (0.333, -0.120), (0.3355, -0.100), (0.340, -0.098),
                (0.340, -0.092), (0.3335, -0.090), (0.3335, -0.078), (0.340, -0.076),
                (0.340, -0.044), (0.3335, -0.042), (0.3335, -0.030), (0.340, -0.028), (0.340, -0.006), (0.3335, -0.004)]
        tyre = half + [(r, -x) for r, x in reversed(half)]
        lathe('s_tyre', tyre, 48, uref=WR, sharp=0.80)
        # shoulder blocks between short cross grooves, on both shoulders
        N = 40
        for s in (1, -1):
            for i in range(N):
                a0 = TAU * (i + (0.5 if s < 0 else 0.0)) / N; da = TAU / N * 0.30
                def P(a, r, x): return Vector((r * math.cos(a), r * math.sin(a), s * x))
                A = [P(a0 - da, 0.340, 0.0985), P(a0 + da, 0.340, 0.0985), P(a0 + da, 0.3375, 0.119), P(a0 - da, 0.3375, 0.119)]
                Bq = [P(a0 - da, 0.3355, 0.0985), P(a0 + da, 0.3355, 0.0985), P(a0 + da, 0.3332, 0.119), P(a0 - da, 0.3332, 0.119)]
                C = [P(a0 - da, 0.3285, 0.1335), P(a0 + da, 0.3285, 0.1335)]
                ref = P(a0, 0.30, 0.10)
                u0, u1 = (a0 - da) * WR, (a0 + da) * WR
                quad_out('s_tyre', A[0], A[1], A[2], A[3], ref, [(u0, s * 0.097), (u1, s * 0.097), (u1, s * 0.118), (u0, s * 0.118)])
                quad_out('s_tyre', A[3], A[2], C[1], C[0], ref, [(u0, s * 0.118), (u1, s * 0.118), (u1, s * 0.131), (u0, s * 0.131)])
                for i0, i1 in ((0, 3), (1, 2)):
                    quad_out('s_tyre', A[i0], A[i1], Bq[i1], Bq[i0], P(a0, 0.3385, 0.11))
        # the rim: outer flange, barrel, inner flange
        lathe('s_rim', [(0.262, 0.118), (0.266, 0.128), (0.259, 0.132), (0.247, 0.128), (0.242, 0.110), (0.236, -0.110),
                        (0.247, -0.128), (0.259, -0.132), (0.266, -0.128), (0.262, -0.118)], 48)
        # ten slim spokes in five pairs that open toward the rim
        SP = [(0.050, 0.020, 0.080, 0.036), (0.085, 0.0155, 0.083, 0.032), (0.125, 0.0125, 0.089, 0.028), (0.165, 0.0115, 0.097, 0.026),
              (0.205, 0.0125, 0.106, 0.026), (0.232, 0.017, 0.113, 0.028), (0.250, 0.027, 0.118, 0.030)]
        with smooth(50.0):
            for i in range(5):
                for da in (-12.5, 12.5):
                    rings = []
                    for r, hw, zf, th in SP:
                        a = TAU * (i + 0.25) / 5 + rad(da) * (0.22 + 0.78 * (r - 0.05) / 0.20)
                        er = Vector((math.cos(a), math.sin(a), 0)); et = Vector((-math.sin(a), math.cos(a), 0)); ez = Vector((0, 0, 1))
                        c = er * r
                        rings.append([c - et * hw + ez * (zf - 0.010), c - et * hw * 0.5 + ez * zf, c + et * hw * 0.5 + ez * zf,
                                      c + et * hw + ez * (zf - 0.010), c + et * hw * 0.8 + ez * (zf - th), c - et * hw * 0.8 + ez * (zf - th)])
                    loft('s_rim', rings)
        # hub, centre cap (blank), wheel nuts
        lathe('s_rim', [(0.070, 0.046), (0.070, 0.082), (0.062, 0.091), (0.030, 0.095), (0.0, 0.0955)], 30)
        lathe('s_metal', [(0.030, 0.095), (0.027, 0.099), (0.0, 0.100)], 20)
        for i in range(5):
            a = TAU * (i + 0.75) / 5
            c = Vector((0.047 * math.cos(a), 0.047 * math.sin(a), 0.090))
            cyl('s_metal', c, c + Vector((0, 0, 0.012)), 0.0085, 6, (False, True))
        # brake: disc with its bell, caliper on top
        lathe('s_brake', [(0.085, 0.008), (0.205, 0.008), (0.205, 0.038), (0.085, 0.038)], 40)
        lathe('s_metal', [(0.0, -0.085), (0.050, -0.085), (0.050, -0.020), (0.092, -0.020), (0.092, 0.046), (0.070, 0.046)], 24)
        arc = [(r * math.cos(rad(a)), r * math.sin(rad(a))) for r, aa in ((0.228, range(58, 123, 8)), (0.130, range(122, 57, -8))) for a in aa]
        prism('s_caliper', arc, -0.014, 0.056, 0.007)
        for a in (72.0, 90.0, 108.0):                    # the pistons' bosses
            c = Vector((0.178 * math.cos(rad(a)), 0.178 * math.sin(rad(a)), 0.056))
            cyl('s_caliper', c, c + Vector((0, 0, 0.004)), 0.017, 10, (False, True))


# ================================================================== the steering wheel
def build_steering():
    """In the frame of steering.mesh after RX(90): axis = Blender -Y = engine +Z."""
    with xf(RX(rad(90.0))):                             # local z (toward the driver) -> Blender -Y
        R = 0.183
        with smooth(60.0):
            pipe('s_dash', [(R * math.cos(TAU * i / 44), R * math.sin(TAU * i / 44), 0.0) for i in range(44)], 0.0165, 10, closed=True)
        for a, w in ((4.0, 0.046), (176.0, 0.046), (270.0, 0.036)):
            c, s = math.cos(rad(a)), math.sin(rad(a))
            obox('s_carbon', (0.045 * c, 0.045 * s - 0.006, -0.026), (0.176 * c, 0.176 * s, -0.003), w, 0.014, (0, 0, 1), 0.004)
        with smooth(50.0):
            lathe('s_dash', [(0.072, -0.052), (0.074, -0.028), (0.066, -0.014), (0.036, -0.008), (0.0, -0.007)], 28)
        lathe('s_carbon', [(0.036, -0.110), (0.042, -0.052), (0.072, -0.052)], 20)
        lathe('s_metal', [(0.076, -0.030), (0.0765, -0.026), (0.074, -0.0255)], 28)        # a bright ring round the pad, no badge
        for sx in (-1, 1):                                                                  # thumb switches
            cbox('s_metal', (sx * 0.085, -0.004, -0.014), (sx * 0.125, 0.016, -0.004), 0.003)


# ================================================================== the body shell
# cross-sections, nose to tail:
#   y, half width, underside, sill top, widest (low), widest (high), shoulder, shoulder inset,
#   the upper points 7, 8, 9: three heights on a deck (bonnet, boot) or three (x, z) of the greenhouse,
#   top centre, sweep of the whole section, sweep of its upper points only
ST = [( 2.375, 0.36,  0.23,  0.27, 0.35, 0.50, 0.600, 0.03,  (0.625, 0.635, 0.64),                    0.64,  0.02, 0.0),
      ( 2.355, 0.62,  0.21,  0.26, 0.35, 0.52, 0.630, 0.04,  (0.66, 0.675, 0.68),                     0.68,  0.10, 0.0),
      ( 2.28,  0.81,  0.18,  0.25, 0.36, 0.55, 0.675, 0.05,  (0.705, 0.72, 0.725),                    0.725, 0.12, 0.0),
      ( 2.06,  0.905, 0.155, 0.24, 0.38, 0.62, 0.735, 0.06,  (0.765, 0.785, 0.79),                    0.79,  0.05, 0.0),
      ( 1.76,  0.925, 0.15,  0.24, 0.42, 0.70, 0.795, 0.07,  (0.825, 0.845, 0.85),                    0.85,  0.0,  0.0),
      ( 1.45,  0.93,  0.15,  0.24, 0.44, 0.74, 0.840, 0.07,  (0.868, 0.888, 0.895),                   0.895, 0.0,  0.0),
      ( 1.16,  0.93,  0.15,  0.24, 0.44, 0.76, 0.880, 0.065, (0.905, 0.928, 0.935),                   0.935, 0.0,  0.02),
      ( 1.00,  0.928, 0.15,  0.24, 0.44, 0.77, 0.905, 0.06,  (0.925, 0.95, 0.96),                     0.96,  0.0,  0.06),
      ( 0.93,  0.927, 0.15,  0.24, 0.44, 0.77, 0.912, 0.055, ((0.815, 0.94), (0.74, 0.962), (0.37, 0.985)),  0.99,  0.0,  0.10),
      ( 0.56,  0.925, 0.15,  0.24, 0.44, 0.77, 0.922, 0.05,  ((0.83, 1.04), (0.75, 1.15), (0.42, 1.19)),  1.20,  0.0,  0.20),
      ( 0.15,  0.93,  0.15,  0.24, 0.44, 0.77, 0.932, 0.05,  ((0.82, 1.17), (0.725, 1.355), (0.42, 1.397)),   1.405, 0.0,  0.14),
      (-0.30,  0.93,  0.15,  0.24, 0.44, 0.77, 0.942, 0.05,  ((0.815, 1.20), (0.715, 1.395), (0.42, 1.442)),    1.45,  0.0,  0.04),
      (-0.80,  0.93,  0.15,  0.24, 0.44, 0.77, 0.952, 0.05,  ((0.815, 1.205), (0.715, 1.39), (0.42, 1.437)), 1.445, 0.0,  0.0),
      (-1.25,  0.93,  0.15,  0.24, 0.44, 0.77, 0.962, 0.05,  ((0.81, 1.18), (0.715, 1.335), (0.42, 1.377)), 1.385, 0.0, -0.10),
      (-1.62,  0.93,  0.15,  0.24, 0.44, 0.77, 0.970, 0.05,  ((0.81, 1.08), (0.725, 1.17), (0.42, 1.212)), 1.22,  0.0, -0.17),
      (-1.95,  0.925, 0.15,  0.24, 0.44, 0.77, 0.975, 0.06,  ((0.79, 1.0), (0.70, 1.035), (0.36, 1.075)),    1.085, 0.0, -0.10),
      (-2.03,  0.915, 0.17,  0.25, 0.44, 0.77, 0.972, 0.06,  (0.995, 1.03, 1.055),                    1.06,  0.0, -0.06),
      (-2.25,  0.90,  0.22,  0.29, 0.46, 0.77, 0.955, 0.06,  (0.985, 1.02, 1.045),                    1.05,  0.0,  0.0),
      (-2.42,  0.86,  0.27,  0.34, 0.50, 0.76, 0.940, 0.06,  (0.975, 1.015, 1.04),                    1.045, 0.04, 0.0),
      (-2.50,  0.70,  0.30,  0.37, 0.52, 0.76, 0.930, 0.05,  (0.97, 1.01, 1.03),                      1.035, 0.10, 0.0),
      (-2.525, 0.42,  0.32,  0.38, 0.52, 0.76, 0.920, 0.04,  (0.955, 0.99, 1.0),                      1.0,   0.03, 0.0)]


def belt(y):
    """Height of the shoulder line (the cage's, near enough the finished one)."""
    return 0.912 + (0.93 - max(-1.95, min(0.93, y))) * 0.0219


def make_shell():
    verts = []; faces = []; fm = []
    NR = 20
    for (y, w, zb, zk, zlo, zhi, zs, ins, top, zc, sw, su) in ST:
        if isinstance(top[0], tuple): p7, p8, p9 = top
        else: p7, p8, p9 = (0.87 * w, top[0]), (0.72 * w, top[1]), (0.37 * w, top[2])
        half = [(0.0, zb), (0.55 * w, zb), (w - 0.07, zb), (w - 0.015, zk), (w, zlo), (w - 0.005, zhi), (w - ins, zs), p7, p8, p9, (0.0, zc)]
        def Y(j):
            f = (half[j][0] / w) ** 2
            return y - sw * f - su * f * (1.0 if j >= 7 else (0.5 if j == 6 else 0.0))
        verts += [(half[j][0], Y(j), half[j][1]) for j in range(11)] + [(-half[j][0], Y(j), half[j][1]) for j in range(9, 0, -1)]
    def gm(k):
        kk = k if k < 10 else 19 - k
        return DARK if kk <= 2 else PAINT
    for i in range(len(ST) - 1):
        for k in range(NR):
            a, b = i * NR + k, i * NR + (k + 1) % NR
            faces.append((a, b, b + NR, a + NR)); fm.append(gm(k))
    for base in (0, (len(ST) - 1) * NR):                 # the end faces, as strips from side to side
        for j in range(1, 9):
            faces.append((base + j, base + j + 1, base + 20 - (j + 1), base + 20 - j)); fm.append(DARK if j <= 2 else PAINT)
        faces.append((base, base + 1, base + 19)); fm.append(DARK)
        faces.append((base + 10, base + 11, base + 9)); fm.append(PAINT)
    ob = new_obj('shell', verts, faces, fm)
    m = ob.modifiers.new('ss', 'SUBSURF'); m.levels = 2; m.render_levels = 2
    ob = bake(ob, 'shell1')
    me = ob.data
    cs = [v.co for v in me.vertices]
    wmax = max(abs(c.x) for c in cs)
    y0, y1 = min(c.y for c in cs), max(c.y for c in cs)
    z0, z1 = min(c.z for c in cs), max(c.z for c in cs)
    for v in me.vertices:
        v.co.x *= HALFW / wmax
        v.co.y = Y_TAIL + (v.co.y - y0) * (Y_NOSE - Y_TAIL) / (y1 - y0)
        v.co.z = Z_UNDER + (v.co.z - z0) * (Z_ROOF - Z_UNDER) / (z1 - z0)
    me.update()
    return ob


def tris_of(ob):
    me = ob.data
    me.calc_loop_triangles()
    for lt in me.loop_triangles:
        yield GROUPS[lt.material_index], [me.vertices[i].co.copy() for i in lt.vertices]


LAMPS = {}
RIDGE = {}


def ridge(y):
    """Where the body side rolls over into the roof / the screens at station y: (x, z)."""
    key = round(y, 3)
    if key in RIDGE: return RIDGE[key]
    def probe(z):
        loc, n, _, _ = BVH.ray_cast(Vector((3.0, y, z)), Vector((-1.0, 0.0, 0.0)))
        if loc is None: return None, True
        return loc.x, n.z > 1.4 * n.x
    z = belt(y) + 0.015
    x, over = probe(z)
    assert x is not None, 'no body side at y %.3f' % y
    best = (x, z)
    while not over and z < 1.6:
        z2 = z + 0.01
        x2, over = probe(z2)
        if not over: z = z2; best = (x2, z2)
    lo, hi = z, z + 0.01
    for _ in range(6):
        mid = (lo + hi) / 2
        x2, ov = probe(mid)
        if ov: hi = mid
        else: lo = mid; best = (x2, mid)
    RIDGE[key] = best
    return best


def P_side(y, z, out=0.0): return Vector((side_x(y, z) + out, y, z))
def P_top(x, y, out=0.0): return Vector((x, y, top_z(x, y) + out))
def P_tail(x, z, out=0.0): return Vector((x, tail_y(x, z) - out, z))
def P_nose(x, z, out=0.0): return Vector((x, nose_y(x, z) + out, z))


def win_bot(y): return belt(y) + 0.03
def win_top(y, y_end=None):
    zt = ridge(y)[1] - 0.042
    if y_end is not None: zt = min(zt, win_bot(y) + (y - y_end) * 1.6)
    return zt


def side_window(ys, nz, y_end=None):
    grid = []
    for y in ys:
        zb_ = win_bot(y); zt = max(win_top(y, y_end), zb_ + 0.03)
        grid.append([ray((3.0, y, lerp(zb_, zt, j / (nz - 1))), (-1, 0, 0)) for j in range(nz)])
    return grid


def screen(YB, cb, YT, ct, ns, nt, margin):
    """A screen's grid [across][up the glass]: base / top edges curved in plan, sides a margin inside the pillars' ridge."""
    grid = []
    for i in range(ns + 1):
        s = -1.0 + 2.0 * i / ns; row = []
        for j in range(nt + 1):
            y = lerp(YB - cb * s * s, YT - ct * s * s, j / nt)
            x = s * (ridge(y)[0] - margin)
            row.append(ray((x, y, 3.0), (0, 0, -1)))
        grid.append(row)
    return grid


def ring_cutter(rings, gfn, gcap=CARBON):
    n = len(rings[0]); verts = [p for r in rings for p in r]; faces = []; fm = []
    for i in range(len(rings) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
            fm.append(gfn(sum(verts[q][2] for q in (a, b, b + n, a + n)) / 4.0))
    faces.append(tuple(range(n))); fm.append(gcap)
    faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n))); fm.append(gcap)
    cutter(verts, faces, fm)


def inner_ring(y, cz, angles):
    """The body's section at y, seen from (0, y, cz), moved WALL inward."""
    pts = []
    for a in angles:
        d = Vector((math.cos(a), 0.0, math.sin(a)))
        loc = None
        for dy in (0.0, 0.0031, -0.0031):
            loc, n, _, _ = BVH.ray_cast(Vector((0.0, y + dy, cz)), d)
            if loc is not None: break
        assert loc is not None, 'cabin ray misses at y %.2f angle %.2f' % (y, a)
        if n.dot(d) < 0.0: n = -n
        npv = Vector((n.x, 0.0, n.z)); l = npv.length
        if l < 0.35: p = loc - d * WALL
        else: p = loc - npv / l * min(WALL / l, 0.11)
        pts.append(Vector((p.x, y, p.z)))
    return pts


def box_cutter(lo, hi, g):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    F = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    cutter(V, F, g)


def build_shell():
    """The subdivided hull with every opening cut into it; fills BVH (the uncut surface)."""
    global BVH
    body = make_shell()
    BVH = bvh_of(body.data)
    for y in (2.0, 1.6, 1.2, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0, -0.3, -0.8, -1.2, -1.4, -1.6, -1.8, -1.9, -2.0, -2.2):
        r = ridge(y) if -2.0 < y < 0.95 else (0, 0)
        print('CAR probe y %.2f: top centre %.3f, top at x 0.5 %.3f, half width at z 0.6 %.3f, at belt %.3f, ridge x %.3f z %.3f'
              % (y, top_z(0.0, y), top_z(0.5, y), side_x(y, 0.6), side_x(y, min(belt(y), top_z(0.0, y) - 0.08)), r[0], r[1]))

    # ---------------------------------------------------------------- the cabin: three overlapping hollows
    def gcab(z): return RUBBER if z < 0.24 else (DASH if z < 0.93 else CARBON)
    N = 72
    ang = [TAU * (k + 0.5) / N for k in range(N)]
    ring_cutter([inner_ring(y, 0.60, ang) for y in (0.90, 0.80, 0.62, 0.42, 0.22, 0.0, -0.25, -0.50, -0.75, -0.95, -1.03)], gcab)
    box_cutter((-0.49, -1.46, FLOOR), (0.49, -1.00, 0.82), CARBON)                  # between the rear wheel housings
    ang = [0.02] + [math.pi * k / 27 for k in range(1, 27)] + [math.pi - 0.02]
    ring_cutter([inner_ring(y, zf, ang) for y, zf in ((-1.00, 0.80), (-1.20, 0.80), (-1.41, 0.80), (-1.47, 0.99), (-1.62, 0.99),
                                                      (-1.74, 0.99), (-1.82, 0.99))], lambda z: CARBON)
    # ---------------------------------------------------------------- wheel housings
    for yc in (AXF, AXR):
        n = 48; V = []
        for x in (0.53, 1.3):
            V += [(x, yc + ARCH * math.cos(TAU * i / n), WR + ARCH * math.sin(TAU * i / n)) for i in range(n)]
        F = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n)), tuple(range(n, 2 * n))]
        cutter(V, F, DARK, True)
    # ---------------------------------------------------------------- window openings
    y = 0.95
    while win_top(y) - win_bot(y) < 0.035: y -= 0.01
    gf = side_window([lerp(y, -0.385, i / 15) for i in range(16)], 7)
    y = -1.20
    while win_top(y - 0.01, -1.50) - win_bot(y - 0.01) > 0.035: y -= 0.01
    gr = side_window([lerp(-0.485, y, i / 12) for i in range(13)], 7, -1.50)
    for g2 in (gf, gr): slab(g2, (1, 0, 0), 0.05, 0.12, RUBBER, RUBBER, True)
    gw = screen(0.875, 0.11, 0.155, 0.10, 16, 8, 0.06)
    gb = screen(-1.90, -0.10, -1.27, -0.09, 14, 7, 0.075)
    for g2 in (gw, gb): slab(g2, (0, 0, 1), 0.05, 0.12, RUBBER, RUBBER)
    LAMPS.update(gf=gf, gr=gr, gw=gw, gb=gb)
    print('CAR side glass: front y %.2f..%.2f, rear ..%.2f; height at the driver %.3f' % (gf[0][0].y, gf[-1][0].y, gr[-1][0].y, gf[10][-1].z - gf[10][0].z))
    # ---------------------------------------------------------------- the nose: slim lamps, grille, intakes
    def hl(u, v):
        t = (u + 1) / 2
        h = 0.014 + 0.021 * math.sin(math.pi * min(1.0, t * 1.12)) ** 0.8
        return (lerp(0.37, 0.855, t), 4.0, 0.612 + 0.050 * t ** 1.4 + v * h)
    grid = cast_grid(hl, 15, 5, (0, -1, 0))
    slab(grid, (0, 1, 0), 0.05, 0.055, DARK, DARK, True)
    LAMPS['head'] = grid
    g2 = cast_grid(bilin((-0.30, 0.445), (0.30, 0.445), (0.335, 0.59), (-0.335, 0.59), (1, 4.0)), 9, 4, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.05, 0.06, DARK, DARK)
    LAMPS['grille'] = g2
    g2 = cast_grid(bilin((-0.50, 0.25), (0.50, 0.25), (0.56, 0.365), (-0.56, 0.365), (1, 4.0)), 9, 4, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.05, 0.07, DARK, DARK)
    LAMPS['intake'] = g2
    g2 = cast_grid(bilin((0.66, 0.27), (0.80, 0.30), (0.80, 0.44), (0.68, 0.43), (1, 4.0)), 4, 5, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.05, 0.05, DARK, DARK, True)
    # ---------------------------------------------------------------- the tail: light bar, plate recess, diffuser
    def tl(u, v):
        a = abs(u)
        return (0.865 * u, -4.0, 0.878 + 0.014 * a * a + v * (0.015 + 0.030 * smoothstep(0.50, 0.92, a)))
    grid = cast_grid(tl, 49, 5, (0, 1, 0))
    slab(grid, (0, -1, 0), 0.05, 0.03, DARK, DARK)
    LAMPS['tail'] = grid
    g2 = cast_grid(bilin((-0.27, 0.565), (0.27, 0.565), (0.27, 0.69), (-0.27, 0.69), (1, -4.0)), 5, 3, (0, 1, 0))
    slab(g2, (0, -1, 0), 0.05, 0.014, PAINT, PAINT)
    g2 = cast_grid(bilin((-0.64, 0.335), (0.64, 0.335), (0.60, 0.455), (-0.60, 0.455), (1, -4.0)), 9, 3, (0, 1, 0))
    slab(g2, (0, -1, 0), 0.05, 0.05, DARK, DARK)
    LAMPS['diffuser'] = g2
    # ---------------------------------------------------------------- door handle recesses
    LAMPS['handles'] = []
    for y0, y1 in ((-0.36, -0.17), (-1.30, -1.11)):
        zc = belt((y0 + y1) / 2) - 0.07
        g2 = cast_grid(bilin((y0, zc - 0.022), (y1, zc - 0.022), (y1, zc + 0.022), (y0, zc + 0.022), (0, 3.0)), 5, 3, (-1, 0, 0))
        slab(g2, (1, 0, 0), 0.03, 0.016, DARK, DARK, True)
        LAMPS['handles'].append((y0, y1, zc))
    # ---------------------------------------------------------------- panel gaps
    zs = 0.275
    groove([P_side(0.90, belt(0.90) - 0.005), P_side(0.90, zs), P_side(-0.98, zs), P_side(-1.04, 0.45), P_side(-1.11, 0.62),
            P_side(-1.25, 0.77), P_side(-1.42, 0.85), P_side(-1.50, belt(-1.50) + 0.03)], False, mirror=True, radius=0.05)
    groove([P_side(-0.435, zs), P_side(-0.435, win_bot(-0.435) + 0.01)], False, mirror=True)
    groove([P_top(0.72, 0.915), P_top(0.775, 1.45), P_top(0.73, 1.95), P_top(0.50, 2.20), P_top(-0.50, 2.20), P_top(-0.73, 1.95),
            P_top(-0.775, 1.45), P_top(-0.72, 0.915), P_top(-0.36, 0.975), P_top(0.36, 0.975)], True, radius=0.08)
    groove([P_top(0.60, -2.035), P_top(0.635, -2.38), P_tail(0.60, 0.82), P_tail(0.58, 0.725), P_tail(-0.58, 0.725), P_tail(-0.60, 0.82),
            P_top(-0.635, -2.38), P_top(-0.60, -2.035)], True, radius=0.06)
    groove([P_side(-1.63, 0.80), P_side(-1.63, 0.92), P_side(-1.79, 0.92), P_side(-1.79, 0.80)], True, radius=0.03)      # fuel flap, right side
    groove([P_side(1.77, 0.62), P_side(1.98, 0.635)], False, mirror=True)                                                # bumper seams
    groove([P_side(-1.77, 0.62), P_side(-2.16, 0.845)], False, mirror=True)
    body = apply_cuts(body, 'shell3')
    return body


# ================================================================== the body
def seat_loft(xc, path, par):
    """A seat as one skin: path = (y, z) of its surface from the cushion's front edge up to the top of
    the back; par = (half width, bolster height, thickness) at each point."""
    rings = []
    for i, ((y, z), (W, B, th)) in enumerate(zip(path, par)):
        a = Vector(path[max(i - 1, 0)]); b = Vector(path[min(i + 1, len(path) - 1)])
        t = (b - a).normalized(); n = Vector((t.y, -t.x))
        bw = min(0.07, W * 0.32); gap = min(0.04, W * 0.16)
        sec = [(-W, -th), (-W, B * 0.7), (-W + 0.02, B), (-W + bw * 0.65, B), (-W + bw, B * 0.4), (-W + bw + gap, 0.0), (0.0, 0.0),
               (W - bw - gap, 0.0), (W - bw, B * 0.4), (W - bw * 0.65, B), (W - 0.02, B), (W, B * 0.7), (W, -th)]
        rings.append([(xc + s, y + n.x * h, z + n.y * h) for s, h in sec])
    gs = [CARBON] + [SEAT] * 10 + [CARBON, CARBON]
    with smooth(50.0):
        loft(gs, rings, (SEAT, SEAT))


def headrest(x, y, z):
    with smooth(60.0):
        ball(SEAT, (x, y, z), 1.0, 12, 8, (0.115, 0.048, 0.082))
    for dx in (-0.05, 0.05):
        cyl(METAL, (x + dx, y + 0.012, z - 0.14), (x + dx, y, z - 0.04), 0.006, 6, False)


def front_seat(xc):
    S = SEAT_TOP
    path = [(0.36, S - 0.045), (0.325, S - 0.004), (0.15, S), (-0.03, S - 0.004), (-0.14, S + 0.004), (-0.20, S + 0.07), (-0.245, S + 0.22),
            (-0.295, S + 0.40), (-0.34, S + 0.55), (-0.365, S + 0.62), (-0.38, S + 0.67), (-0.386, S + 0.695)]
    par = [(0.235, 0.015, 0.075), (0.25, 0.035, 0.105), (0.25, 0.04, 0.105), (0.25, 0.04, 0.10), (0.25, 0.045, 0.10), (0.25, 0.06, 0.09),
           (0.25, 0.07, 0.08), (0.25, 0.065, 0.07), (0.235, 0.045, 0.065), (0.205, 0.028, 0.06), (0.175, 0.016, 0.055), (0.15, 0.006, 0.045)]
    seat_loft(xc, path, par)
    headrest(xc, -0.425, S + 0.80)
    cbox(CARBON, (xc - 0.215, -0.15, FLOOR - 0.005), (xc + 0.215, 0.30, S - 0.085), 0.012)       # the base on its runners


def rear_bench():
    S = SEAT_TOP + 0.02
    path = [(-0.575, S - 0.05), (-0.61, S - 0.004), (-0.78, S), (-0.93, S - 0.008), (-1.025, S + 0.004), (-1.075, S + 0.07), (-1.125, S + 0.22),
            (-1.18, S + 0.385), (-1.205, S + 0.45), (-1.25, S + 0.56), (-1.285, S + 0.64), (-1.30, S + 0.675)]
    par = [(0.62, 0.012, 0.09), (0.63, 0.03, 0.12), (0.63, 0.035, 0.13), (0.63, 0.035, 0.13), (0.46, 0.035, 0.13), (0.46, 0.05, 0.10),
           (0.46, 0.055, 0.09), (0.46, 0.055, 0.08), (0.61, 0.05, 0.075), (0.61, 0.04, 0.07), (0.60, 0.025, 0.06), (0.58, 0.008, 0.05)]
    seat_loft(0.0, path, par)
    for x in (-0.36, 0.36): headrest(x, -1.335, S + 0.775)
    for x in (-0.19, 0.19):                                                                        # the seams between the three places
        obox(CARBON, (x, -0.62, S + 0.002), (x, -1.01, S - 0.004), 0.008, 0.006, (0, 0, 1))
        obox(CARBON, (x, -1.085, S + 0.10), (x, -1.275, S + 0.62), 0.008, 0.006, (0, 1, 0.3))
    box(CARBON, (-0.63, -1.00, FLOOR - 0.005), (0.63, -0.60, S - 0.10), '-z')                    # the bench's base


def build_body():
    body = build_shell()
    # ---------------------------------------------------------------- the hull
    with smooth(38.0, 'box'):
        for g, P in tris_of(body): tri(g, P)
    hull = bounds(G)
    X1 = Vector((1, 0, 0)); Y1 = Vector((0, 1, 0)); Z1 = Vector((0, 0, 1))

    def sheet(g, grid, off, two=True, sm=60.0):
        with smooth(sm, 'box'):
            for i in range(len(grid) - 1):
                for j in range(len(grid[0]) - 1):
                    a, b, c, d = [p + off for p in (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])]
                    if two: quad2(g, a, b, c, d)
                    else: quad_out(g, a, b, c, d, (a + c) / 2 - off.normalized())

    # ---------------------------------------------------------------- glazing, flush with the body
    gf, gr, gw, gb = LAMPS['gf'], LAMPS['gr'], LAMPS['gw'], LAMPS['gb']
    for m in (IDN, MIR):
        with xf(m):
            sheet(GLASS, gf, -X1 * 0.003); sheet(GLASS, gr, -X1 * 0.003)
            # the black centre pillar and the bright strip round the side glass
            sheet(DARK, [gf[-1], gr[0]], X1 * 0.003, False)
            loop = [r[0] for r in gf] + [r[0] for r in gr] + [r[-1] for r in reversed(gr)] + [r[-1] for r in reversed(gf)]
            with smooth(70.0):
                pipe(METAL, [p + X1 * 0.001 for p in loop], 0.0055, 6, closed=True)
    sheet(GLASS, gw, -Z1 * 0.003); sheet(GLASS, gb, -Z1 * 0.003)
    # wipers, parked along the windscreen's lower edge
    ns = len(gw) - 1
    gn = (gw[ns // 2][2] - gw[ns // 2][0]).cross(X1).normalized()
    if gn.z < 0: gn = -gn
    for i0, i1, ip in ((2, 8, 8), (9, 14, 14)):
        pts = [gw[i][0].lerp(gw[i][1], 0.75) + gn * 0.016 for i in range(i0, i1 + 1)]
        with smooth(70.0):
            pipe(RUBBER, pts, 0.007, 6)
        piv = gw[ip][0] + Vector((0.03, 0.035, 0.0)); piv.z = top_z(piv.x, piv.y) + 0.004
        cyl(RUBBER, piv - Z1 * 0.01, piv + Z1 * 0.016, 0.013, 10)
        mid = pts[len(pts) // 2] + gn * 0.012
        obox(RUBBER, piv + Z1 * 0.014, mid, 0.014, 0.006, gn)
    # the high brake light behind the rear screen's top edge, the mirror on the windscreen
    zh = top_z(0.0, -1.33) - WALL - 0.012
    box('s_light_red', (-0.17, -1.345, zh - 0.018), (0.17, -1.325, zh))
    zh = top_z(0.0, 0.20) - WALL
    cbox(CARBON, (-0.105, 0.17, zh - 0.105), (0.105, 0.19, zh - 0.045), 0.006)
    cyl(CARBON, (0.0, 0.185, zh - 0.06), (0.0, 0.215, zh + 0.01), 0.011, 8)

    # ---------------------------------------------------------------- head lights
    grid = LAMPS['head']; nu, nv = len(grid), len(grid[0])
    for m in (IDN, MIR):
        with xf(m):
            sheet(GLASS, grid, -Y1 * 0.002, False)
            for i in (3, 6, 9):                                              # three projectors
                p = grid[i][2] - Y1 * 0.045; r = 0.013
                with xf(T(*p) @ RX(rad(-90.0))):
                    lathe(METAL, [(r * 1.35, -0.008), (r * 1.35, 0.014), (r, 0.018)], 14)
                    lathe('s_light', [(r, 0.018), (r * 0.85, 0.018 + r * 0.25), (r * 0.5, 0.018 + r * 0.42), (0.0, 0.018 + r * 0.48)], 14)
            pipe('s_light', [grid[i][nv - 2].lerp(grid[i][nv - 1], 0.25) - Y1 * 0.02 for i in range(1, nu - 1)], 0.0045, 6)   # running-light blade
            pipe('s_light_amber', [grid[i][1].lerp(grid[i][0], 0.25) - Y1 * 0.024 for i in range(8, nu - 1)], 0.0045, 6)      # indicator
            pipe(METAL, [grid[i][2] - Y1 * 0.05 for i in range(1, nu - 1)], 0.008, 6)
    # the grille: bright bars and a bright surround; plain bars in the lower intake
    g2 = LAMPS['grille']
    for j, f in ((0, 0.5), (1, 0.0), (1, 0.5), (2, 0.0), (2, 0.5)):
        a = g2[0][j].lerp(g2[0][j + 1], f); b = g2[-1][j].lerp(g2[-1][j + 1], f); mid_ = g2[4][j].lerp(g2[4][j + 1], f)
        yb = min(a.y, b.y, mid_.y) - 0.012
        obox(METAL, (a.x - 0.005, yb, a.z), (b.x + 0.005, yb, b.z), 0.016, 0.007, (0, 0, 1))
    rim_ = [g2[i][0] for i in range(9)] + [g2[8][j] for j in range(1, 4)] + [g2[i][3] for i in range(7, -1, -1)] + [g2[0][j] for j in (2, 1)]
    with smooth(70.0):
        pipe(METAL, [p + Y1 * 0.001 for p in rim_], 0.006, 6, closed=True)
    g2 = LAMPS['intake']
    for j in (1, 2):
        a = g2[0][j]; b = g2[-1][j]
        yb = min(a.y, b.y, g2[4][j].y) - 0.02
        obox(CARBON, (a.x - 0.005, yb, a.z), (b.x + 0.005, yb, b.z), 0.03, 0.008, (0, 0, 1))

    # ---------------------------------------------------------------- tail lights: one bar from side to side
    grid = LAMPS['tail']; nu, nv = len(grid), len(grid[0])
    with smooth(60.0, 'box'):
        for i in range(nu - 1):
            u = abs(-1.0 + 2.0 * (i + 0.5) / (nu - 1))
            for j in range(nv - 1):
                a, b, c, d = [p + Y1 * 0.007 for p in (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])]
                g = 's_light_amber' if (u > 0.80 and j == 0) else 's_light_red'
                quad_out(g, a, b, c, d, (a + c) / 2 + Y1)
    # exhaust tips in the diffuser, with fins between them
    g2 = LAMPS['diffuser']
    ty = max(p.y for row in g2 for p in row)
    for sx in (-1, 1):
        ye = tail_y(sx * 0.47, 0.395)
        with xf(T(sx * 0.47, ye + 0.045, 0.395) @ RX(rad(90.0)) @ Matrix.Diagonal((1.75, 1.0, 1.0, 1.0))):
            lathe(METAL, [(0.036, 0.0), (0.036, 0.062), (0.031, 0.062), (0.031, 0.010)], 24)
            lathe(DARK, [(0.031, 0.010), (0.0, 0.010)], 24)
    for x in (-0.27, -0.135, 0.0, 0.135, 0.27):
        ye = tail_y(x, 0.395)
        with xf(PX): prism(DARK, [(ye + 0.06, 0.34), (ye + 0.004, 0.345), (ye + 0.004, 0.45), (ye + 0.06, 0.45)], x - 0.006, x + 0.006)

    # ---------------------------------------------------------------- door handles, mirrors, roof fin
    for m in (IDN, MIR):
        with xf(m):
            for y0, y1, zc in LAMPS['handles']:
                a = P_side(y0 + 0.004, zc + 0.008, 0.002); b = P_side(y1 - 0.004, zc + 0.008, 0.002)
                obox(METAL, a, b, 0.012, 0.020, (0, 0, 1), 0.003)
            my = gf[0][0].y + 0.05; mz = belt(my) - 0.012
            p0 = P_side(my, mz, -0.01)
            hc = p0 + Vector((0.135, -0.015, 0.085))
            with smooth(60.0):
                pipe(DARK, [p0, p0 + Vector((0.05, -0.005, 0.035)), hc - Vector((0.03, 0, 0.03))], 0.017, 8)
                ball(PAINT, hc, 1.0, 14, 10, (0.095, 0.052, 0.058))
            ball(METAL, hc - Y1 * 0.034, 1.0, 14, 8, (0.080, 0.022, 0.046))                 # the mirror glass
            ball('s_light_amber', hc + Vector((0.035, 0.036, 0.0)), 1.0, 8, 6, (0.045, 0.014, 0.008))   # repeater
    rings = []
    for i in range(8):
        t = i / 7; y = lerp(-0.96, -1.19, t)
        hw = 0.006 + 0.022 * math.sin(math.pi * min(1.0, t * 1.25)) ** 0.7; h = 0.004 + 0.062 * t ** 1.3 * (1.0 if t < 0.93 else 0.75)
        zb_ = top_z(0.0, y) - 0.006
        rings.append([(hw * math.cos(math.pi * k / 8), y, zb_ + h * math.sin(math.pi * k / 8) ** 0.8) for k in range(9)])
    with smooth(50.0, 'box'):
        loft(PAINT, rings, (PAINT, PAINT))

    # ---------------------------------------------------------------- the cabin
    front_seat(-SEAT_X); front_seat(SEAT_X)
    rear_bench()
    # dashboard: one skin from door to door, its top following the base of the windscreen
    rings = []
    for i in range(17):
        x = -0.85 + 1.70 * i / 16
        zt = top_z(x, 0.90) - 0.04
        zr = min(0.965, top_z(x, 0.70) - 0.05)
        prof = [(0.90, zt), (0.70, zr), (0.635, zr - 0.012), (0.60, zr - 0.05), (0.585, 0.72), (0.60, 0.63), (0.68, 0.565), (0.90, 0.55)]
        rings.append([(x, y, z) for y, z in prof])
    with smooth(50.0):
        loft(DASH, rings, (DASH, DASH))
    dx = -SEAT_X; GZ = 0.895
    arch_o = [(dx + 0.20 * math.cos(math.pi * k / 16), GZ - 0.01 + 0.105 * math.sin(math.pi * k / 16)) for k in range(17)]
    arch_i = [(dx + 0.188 * math.cos(math.pi * k / 16), GZ - 0.01 + 0.093 * math.sin(math.pi * k / 16)) for k in range(17)]
    with smooth(50.0):
        with xf(PY):
            prism(DASH, arch_o + arch_i[::-1], 0.545, 0.72)                 # the instrument hood
            prism(CARBON, arch_i, 0.592, 0.61)                              # the panel the dials sit on
    gauge(dx - 0.082, 0.592, GZ + 0.022, 0.050, 110.0)
    gauge(dx + 0.082, 0.592, GZ + 0.022, 0.050, 200.0)
    for x in (-0.76, -0.13, 0.13, 0.76): vent(x, 0.584, 0.865, 0.034)
    cbox(CARBON, (-0.135, 0.555, 0.885), (0.135, 0.585, 1.005), 0.006)      # the centre display, a dark slab in a bright frame
    cbox(METAL, (-0.141, 0.560, 0.879), (0.141, 0.588, 1.011), 0.006)
    cbox(METAL, (0.20, 0.578, 0.80), (0.72, 0.588, 0.812), 0.003)           # a bright line across the passenger's side
    cbox(CARBON, (-0.12, 0.575, 0.70), (0.12, 0.59, 0.80), 0.006)           # the heater panel
    for x in (-0.07, 0.0, 0.07): cyl(METAL, (x, 0.578, 0.75), (x, 0.562, 0.75), 0.016, 12, (False, True))
    # steering column and stalks
    t = rad(STEER_TILT)
    hub = Vector(STEER_HUB); A = Vector((0.0, -math.cos(t), math.sin(t)))   # the column's axis, toward the driver
    cyl(CARBON, hub - A * 0.30, hub - A * 0.105, 0.045, 14)
    with xf(T(*STEER_HUB) @ RX(-t)):
        for sx in (-1, 1):
            obox(CARBON, (sx * 0.04, 0.075, 0.0), (sx * 0.20, 0.060, 0.012), 0.014, 0.012, (0, 1, 0), 0.003)
    # centre console, arm rest, gear selector
    with xf(PX):
        prism(DASH, [(0.62, FLOOR - 0.005), (0.62, 0.68), (0.50, 0.66), (0.40, 0.555), (0.12, 0.50), (-0.20, 0.50), (-0.54, 0.50),
                     (-0.56, FLOOR - 0.005)], -0.115, 0.115, 0.012)
    obox(CARBON, (0, 0.38, 0.555), (0, 0.13, 0.5065), 0.16, 0.006, (0, 0, 1))
    cbox(SEAT, (-0.10, -0.52, 0.498), (0.10, -0.14, 0.548), 0.014)
    with xf(T(0.0, 0.215, 0.512) @ RX(rad(-11.0))):
        cbox(METAL, (-0.022, -0.035, 0.0), (0.022, 0.035, 0.03), 0.008)
    for y in (0.02, -0.06):
        cyl(CARBON, (0.045, y, 0.494), (0.045, y, 0.501), 0.034, 14, (False, True))        # cup holders
        cyl(METAL, (0.045, y, 0.494), (0.045, y, 0.5015), 0.037, 14, (False, False))
    # door trims: arm rests, pulls, speaker grilles
    for sx in (-1, 1):
        for y0, y1 in ((-0.30, 0.42), (-0.97, -0.53)):
            xi = side_x((y0 + y1) / 2, 0.66) - WALL
            cbox(DASH, (sx * (xi - 0.075), y0, 0.60), (sx * (xi + 0.01), y1, 0.665), 0.014)
            cbox(SEAT, (sx * (xi - 0.07), y0 + 0.06, 0.663), (sx * (xi - 0.005), y1 - 0.06, 0.672), 0.004)
            cyl(METAL, (sx * (xi - 0.012), y1 - 0.10, 0.76), (sx * (xi - 0.012), y1 - 0.22, 0.76), 0.008, 8)
            ys = y1 - 0.14
            cyl(CARBON, (sx * (side_x(ys, 0.40) - WALL + 0.004), ys, 0.40), (sx * (side_x(ys, 0.40) - WALL - 0.008), ys, 0.40), 0.075, 16, (False, True))
    # pedals and the footrest
    for x0, x1, z0 in ((-0.455, -0.375, 0.29), (-0.315, -0.26, 0.25)):
        with xf(T((x0 + x1) / 2, 0.80, z0 + 0.06) @ RX(rad(-20.0))):
            cbox(METAL, (-(x1 - x0) / 2, -0.008, -0.055), ((x1 - x0) / 2, 0.004, 0.055), 0.003)
        cyl(METAL, ((x0 + x1) / 2, 0.80, z0 + 0.07), ((x0 + x1) / 2, 0.87, z0 + 0.16), 0.007, 6)
    cbox(METAL, (-0.66, 0.79, FLOOR + 0.02), (-0.575, 0.81, FLOOR + 0.17), 0.004)
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
blo, bhi = HULL                                          # the hull alone (without mirrors, roof fin)
flo, fhi = bounds(BODY)
L_, W_, H_ = bhi.y - blo.y, bhi.x - blo.x, bhi.z
HEADLINER = top_z(HIP[0], HIP[1] - 0.25) - WALL
print('CAR hull L %.3f W %.3f H %.3f, clearance %.3f, full bounds %s %s' % (L_, W_, H_, blo.z, tuple(flo), tuple(fhi)))
print('CAR cushion top %.3f, headliner over the driver %.3f (%.3f above the cushion)' % (SEAT_TOP, HEADLINER, HEADLINER - SEAT_TOP))
assert abs(L_ - 4.85) < 0.02, 'length %.3f' % L_
assert abs(W_ - 1.86) < 0.01, 'width %.3f' % W_
assert abs(H_ - 1.45) < 0.01, 'height %.3f' % H_
assert 0.14 <= blo.z <= 0.16, 'ground clearance %.3f' % blo.z
assert HEADLINER - SEAT_TOP >= 0.98, 'head room %.3f' % (HEADLINER - SEAT_TOP)
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
HEAD = LAMPS['head'][6][2]
TAILP = LAMPS['tail'][44][2]


def cbx(lo, hi):
    return {'center': E(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)),
            'half_extents': [round((hi[0] - lo[0]) / 2, 4), round((hi[2] - lo[2]) / 2, 4), round((hi[1] - lo[1]) / 2, 4)]}

CB = 0.24                                                # the boxes' underside (the wheels are simulated on their own)
INFO = {
    'comment': 'engine frame: +y up, front = -z, +x = right; metres. body.mesh origin = on the ground under the middle of the wheelbase',
    'wheel_radius': WR, 'wheel_width': WW, 'wheelbase': WB, 'track': TRACK,
    'wheel_mesh_axle_axis': [1, 0, 0],
    'wheel_mesh_outer_side': '+x (draw as is on the right, rotated 180 deg about y on the left)',
    'wheels': {'front_left': E((-WX, AXF, WR)), 'front_right': E((WX, AXF, WR)), 'rear_left': E((-WX, AXR, WR)), 'rear_right': E((WX, AXR, WR))},
    'seat_driver': E(EYE_L), 'seat_passenger': E(EYE_R),
    'hip_driver': E(HIP),
    'legs': LEGS,
    'steering_hub': E(STEER_HUB),
    'steering_axis': [0.0, round(math.sin(t_), 5), round(math.cos(t_), 5)],
    'steering_tilt_deg': STEER_TILT,
    'steering_mesh_axis': [0, 0, 1],
    'steering_placement': 'spin about mesh +z, then rotate about +x by -steering_tilt_deg, then translate to steering_hub',
    'headlights': [E((-HEAD.x, HEAD.y + 0.02, HEAD.z)), E((HEAD.x, HEAD.y + 0.02, HEAD.z))],
    'tail_lights': [E((-TAILP.x, TAILP.y - 0.02, TAILP.z)), E((TAILP.x, TAILP.y - 0.02, TAILP.z))],
    'exit_left': E((-(HALFW + 1.0), -0.15, 0.0)), 'exit_right': E((HALFW + 1.0, -0.15, 0.0)),
    'bounds_min': E((flo.x, fhi.y, flo.z)), 'bounds_max': E((fhi.x, flo.y, fhi.z)),
    'bounds_note': 'bounds are of the whole body.mesh, door mirrors included; the hull alone is %.3f wide' % W_,
    'collision_boxes': [
        dict(name='nose', **cbx((-0.86, 1.95, CB), (0.86, bhi.y - 0.03, round(top_z(0.0, 2.15), 3)))),
        dict(name='bonnet', **cbx((-HALFW, 0.85, CB), (HALFW, 1.95, round(top_z(0.0, 1.30), 3)))),
        dict(name='cabin', **cbx((-HALFW, -1.85, CB), (HALFW, 0.85, round(belt(-0.5), 3)))),
        dict(name='roof', **cbx((-0.70, -1.25, round(belt(-0.5), 3)), (0.70, 0.25, bhi.z - 0.01))),
        dict(name='rear_screen', **cbx((-0.70, -1.80, round(belt(-0.5), 3)), (0.70, -1.25, round(top_z(0.0, -1.55), 3)))),
        dict(name='boot', **cbx((-HALFW + 0.02, blo.y + 0.03, CB + 0.06), (HALFW - 0.02, -1.85, round(top_z(0.0, -2.2), 3)))),
    ],
}
with open(os.path.join(OUT, 'car.json'), 'w') as f:
    json.dump(INFO, f, indent=2)
    f.write('\n')
print('CAR written to', OUT)


# ================================================================== preview renders (SALOON_PREVIEW=1)
def preview():
    COLS = {'s_paint': (0.10, 0.17, 0.30), 's_paint_dark': (0.035, 0.035, 0.04), 's_carbon': (0.10, 0.10, 0.11), 's_glass': (0.55, 0.75, 0.85),
            's_rim': (0.60, 0.61, 0.63), 's_tyre': (0.03, 0.03, 0.03), 's_brake': (0.42, 0.41, 0.40), 's_caliper': (0.75, 0.08, 0.05),
            's_seat': (0.66, 0.50, 0.33), 's_dash': (0.07, 0.07, 0.075), 's_metal': (0.78, 0.78, 0.77), 's_rubber': (0.015, 0.015, 0.015),
            's_light': (1.0, 0.98, 0.88), 's_light_red': (0.85, 0.04, 0.03), 's_light_amber': (1.0, 0.55, 0.05), 's_gauge': (0.92, 0.90, 0.84)}
    if os.environ.get('SALOON_COLOR'): COLS['s_paint'] = tuple(float(c) for c in os.environ['SALOON_COLOR'].split(','))
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    mats = {}
    for g, c in COLS.items():
        m = bpy.data.materials.new('pv_' + g)
        m.diffuse_color = (c[0], c[1], c[2], 0.30 if g == 's_glass' else 1.0)
        m.roughness = 0.25 if g in ('s_paint', 's_glass', 's_rim', 's_metal') else 0.7
        m.metallic = 0.6 if g in ('s_rim', 's_metal') else 0.0
        mats[g] = m

    def add(name, groups, mat):
        for g, v in groups.items():
            if g in os.environ.get("SALOON_HIDE", "").split(","): continue
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

    add('body', BODY, IDN)
    add('steer', STEER, T(*STEER_HUB) @ RX(-t_))
    for sx, y in ((1, AXF), (1, AXR), (-1, AXF), (-1, AXR)):
        add('wheel', WHEEL, T(sx * WX, y, WR) @ (IDN if sx > 0 else RZ(math.pi)))
    gm = bpy.data.meshes.new('ground')
    gm.from_pydata([(-12, -12, 0), (12, -12, 0), (12, 12, 0), (-12, 12, 0)], [], [(0, 1, 2, 3)])
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
    cam.clip_end = 200.0
    # name, camera, target, lens, near clip (the "interior" view clips the near side of the car away), hide the ground
    views = [('front34', (3.2, 5.2, 1.55), (0.0, 0.1, 0.60), 40.0, 0.03, False),
             ('rear34', (-3.3, -5.4, 1.7), (0.0, -0.2, 0.62), 40.0, 0.03, False),
             ('side', (7.4, 0.0, 0.95), (0.0, -0.05, 0.68), 50.0, 0.03, False),
             ('top', (0.0, -0.05, 13.5), (0.0, -0.05, 0.5), 50.0, 0.03, False),
             ('interior', (-3.6, -0.30, 1.25), (0.0, -0.30, 0.78), 34.0, 2.86, False),
             ('front', (0.5, 6.2, 0.9), (0.0, 1.0, 0.55), 45.0, 0.03, False),
             ('rear', (-0.5, -6.2, 1.0), (0.0, -1.0, 0.60), 45.0, 0.03, False),
             ('wheel', (2.1, 2.5, 0.6), (0.85, 1.45, 0.40), 50.0, 0.03, False),
             ('driver', (EYE_L[0], EYE_L[1], EYE_L[2] + 0.12), (-0.25, 2.5, 0.80), 14.0, 0.03, False),
             ('cabin', (0.0, -0.62, 1.22), (-0.15, 0.6, 0.72), 14.0, 0.03, False),
             ('under', (2.8, 2.8, -2.2), (0.0, 0.0, 0.4), 35.0, 0.03, True)]
    if os.environ.get("SALOON_CAM"):                    # a throw-away view: x,y,z,lookx,looky,lookz,lens,near
        c = [float(a) for a in os.environ["SALOON_CAM"].split(",")]
        views.append(("debug", tuple(c[0:3]), tuple(c[3:6]), c[6], c[7], False))
    only = os.environ.get('SALOON_VIEWS')
    for name, pos, look, lens, near, hide_ground in views:
        if only and name not in only.split(','): continue
        cob.location = Vector(pos)
        d = Vector(look) - Vector(pos)
        cob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        cam.lens = lens; cam.clip_start = near
        ground.hide_render = hide_ground
        sc.render.filepath = os.path.join(OUT, 'preview_%s.png' % name)
        bpy.ops.render.render(write_still=True)
        print('CAR preview', sc.render.filepath)


if os.environ.get('SALOON_PREVIEW') == '1':
    preview()
