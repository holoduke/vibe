# Builds the sports car - an original open-top mid-engine spider - in Blender and exports it
# for the game (same pipeline and file formats as tools/jeep/make_jeep.py):
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --python tools/sportscar/make_sportscar.py
# (set SPORTSCAR_PREVIEW=1 to also render assets/sportscar/preview_*.png with back-face culling on)
# Output (assets/sportscar/):
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
#   car.json       the numbers the engine needs (same keys as assets/jeep/jeep.json).
# File format: "mesh <name> <triangles>", then per vertex: position, normal, uv (uv in metres).
#
# How the body is made: a coarse cage of cross-sections -> Catmull-Clark subdivision -> boolean
# cuts (wheel housings, cockpit tub, side intakes, light pockets, panel gaps ...) -> triangles
# with normals smoothed below a crease angle. Everything else is put together from primitives.
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
OUT = os.path.join(ROOT, 'assets', 'sportscar')
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
WR, WW = 0.34, 0.28            # wheel radius, tyre width
WB, TRACK = 2.65, 1.66
AXF, AXR = WB / 2, -WB / 2     # axle lines (Y)
WX = TRACK / 2
HALFW = 0.975                  # half the body's width
ARCH = 0.385                   # wheel housing radius
TUB = 0.70                     # the cockpit's half width
C = 0.22                       # how far the whole cabin sits forward of the first layout
TUB_Y0, TUB_Y1 = -0.80 + C, 0.56 + C   # the cockpit opening (rear bulkhead, cowl edge)
FLOOR = 0.17
SEAT_X = 0.37
SEAT_TOP = 0.30                # top of the seat cushion
STEER_TILT = 22.0              # the column rises toward the driver at this angle from the horizontal
STEER_HUB = (-SEAT_X, 0.16 + C, 0.67)
HIP = (-SEAT_X, -0.29 + C, SEAT_TOP + 0.13)
EYE_L = (-SEAT_X, HIP[1], HIP[2] + 0.52)     # "seat_driver": the engine puts the hip 0.52 under it
EYE_R = (SEAT_X, HIP[1], HIP[2] + 0.52)
PAINT, DARK, CARBON, METAL, RUBBER, GLASS, DASH, SEAT = 's_paint', 's_paint_dark', 's_carbon', 's_metal', 's_rubber', 's_glass', 's_dash', 's_seat'
GROUPS = ['s_paint', 's_paint_dark', 's_carbon', 's_glass', 's_rim', 's_tyre', 's_brake', 's_caliper', 's_seat', 's_dash', 's_metal',
          's_rubber', 's_light', 's_light_red', 's_light_amber', 's_gauge']
GI = {g: i for i, g in enumerate(GROUPS)}
MAT = {g: bpy.data.materials.new(g) for g in GROUPS}
rad = math.radians


# ================================================================== the wheel
def build_wheel():
    """In the frame of wheel.mesh: axle = X, outer face toward +X."""
    with xf(PX):                                        # lathe axis (local z) -> X
        half = [(0.258, -0.128), (0.285, -0.139), (0.312, -0.140), (0.325, -0.134), (0.333, -0.120), (0.3355, -0.100), (0.340, -0.098),
                (0.340, -0.092), (0.3335, -0.090), (0.3335, -0.078), (0.340, -0.076),
                (0.340, -0.044), (0.3335, -0.042), (0.3335, -0.030), (0.340, -0.028), (0.340, -0.006), (0.3335, -0.004)]
        tyre = half + [(r, -x) for r, x in reversed(half)]
        lathe('s_tyre', tyre, 48, uref=WR, sharp=0.80)
        # shoulder blocks: short cross grooves on both shoulders, as dark sunk-looking ribs
        N = 36
        for s in (1, -1):
            for k in range(N):
                a0 = TAU * (k + (0.5 if s < 0 else 0.0)) / N; da = TAU / N * 0.30
                def P(a, r, x): return Vector((r * math.cos(a), r * math.sin(a), s * x))
                # a raised block on the shoulder between two cross grooves
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
        lathe('s_rim', [(0.258, 0.118), (0.262, 0.128), (0.255, 0.132), (0.243, 0.128), (0.238, 0.110), (0.232, -0.110),
                        (0.243, -0.128), (0.255, -0.132), (0.262, -0.128), (0.258, -0.118)], 48)
        # five spokes
        ST = [(0.040, 0.044, 0.078, 0.040), (0.075, 0.032, 0.081, 0.036), (0.115, 0.025, 0.088, 0.032), (0.155, 0.022, 0.097, 0.030),
              (0.195, 0.024, 0.106, 0.030), (0.226, 0.029, 0.113, 0.032), (0.246, 0.038, 0.118, 0.032)]
        with smooth(50.0):
            for k in range(5):
                a = TAU * (k + 0.25) / 5
                er = Vector((math.cos(a), math.sin(a), 0)); et = Vector((-math.sin(a), math.cos(a), 0)); ez = Vector((0, 0, 1))
                rings = []
                for r, hw, zf, th in ST:
                    c = er * r
                    rings.append([c - et * hw + ez * (zf - 0.012), c - et * hw * 0.55 + ez * zf, c + et * hw * 0.55 + ez * zf,
                                  c + et * hw + ez * (zf - 0.012), c + et * hw * 0.8 + ez * (zf - th), c - et * hw * 0.8 + ez * (zf - th)])
                loft('s_rim', rings)
        # hub, centre cap (blank), wheel nuts
        lathe('s_rim', [(0.068, 0.046), (0.068, 0.082), (0.060, 0.091), (0.030, 0.095), (0.0, 0.0955)], 30)
        lathe('s_metal', [(0.030, 0.095), (0.027, 0.099), (0.0, 0.100)], 20)
        for k in range(5):
            a = TAU * (k + 0.75) / 5
            c = Vector((0.047 * math.cos(a), 0.047 * math.sin(a), 0.090))
            cyl('s_metal', c, c + Vector((0, 0, 0.012)), 0.0085, 6, (False, True))
        # brake: disc with its bell, caliper on top
        lathe('s_brake', [(0.085, 0.008), (0.198, 0.008), (0.198, 0.038), (0.085, 0.038)], 40)
        lathe('s_metal', [(0.0, -0.085), (0.050, -0.085), (0.050, -0.020), (0.092, -0.020), (0.092, 0.046), (0.068, 0.046)], 24)
        arc = [(r * math.cos(rad(a)), r * math.sin(rad(a))) for r, aa in ((0.224, range(58, 123, 8)), (0.128, range(122, 57, -8))) for a in aa]
        prism('s_caliper', arc, -0.014, 0.056, 0.007)
        for a in (72.0, 90.0, 108.0):                    # the pistons' bosses
            c = Vector((0.176 * math.cos(rad(a)), 0.176 * math.sin(rad(a)), 0.056))
            cyl('s_caliper', c, c + Vector((0, 0, 0.004)), 0.017, 10, (False, True))


# ================================================================== the steering wheel
def build_steering():
    """In the frame of steering.mesh after RX(90): axis = Blender -Y = engine +Z."""
    with xf(RX(rad(90.0))):                             # local z (toward the driver) -> Blender -Y
        R = 0.172
        with smooth(60.0):
            pipe('s_dash', [(R * math.cos(TAU * i / 40), R * math.sin(TAU * i / 40), 0.0) for i in range(40)], 0.0165, 10, closed=True)
        for a, w in ((8.0, 0.040), (172.0, 0.040), (270.0, 0.034)):
            c, s = math.cos(rad(a)), math.sin(rad(a))
            obox('s_carbon', (0.045 * c, 0.045 * s - 0.006, -0.024), (0.166 * c, 0.166 * s, -0.003), w, 0.013, (0, 0, 1), 0.004)
        lathe('s_dash', [(0.064, -0.050), (0.066, -0.026), (0.058, -0.013), (0.032, -0.007), (0.0, -0.006)], 24)
        lathe('s_carbon', [(0.034, -0.105), (0.040, -0.050), (0.064, -0.050)], 20)
        lathe('s_metal', [(0.021, -0.0085), (0.018, -0.0045), (0.0, -0.004)], 16)          # a plain boss, no badge
        for sx in (-1, 1):                                                                  # thumb switches
            cyl('s_metal', (sx * 0.092, 0.004, -0.012), (sx * 0.092, 0.004, -0.004), 0.009, 10, (False, True))
            cyl('s_metal', (sx * 0.120, 0.006, -0.008), (sx * 0.120, 0.006, -0.0005), 0.007, 10, (False, True))


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
    if os.environ.get("SPORTSCAR_DEBUG") == "1":
        for k, c in enumerate(CUTS):
            m = body.modifiers.new("cut", "BOOLEAN")
            m.operation = "DIFFERENCE"; m.object = c; m.solver = "EXACT"
            body = bake(body, name)
            xs = [v.co for v in body.data.vertices]
            print("CAR cut %d (%d verts in cutter, first %s): body %d polys, y %.2f..%.2f z %.2f..%.2f" % (k, len(c.data.vertices), tuple(round(a, 2) for a in c.data.vertices[0].co), len(body.data.polygons), min(p.y for p in xs), max(p.y for p in xs), min(p.z for p in xs), max(p.z for p in xs)))
    for c in ([] if os.environ.get("SPORTSCAR_DEBUG") == "1" else CUTS):
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


# ================================================================== the body shell
# cross-sections, nose to tail:
#   y, half width, underside, sill top, widest (low), widest (high), crest, crest inset, top at 0.62 w, top at 0.32 w, top centre, sweep, rake
ST = [( 2.385, 0.34,  0.17, 0.19,  0.24, 0.30, 0.355, 0.04, 0.37,  0.375, 0.375, 0.02, 0.0),
      ( 2.36,  0.60,  0.16, 0.19,  0.25, 0.32, 0.385, 0.06, 0.395, 0.40,  0.40,  0.13, 0.0),
      ( 2.27,  0.80,  0.13, 0.19,  0.27, 0.35, 0.42,  0.08, 0.43,  0.435, 0.435, 0.12, 0.0),
      ( 2.05,  0.93,  0.12, 0.20,  0.30, 0.42, 0.51,  0.09, 0.50,  0.49,  0.49,  0.04, 0.0),
      ( 1.74,  0.975, 0.12, 0.22,  0.38, 0.62, 0.69,  0.08, 0.62,  0.58,  0.57,  0.0, 0.0),
      ( 1.325, 0.985, 0.12, 0.22,  0.42, 0.75, 0.795, 0.07, 0.745, 0.675, 0.66,  0.0, 0.0),
      ( 0.98,  0.975, 0.12, 0.22,  0.44, 0.72, 0.79,  0.09, 0.765, 0.745, 0.74,  0.0, 0.0),
      ( 0.78,  0.95,  0.12, 0.22,  0.46, 0.68, 0.775, 0.11, 0.80,  0.81,  0.81,  0.0, 0.0),
      ( 0.30,  0.93,  0.12, 0.22,  0.46, 0.63, 0.73,  0.12, 0.745, 0.75,  0.75,  0.0, 0.0),
      (-0.25,  0.945, 0.12, 0.22,  0.48, 0.65, 0.76,  0.12, 0.80,  0.81,  0.81,  0.0, 0.0),
      (-0.65,  0.975, 0.12, 0.22,  0.50, 0.74, 0.86,  0.11, 0.88,  0.89,  0.89,  0.0, 0.0),
      (-1.325, 0.985, 0.12, 0.24,  0.50, 0.80, 0.91,  0.09, 0.905, 0.90,  0.90,  0.0, 0.0),
      (-1.78,  0.965, 0.15, 0.30,  0.50, 0.76, 0.865, 0.10, 0.865, 0.86,  0.86,  0.0, 0.0),
      (-2.10,  0.93,  0.27, 0.37,  0.52, 0.72, 0.82,  0.09, 0.82,  0.82,  0.82,  0.0, 0.10),
      (-2.19,  0.90,  0.32, 0.40,  0.52, 0.71, 0.805, 0.07, 0.805, 0.805, 0.805, 0.0, 0.15),
      (-2.235, 0.80,  0.34, 0.41,  0.52, 0.70, 0.795, 0.06, 0.80,  0.80,  0.80,  0.0, 0.15)]


def make_shell():
    verts = []; faces = []; fm = []
    NR = 18
    for (y, w, zb, zk, zlo, zhi, zs, ins, zm, z8, zc, sw, rk) in ST:
        half = [(0.0, zb), (0.55 * w, zb), (w - 0.07, zb), (w - 0.015, zk), (w, zlo), (w - 0.005, zhi), (w - ins, zs),
                (0.62 * w, zm), (0.32 * w, z8), (0.0, zc)]
        def yy(x, z): return y - sw * (x / w) ** 2 + rk * (zc - z)
        ring = [(x, yy(x, z), z) for x, z in half] + [(-half[j][0], yy(half[j][0], half[j][1]), half[j][1]) for j in range(8, 0, -1)]
        verts += ring
    def gm(k):
        kk = k if k < 9 else 17 - k
        return DARK if kk <= 2 else PAINT
    for i in range(len(ST) - 1):
        for k in range(NR):
            a, b = i * NR + k, i * NR + (k + 1) % NR
            faces.append((a, b, b + NR, a + NR)); fm.append(gm(k))
    for base in (0, (len(ST) - 1) * NR):                 # the end faces, as strips from side to side
        for j in range(1, 8):
            faces.append((base + j, base + j + 1, base + 18 - (j + 1), base + 18 - j)); fm.append(DARK if j <= 2 else PAINT)
        faces.append((base, base + 1, base + 17)); fm.append(DARK)
        faces.append((base + 9, base + 10, base + 8)); fm.append(PAINT)
    ob = new_obj('shell', verts, faces, fm)
    m = ob.modifiers.new('ss', 'SUBSURF'); m.levels = 2; m.render_levels = 2
    ob = bake(ob, 'shell1')
    me = ob.data
    wmax = max(abs(v.co.x) for v in me.vertices)
    for v in me.vertices: v.co.x *= HALFW / wmax
    me.update()
    return ob


def tris_of(ob):
    me = ob.data
    me.calc_loop_triangles()
    for lt in me.loop_triangles:
        yield GROUPS[lt.material_index], [me.vertices[i].co.copy() for i in lt.vertices]


LAMPS = {}
TAIL_LAMPS = ((0.63, 0.082, False), (0.445, 0.064, True))
TAIL_Z = 0.665        # x, radius, amber centre
MOUTH_Y = -0.62                                                     # where the side intakes open


def build_shell():
    """The subdivided hull with every opening cut into it; fills BVH (the uncut surface)."""
    global BVH
    body = make_shell()
    BVH = bvh_of(body.data)
    # the scallops in the flanks that feed the side intakes
    V = [(0.975, 0.72, 0.49), (0.975, 0.72, 0.53), (1.3, 0.72, 0.53), (1.3, 0.72, 0.49),
         (0.845, MOUTH_Y, 0.33), (0.845, MOUTH_Y, 0.66), (1.3, MOUTH_Y, 0.66), (1.3, MOUTH_Y, 0.33)]
    F = [(0, 1, 5, 4), (1, 2, 6, 5), (0, 4, 7, 3), (4, 5, 6, 7), (0, 3, 2, 1), (2, 3, 7, 6)]
    cutter(V, F, [PAINT, PAINT, PAINT, DARK, PAINT, PAINT], True, 0.022)
    # wheel housings
    for yc in (AXF, AXR):
        n = 48; V = []
        for x, r in ((0.60, 0.33), (0.72, ARCH), (1.3, ARCH)):      # narrower at the inner end, under the bonnet
            V += [(x, yc + r * math.cos(TAU * i / n), WR + r * math.sin(TAU * i / n)) for i in range(n)]
        F = [(k * n + i, k * n + (i + 1) % n, (k + 1) * n + (i + 1) % n, (k + 1) * n + i) for k in (0, 1) for i in range(n)] + [tuple(range(n)), tuple(range(2 * n, 3 * n))]
        cutter(V, F, DARK, True)
    # the cockpit tub, and the footwells that reach forward between the front wheels
    prof = [(0.90, FLOOR), (0.90, 0.50), (0.60 + C, 0.56), (TUB_Y1, 0.62), (TUB_Y1, 1.5), (TUB_Y0, 1.5), (TUB_Y0, 0.55), (-0.70 + C, FLOOR)]
    n = len(prof)
    V = [(-TUB, y, z) for y, z in prof] + [(TUB, y, z) for y, z in prof]
    F = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n)), tuple(range(n, 2 * n))]
    cutter(V, F, CARBON)
    V = [(x, y, z) for x in (-0.55, 0.55) for y in (0.85, 1.10) for z in (FLOOR, 0.49)]
    cutter(V, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)], CARBON)
    # headlight pockets on the wings
    c0, c1 = Vector((0.615, 2.115)), Vector((0.80, 1.69))
    dl = (c1 - c0).normalized(); pp = Vector((-dl.y, dl.x))
    def hl(u, v):
        t = (u + 1) / 2
        h = 0.012 + 0.072 * math.sin(math.pi * min(1.0, t * 1.15)) ** 0.7
        c = c0.lerp(c1, t) + pp * (v * h)
        return (c.x, c.y, 3.0)
    grid = cast_grid(hl, 15, 7, (0, 0, -1))
    slab(grid, (0, 0, 1), 0.05, 0.055, DARK, DARK, True)
    LAMPS['head'] = grid
    # the nose: two big intakes and a small one between them
    g2 = cast_grid(bilin((0.17, 0.20), (0.72, 0.215), (0.76, 0.325), (0.19, 0.33), (1, 4.0)), 9, 5, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.06, 0.085, DARK, DARK, True)
    LAMPS['intake'] = g2
    g2 = cast_grid(bilin((-0.115, 0.20), (0.115, 0.20), (0.10, 0.31), (-0.10, 0.31), (1, 4.0)), 5, 4, (0, -1, 0))
    slab(g2, (0, 1, 0), 0.06, 0.06, DARK)
    # the tail: round lamp pockets, a grille between them, the exhaust recess
    for lx, lr, _ in TAIL_LAMPS:
        n = 32; V = []
        ty = tail_y(lx, TAIL_Z)
        for y in (ty - 0.3, ty + 0.045):
            V += [(lx + (lr + 0.009) * math.cos(TAU * i / n), y, TAIL_Z + (lr + 0.009) * math.sin(TAU * i / n)) for i in range(n)]
        F = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n)), tuple(range(n, 2 * n))]
        cutter(V, F, DARK, True)
    g2 = cast_grid(bilin((-0.325, 0.612), (0.325, 0.612), (0.325, 0.718), (-0.325, 0.718), (1, -4.0)), 5, 3, (0, 1, 0))
    slab(g2, (0, -1, 0), 0.05, 0.035, DARK)
    g2 = cast_grid(bilin((-0.40, 0.43), (0.40, 0.43), (0.36, 0.56), (-0.36, 0.56), (1, -4.0)), 5, 3, (0, 1, 0))
    slab(g2, (0, -1, 0), 0.05, 0.05, DARK)
    # panel gaps: doors (they end at the intake's mouth), bonnet, engine cover
    zs = 0.245
    groove([(0.64, 0.74, 0.82), (0.90, 0.755, 0.77), (0.98, 0.82, 0.52), (0.97, 0.865, zs), (0.97, MOUTH_Y, zs), (0.97, MOUTH_Y, 0.305)],
           False, mirror=True, radius=0.06)
    groove([(0.97, MOUTH_Y, 0.69), (0.93, MOUTH_Y + 0.02, 0.76), (0.64, MOUTH_Y + 0.07, 0.84)], False, mirror=True, radius=0.04)
    groove([(0.60, 1.07, 0.9), (0.585, 1.55, 0.9), (0.44, 2.06, 0.9), (-0.44, 2.06, 0.9), (-0.585, 1.55, 0.9), (-0.60, 1.07, 0.9)], True, radius=0.07)
    groove([(0.64, -0.74, 1.0), (0.61, -2.03, 1.0), (-0.61, -2.03, 1.0), (-0.64, -0.74, 1.0)], True, radius=0.07)
    # bonnet outlets, the engine cover's vent, door handle recesses
    g2 = cast_grid(bilin((0.16, 1.71), (0.40, 1.75), (0.385, 1.87), (0.16, 1.83), (2, 3.0)), 5, 4, (0, 0, -1))
    slab(g2, (0, 0, 1), 0.05, 0.035, DARK, DARK, True)
    LAMPS['outlet'] = g2
    g2 = cast_grid(bilin((-0.135, -1.90), (0.135, -1.90), (0.135, -0.80), (-0.135, -0.80), (2, 3.0)), 3, 11, (0, 0, -1))
    slab(g2, (0, 0, 1), 0.05, 0.03, DARK)
    g2 = cast_grid(bilin((-0.42, 0.668), (-0.27, 0.668), (-0.27, 0.698), (-0.42, 0.698), (0, 3.0)), 5, 3, (-1, 0, 0))
    slab(g2, (1, 0, 0), 0.03, 0.022, DARK, DARK, True)
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


def seat(xc):
    """A bucket seat: the surface the occupant touches is swept along the side profile `path`."""
    path = [(0.13, 0.262), (0.09, 0.303), (-0.10, 0.300), (-0.30, 0.280), (-0.40, 0.287), (-0.455, 0.36), (-0.52, 0.55), (-0.585, 0.75),
            (-0.625, 0.88), (-0.650, 0.95), (-0.668, 1.01), (-0.680, 1.06), (-0.688, 1.09), (-0.692, 1.102)]
    par = [(0.235, 0.02, 0.085), (0.25, 0.055, 0.12), (0.25, 0.07, 0.12), (0.25, 0.075, 0.105), (0.25, 0.085, 0.10), (0.25, 0.10, 0.09),
           (0.25, 0.11, 0.07), (0.245, 0.09, 0.06), (0.215, 0.055, 0.058), (0.165, 0.03, 0.058), (0.135, 0.018, 0.056), (0.125, 0.012, 0.05),
           (0.105, 0.006, 0.04), (0.06, 0.0, 0.022)]
    rings = []
    for i, ((y, z), (W, B, th)) in enumerate(zip(path, par)):
        a = Vector(path[max(i - 1, 0)]); b = Vector(path[min(i + 1, len(path) - 1)])
        t = (b - a).normalized(); n = Vector((t.y, -t.x))
        bw = min(0.07, W * 0.32); gap = min(0.04, W * 0.16)
        sec = [(-W, -th * 0.75), (-W, B * 0.7), (-W + 0.02, B), (-W + bw * 0.65, B), (-W + bw, B * 0.4), (-W + bw + gap, 0.0), (0.0, 0.0),
               (W - bw - gap, 0.0), (W - bw, B * 0.4), (W - bw * 0.65, B), (W - 0.02, B), (W, B * 0.7), (W, -th * 0.75),
               (W * 0.62, -th * 1.25), (-W * 0.62, -th * 1.25)]
        rings.append([(xc + s, y + C + n.x * h, z + n.y * h) for s, h in sec])
    gs = [SEAT] * 15
    with smooth(62.0):
        loft(gs, rings, (SEAT, SEAT))
    for dx in (-0.045, 0.045):                          # two seams down the backrest
        obox(CARBON, (xc + dx, -0.468 + C, 0.42), (xc + dx, -0.573 + C, 0.74), 0.008, 0.004, (0, 1, 0.3))


def build_body():
    body = build_shell()
    # ---------------------------------------------------------------- the hull
    with smooth(38.0, 'box'):
        for g, P in tris_of(body): tri(g, P)
    hull = bounds(G)

    # ---------------------------------------------------------------- lights
    grid = LAMPS['head']; nu, nv = len(grid), len(grid[0])
    for m in (IDN, MIR):
        with xf(m):
            with smooth(60.0, 'box'):                                        # the clear cover, flush with the wing
                for i in range(nu - 1):
                    for j in range(nv - 1):
                        a, b, c, d = [p - Vector((0, 0, 0.002)) for p in (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])]
                        quad_out(GLASS, a, b, c, d, (a + c) / 2 - Vector((0, 0, 1)))
            mid = nv // 2
            for i, r in ((4, 0.028), (7, 0.026)):                            # two projectors
                p = grid[i][mid] - Vector((0, 0, 0.034))
                with xf(T(*p) @ RX(rad(-68.0))):
                    lathe(METAL, [(r * 1.28, -0.02), (r * 1.28, 0.012), (r, 0.016)], 18)
                    lathe('s_light', [(r, 0.016), (r * 0.85, 0.016 + r * 0.25), (r * 0.5, 0.016 + r * 0.42), (0.0, 0.016 + r * 0.48)], 18)
            pipe('s_light', [grid[i][1] - Vector((0, 0, 0.020)) for i in range(1, nu - 3)], 0.006, 6)           # running-light strip
            pipe('s_light_amber', [grid[i][nv - 2] - Vector((0, 0, 0.022)) for i in range(7, nu - 2)], 0.007, 6)  # indicator
            pipe(METAL, [grid[i][mid] - Vector((0, 0, 0.050)) for i in range(1, nu - 1)], 0.010, 6)
    for sx in (-1, 1):
        for lx, lr, amber in TAIL_LAMPS:
            k = lr / 0.090
            with xf(T(sx * lx, tail_y(lx, TAIL_Z) + 0.045, TAIL_Z) @ RX(rad(90.0))):           # local z -> rearward
                lathe('s_light_red', [(lr, 0.0), (lr, 0.024), (lr - 0.008 * k, 0.034), (0.058 * k, 0.034), (0.049 * k, 0.024)], 32)
                lathe(METAL, [(0.049 * k, 0.024), (0.049 * k, 0.029), (0.042 * k, 0.029), (0.042 * k, 0.014)], 32)
                lathe('s_light_amber' if amber else 's_light_red', [(0.042 * k, 0.014), (0.030 * k, 0.021), (0.0, 0.023)], 32)
    ty = tail_y(0.0, 0.762)
    box("s_light_red", (-0.16, ty - 0.004, 0.757), (0.16, ty + 0.01, 0.767))                 # high brake light
    for z in (0.638, 0.665, 0.692):                                                           # the grille's slats
        ty = tail_y(0.0, z)
        box(CARBON, (-0.32, ty + 0.012, z - 0.006), (0.32, ty + 0.03, z + 0.006))
    # exhaust tips in their recess
    for sx in (-1, 1):
        with xf(T(sx * 0.18, tail_y(0.18, 0.495) + 0.05, 0.495) @ RX(rad(90.0))):
            lathe(METAL, [(0.054, 0.0), (0.054, 0.070), (0.047, 0.070), (0.047, 0.012)], 24)
            lathe(DARK, [(0.047, 0.012), (0.0, 0.012)], 24)
    # diffuser fins
    for x in (-0.60, -0.36, -0.12, 0.12, 0.36, 0.60):
        with xf(PX): prism(DARK, [(-1.80, 0.16), (tail_y(x, 0.39) - 0.012, 0.265), (tail_y(x, 0.39) - 0.012, 0.39), (-1.80, 0.39)], x - 0.007, x + 0.007)
    # slats in the nose intakes and bonnet outlets, vanes in the side intakes, louvres on the engine cover
    g2 = LAMPS['intake']; g3 = LAMPS['outlet']
    for m in (IDN, MIR):
        with xf(m):
            for j in (1, 2, 3):
                pts = [g2[i][j] - Vector((0, 0.035, 0)) for i in range(len(g2))]
                for a, b in zip(pts[:-1], pts[1:]): obox(CARBON, a, b, 0.05, 0.006, (0, 0, 1))
            a = (g3[0][1] + g3[0][2]) / 2 - Vector((0, 0, 0.012)); b = (g3[-1][1] + g3[-1][2]) / 2 - Vector((0, 0, 0.012))
            obox(CARBON, a, b, 0.03, 0.005, (0, 0.5, 1))
            for z in (0.44, 0.55):
                box(CARBON, (0.85, MOUTH_Y - 0.04, z - 0.005), (side_x(MOUTH_Y - 0.08, z) - 0.004, MOUTH_Y + 0.02, z + 0.005))
    # the front splitter: a flat blade that follows the nose and stands a little proud of it
    fr = [Vector((x, nose_y(x, 0.185) + 0.035, 0.0)) for x in [-0.82 + 1.64 * i / 24 for i in range(25)]]
    with smooth(50.0):
        for a, b in zip(fr[:-1], fr[1:]):
            P = [Vector((p.x, p.y - dy, z)) for p in (a, b) for dy in (0.0, 0.16) for z in (0.150, 0.168)]
            ref = (P[0] + P[7]) / 2
            quad_out(DARK, P[1], P[5], P[7], P[3], ref); quad_out(DARK, P[0], P[4], P[6], P[2], ref); quad_out(DARK, P[0], P[4], P[5], P[1], ref)
    for p in (fr[0], fr[-1]):
        quad_out(DARK, (p.x, p.y, 0.150), (p.x, p.y, 0.168), (p.x, p.y - 0.16, 0.168), (p.x, p.y - 0.16, 0.150), (0, p.y, 0.16))
    for i in range(14):
        y = -0.85 - 0.076 * i
        z = top_z(0.0, y)
        with xf(T(0, y, z - 0.012) @ RX(rad(-22.0))):
            box(CARBON, (-0.135, -0.03, -0.004), (0.135, 0.03, 0.004))

    # ---------------------------------------------------------------- windscreen
    def Bc(s):
        x = 0.74 * s; y = 0.72 + C - 0.14 * s * s
        return Vector((x, y, top_z(x, y) + 0.004))
    def Tc(s):
        return Vector((0.655 * s, 0.10 + C - 0.10 * s * s, 1.133 - 0.03 * s * s))
    NS, NV = 16, 7
    GP = [[Bc(-1 + 2 * i / NS).lerp(Tc(-1 + 2 * i / NS), j / NV) for j in range(NV + 1)] for i in range(NS + 1)]
    with smooth(60.0):
        for i in range(NS):
            for j in range(NV):
                quad2(GLASS, GP[i][j], GP[i + 1][j], GP[i + 1][j + 1], GP[i][j + 1])
    with smooth(70.0):
        for i in (0, NS):
            pipe(DARK, [GP[i][0] - Vector((0, 0, 0.03))] + GP[i][1:], 0.021, 10)
            ball(DARK, GP[i][NV], 0.021, 10, 8)
        pipe(DARK, [GP[i][NV] for i in range(NS + 1)], 0.019, 10)
        pipe(RUBBER, [GP[i][0] for i in range(NS + 1)], 0.011, 8)
    # the rear-view mirror under the header
    hm = GP[NS // 2][NV]
    cyl(DARK, hm, hm + Vector((0, -0.035, -0.04)), 0.008, 8)
    cbox(DARK, (-0.085, hm.y - 0.05, hm.z - 0.085), (0.085, hm.y - 0.03, hm.z - 0.035), 0.008)
    box(METAL, (-0.075, hm.y - 0.0515, hm.z - 0.078), (0.075, hm.y - 0.05, hm.z - 0.042), '+y')
    # one wiper, parked along the lower edge
    gn = (GP[NS // 2][1] - GP[NS // 2][0]).cross(Vector((1, 0, 0))).normalized()
    if gn.z < 0: gn = -gn
    wa = GP[3][1].lerp(GP[3][2], 0.2) + gn * 0.014; wb = GP[9][1].lerp(GP[9][2], 0.2) + gn * 0.014
    obox(RUBBER, wa, wb, 0.014, 0.012, gn)
    piv = GP[7][0] + gn * 0.012 + Vector((0, 0.03, 0))
    cyl(METAL, piv - gn * 0.02, piv + gn * 0.012, 0.012, 10)
    obox(METAL, piv + gn * 0.008, (wa + wb) / 2 + gn * 0.008, 0.012, 0.005, gn)
    # quarter lights behind the pillars, door mirrors
    for sx in (-1, 1):
        i = NS if sx > 0 else 0
        p0, p1 = GP[i][0], GP[i][3]
        p2 = Vector((sx * 0.75, 0.17 + C, top_z(sx * 0.75, 0.17 + C) - 0.004))
        tri(GLASS, (p0, p1, p2)); tri(GLASS, (p2, p1, p0))
        pipe(RUBBER, [p1, p2], 0.007, 6)
        my = 0.42 + C
        mz = top_z(sx * 0.87, my)
        with smooth(60.0):
            pipe(DARK, [(sx * 0.87, my, mz - 0.015), (sx * 0.935, my - 0.01, mz + 0.035), (sx * 0.985, my - 0.015, mz + 0.06)], 0.013, 8)
            ball(PAINT, (sx * 1.035, my - 0.02, mz + 0.085), 1.0, 16, 10, (0.092, 0.046, 0.050))
            ball(METAL, (sx * 1.035, my - 0.02 - 0.030, mz + 0.085), 1.0, 16, 8, (0.078, 0.021, 0.040))      # the mirror glass

    # ---------------------------------------------------------------- roll hoops and their fairings
    for sx in (-1, 1):
        xc = sx * SEAT_X; y0, y1, n = TUB_Y0 - 0.045, -1.93, 16
        rings = []
        for i in range(n + 1):
            t = i / n; y = lerp(y0, y1, t)
            hw = 0.175 - 0.085 * t ** 0.8; h = 0.255 * (1.0 - t) ** 2.0 + 0.008
            zd = min(top_z(xc - hw, y), top_z(xc + hw, y), top_z(xc, y)) - 0.02
            zt = top_z(xc, y) + h
            rings.append([(xc + hw * math.cos(math.pi * k / 12), y, zd + (zt - zd) * math.sin(math.pi * k / 12) ** 0.85) for k in range(13)])
        with smooth(50.0, 'box'):
            loft(PAINT, rings, (CARBON, PAINT))
        r0 = rings[0]
        with smooth(60.0):
            zb_, zt_ = r0[0][2] - 0.03, r0[6][2] + 0.006
            pipe(METAL, [(xc + 0.182 * math.cos(math.pi * k / 16), y0 + 0.022, zb_ + (zt_ - zb_) * math.sin(math.pi * k / 16) ** 0.85) for k in range(17)], 0.019, 10)

    # ---------------------------------------------------------------- the cabin
    seat(-SEAT_X); seat(SEAT_X)
    # dashboard
    rings = []
    for i in range(15):
        x = -TUB + 2 * TUB * i / 14
        zt = top_z(x, 0.575 + C) + 0.002
        prof = [(0.575, zt), (0.45, zt - 0.004), (0.385, zt - 0.02), (0.348, zt - 0.06), (0.340, 0.68), (0.355, 0.60), (0.40, 0.555), (0.575, 0.54)]
        rings.append([(x, y + C, z) for y, z in prof])
    with smooth(50.0):
        loft(DASH, rings, (DASH, DASH))
    dx = -SEAT_X
    arch_o = [(dx + 0.20 * math.cos(math.pi * k / 16), 0.70 + 0.145 * math.sin(math.pi * k / 16)) for k in range(17)]
    arch_i = [(dx + 0.188 * math.cos(math.pi * k / 16), 0.70 + 0.133 * math.sin(math.pi * k / 16)) for k in range(17)]
    with smooth(50.0):
        with xf(PY):
            prism(DASH, arch_o + arch_i[::-1], 0.305 + C, 0.50 + C)         # the instrument hood
            prism(CARBON, arch_i, 0.352 + C, 0.37 + C)                      # the panel the dials sit on
    gauge(dx, 0.352 + C, 0.765, 0.054, 100.0)
    gauge(dx - 0.105, 0.352 + C, 0.752, 0.036, 200.0); gauge(dx + 0.105, 0.352 + C, 0.752, 0.036, 310.0)
    for x in (-0.635, -0.085, 0.085, 0.635): vent(x, 0.3405 + C, 0.69)
    cbox(CARBON, (0.16, 0.332 + C, 0.645), (0.56, 0.345 + C, 0.735), 0.005)  # trim panel in front of the passenger
    # steering column and shift paddles
    t = rad(STEER_TILT)
    hub = Vector(STEER_HUB); A = Vector((0.0, -math.cos(t), math.sin(t)))   # the column's axis, toward the driver
    cyl(CARBON, hub - A * 0.42, hub - A * 0.10, 0.032, 14)
    with xf(T(*STEER_HUB) @ RX(-t)):
        for sx in (-1, 1):
            cbox(METAL, (sx * 0.055, 0.050, -0.022), (sx * 0.150, 0.058, 0.075), 0.003)
    # centre console
    with xf(PX):
        prism(DASH, [(0.60 + C, FLOOR), (0.60 + C, 0.56), (0.40 + C, 0.56), (0.30 + C, 0.50), (0.05 + C, 0.42), (-0.45 + C, 0.40),
                     (-0.765 + C, 0.44), (-0.765 + C, FLOOR)], -0.10, 0.10, 0.012)
    obox(CARBON, (0, 0.035 + C, 0.4225), (0, -0.43 + C, 0.4035), 0.14, 0.006, (0, 0, 1))
    for i, y in enumerate((-0.04, -0.11, -0.18)):
        zc = 0.4245 - (0.035 - y) * 0.0409
        cyl(METAL, (0, y + C, zc), (0, y + C, zc + 0.012), 0.017, 14, (False, True))
    cyl('s_light_red', (0.0, -0.27 + C, 0.412), (0.0, -0.27 + C, 0.423), 0.014, 12, (False, True))    # the starter button
    cbox(SEAT, (-0.085, -0.745 + C, 0.435), (0.085, -0.47 + C, 0.475), 0.012)
    # door trims, pedals, floor mats
    for sx in (-1, 1):
        cbox(SEAT, (sx * 0.648, -0.36 + C, 0.50), (sx * 0.70, 0.26 + C, 0.555), 0.012)
        cyl(METAL, (sx * 0.693, 0.31 + C, 0.60), (sx * 0.693, 0.40 + C, 0.60), 0.008, 8)
        box(RUBBER, (sx * 0.13, -0.30 + C, FLOOR), (sx * 0.54, 1.02, FLOOR + 0.006), '-z')
    for x0, x1, z0 in ((-0.45, -0.38, 0.25), (-0.33, -0.275, 0.215)):
        with xf(T((x0 + x1) / 2, 1.02, z0 + 0.06) @ RX(rad(-20.0))):
            cbox(METAL, (-(x1 - x0) / 2, -0.008, -0.055), ((x1 - x0) / 2, 0.004, 0.055), 0.003)
        cyl(METAL, ((x0 + x1) / 2, 1.02, z0 + 0.07), ((x0 + x1) / 2, 1.09, z0 + 0.16), 0.007, 6)
    cbox(METAL, (-0.545, 1.00, 0.20), (-0.485, 1.02, 0.34), 0.004)           # footrest
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
blo, bhi = HULL                                          # the hull alone (without mirrors, screen, hoops)
flo, fhi = bounds(BODY)
L_, W_, H_ = bhi.y - blo.y, bhi.x - blo.x, fhi.z
print('CAR hull L %.3f W %.3f, deck %.3f, overall height %.3f, clearance %.3f, full bounds %s %s' % (L_, W_, bhi.z, H_, blo.z, tuple(flo), tuple(fhi)))
for y_, z_ in ((AXF, 0.34), (AXF, 0.66), (AXF, 0.72), (AXR, 0.34), (AXR, 0.66), (AXR, 0.72), (0.1, 0.5)):
    print('CAR half width at y %.2f z %.2f: %.3f' % (y_, z_, side_x(y_, z_)))
assert 4.6 * 0.95 <= L_ <= 4.6 * 1.05, 'length %.3f' % L_
assert abs(W_ - 1.95) < 0.01, 'width %.3f' % W_
assert 1.10 <= H_ <= 1.20, 'height %.3f' % H_
assert 0.10 <= blo.z <= 0.14, 'ground clearance %.3f' % blo.z
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
TY = tail_y(TAIL_LAMPS[0][0], TAIL_Z)
HEAD = LAMPS['head'][4][len(LAMPS['head'][0]) // 2]


def cbx(lo, hi):
    return {'center': E(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)),
            'half_extents': [round((hi[0] - lo[0]) / 2, 4), round((hi[2] - lo[2]) / 2, 4), round((hi[1] - lo[1]) / 2, 4)]}

CB = 0.20                                                # the boxes' underside (the wheels are simulated on their own)
INFO = {
    'comment': 'engine frame: +y up, front = -z, +x = right; metres. body.mesh origin = on the ground under the middle of the wheelbase',
    'wheel_radius': WR, 'wheel_width': WW, 'wheelbase': WB, 'track': TRACK,
    'wheel_mesh_axle_axis': [1, 0, 0],
    'wheel_mesh_outer_side': '+x (draw as is on the right, rotated 180 deg about y on the left)',
    'wheels': {'front_left': E((-WX, AXF, WR)), 'front_right': E((WX, AXF, WR)), 'rear_left': E((-WX, AXR, WR)), 'rear_right': E((WX, AXR, WR))},
    'seat_driver': E(EYE_L), 'seat_passenger': E(EYE_R),
    'steering_hub': E(STEER_HUB),
    'steering_axis': [0.0, round(math.sin(t_), 5), round(math.cos(t_), 5)],
    'steering_tilt_deg': STEER_TILT,
    'steering_mesh_axis': [0, 0, 1],
    'steering_placement': 'spin about mesh +z, then rotate about +x by -steering_tilt_deg, then translate to steering_hub',
    'headlights': [E((-HEAD.x, HEAD.y + 0.03, HEAD.z - 0.03)), E((HEAD.x, HEAD.y + 0.03, HEAD.z - 0.03))],
    'tail_lights': [E((-TAIL_LAMPS[0][0], TY - 0.01, TAIL_Z)), E((TAIL_LAMPS[0][0], TY - 0.01, TAIL_Z))],
    'exit_left': E((-(HALFW + 1.0), -0.15, 0.0)), 'exit_right': E((HALFW + 1.0, -0.15, 0.0)),
    'bounds_min': E((flo.x, fhi.y, flo.z)), 'bounds_max': E((fhi.x, flo.y, fhi.z)),
    'bounds_note': 'bounds are of the whole body.mesh, door mirrors included; the hull alone is %.3f wide' % W_,
    'collision_boxes': [
        dict(name='nose', **cbx((-0.90, 1.75, CB), (0.90, bhi.y - 0.03, 0.50))),
        dict(name='front', **cbx((-HALFW, TUB_Y1, CB), (HALFW, 1.75, 0.77))),
        dict(name='cabin', **cbx((-HALFW, TUB_Y0, CB), (HALFW, TUB_Y1, 0.74))),
        dict(name='rear', **cbx((-HALFW, blo.y, CB + 0.04), (HALFW, TUB_Y0, 0.89))),
        dict(name='windscreen', **cbx((-0.68, 0.06 + C, 0.74), (0.68, 0.40 + C, 1.15))),
        dict(name='roll_hoops', **cbx((-0.57, TUB_Y0 - 0.30, 0.89), (0.57, TUB_Y0, fhi.z))),
    ],
}
with open(os.path.join(OUT, 'car.json'), 'w') as f:
    json.dump(INFO, f, indent=2)
    f.write('\n')
print('CAR written to', OUT)


# ================================================================== preview renders (SPORTSCAR_PREVIEW=1)
def preview():
    COLS = {'s_paint': (0.62, 0.02, 0.02), 's_paint_dark': (0.035, 0.035, 0.04), 's_carbon': (0.09, 0.09, 0.10), 's_glass': (0.55, 0.75, 0.85),
            's_rim': (0.55, 0.56, 0.58), 's_tyre': (0.03, 0.03, 0.03), 's_brake': (0.42, 0.41, 0.40), 's_caliper': (0.95, 0.75, 0.05),
            's_seat': (0.62, 0.42, 0.24), 's_dash': (0.06, 0.06, 0.065), 's_metal': (0.75, 0.75, 0.74), 's_rubber': (0.015, 0.015, 0.015),
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
    cam.clip_start = 0.03; cam.clip_end = 200.0
    views = [('front34', (3.0, 4.6, 1.45), (0.0, 0.1, 0.48), 40.0, False),
             ('rear34', (-3.1, -4.7, 1.6), (0.0, -0.2, 0.50), 40.0, False),
             ("side", (7.5, 0.05, 0.8), (0.0, 0.05, 0.5), 38.0, False),
             ("top", (0.0, 0.05, 9.0), (0.0, 0.05, 0.5), 40.0, False),
             ("interior", (-1.0, -1.45, 1.95), (-0.15, 0.45, 0.60), 32.0, False),
             ('front', (0.5, 5.6, 0.8), (0.0, 1.0, 0.45), 45.0, False),
             ('rear', (-0.5, -5.4, 0.9), (0.0, -1.0, 0.50), 45.0, False),
             ('wheel', (2.0, 2.35, 0.55), (0.85, 1.33, 0.36), 50.0, False),
             ('driver', EYE_L, (-0.25, 2.5, 0.72), 14.0, False),
             ('under', (2.6, 2.6, -2.0), (0.0, 0.0, 0.4), 35.0, True)]
    only = os.environ.get('SPORTSCAR_VIEWS')
    for name, pos, look, lens, hide_ground in views:
        if only and name not in only.split(','): continue
        cob.location = Vector(pos)
        d = Vector(look) - Vector(pos)
        cob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        if name == "top": cob.rotation_euler = (0.0, 0.0, rad(-90.0))
        cam.lens = lens
        ground.hide_render = hide_ground
        sc.render.filepath = os.path.join(OUT, 'preview_%s.png' % name)
        bpy.ops.render.render(write_still=True)
        print('CAR preview', sc.render.filepath)


if os.environ.get('SPORTSCAR_PREVIEW') == '1':
    preview()
