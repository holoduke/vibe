# Builds the jeep - a generic open 4x4 utility vehicle - in Blender and exports it for the game:
#   blender -b --python tools/jeep/make_jeep.py
# (set JEEP_PREVIEW=1 to also render assets/jeep/preview_*.png with back-face culling on)
# Output (assets/jeep/):
#   body.mesh      everything except the four road wheels and the steering wheel.
#                  Origin: on the ground under the middle of the wheelbase.
#   wheel.mesh     ONE wheel (tyre, rim, hub, brake drum), centred on the origin, axle along
#                  the engine's X axis. Its OUTER face (hub cap, nuts) looks toward +X: draw it
#                  as it is for the right-hand wheels, turned 180 degrees about Y for the left.
#   steering.mesh  the steering wheel alone. Origin: the centre of the rim's plane (its hub).
#                  In the file its rotation axis is the engine's +Z (0, 0, 1) - the side the
#                  driver is on - and it is NOT tilted. To place it: spin it about its own Z,
#                  then rotate about the engine's +X by -STEER_TILT degrees (so the axis
#                  becomes jeep.json's "steering_axis" = (0, sin t, cos t)), then translate to
#                  "steering_hub".
#   jeep.json      the numbers the engine needs (wheel centres, seats, lights, colliders ...).
# The file format is the one of tools/office/make_office.py / tools/rocket/make_rocket.py:
# "mesh <name> <triangles>", then per vertex: position, normal, uv (uv in metres).
#
# Blender space: X = the vehicle's right, Y = forward, Z up.
# Game space:    x = X, y = Z, z = -Y   (so the front points to -z, +x is the right side).
import bpy, math, os, json
from mathutils import Vector, Matrix
from mathutils.geometry import tessellate_polygon

bpy.ops.wm.read_factory_settings(use_empty=True)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'assets', 'jeep')
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


def bolt(p, n, r=0.011, h=0.007, g='j_metal'):
    p = Vector(p); n = Vector(n).normalized()
    cyl(g, p, p + n * h, r, 6, (False, True))


def bolts(p0, p1, count, n, r=0.011, h=0.007, g='j_metal'):
    p0, p1 = Vector(p0), Vector(p1)
    for i in range(count):
        s = (i + 0.5) / count
        bolt(p0.lerp(p1, s), n, r, h, g)


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


# ================================================================== the measures
WR, WW = 0.40, 0.26            # wheel radius, tyre width
WB, TRACK = 2.40, 1.50
AXF, AXR = WB / 2, -WB / 2     # axle lines (Y)
WX = TRACK / 2
TW, TI = 0.72, 0.69            # tub half-width, outside / inside of its skin
BELT, SILL = 1.04, 0.40        # top edge of the tub, its bottom edge
YF, YR = 0.52, -1.78           # the tub's front (firewall) and rear
FLOOR, RFLOOR = 0.44, 0.56     # floor in front, the raised floor over the rear axle
FEND = 0.88                    # top of the front wings
FOUT = 0.89                    # outer edge of wings / flares / running boards
ARCH = 0.47                    # rear wheel arch radius
BON_Y1 = 1.71                  # the bonnet's nose
STEER_TILT = 38.0              # the column rises toward the driver at this angle from the horizontal
STEER_HUB = (-0.40, 0.02, 1.00)
EYE_Z = 1.28
PAINT, DARK, STEEL, METAL = 'j_paint', 'j_paint_dark', 'j_steel', 'j_metal'


# ================================================================== the wheel
def build_wheel():
    """In the frame of wheel.mesh: axle = X, outer face toward +X."""
    with xf(PX):                                        # lathe axis (local z) -> X
        tyre = [(0.212, -0.100), (0.232, -0.112), (0.262, -0.124), (0.300, -0.130), (0.335, -0.127), (0.360, -0.118),
                (0.376, -0.106), (0.384, -0.088), (0.385, -0.045), (0.385, 0.0)]
        tyre = tyre + [(r, -x) for r, x in reversed(tyre[:-1])]
        lathe('j_tyre', tyre, 64, uref=WR)
        # the tread: two staggered rows of lugs that run from the middle over the shoulder
        N = 22; pitch = TAU / N
        ST = [(0.004, 0.400, 0.378, 0.30), (0.086, 0.400, 0.378, 0.0), (0.114, 0.386, 0.360, -0.08), (0.1285, 0.352, 0.334, -0.12)]
        for s in (1, -1):
            for k in range(N):
                th0 = pitch * (k + (0.5 if s < 0 else 0.0))
                def P(i, side, top):
                    x, rt, rb, sk = ST[i]
                    th = th0 + (sk + side * 0.27) * pitch; r = rt if top else rb
                    return Vector((r * math.cos(th), r * math.sin(th), s * x)), (th * WR, s * x)
                for i in range(3):
                    c8 = [P(a, sd, tp) for a in (i, i + 1) for sd in (-1, 1) for tp in (0, 1)]
                    ref = sum((c[0] for c in c8), Vector((0, 0, 0))) / 8.0
                    def Q(a, b, c, d):
                        quad_out('j_tyre', a[0], b[0], c[0], d[0], ref, [a[1], b[1], c[1], d[1]])
                    Q(P(i, -1, 1), P(i, 1, 1), P(i + 1, 1, 1), P(i + 1, -1, 1))             # top
                    for sd in (-1, 1): Q(P(i, sd, 0), P(i, sd, 1), P(i + 1, sd, 1), P(i + 1, sd, 0))
                    if i == 0: Q(P(0, -1, 0), P(0, 1, 0), P(0, 1, 1), P(0, -1, 1))
                    if i == 2: Q(P(3, -1, 0), P(3, 1, 0), P(3, 1, 1), P(3, -1, 1))
        # the rim: outer face (a dished disc, the hub cap), then the back
        lathe('j_rim', [(0.232, 0.100), (0.229, 0.108), (0.216, 0.108), (0.205, 0.095), (0.200, 0.062), (0.186, 0.046),
                        (0.150, 0.040), (0.112, 0.052), (0.076, 0.056), (0.058, 0.056), (0.058, 0.076), (0.050, 0.089),
                        (0.030, 0.097), (0.0, 0.099)], 40)
        lathe('j_rim', [(0.150, 0.020), (0.195, 0.030), (0.195, -0.095), (0.216, -0.108), (0.229, -0.108), (0.232, -0.100)], 40)
        lathe('j_steel', [(0.0, -0.088), (0.060, -0.088), (0.150, -0.080), (0.150, 0.020)], 32)      # brake drum
        for k in range(5):
            a = TAU * (k + 0.25) / 5
            c = Vector((0.092 * math.cos(a), 0.092 * math.sin(a), 0.048))
            cyl('j_metal', c, c + Vector((0, 0, 0.022)), 0.0135, 6, (False, True))
            cyl('j_metal', c, c + Vector((0, 0, 0.028)), 0.006, 6, (False, True))
        # ventilation slots in the disc: dark dimples
        for k in range(5):
            a = TAU * (k + 0.75) / 5
            c = Vector((0.150 * math.cos(a), 0.150 * math.sin(a), 0.0405))
            cyl('j_steel', c, c + Vector((0, 0, 0.002)), 0.016, 10, (False, True))


# ================================================================== the steering wheel
def build_steering():
    """In the frame of steering.mesh after RX(90): axis = Blender -Y = engine +Z."""
    with xf(RX(math.radians(90.0))):                    # local z (toward the driver) -> Blender -Y
        R = 0.19
        pipe('j_rubber', [(R * math.cos(TAU * i / 36), R * math.sin(TAU * i / 36), 0.0) for i in range(36)], 0.015, 8, closed=True)
        for a in (90.0, 210.0, 330.0):
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            cyl(METAL, (0.032 * c, 0.032 * s, -0.045), (0.180 * c, 0.180 * s, -0.002), 0.008, 8, False)
        lathe(STEEL, [(0.0, -0.075), (0.024, -0.075), (0.042, -0.062), (0.042, -0.036), (0.032, -0.030)], 16)
        lathe('j_dash', [(0.032, -0.030), (0.029, -0.021), (0.0, -0.018)], 16)


# ================================================================== the body
def lamp(pos, r, g_lens, front=True, depth=0.03):
    with xf(T(*pos) @ RX(math.radians(-90.0 if front else 90.0))):
        lathe(METAL, [(r * 1.14, 0.0), (r * 1.14, depth * 0.8), (r, depth)], 20)
        lathe(g_lens, [(r, depth), (r * 0.86, depth + r * 0.20), (r * 0.5, depth + r * 0.36), (0.0, depth + r * 0.42)], 20)


def gauge(x, z, r, needle_deg):
    with xf(T(x, 0.30, z) @ RX(math.radians(90.0))):    # local z -> toward the driver (-Y)
        lathe(METAL, [(r + 0.008, 0.0), (r + 0.008, 0.009), (r, 0.011)], 20)
        lathe('j_gauge', [(r, 0.006), (0.0, 0.006)], 20)
        for i in range(9):
            a = math.radians(225.0 - 270.0 * i / 8)
            c, s = math.cos(a), math.sin(a)
            obox('j_dash', (r * 0.74 * c, r * 0.74 * s, 0.0065), (r * 0.93 * c, r * 0.93 * s, 0.0065), r * 0.07, 0.001)
        a = math.radians(needle_deg)
        obox('j_dash', (0, 0, 0.0075), (r * 0.80 * math.cos(a), r * 0.80 * math.sin(a), 0.0075), r * 0.07, 0.0015)
        cyl('j_dash', (0, 0, 0.006), (0, 0, 0.010), r * 0.13, 8, (False, True))


def seat_pads(x0, x1, y0, y1, z0, z1, n, c=0.02):
    w = (x1 - x0) / n
    for i in range(n):
        cbox('j_seat', (x0 + w * i + 0.003, y0, z0), (x0 + w * (i + 1) - 0.003, y1, z1), c)


def front_seat(xc):
    # a pan on runners, a tube hoop for the back, ribbed canvas cushions
    box(STEEL, (xc - 0.21, -0.61, 0.475), (xc + 0.21, -0.15, 0.50))
    for sx in (-0.19, 0.19):
        obox(STEEL, (xc + sx, -0.60, 0.458), (xc + sx, -0.16, 0.458), 0.03, 0.036)
    seat_pads(xc - 0.22, xc + 0.22, -0.63, -0.14, 0.50, 0.63, 4)
    pipe(STEEL, fillet([(xc - 0.215, -0.60, 0.48), (xc - 0.215, -0.705, 1.13), (xc + 0.215, -0.705, 1.13), (xc + 0.215, -0.60, 0.48)], 0.06, 4), 0.013, 8)
    with xf(T(xc, -0.605, 0.61) @ RX(math.radians(10.0))):
        seat_pads(-0.21, 0.21, -0.065, 0.02, 0.0, 0.50, 4)
        box(STEEL, (-0.20, -0.075, 0.02), (0.20, -0.06, 0.48))          # the back's sheet


def leaf_spring(xs, yc):
    for k in range(3):
        L = 0.50 * (1.0 - 0.27 * k); zc = 0.355 - 0.013 * k
        top = [(yc - L + 2 * L * i / 8, zc + 0.085 * ((-L + 2 * L * i / 8) / 0.5) ** 2) for i in range(9)]
        poly = top + [(y, z - 0.011) for y, z in reversed(top)]
        with xf(PX): prism(STEEL, poly, xs - 0.03, xs + 0.03)
    for e in (-1, 1):
        y = yc + e * 0.50
        cyl(STEEL, (xs - 0.036, y, 0.434), (xs + 0.036, y, 0.434), 0.018, 8)
        box(STEEL, (xs - 0.034, y - 0.012, 0.43), (xs - 0.028, y + 0.012, 0.50))
        box(STEEL, (xs + 0.028, y - 0.012, 0.43), (xs + 0.034, y + 0.012, 0.50))
    box(STEEL, (xs - 0.045, yc - 0.05, 0.305), (xs + 0.045, yc + 0.05, 0.318))          # the U-bolt plate
    for dx in (-0.038, 0.038):
        for dy in (-0.04, 0.04):
            cyl(METAL, (xs + dx, yc + dy, 0.296), (xs + dx, yc + dy, 0.44), 0.006, 6)


def axle(yc, diff_x):
    cyl(STEEL, (-0.60, yc, WR), (0.60, yc, WR), 0.04, 12)
    ball(STEEL, (diff_x, yc, WR), 0.115, 14, 8, (1.0, 1.15, 1.0))
    for sx in (-1, 1):
        cyl(STEEL, (sx * 0.52, yc, WR), (sx * 0.615, yc, WR), 0.075, 12)       # hub carrier
        cyl(STEEL, (diff_x + sx * 0.10, yc, WR), (diff_x + sx * 0.20, yc, WR), 0.06, 12, False, 0.04)
    # the cover's bolt ring
    for i in range(8):
        a = TAU * i / 8
        bolt((diff_x + 0.085 * math.cos(a), yc - math.copysign(0.115, yc), WR + 0.085 * math.sin(a)), (0, -math.copysign(1.0, yc), 0), 0.008, 0.012)


def build_body():
    rad = math.radians

    def side_g(n, c):                                   # olive outside, darker inside the tub
        return DARK if n.x * (1.0 if c.x > 0 else -1.0) < -0.5 else PAINT

    # ---------------------------------------------------------------- the tub
    side = [(YF, SILL), (YF, BELT), (0.30, BELT), (0.21, 0.76), (0.16, 0.72), (-0.44, 0.72), (-0.51, 0.77), (-0.62, BELT),
            (YR, BELT), (YR, SILL)]
    side += [(AXR + ARCH * math.cos(math.pi - math.pi * i / 14), WR + ARCH * math.sin(math.pi * i / 14)) for i in range(15)]
    for m in (IDN, MIR):
        with xf(m @ PX): prism(side_g, side, TI, TW, 0.015, (False, True))
    box(lambda n, c: DARK if n.y > 0.5 else PAINT, (-TI, YR, SILL), (TI, YR + 0.03, BELT))          # rear panel
    cbox(PAINT, (-0.60, YR - 0.015, 0.47), (0.60, YR + 0.005, 1.015), 0.012)                        # the tailgate, proud of it
    for z in (0.60, 0.93):
        cbox(PAINT, (-0.56, YR - 0.023, z - 0.02), (0.56, YR - 0.010, z + 0.02), 0.008)             # its pressed ribs
    for sx in (-1, 1):
        cyl(METAL, (sx * 0.44 - 0.05, YR - 0.02, 0.47), (sx * 0.44 + 0.05, YR - 0.02, 0.47), 0.014, 8)   # hinges
        box(METAL, (sx * 0.44 - 0.04, YR - 0.019, 0.47), (sx * 0.44 + 0.04, YR - 0.013, 0.53))
        box(METAL, (sx * 0.63 - 0.025, YR - 0.012, 0.955), (sx * 0.63 + 0.025, YR - 0.0, 0.985))    # latches
        cyl(METAL, (sx * 0.61, YR - 0.012, 0.97), (sx * 0.61, YR - 0.03, 0.97), 0.008, 6)
    box(DARK, (-TI, YF - 0.03, SILL), (TI, YF, BELT))                                               # firewall
    box(lambda n, c: PAINT if n.z > 0.5 else DARK, (-TI, 0.30, 1.01), (TI, YF, BELT), '+y')         # cowl top
    box('j_dash', (-TI, 0.30, 0.80), (TI, 0.33, 1.01), '+z')                                        # dashboard
    box('j_dash', (-TI, 0.33, 0.80), (TI, YF - 0.03, 0.82), '-y+y')                                 # the shelf under it
    box(DARK, (-TI, -0.72, SILL), (TI, YF - 0.03, FLOOR))                                           # front floor
    box(DARK, (-0.56, -0.72, SILL), (0.56, -0.70, RFLOOR), '+z')                                    # the step up
    box(DARK, (-0.56, YR + 0.03, RFLOOR - 0.04), (0.56, -0.70, RFLOOR))                             # rear floor
    for sx in (-1, 1):                                                                              # wheel housings
        box(DARK, (sx * 0.56, YR + 0.03, SILL), (sx * TI, -0.70, 0.92), '-z-y' + ('+x' if sx > 0 else '-x'))
        box(DARK, (sx * 0.59, -1.70, 0.36), (sx * (TW - 0.004), -0.715, 0.90), '-z' + ('+x' if sx > 0 else '-x'), inv=True)
    # pressed ribs in the floors
    for i in range(9):
        x = -0.48 + 0.12 * i
        box(DARK, (x - 0.02, YR + 0.08, RFLOOR), (x + 0.02, -0.76, RFLOOR + 0.006), '-z')
    for sx in (-1, 1):
        for i in range(4):
            x = sx * (0.24 + 0.12 * i)
            box(DARK, (x - 0.02, -0.66, FLOOR), (x + 0.02, 0.20, FLOOR + 0.006), '-z')
    with xf(PY): prism(DARK, [(-0.15, FLOOR), (0.15, FLOOR), (0.10, 0.515), (-0.10, 0.515)], -0.45, YF - 0.03)   # tunnel
    # running boards, with tread strips
    for sx in (-1, 1):
        cbox(PAINT, (sx * 0.70, -0.735, 0.40), (sx * FOUT, 0.71, 0.43), 0.008)
        for i in range(3):
            x = sx * (0.755 + 0.045 * i)
            box('j_rubber', (x - 0.012, -0.68, 0.43), (x + 0.012, 0.62, 0.436), '-z')
    # rear wheel-arch flares and mud flaps
    band = [(AXR + (ARCH + 0.03) * math.cos(math.pi * i / 14), WR + (ARCH + 0.03) * math.sin(math.pi * i / 14)) for i in range(15)]
    band += [(AXR + ARCH * math.cos(math.pi - math.pi * i / 14), WR + ARCH * math.sin(math.pi * i / 14)) for i in range(15)]
    def flare_g(n, c):
        return DARK if abs(n.x) < 0.5 and (n.y * (c.y - AXR) + n.z * (c.z - WR)) < 0.0 else PAINT
    for m in (IDN, MIR):
        with xf(m @ PX): prism(flare_g, band, TW - 0.005, FOUT, 0.012, (False, True))
        with xf(m):
            box('j_rubber', (0.60, AXR - ARCH - 0.035, 0.24), (0.885, AXR - ARCH - 0.022, 0.42))
            for i in range(14):                                             # rivets round the flare
                a = math.pi * (i + 0.5) / 14
                bolt((TW + 0.03, AXR + (ARCH + 0.03) * math.cos(a), WR + (ARCH + 0.03) * math.sin(a)), (math.cos(a) * 0, math.cos(a), math.sin(a)), 0.008, 0.005)
    # rivet rows along the tub
    for sx in (-1, 1):
        bolts((sx * TW, -0.66, 0.995), (sx * TW, YR + 0.04, 0.995), 10, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * TW, YF - 0.02, 0.995), (sx * TW, 0.32, 0.995), 3, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * TW, YF - 0.035, 0.47), (sx * TW, YF - 0.035, 0.95), 5, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * TW, 0.10, 0.455), (sx * TW, -0.68, 0.455), 7, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * TW, YR + 0.035, 0.47), (sx * TW, YR + 0.035, 0.95), 5, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * 0.655, YR, 0.50), (sx * 0.655, YR, 0.90), 4, (0, -1, 0), 0.008, 0.005)
        # a grab handle on the flank and a lashing cleat
        pipe(STEEL, fillet([(sx * TW, -0.74, 0.93), (sx * (TW + 0.04), -0.74, 0.93), (sx * (TW + 0.04), -0.98, 0.93), (sx * TW, -0.98, 0.93)], 0.02, 3), 0.009, 6, False)
        pipe(STEEL, fillet([(sx * TW, -1.42, 0.93), (sx * (TW + 0.04), -1.42, 0.93), (sx * (TW + 0.04), -1.66, 0.93), (sx * TW, -1.66, 0.93)], 0.02, 3), 0.009, 6, False)
    cyl(METAL, (-TW, -0.665, 0.90), (-TW - 0.022, -0.665, 0.90), 0.036, 12)                        # fuel filler cap
    cyl(METAL, (-TW - 0.022, -0.665, 0.90), (-TW - 0.030, -0.665, 0.90), 0.022, 8)

    # ---------------------------------------------------------------- the bonnet and the engine bay
    def sect(y, hw, zt):
        return [Vector(p) for p in ((hw, y, 0.86), (hw, y, zt - 0.015), (hw - 0.015, y, zt), (hw * 0.5, y, zt + 0.014), (0.0, y, zt + 0.02),
                                    (-hw * 0.5, y, zt + 0.014), (-hw + 0.015, y, zt), (-hw, y, zt - 0.015), (-hw, y, 0.86))]
    S = [sect(YF, 0.66, BELT), sect(BON_Y1 - 0.015, 0.52, 1.015), sect(BON_Y1, 0.505, 1.0)]
    for a, b in ((S[0], S[1]), (S[1], S[2])):
        for i in range(8):
            quad_out(PAINT, a[i], a[i + 1], b[i + 1], b[i], (0.0, (a[0].y + b[0].y) / 2, 0.80))
    for i in range(1, 7): tri(PAINT, (S[2][0], S[2][i], S[2][i + 1]), [(0, 1, 0)] * 3)
    tri(PAINT, (S[2][0], S[2][7], S[2][8]), [(0, 1, 0)] * 3)
    cyl(PAINT, (0, YF + 0.01, BELT + 0.022), (0, BON_Y1 - 0.03, 1.015 + 0.022), 0.008, 6)           # the centre hinge line
    for sx in (-1, 1):
        for y in (0.74, 1.12, 1.48):                                                                # side hinges
            hw = 0.66 - 0.14 * (y - YF) / (BON_Y1 - 0.015 - YF); zt = BELT - 0.025 * (y - YF) / (BON_Y1 - 0.015 - YF)
            dy = Vector((-0.14 * 0.12 / 1.175, 0.12, -0.025 * 0.12 / 1.175))
            p = Vector((sx * (hw + 0.002), y, zt - 0.012))
            cyl(METAL, p - Vector((sx * dy.x, dy.y, dy.z)) * 0.5, p + Vector((sx * dy.x, dy.y, dy.z)) * 0.5, 0.010, 8)
            box(METAL, (sx * (hw - 0.001), y - 0.05, zt - 0.075), (sx * (hw + 0.005), y + 0.05, zt - 0.015))
        for y in (0.95, 1.45):                                                                      # bonnet catches
            hw = 0.66 - 0.14 * (y - YF) / (BON_Y1 - 0.015 - YF)
            box(METAL, (sx * (hw - 0.001), y - 0.012, 0.87), (sx * (hw + 0.012), y + 0.012, 0.95))
            cyl(METAL, (sx * (hw + 0.012), y - 0.03, 0.945), (sx * (hw + 0.012), y + 0.03, 0.945), 0.007, 6)
    box(DARK, (-0.42, YF, 0.50), (0.42, BON_Y1 - 0.01, 0.86), '+z')                                 # engine bay
    # front wings: a flat top over the wheel, an apron in front, a skirt down to the running board
    wing = thick([(1.765, 0.72), (1.65, FEND), (0.80, FEND), (0.70, 0.43)], 0.02)
    for m in (IDN, MIR):
        with xf(m @ PX): prism(lambda n, c: DARK if n.z < -0.3 else PAINT, wing, 0.53, FOUT, 0.015, (False, True))
        with xf(m):
            box(DARK, (0.42, 0.78, 0.845), (0.67, 1.60, 0.875))                                     # closes the wing to the bay
            box(lambda n, c: PAINT if abs(n.x) > 0.5 else DARK, (0.42, YF, 0.45), (0.62, 0.80, 0.872))       # the box behind the wheel well
            box(DARK, (0.42, 1.60, 0.845), (0.535, BON_Y1, 0.875))
            box(PAINT, (FOUT - 0.012, 0.80, FEND - 0.055), (FOUT, 1.63, FEND - 0.012))              # the wing's turned-down edge
            bolts((0.60, 1.56, FEND), (0.60, 0.86, FEND), 5, (0, 0, 1), 0.008, 0.005)
            bolts((0.84, 1.56, FEND), (0.84, 0.86, FEND), 5, (0, 0, 1), 0.008, 0.005)

    # ---------------------------------------------------------------- the grille
    GY0, GY1 = BON_Y1, BON_Y1 + 0.03
    for sx in (-1, 1):
        cbox(PAINT, (sx * 0.25, GY0, 0.58), (sx * 0.53, GY1, 1.0), 0.01)
        lamp((sx * 0.39, GY1, 0.855), 0.082, 'j_light', True, 0.03)
        lamp((sx * 0.39, GY1, 0.675), 0.034, 'j_light_amber', True, 0.018)
        bolts((sx * 0.515, GY1, 0.61), (sx * 0.515, GY1, 0.97), 5, (0, 1, 0), 0.007, 0.004)
    cbox(PAINT, (-0.25, GY0, 0.93), (0.25, GY1, 1.0), 0.01)
    cbox(PAINT, (-0.25, GY0, 0.58), (0.25, GY1, 0.64), 0.01)
    for i in range(7):
        x = -0.1875 + 0.0625 * i
        cbox(PAINT, (x - 0.015, GY0 + 0.004, 0.63), (x + 0.015, GY1 - 0.003, 0.94), 0.006)
    box(STEEL, (-0.25, GY0 - 0.004, 0.63), (0.25, GY0 + 0.008, 0.94))                               # the radiator behind the slots
    bolts((-0.20, GY1, 0.985), (0.20, GY1, 0.985), 5, (0, 1, 0), 0.007, 0.004)

    # ---------------------------------------------------------------- frame, bumpers
    rail = [(-1.86, 0.49), (-0.78, 0.49), (-0.58, 0.35), (0.56, 0.35), (0.76, 0.51), (1.86, 0.51)]
    rail_poly = [(y, z + 0.04) for y, z in rail] + [(y, z - 0.04) for y, z in reversed(rail)]
    for m in (IDN, MIR):
        with xf(m @ PX): prism(STEEL, rail_poly, 0.31, 0.37)
    for y, z in ((-1.72, 0.49), (-0.94, 0.49), (-0.40, 0.35), (0.30, 0.35), (1.00, 0.51), (1.76, 0.51)):
        box(STEEL, (-0.31, y - 0.03, z - 0.03), (0.31, y + 0.03, z + 0.03))
    fb = [(1.82, 0.45), (1.90, 0.45), (1.90, 0.57), (1.82, 0.57), (1.82, 0.556), (1.886, 0.556), (1.886, 0.464), (1.82, 0.464)]
    with xf(PX): prism(STEEL, fb, -0.80, 0.80)
    rb = [(-1.82, 0.43), (-1.90, 0.43), (-1.90, 0.55), (-1.82, 0.55), (-1.82, 0.536), (-1.886, 0.536), (-1.886, 0.444), (-1.82, 0.444)]
    with xf(PX): prism(STEEL, rb, -0.74, 0.74)
    for sx in (-1, 1):
        # front tow hooks, standing on the bumper
        x = sx * 0.34
        box(STEEL, (x - 0.035, 1.825, 0.57), (x + 0.035, 1.895, 0.58))
        pipe(STEEL, fillet([(x, 1.84, 0.575), (x, 1.84, 0.635), (x, 1.885, 0.665), (x, 1.925, 0.635), (x, 1.915, 0.60)], 0.02, 3), 0.012, 8)
        bolts((x - 0.05, 1.90, 0.51), (x + 0.05, 1.90, 0.51), 2, (0, 1, 0), 0.010, 0.007)
        bolts((sx * 0.66, 1.90, 0.51), (sx * 0.74, 1.90, 0.51), 2, (0, 1, 0), 0.010, 0.007)
        if sx < 0: bolts((sx * 0.30, -1.90, 0.49), (sx * 0.38, -1.90, 0.49), 2, (0, -1, 0), 0.010, 0.007)
        # rear lifting rings
        x = sx * 0.62
        box(STEEL, (x - 0.03, -1.908, 0.45), (x + 0.03, -1.90, 0.53))
        pipe(STEEL, [(x + 0.028 * math.cos(TAU * i / 12), -1.925, 0.47 + 0.028 * math.sin(TAU * i / 12)) for i in range(12)], 0.008, 6, closed=True)
        cyl(STEEL, (x, -1.90, 0.50), (x, -1.935, 0.50), 0.012, 6)
    # the towing pintle
    box(STEEL, (-0.06, -1.915, 0.44), (0.06, -1.90, 0.54))
    cyl(STEEL, (0, -1.90, 0.49), (0, -1.96, 0.49), 0.022, 8)
    pipe(STEEL, fillet([(0, -1.955, 0.49), (0, -1.985, 0.49), (0, -1.985, 0.545), (0, -1.945, 0.545)], 0.015, 3), 0.014, 8)
    # the number plate: a blank panel in a bright frame
    box(DARK, (0.20, -1.906, 0.435), (0.50, -1.90, 0.545), '+y')
    for a, b in (((0.20, 0.435), (0.50, 0.443)), ((0.20, 0.537), (0.50, 0.545)), ((0.20, 0.435), (0.208, 0.545)), ((0.492, 0.435), (0.50, 0.545))):
        box(METAL, (a[0], -1.909, a[1]), (b[0], -1.906, b[1]), '+y')

    # ---------------------------------------------------------------- running gear
    axle(AXF, 0.16); axle(AXR, 0.0)
    for sx in (-1, 1):
        leaf_spring(sx * 0.34, AXF); leaf_spring(sx * 0.34, AXR)
        cyl(STEEL, (sx * 0.26, AXR + 0.07, 0.37), (sx * 0.26, AXR + 0.24, 0.51), 0.022, 8)          # dampers
        cyl(STEEL, (sx * 0.26, AXR + 0.24 * 0.5 + 0.035, 0.44), (sx * 0.26, AXR + 0.24, 0.51), 0.028, 8)
        cyl(STEEL, (sx * 0.44, AXF - 0.07, 0.37), (sx * 0.41, AXF - 0.22, 0.56), 0.022, 8)
        cyl(STEEL, (sx * 0.425, AXF - 0.145, 0.465), (sx * 0.41, AXF - 0.22, 0.56), 0.028, 8)
        cyl(STEEL, (sx * 0.56, AXF + 0.15, 0.35), (sx * 0.56, AXF, 0.37), 0.012, 6)                 # steering arms
    cyl(STEEL, (-0.56, AXF + 0.15, 0.35), (0.56, AXF + 0.15, 0.35), 0.012, 8)                       # tie rod
    cbox(STEEL, (-0.15, 0.62, 0.36), (0.15, 1.05, 0.52), 0.03)                                      # sump
    cyl(STEEL, (0, 0.44, 0.40), (0, 0.62, 0.42), 0.13, 12, True, 0.16)                              # bell housing
    cbox(STEEL, (-0.12, 0.06, 0.27), (0.12, 0.46, 0.41), 0.03)                                      # gearbox
    cbox(STEEL, (-0.10, -0.28, 0.25), (0.25, 0.08, 0.41), 0.03)                                     # transfer case
    cyl(STEEL, (0.0, -0.28, 0.33), (0.0, AXR + 0.21, WR), 0.026, 10)                                # rear propeller shaft
    cyl(STEEL, (0.0, AXR + 0.21, WR), (0.0, AXR + 0.10, WR), 0.05, 10)
    cyl(STEEL, (0.0, -0.28, 0.33), (0.0, -0.36, 0.337), 0.045, 10)
    cyl(STEEL, (0.16, 0.08, 0.33), (0.16, AXF - 0.21, WR), 0.022, 10)                               # front propeller shaft
    cyl(STEEL, (0.16, AXF - 0.21, WR), (0.16, AXF - 0.10, WR), 0.05, 10)
    # exhaust: over the rail, a silencer under the right-hand sill, then over the rear axle
    pipe(STEEL, fillet([(0.36, 0.66, 0.50), (0.50, 0.50, 0.36), (0.50, 0.25, 0.325)], 0.05, 3), 0.022, 8)
    cyl(STEEL, (0.50, 0.25, 0.325), (0.50, -0.35, 0.325), 0.066, 14)
    for y in (0.22, -0.32): cyl(STEEL, (0.50, y - 0.012, 0.325), (0.50, y + 0.012, 0.325), 0.071, 14)
    pipe(STEEL, fillet([(0.50, -0.35, 0.325), (0.50, -0.74, 0.345), (0.50, -0.95, 0.485), (0.50, -1.45, 0.485), (0.50, -1.62, 0.36), (0.50, -1.86, 0.355)], 0.06, 3), 0.022, 8)
    cyl(METAL, (0.50, -1.85, 0.355), (0.50, -1.95, 0.354), 0.026, 10)
    cyl(STEEL, (0.50, -1.948, 0.354), (0.50, -1.90, 0.354), 0.020, 10, (True, False))               # the dark bore
    box(STEEL, (0.37, -1.40, 0.49), (0.51, -1.37, 0.52))                                            # its hanger
    cbox(STEEL, (-0.29, -1.74, 0.36), (0.29, -1.40, 0.50), 0.025)                                   # fuel tank
    for x in (-0.18, 0.18): box(STEEL, (x - 0.02, -1.745, 0.352), (x + 0.02, -1.395, 0.53))         # its straps
    cyl(METAL, (0.0, -1.57, 0.36), (0.0, -1.57, 0.345), 0.02, 6)                                    # drain plug

    # ---------------------------------------------------------------- windscreen (hinged on the cowl, folds forward)
    H = 0.71
    with xf(T(0, 0.485, BELT) @ RX(rad(8.0))):
        for sx in (-1, 1):
            cbox(PAINT, (sx * 0.655, -0.02, 0.0), (sx * 0.70, 0.02, H), 0.006)
            quad2('j_glass', (sx * 0.015, 0.0, 0.05), (sx * 0.655, 0.0, 0.05), (sx * 0.655, 0.0, H - 0.045), (sx * 0.015, 0.0, H - 0.045))
            cyl(METAL, (sx * 0.54, 0.03, 0.0), (sx * 0.66, 0.03, 0.0), 0.013, 8)                    # hinge barrels
            box(METAL, (sx * 0.56, 0.018, 0.0), (sx * 0.64, 0.024, 0.06))
            cyl(METAL, (sx * 0.70, 0.0, 0.10), (sx * 0.725, 0.0, 0.10), 0.02, 8)                    # the clamp knobs
            # a wiper hanging from the top rail
            box(STEEL, (sx * 0.34 - 0.03, 0.02, H - 0.04), (sx * 0.34 + 0.03, 0.045, H - 0.005))
            cyl(METAL, (sx * 0.34, 0.03, H - 0.022), (sx * 0.34, 0.05, H - 0.022), 0.008, 6)
            obox(METAL, (sx * 0.34, 0.048, H - 0.022), (sx * 0.34 - 0.07, 0.022, 0.34), 0.010, 0.004, (0, 1, 0))
            obox('j_rubber', (sx * 0.34 - 0.035, 0.012, 0.52), (sx * 0.34 - 0.105, 0.012, 0.16), 0.012, 0.016, (0, 1, 0))
            # wing mirror
            pipe(STEEL, [(sx * 0.70, 0.0, 0.36), (sx * 0.78, -0.01, 0.40), (sx * 0.80, -0.01, 0.42)], 0.008, 6)
            cyl(STEEL, (sx * 0.85, 0.008, 0.44), (sx * 0.85, -0.012, 0.44), 0.066, 14)
            cyl(METAL, (sx * 0.85, -0.012, 0.44), (sx * 0.85, -0.0135, 0.44), 0.058, 14, (False, True))
            bolts((sx * 0.6775, 0.02, 0.12), (sx * 0.6775, 0.02, H - 0.06), 5, (0, 1, 0), 0.006, 0.004)
        cbox(PAINT, (-0.655, -0.02, H - 0.045), (0.655, 0.02, H), 0.006)
        cbox(PAINT, (-0.655, -0.02, 0.0), (0.655, 0.02, 0.05), 0.006)
        cbox(PAINT, (-0.015, -0.018, 0.05), (0.015, 0.018, H - 0.045), 0.005)

    # ---------------------------------------------------------------- roll bar
    pipe(STEEL, fillet([(0.63, -0.80, 0.92), (0.63, -0.80, 1.70), (-0.63, -0.80, 1.70), (-0.63, -0.80, 0.92)], 0.16, 6), 0.028, 10)
    for sx in (-1, 1):
        cbox(STEEL, (sx * 0.575, -0.86, 0.92), (sx * 0.685, -0.74, 0.932), 0.004)
        for dx in (-0.04, 0.04):
            for dy in (-0.045, 0.045): bolt((sx * 0.63 + dx, -0.80 + dy, 0.932), (0, 0, 1), 0.008, 0.006)
        cyl(STEEL, (sx * 0.63, -0.82, 1.46), (sx * 0.63, -1.52, 0.925), 0.022, 10)
        cbox(STEEL, (sx * 0.585, -1.58, 0.92), (sx * 0.675, -1.47, 0.932), 0.004)

    # ---------------------------------------------------------------- seats
    front_seat(-0.40); front_seat(0.40)
    box(STEEL, (-0.53, -1.62, RFLOOR), (0.53, -1.20, 0.60))                                          # bench box
    seat_pads(-0.54, 0.54, -1.64, -1.17, 0.60, 0.72, 6)
    with xf(T(0, -1.63, 0.71) @ RX(rad(8.0))):
        seat_pads(-0.54, 0.54, -0.07, 0.015, 0.0, 0.39, 6)
        box(STEEL, (-0.53, -0.08, 0.0), (0.53, -0.065, 0.37))

    # ---------------------------------------------------------------- controls
    t = rad(STEER_TILT)
    hub = Vector(STEER_HUB); A = Vector((0.0, -math.cos(t), math.sin(t)))      # the column's axis, toward the driver
    cyl(STEEL, hub - A * 0.60, hub - A * 0.07, 0.019, 10)
    cyl(STEEL, hub - A * 0.60, hub - A * 0.30, 0.026, 10)
    pd = hub - A * 0.34                                                         # a stay from the dash's shelf to the column
    box(STEEL, (pd.x - 0.035, pd.y - 0.015, pd.z), (pd.x + 0.035, pd.y + 0.015, 0.80))
    for x in (-0.58, -0.49):                                                    # clutch, brake
        with xf(T(x, 0.36, 0.56) @ RX(rad(-35.0))):
            cbox('j_rubber', (-0.032, -0.012, -0.03), (0.032, 0.0, 0.03), 0.004)
            box(STEEL, (-0.03, 0.0, -0.028), (0.03, 0.004, 0.028))
        pipe(STEEL, [(x, 0.365, 0.56), (x, 0.43, 0.60), (x, 0.49, 0.74)], 0.008, 6)
    obox('j_rubber', (-0.30, 0.31, 0.455), (-0.30, 0.42, 0.60), 0.05, 0.012, (0, -1, 1))     # accelerator
    cyl(STEEL, (-0.30, 0.40, 0.56), (-0.30, 0.49, 0.58), 0.006, 6)
    # gear lever, transfer levers, handbrake
    cyl('j_rubber', (0.0, 0.12, 0.515), (0.0, 0.11, 0.565), 0.045, 12, (False, True), 0.014)
    cyl(STEEL, (0.0, 0.115, 0.55), (0.02, -0.03, 0.83), 0.009, 8)
    ball('j_rubber', (0.02, -0.03, 0.845), 0.026, 10, 7)
    for i, x in enumerate((0.065, 0.10)):
        cyl(STEEL, (x, -0.06, 0.515), (x + 0.01, -0.12 - 0.02 * i, 0.70), 0.007, 6)
        ball('j_rubber', (x + 0.01, -0.12 - 0.02 * i, 0.71), 0.018, 8, 6)
    box(STEEL, (0.155, -0.14, FLOOR), (0.185, -0.06, 0.48))
    obox(STEEL, (0.17, -0.10, 0.47), (0.17, -0.30, 0.59), 0.012, 0.022)
    cyl('j_rubber', (0.17, -0.29, 0.584), (0.17, -0.38, 0.638), 0.016, 8)
    # the dashboard: instruments in the middle, switches, a grab handle and data plates for the passenger
    gauge(0.0, 0.915, 0.058, 150.0)
    gauge(-0.125, 0.955, 0.030, 60.0); gauge(0.125, 0.955, 0.030, 120.0)
    gauge(-0.125, 0.872, 0.030, 200.0); gauge(0.125, 0.872, 0.030, 75.0)
    for i in range(4):
        x = -0.27 + 0.045 * i - (0.0 if i < 2 else 0.0)
        cyl(METAL, (x, 0.30, 0.855), (x, 0.275, 0.855), 0.011, 8)
    for x in (0.22, 0.245):
        cyl(METAL, (x, 0.30, 0.86), (x, 0.285, 0.86), 0.007, 6)
        obox(METAL, (x, 0.285, 0.86), (x, 0.268, 0.878), 0.006, 0.006)
    pipe(METAL, fillet([(0.32, 0.30, 0.965), (0.32, 0.25, 0.965), (0.60, 0.25, 0.965), (0.60, 0.30, 0.965)], 0.025, 3), 0.009, 8, False)
    for x0, x1, z0, z1 in ((0.32, 0.44, 0.84, 0.905), (0.47, 0.60, 0.84, 0.905)):
        box(METAL, (x0, 0.2985, z0), (x1, 0.30, z1), '+y')
        for bx in (x0 + 0.008, x1 - 0.008):
            for bz in (z0 + 0.008, z1 - 0.008): bolt((bx, 0.2985, bz), (0, -1, 0), 0.003, 0.002, 'j_dash')
    bolts((-0.66, 0.30, 0.985), (0.66, 0.30, 0.985), 9, (0, -1, 0), 0.006, 0.004)

    # ---------------------------------------------------------------- on the back: spare wheel, jerrycan, lights
    TG = YR - 0.015                                                             # the tailgate's face
    SPARE = (0.22, TG - 0.03 - WW / 2, 0.95)
    with xf(T(*SPARE) @ RZ(rad(-90.0))): build_wheel()                          # outer face to the rear
    cyl(STEEL, (SPARE[0], TG, SPARE[2]), (SPARE[0], TG - 0.08, SPARE[2]), 0.075, 12, False)
    cbox(STEEL, (SPARE[0] - 0.13, TG - 0.012, SPARE[2] - 0.13), (SPARE[0] + 0.13, TG, SPARE[2] + 0.13), 0.004)
    JX0, JX1, JY0, JY1, JZ0, JZ1 = -0.57, -0.23, TG - 0.16, TG - 0.01, 0.58, 1.01
    cbox(PAINT, (JX0, JY0, JZ0), (JX1, JY1, JZ1), 0.028)
    jm = (JY0 + JY1) / 2
    for dy in (-0.045, 0.0, 0.045):                                             # three handles
        pipe(PAINT, fillet([(JX0 + 0.065, jm + dy, JZ1 - 0.01), (JX0 + 0.065, jm + dy, JZ1 + 0.04), (JX0 + 0.215, jm + dy, JZ1 + 0.04), (JX0 + 0.215, jm + dy, JZ1 - 0.01)], 0.02, 3), 0.010, 6)
    cyl(PAINT, (JX1 - 0.06, jm, JZ1 - 0.02), (JX1 - 0.04, jm, JZ1 + 0.035), 0.030, 10)    # spout
    cyl(METAL, (JX1 - 0.04, jm, JZ1 + 0.035), (JX1 - 0.035, jm, JZ1 + 0.05), 0.034, 10)
    for a, b in (((JX0 + 0.045, JZ0 + 0.07), (JX1 - 0.045, JZ0 + 0.36)), ((JX1 - 0.045, JZ0 + 0.07), (JX0 + 0.045, JZ0 + 0.36))):    # the pressed cross
        obox(PAINT, (a[0], JY0 - 0.001, a[1]), (b[0], JY0 - 0.001, b[1]), 0.022, 0.008, (0, 1, 0))
    box(STEEL, (JX0 - 0.012, JY0 - 0.006, JZ0 + 0.20), (JX1 + 0.012, TG, JZ0 + 0.235))      # the strap
    box(METAL, (JX0 + 0.145, JY0 - 0.014, JZ0 + 0.195), (JX0 + 0.195, JY0 - 0.006, JZ0 + 0.24))
    box(STEEL, (JX0 - 0.012, JY0 - 0.006, JZ0 - 0.015), (JX1 + 0.012, TG, JZ0 + 0.03))      # the tray
    for sx in (-1, 1):
        lamp((sx * 0.648, YR, 0.60), 0.034, 'j_light_red', False, 0.02)
        box(STEEL, (sx * 0.648 - 0.045, YR - 0.004, 0.555), (sx * 0.648 + 0.045, YR, 0.645), '+y')

        cyl('j_light_red', (sx * 0.648, YR, 0.72), (sx * 0.648, YR - 0.008, 0.72), 0.024, 12, (False, True))    # reflectors
        cyl(METAL, (sx * 0.648, YR, 0.72), (sx * 0.648, YR - 0.005, 0.72), 0.029, 12, (False, True))

    # ---------------------------------------------------------------- small fittings
    for sx in (-1, 1):
        # marker lamps on the wings
        cyl(STEEL, (sx * 0.73, 1.46, 0.925), (sx * 0.73, 1.53, 0.925), 0.028, 10, (True, False))
        cyl(STEEL, (sx * 0.73, 1.49, FEND), (sx * 0.73, 1.49, 0.905), 0.010, 6, False)
        with xf(T(sx * 0.73, 1.53, 0.925) @ RX(rad(-90.0))):
            lathe('j_light', [(0.028, 0.0), (0.022, 0.008), (0.0, 0.012)], 10)
        # brush guards over the headlights
        for dx in (-0.04, 0.04):
            x = sx * 0.39 + dx
            pipe(STEEL, fillet([(x, GY1, 0.955), (x, GY1 + 0.115, 0.955), (x, GY1 + 0.115, 0.755), (x, GY1, 0.755)], 0.035, 3), 0.006, 6)
        pipe(STEEL, [(sx * 0.39 - 0.04, GY1 + 0.115, 0.855), (sx * 0.39 + 0.04, GY1 + 0.115, 0.855)], 0.006, 6)
        # rivets: round the door openings, along the bonnet's shoulders, across the rear panel
        bolts((sx * TW, -0.40, 0.69), (sx * TW, 0.14, 0.69), 6, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * TW, 0.27, 0.80), (sx * TW, 0.335, 1.0), 2, (sx, 0, 0), 0.008, 0.005)
        bolts((sx * TW, -0.565, 0.80), (sx * TW, -0.635, 1.0), 2, (sx, 0, 0), 0.008, 0.005)
        for i in range(8):
            s = (i + 0.5) / 8; y = YF + s * (BON_Y1 - 0.015 - YF)
            bolt((sx * (0.66 - 0.14 * s - 0.045), y, BELT - 0.025 * s + 0.0015), (0, 0, 1), 0.008, 0.005)
        bolts((sx * 0.625, YR, 0.80), (sx * 0.685, YR, 1.01), 2, (0, -1, 0), 0.008, 0.005)
        # rubber rests for the folded windscreen
        cyl('j_rubber', (sx * 0.30, 1.30, 1.025), (sx * 0.30, 1.30, 1.06), 0.022, 8, (False, True))
    # pioneer tools under the door openings: an axe on the right flank, a shovel on the left
    cyl(STEEL, (TW + 0.016, -0.40, 0.56), (TW + 0.016, 0.10, 0.56), 0.014, 8)
    cbox(METAL, (TW + 0.002, 0.03, 0.50), (TW + 0.03, 0.13, 0.62), 0.006)
    cyl(STEEL, (-TW - 0.016, -0.40, 0.56), (-TW - 0.016, 0.02, 0.56), 0.014, 8)
    with xf(PX): prism(METAL, [(0.0, 0.535), (0.03, 0.49), (0.17, 0.50), (0.20, 0.56), (0.17, 0.62), (0.03, 0.63), (0.0, 0.585)], -TW - 0.014, -TW - 0.004)
    for sx, ys in ((1, (-0.30, -0.08)), (-1, (-0.30, -0.10))):
        for y in ys:
            box(METAL, (sx * TW, y - 0.012, 0.535), (sx * (TW + 0.034), y + 0.012, 0.585), '-x' if sx > 0 else '+x')

    hull = bounds(G)
    # ---------------------------------------------------------------- radio aerial on the rear left corner
    ax_, ay_ = -0.655, -1.735
    box(STEEL, (ax_ - 0.035, ay_ - 0.03, BELT), (ax_ + 0.035, ay_ + 0.03, BELT + 0.01))
    cyl(STEEL, (ax_, ay_, BELT + 0.01), (ax_, ay_, BELT + 0.05), 0.022, 10)
    for i in range(5):
        cyl(STEEL, (ax_, ay_, BELT + 0.05 + 0.014 * i), (ax_, ay_, BELT + 0.062 + 0.014 * i), 0.016, 8)   # the spring
    cyl(STEEL, (ax_, ay_, BELT + 0.12), (ax_, ay_ - 0.05, 2.05), 0.005, 5, True, 0.003)
    return hull


# ================================================================== build, check, write
build_wheel(); WHEEL = take()
build_steering(); STEER = take()
HULL = build_body(); BODY = take()


def ntri(groups): return sum(len(v) for v in groups.values()) // 3

wlo, whi = bounds(WHEEL)
for i in range(3): assert abs(wlo[i] + whi[i]) * 0.5 < 0.005, 'wheel not centred on the origin'
wr = max(math.hypot(p.y, p.z) for v in WHEEL.values() for p, _, _ in v)
assert abs(wr - WR) < 0.005, 'wheel radius %.4f' % wr
assert abs(whi.y - WR) < 0.005 and abs(whi.z - WR) < 0.005 and abs(wlo.y + WR) < 0.005 and abs(wlo.z + WR) < 0.005
assert abs((whi.x - wlo.x) - WW) < 0.005, 'tyre width %.4f' % (whi.x - wlo.x)
slo, shi = bounds(STEER)
for i in (0, 2): assert abs(slo[i] + shi[i]) * 0.5 < 0.005, 'steering wheel not centred'
blo, bhi = HULL                                          # the body without the aerial's whip
L_, W_, H_ = bhi.y - blo.y, bhi.x - blo.x, bhi.z
assert 3.9 * 0.9 <= L_ <= 3.9 * 1.1, 'length %.3f' % L_
assert 1.75 * 0.9 <= W_ <= 1.75 * 1.1, 'width %.3f' % W_
assert 1.75 * 0.9 <= H_ <= 1.75 * 1.1, 'height %.3f' % H_
assert 0.20 <= blo.z <= 0.30, 'ground clearance %.3f' % blo.z
NEED = {'body': ['j_paint', 'j_paint_dark', 'j_steel', 'j_metal', 'j_rim', 'j_tyre', 'j_seat', 'j_rubber', 'j_glass', 'j_light',
                 'j_light_red', 'j_light_amber', 'j_dash', 'j_gauge'],
        'wheel': ['j_tyre', 'j_rim', 'j_metal', 'j_steel'], 'steering': ['j_rubber', 'j_metal', 'j_steel', 'j_dash']}
for nm, grp in (('body', BODY), ('wheel', WHEEL), ('steering', STEER)):
    for g in NEED[nm]: assert len(grp.get(g, [])) >= 3, '%s: group %s is empty' % (nm, g)
    for g, v in grp.items(): assert len(v) >= 3 and len(g) < 32
    write_mesh(nm, grp)
assert 25000 <= ntri(BODY) <= 45000, 'body triangles %d' % ntri(BODY)
assert 3000 <= ntri(WHEEL) <= 6000, 'wheel triangles %d' % ntri(WHEEL)
assert ntri(STEER) < 1500, 'steering triangles %d' % ntri(STEER)

t_ = math.radians(STEER_TILT)
flo, fhi = bounds(BODY)


def cbx(lo, hi):
    return {'center': E(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)),
            'half_extents': [round((hi[0] - lo[0]) / 2, 4), round((hi[2] - lo[2]) / 2, 4), round((hi[1] - lo[1]) / 2, 4)]}

EYE_L = (-0.40, -0.42, EYE_Z); EYE_R = (0.40, -0.42, EYE_Z)
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
    'headlights': [E((-0.39, BON_Y1 + 0.095, 0.855)), E((0.39, BON_Y1 + 0.095, 0.855))],
    'tail_lights': [E((-0.648, YR - 0.035, 0.60)), E((0.648, YR - 0.035, 0.60))],
    'exit_left': E((-(TW + 1.2), -0.15, 0.0)), 'exit_right': E((TW + 1.2, -0.15, 0.0)),
    'bounds_min': E((blo.x, bhi.y, blo.z)), 'bounds_max': E((bhi.x, blo.y, bhi.z)),
    'bounds_note': 'bounds exclude the thin radio aerial, whose tip is at aerial_tip',
    'aerial_tip': E((-0.655, -1.785, 2.05)),
    'collision_boxes': [
        dict(name='tub', **cbx((-TW, YR - 0.02, SILL), (TW, YF, BELT))),
        dict(name='bonnet', **cbx((-0.66, YF, 0.50), (0.66, BON_Y1 + 0.03, 1.06))),
        dict(name='wings_bumper', **cbx((-FOUT, 0.70, 0.43), (FOUT, 1.90, FEND))),
        dict(name='windscreen', **cbx((-0.70, 0.36, BELT), (0.70, 0.52, 1.745))),
        dict(name='roll_bar', **cbx((-0.66, -0.84, BELT), (0.66, -0.76, 1.73))),
    ],
}
with open(os.path.join(OUT, 'jeep.json'), 'w') as f:
    json.dump(INFO, f, indent=2)
    f.write('\n')

print('JEEP body %d tris %s' % (ntri(BODY), {g: len(v) // 3 for g, v in sorted(BODY.items())}))
print('JEEP wheel %d tris %s' % (ntri(WHEEL), {g: len(v) // 3 for g, v in sorted(WHEEL.items())}))
print('JEEP steering %d tris %s' % (ntri(STEER), {g: len(v) // 3 for g, v in sorted(STEER.items())}))
print('JEEP size L %.3f W %.3f H %.3f clearance %.3f | wheel.mesh: axle = engine X, outer face +X | steering.mesh: axis = engine +Z, '
      'untilted; in the car the column axis is (0, %.4f, %.4f), tilt %.0f deg above the horizontal'
      % (L_, W_, H_, blo.z, math.sin(t_), math.cos(t_), STEER_TILT))


# ================================================================== preview renders (JEEP_PREVIEW=1)
def preview():
    COLS = {'j_paint': (0.33, 0.37, 0.20), 'j_paint_dark': (0.20, 0.23, 0.13), 'j_steel': (0.16, 0.16, 0.17), 'j_metal': (0.75, 0.75, 0.72),
            'j_rim': (0.36, 0.40, 0.24), 'j_tyre': (0.06, 0.06, 0.06), 'j_seat': (0.55, 0.47, 0.30), 'j_rubber': (0.03, 0.03, 0.03),
            'j_glass': (0.60, 0.80, 0.90), 'j_light': (1.0, 0.98, 0.85), 'j_light_red': (0.85, 0.05, 0.03), 'j_light_amber': (1.0, 0.55, 0.05),
            'j_dash': (0.09, 0.10, 0.08), 'j_gauge': (0.92, 0.90, 0.82)}
    sc = bpy.context.scene
    mats = {}
    for g, c in COLS.items():
        m = bpy.data.materials.new(g)
        m.diffuse_color = (c[0], c[1], c[2], 0.25 if g == 'j_glass' else 1.0)
        mats[g] = m

    def add(name, groups, mat):
        for g, v in groups.items():
            me = bpy.data.meshes.new(name + '_' + g)
            me.from_pydata([tuple(p) for p, _, _ in v], [], [(i, i + 1, i + 2) for i in range(0, len(v), 3)])
            me.update()
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
    sc.world = bpy.data.worlds.new('w'); sc.world.color = (0.75, 0.80, 0.86)
    cam = bpy.data.cameras.new('cam'); cob = bpy.data.objects.new('cam', cam)
    sc.collection.objects.link(cob); sc.camera = cob
    cam.clip_start = 0.03; cam.clip_end = 200.0
    views = [('front34', (3.3, 4.4, 2.0), (0.0, 0.0, 0.75), 38.0, False),
             ('rear34', (-3.4, -4.6, 2.2), (0.0, -0.2, 0.75), 38.0, False),
             ('side', (6.5, 0.0, 1.0), (0.0, -0.05, 0.85), 45.0, False),
             ('driver', EYE_L, (-0.15, 2.5, 0.85), 13.0, False),
             ('top', (0.0, -0.1, 9.0), (0.0, 0.0, 0.5), 35.0, False),
             ('under', (2.6, 2.2, -2.2), (0.0, 0.0, 0.5), 35.0, True),
             ('cabin', (-1.9, -1.5, 2.3), (0.0, -0.2, 0.75), 45.0, False),
             ('wheel', (1.75, 2.1, 0.75), (0.75, 1.2, 0.42), 50.0, False),
             ('rear', (0.3, -4.2, 1.2), (0.0, -1.0, 0.75), 50.0, False)]
    for name, pos, look, lens, hide_ground in views:
        cob.location = Vector(pos)
        d = Vector(look) - Vector(pos)
        cob.rotation_euler = d.to_track_quat('-Z', 'Z' if abs(d.normalized().z) > 0.98 else 'Y').to_euler() if False else d.to_track_quat('-Z', 'Y').to_euler()
        cam.lens = lens
        ground.hide_render = hide_ground
        sc.render.filepath = os.path.join(OUT, 'preview_%s.png' % name)
        bpy.ops.render.render(write_still=True)
        print('JEEP preview', sc.render.filepath)


if os.environ.get('JEEP_PREVIEW') == '1':
    preview()
