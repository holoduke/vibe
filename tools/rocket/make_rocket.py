# Builds the launch site in Blender and exports it for the game:
#   blender -b --python tools/rocket/make_rocket.py
# Output (assets/rocket/):
#   rocket.mesh  the vehicle (origin: centre of its base; drawn with a model matrix)
#   site.mesh    apron, launch mount, tower with its arms, tank farm, light masts (origin: pad
#                centre at ground level)
#   house.mesh   the control house with the launch button (origin: floor centre at ground level)
#   hull.png / hull_h.png, concrete.png / concrete_h.png   colour + height (the game derives the
#                normal maps from the heights)
#   rocket.blend the same geometry as Blender objects, to look at
# File format: text, one block per material - "mesh <name> <triangles>", then three lines per
# triangle: position, normal, uv (uv in METRES along the surface; the game divides by its tile).
#
# Blender space: Z up. Game space: x = X, y = Z, z = -Y.
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector, Matrix

bpy.ops.wm.read_factory_settings(use_empty=True)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'rocket')
os.makedirs(OUT, exist_ok=True)
TAU = 2.0 * math.pi

# ------------------------------------------------------------------ triangle soup
G = {}                                   # material name -> [(p, n, uv) * 3 per triangle]
XF = [Matrix.Identity(4)]


class xf:                                # with xf(matrix): ... builds in a local frame
    def __init__(self, m): self.m = m
    def __enter__(self): XF.append(XF[-1] @ self.m)
    def __exit__(self, *a): XF.pop()


def T(x, y, z): return Matrix.Translation(Vector((x, y, z)))
def RZ(a): return Matrix.Rotation(a, 4, 'Z')
def RX(a): return Matrix.Rotation(a, 4, 'X')
def RY(a): return Matrix.Rotation(a, 4, 'Y')


def planar_uv(p, n):
    if abs(n.z) > 0.95: t, b = Vector((1, 0, 0)), Vector((0, 1, 0))
    else:
        t = Vector((0, 0, 1)).cross(n).normalized()
        b = n.cross(t).normalized()
    return (p.dot(t), -p.dot(b))


def tri(g, pts, nrms=None, uvs=None):
    m = XF[-1]; m3 = m.to_3x3()
    P = [m @ Vector(p) for p in pts]
    fn = (P[1] - P[0]).cross(P[2] - P[0])
    if fn.length < 1e-9: return
    fn.normalize()
    if nrms is None: N = [fn, fn, fn]
    else:
        N = [(m3 @ Vector(n)).normalized() for n in nrms]
        if fn.dot(N[0] + N[1] + N[2]) < 0.0: fn = -fn
    if uvs is None: uvs = [planar_uv(p, fn) for p in P]
    G.setdefault(g, []).extend(zip(P, N, uvs))


def quad(g, a, b, c, d, nrms=None, uvs=None):
    tri(g, (a, b, c), None if nrms is None else (nrms[0], nrms[1], nrms[2]), None if uvs is None else (uvs[0], uvs[1], uvs[2]))
    tri(g, (a, c, d), None if nrms is None else (nrms[0], nrms[2], nrms[3]), None if uvs is None else (uvs[0], uvs[2], uvs[3]))


def box(g, lo, hi, skip=''):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    F = {'-x': ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
         '+x': ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1, 0, 0)),
         '-y': ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
         '+y': ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (0, 1, 0)),
         '-z': ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (0, 0, -1)),
         '+z': ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1))}
    for k, (a, b, c, d, n) in F.items():
        if k in skip: continue
        quad(g, a, b, c, d, [n] * 4)


def obox(g, p0, p1, w, h, up=(0, 0, 1)):
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
    with xf(m): box(g, (-w / 2, -h / 2, 0), (w / 2, h / 2, L))


def lathe(g, prof, seg=32, c=(0.0, 0.0), inside=False, uref=None, sharp=0.75, a0=0.0, a1=TAU):
    """Revolve the (r, z) profile about the vertical axis through c."""
    n = len(prof)
    sn = []                                              # segment normals (r, z)
    for i in range(n - 1):
        dr, dz = prof[i + 1][0] - prof[i][0], prof[i + 1][1] - prof[i][1]
        l = math.hypot(dr, dz) or 1.0
        sn.append((dz / l, -dr / l))
    vl = [0.0]
    for i in range(n - 1): vl.append(vl[-1] + math.hypot(prof[i + 1][0] - prof[i][0], prof[i + 1][1] - prof[i][1]))
    ur = uref if uref is not None else max(p[0] for p in prof)

    def vn(i, s):                                        # normal at profile vertex i for segment s
        o = s - 1 if i == s else s + 1
        if o < 0 or o >= n - 1: return sn[s]
        if sn[s][0] * sn[o][0] + sn[s][1] * sn[o][1] < sharp: return sn[s]
        x, z = sn[s][0] + sn[o][0], sn[s][1] + sn[o][1]
        l = math.hypot(x, z) or 1.0
        return (x / l, z / l)

    sgn = -1.0 if inside else 1.0
    for s in range(n - 1):
        (r0, z0), (r1, z1) = prof[s], prof[s + 1]
        n0, n1 = vn(s, s), vn(s + 1, s)
        for k in range(seg):
            b0 = a0 + (a1 - a0) * k / seg; b1 = a0 + (a1 - a0) * (k + 1) / seg
            c0, s0, c1, s1 = math.cos(b0), math.sin(b0), math.cos(b1), math.sin(b1)
            pa = (c[0] + r0 * c0, c[1] + r0 * s0, z0); pb = (c[0] + r0 * c1, c[1] + r0 * s1, z0)
            pc = (c[0] + r1 * c1, c[1] + r1 * s1, z1); pd = (c[0] + r1 * c0, c[1] + r1 * s0, z1)
            na = (sgn * n0[0] * c0, sgn * n0[0] * s0, sgn * n0[1]); nb = (sgn * n0[0] * c1, sgn * n0[0] * s1, sgn * n0[1])
            nc = (sgn * n1[0] * c1, sgn * n1[0] * s1, sgn * n1[1]); nd = (sgn * n1[0] * c0, sgn * n1[0] * s0, sgn * n1[1])
            ua, ub = b0 * ur, b1 * ur
            quad(g, pa, pb, pc, pd, [na, nb, nc, nd], [(ua, vl[s]), (ub, vl[s]), (ub, vl[s + 1]), (ua, vl[s + 1])])


def disc(g, c, z, r0, r1, seg=32, up=True):
    nz = (0, 0, 1.0 if up else -1.0)
    for k in range(seg):
        b0, b1 = TAU * k / seg, TAU * (k + 1) / seg
        c0, s0, c1, s1 = math.cos(b0), math.sin(b0), math.cos(b1), math.sin(b1)
        quad(g, (c[0] + r0 * c0, c[1] + r0 * s0, z), (c[0] + r1 * c0, c[1] + r1 * s0, z),
             (c[0] + r1 * c1, c[1] + r1 * s1, z), (c[0] + r0 * c1, c[1] + r0 * s1, z), [nz] * 4)


def write_mesh(name):
    global G
    with open(os.path.join(OUT, name + '.mesh'), 'w') as f:
        for g, v in G.items():
            f.write('mesh %s %d\n' % (g, len(v) // 3))
            for p, n, uv in v:
                f.write('%.4f %.4f %.4f %.3f %.3f %.3f %.4f %.4f\n' % (p.x, p.z, -p.y, n.x, n.z, -n.y, uv[0], uv[1]))
    # the same as Blender objects (one per material) for the .blend
    col = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(col)
    for g, v in G.items():
        me = bpy.data.meshes.new(name + '_' + g)
        me.from_pydata([tuple(p) for p, _, _ in v], [], [(i, i + 1, i + 2) for i in range(0, len(v), 3)])
        me.update()
        col.objects.link(bpy.data.objects.new(name + '_' + g, me))
    print(name.upper(), {g: len(v) // 3 for g, v in G.items()})
    G = {}


# ================================================================== the rocket
R = 4.5                    # body radius
HB = 62.0                  # cylinder height
HN = 14.0                  # nose height
SEG = 72


def build_rocket():
    # Body: one stainless cylinder (the weld rings and panels are in the texture).
    lathe('hull', [(R, 0.0), (R, HB)], SEG)
    # Nose: a tangent ogive, rounded at the tip.
    prof = []
    for i in range(15):
        t = i / 14.0
        prof.append((max(R * (1.0 - t ** 2.1) ** 0.72, 0.0), HB + HN * t))
    prof[-1] = (0.0, HB + HN)
    lathe('hull', prof, SEG, uref=R)
    # Dark bands: the vented ring under the nose and the engine skirt's heat shielding.
    lathe('dark', [(R + 0.045, HB - 2.2), (R + 0.045, HB - 0.5)], SEG)
    lathe('dark', [(R, HB - 2.3), (R + 0.045, HB - 2.2)], SEG); lathe('dark', [(R + 0.045, HB - 0.5), (R, HB - 0.4)], SEG)
    for k in range(36):                                   # the ring's vents
        a = TAU * (k + 0.5) / 36
        with xf(RZ(a)): box('vent', (R + 0.03, -0.20, HB - 1.9), (R + 0.06, 0.20, HB - 0.8), skip='-x')
    lathe('dark', [(R + 0.05, 0.0), (R + 0.05, 2.6), (R, 2.75)], SEG)
    # Stringers on the aft and forward sections.
    for k in range(60):
        a = TAU * k / 60
        with xf(RZ(a)):
            box('hull', (R - 0.02, -0.06, 2.9), (R + 0.09, 0.06, 8.6), skip='-x')
            box('hull', (R - 0.02, -0.05, 50.5), (R + 0.07, 0.05, 59.2), skip='-x')
    # Four aero covers (chines) low on the body.
    for k in range(4):
        with xf(RZ(TAU * (k + 0.5) / 4)):
            x0, x1, w = R - 0.1, R + 0.62, 0.85
            b = [(x0, -w, 1.2), (x1, -w * 0.8, 1.2), (x1, w * 0.8, 1.2), (x0, w, 1.2)]
            t = [(x0, -w, 18.0), (x1, -w * 0.8, 18.0), (x1, w * 0.8, 18.0), (x0, w, 18.0)]
            p = (x0, 0.0, 21.5)
            quad('hull', b[1], b[2], t[2], t[1]); quad('hull', b[0], b[1], t[1], t[0]); quad('hull', b[2], b[3], t[3], t[2])
            quad('hull', b[0], b[3], b[2], b[1])
            tri('hull', (t[1], t[2], p)); tri('hull', (t[0], t[1], p)); tri('hull', (t[2], t[3], p))
    # Raceway (cable cover) up one side.
    with xf(RZ(math.radians(290.0))): box('hull', (R - 0.05, -0.28, 8.6), (R + 0.20, 0.28, 50.5), skip='-x')
    # Grid fins: a lattice slab each, on a hinge block.
    for k in range(4):
        with xf(RZ(TAU * k / 4)):
            z0, z1 = 56.6, 57.05
            x0, x1, w = R + 0.35, R + 3.9, 1.7
            box('fin', (R - 0.1, -0.7, 56.3), (R + 0.45, 0.7, 57.35))          # hinge
            for y in (-w, w - 0.10): box('fin', (x0, y, z0), (x1, y + 0.10, z1))
            for x in (x0, x1 - 0.10): box('fin', (x, -w, z0), (x + 0.10, w, z1))
            # diagonal lattice
            n = 7
            for i in range(-n, n + 1):
                for sgn in (1.0, -1.0):
                    # line y = sgn * (x - xm) + i * step, clipped to the frame
                    xm, step = (x0 + x1) * 0.5, 0.62
                    pts = []
                    for xx in (x0 + 0.1, x1 - 0.1):
                        yy = sgn * (xx - xm) + i * step
                        if -w + 0.1 <= yy <= w - 0.1: pts.append((xx, yy))
                    for yy in (-w + 0.1, w - 0.1):
                        xx = xm + sgn * (yy - i * step)
                        if x0 + 0.1 <= xx <= x1 - 0.1: pts.append((xx, yy))
                    if len(pts) >= 2:
                        pts.sort()
                        a, b = pts[0], pts[-1]
                        if math.hypot(b[0] - a[0], b[1] - a[1]) > 0.3:
                            obox('fin', (a[0], a[1], (z0 + z1) / 2), (b[0], b[1], (z0 + z1) / 2), 0.035, z1 - z0 - 0.04)
    # Catch pins under two of the fins.
    for a in (0.0, math.pi):
        with xf(RZ(a)): box('fin', (R - 0.05, -0.22, 54.9), (R + 0.75, 0.22, 55.35))
    # Aft: the engine bay's dome and thirteen engines.
    disc('dark', (0, 0), 0.55, 0.0, R, SEG, up=False)
    eng = [(1.25 * math.cos(TAU * k / 3), 1.25 * math.sin(TAU * k / 3)) for k in range(3)]
    eng += [(3.35 * math.cos(TAU * (k + 0.5) / 10), 3.35 * math.sin(TAU * (k + 0.5) / 10)) for k in range(10)]
    bell = [(0.26, 0.75), (0.30, 0.45), (0.44, -0.05), (0.60, -0.65), (0.70, -1.25)]
    for (ex, ey) in eng:
        lathe('engine', bell, 18, c=(ex, ey))
        lathe('engine_in', bell, 18, c=(ex, ey), inside=True)
        lathe('engine', [(0.0, 1.0), (0.42, 0.95), (0.42, 0.6)], 12, c=(ex, ey))      # power head
    # Lettering, wrapped on the hull facing the control house (-X).
    wrap_text('VIBE', 3.4, math.pi, 37.0)


def wrap_text(s, size, a_mid, z0):
    cu = bpy.data.curves.new('txt', 'FONT')
    cu.body = s; cu.size = size; cu.align_x = 'CENTER'
    ob = bpy.data.objects.new('txt', cu)
    bpy.context.scene.collection.objects.link(ob)
    deps = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(deps))
    me.calc_loop_triangles()
    Rt = R + 0.03

    def put(a, b, c, depth):
        if depth < 6 and max((a - b).length, (b - c).length, (c - a).length) > 0.30:
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            put(a, ab, ca, depth + 1); put(ab, b, bc, depth + 1); put(ca, bc, c, depth + 1); put(ab, bc, ca, depth + 1)
            return
        P, N = [], []
        for q in (a, b, c):
            ang = a_mid + q.x / Rt
            P.append((Rt * math.cos(ang), Rt * math.sin(ang), z0 + q.y)); N.append((math.cos(ang), math.sin(ang), 0.0))
        tri('text', P, N, [(q.x, q.y) for q in (a, b, c)])

    for t in me.loop_triangles:
        v = [Vector((me.vertices[i].co.x, me.vertices[i].co.y)) for i in t.vertices]
        put(v[0], v[1], v[2], 0)
    bpy.data.objects.remove(ob)


def build_flame():
    # The exhaust, one unit long (the game scales it): an outer plume and a hot core. Origin at
    # the nozzle exits, pointing down.
    lathe('flame', [(3.7, 0.0), (4.3, -0.12), (3.9, -0.45), (2.2, -0.8), (0.0, -1.0)], 36)
    lathe('flame_core', [(3.3, 0.0), (3.0, -0.2), (1.6, -0.5), (0.0, -0.72)], 36)


# ================================================================== the site
MZ = 20.0                  # the mount's table top: the rocket stands on it
TY = 14.5                  # tower centre (Y)
TH = 108.0                 # tower height
ARM_Z = MZ + 55.6          # the arms' top rail (just under the catch pins of a standing rocket)


def tank_v(c, r, h):
    prof = [(0.0, 0.0), (r, 0.0), (r, h)]
    for i in range(1, 7):
        a = (math.pi / 2) * i / 6
        prof.append((r * math.cos(a), h + 0.55 * r * math.sin(a)))
    lathe('white', prof, 28, c=c)
    lathe('steel_grey', [(r + 0.06, 0.0), (r + 0.06, 0.5)], 28, c=c)
    lathe('steel_grey', [(r + 0.05, h * 0.5), (r + 0.05, h * 0.5 + 0.2)], 28, c=c)


def truss(p0, p1, w, h, bays, chord=0.34, web=0.2, g='steel_dark'):
    """A box truss from p0 to p1 (section w across, h vertical)."""
    p0, p1 = Vector(p0), Vector(p1)
    ax = (p1 - p0); L = ax.length; ax /= L
    side = ax.cross(Vector((0, 0, 1))).normalized()
    up = Vector((0, 0, 1))
    cs = [side * (sx * w / 2) + up * (sz * h / 2) for sx in (-1, 1) for sz in (-1, 1)]
    for c in cs: obox(g, p0 + c, p1 + c, chord, chord)
    for b in range(bays + 1):
        q = p0 + ax * (L * b / bays)
        obox(g, q + cs[0], q + cs[1], web, web, up=side); obox(g, q + cs[2], q + cs[3], web, web, up=side)
        obox(g, q + cs[0], q + cs[2], web, web); obox(g, q + cs[1], q + cs[3], web, web)
        if b < bays:
            q2 = p0 + ax * (L * (b + 1) / bays)
            e = b % 2
            obox(g, q + cs[0 + e], q2 + cs[1 - e], web, web, up=side); obox(g, q + cs[2 + e], q2 + cs[3 - e], web, web, up=side)
            obox(g, q + cs[0 + 2 * e], q2 + cs[2 - 2 * e], web, web); obox(g, q + cs[1 + 2 * e], q2 + cs[3 - 2 * e], web, web)


def build_site():
    # Apron.
    box('concrete', (-70, -70, -2.5), (70, 70, 0.35), skip='-z')       # (its top clears the terrain's own relief)
    # Flame diverter under the mount: a low steel cone on a plate.
    lathe('steel_plate', [(0.0, 2.6), (2.0, 2.1), (6.0, 0.9), (11.0, 0.3), (12.5, 0.15)], 48)
    # Launch mount: six legs carrying a ring table; the rocket stands over the hole.
    for k in range(6):
        with xf(RZ(TAU * (k + 0.5) / 6)):
            box('concrete_w', (6.5, -1.3, 0.0), (9.1, 1.3, MZ - 3.6))
            box('steel_grey', (6.3, -1.5, MZ - 4.2), (9.3, 1.5, MZ - 3.6))
    lathe('steel_grey', [(9.6, MZ - 3.6), (9.6, MZ)], 48)
    lathe('steel_grey', [(4.95, MZ - 3.6), (4.95, MZ)], 48, inside=True)
    disc('steel_grey', (0, 0), MZ, 4.95, 9.6, 48, up=True)
    disc('steel_grey', (0, 0), MZ - 3.6, 4.95, 9.6, 48, up=False)
    lathe('hazard', [(9.63, MZ - 0.5), (9.63, MZ - 0.05)], 48)
    for k in range(20):                                    # hold-down clamps
        with xf(RZ(TAU * k / 20)): box('steel_dark', (4.62, -0.22, MZ), (5.5, 0.22, MZ + 0.42))
    for k in range(24):                                    # deluge pipes round the table
        with xf(RZ(TAU * k / 24)): box('pipe', (9.6, -0.09, MZ - 3.0), (9.9, 0.09, MZ - 2.82))
    lathe('pipe', [(9.9, MZ - 3.1), (10.1, MZ - 2.9), (9.9, MZ - 2.7)], 48)
    # Tower: four columns, ring beams and X-bracing, a core, the top deck and its crane.
    hx = 4.3
    for sx in (-1, 1):
        for sy in (-1, 1): box('steel_dark', (sx * hx - 0.7, TY + sy * hx - 0.7, 0.0), (sx * hx + 0.7, TY + sy * hx + 0.7, TH))
    bays = 12
    for b in range(bays + 1):
        z = TH * b / bays
        if b > 0:
            for s in (-1, 1):
                obox('steel_dark', (-hx, TY + s * hx, z - 0.35), (hx, TY + s * hx, z - 0.35), 0.6, 0.7)
                obox('steel_dark', (s * hx, TY - hx, z - 0.35), (s * hx, TY + hx, z - 0.35), 0.6, 0.7)
        if b < bays:
            z2 = TH * (b + 1) / bays
            e = 1 if b % 2 else -1
            for s in (-1, 1):
                obox('steel_dark', (-e * hx, TY + s * hx, z), (e * hx, TY + s * hx, z2 - 0.7), 0.36, 0.36)
                obox('steel_dark', (s * hx, TY - e * hx, z), (s * hx, TY + e * hx, z2 - 0.7), 0.36, 0.36)
    box('steel_grey', (-1.8, TY - 1.8, 0.0), (1.8, TY + 1.8, TH - 2.0))               # lift shaft
    box('steel_grey', (-6.0, TY - 6.0, TH), (6.0, TY + 6.0, TH + 0.6))
    for s in (-1, 1):                                                                 # deck railing
        obox('hazard', (-6.0, TY + s * 5.9, TH + 1.6), (6.0, TY + s * 5.9, TH + 1.6), 0.08, 0.08)
        obox('hazard', (s * 5.9, TY - 6.0, TH + 1.6), (s * 5.9, TY + 6.0, TH + 1.6), 0.08, 0.08)
        for i in range(7):
            x = -6.0 + 2.0 * i
            box('steel_dark', (x - 0.04, TY + s * 5.9 - 0.04, TH + 0.6), (x + 0.04, TY + s * 5.9 + 0.04, TH + 1.6))
            box('steel_dark', (s * 5.9 - 0.04, TY + x - 0.04, TH + 0.6), (s * 5.9 + 0.04, TY + x + 0.04, TH + 1.6))
    box('steel_grey', (-1.2, TY - 1.2, TH + 0.6), (1.2, TY + 1.2, TH + 7.0))          # crane house
    truss((0.0, TY + 4.0, TH + 6.0), (0.0, TY - 17.0, TH + 9.0), 1.2, 1.4, 9, 0.22, 0.12)
    obox('steel_grey', (0.0, TY, TH + 7.0), (0.0, TY, TH + 24.0), 0.3, 0.3)            # lightning rod
    obox('steel_grey', (0.0, TY, TH + 24.0), (0.0, TY, TH + 30.0), 0.1, 0.1)
    for z in (TH + 23.6, 60.0, 30.0):                                                 # obstruction lights
        box('lamp_red', (hx + 0.7, TY + hx + 0.7, z), (hx + 1.0, TY + hx + 1.0, z + 0.4)) if z < TH else \
            box('lamp_red', (-0.2, TY - 0.2, z), (0.2, TY + 0.2, z + 0.4))
    # The arms ("chopsticks"): a carriage on the tower's front, two truss arms held open.
    box('steel_grey', (-5.6, TY - hx - 1.5, ARM_Z - 6.0), (5.6, TY - hx - 0.7, ARM_Z + 1.5))
    for s in (-1, 1):
        box('steel_dark', (s * 4.4 - 0.9, TY - hx - 2.6, ARM_Z - 3.2), (s * 4.4 + 0.9, TY - hx - 1.5, ARM_Z + 0.4))    # hinge
        a = math.radians(33.0)
        p0 = Vector((s * 4.4, TY - hx - 2.2, ARM_Z - 1.3))
        p1 = p0 + Vector((s * math.sin(a), -math.cos(a), 0.0)) * 36.0
        truss(p0, p1, 1.7, 2.4, 10)
        d = (p1 - p0).normalized(); side = Vector((-d.y, d.x, 0.0)) * (-s)
        obox('hazard', p0 + side * 0.85 + Vector((0, 0, 1.38)), p1 + side * 0.85 + Vector((0, 0, 1.38)), 0.5, 0.16)       # landing rail
    # Quick-disconnect arm to the rocket's side.
    truss((0.0, TY - hx - 0.7, MZ + 31.0), (0.0, R + 1.6, MZ + 31.0), 1.6, 1.8, 3, 0.26, 0.14, 'steel_grey')
    box('steel_dark', (-1.1, R + 0.5, MZ + 29.8), (1.1, R + 1.6, MZ + 32.2))
    # Tank farm, with a pipe rack to the mount.
    for i in range(5): tank_v((-58.0 + 8.2 * i, -52.0), 3.2, 17.0)
    for i in range(2):
        with xf(T(-52.0 + 20.0 * i, -38.0, 3.2) @ RY(math.pi / 2)):
            prof = [(0.0, -8.6), (1.4, -8.2), (2.2, -7.0), (2.2, 7.0), (1.4, 8.2), (0.0, 8.6)]
            lathe('white', prof, 24)
        for sx in (-5.0, 5.0): box('concrete_w', (-52.0 + 20.0 * i + sx - 0.5, -40.0, 0.0), (-52.0 + 20.0 * i + sx + 0.5, -36.0, 1.6))
    for j, y in enumerate((-33.0, -32.4, -31.8)):
        pts = [(-56.0, y, 1.1), (-9.0 - 0.6 * j, y, 1.1), (-9.0 - 0.6 * j, -6.0 + 0.6 * j, 1.1)]
        for a, b in zip(pts, pts[1:]): obox('pipe' if j else 'white', a, b, 0.42, 0.42)
        obox('pipe' if j else 'white', pts[-1], (pts[-1][0], pts[-1][1], MZ - 3.0), 0.42, 0.42, up=(1, 0, 0))
    for x in range(-54, -8, 6): box('steel_grey', (x - 0.15, -33.6, 0.0), (x + 0.15, -31.2, 0.9))
    box('white', (-66.0, -48.0, 0.0), (-62.0, -40.0, 3.4))                            # pump house
    box('steel_dark', (-66.2, -48.2, 3.4), (-61.8, -39.8, 3.6))
    # Light masts at the apron's corners, their heads turned to the pad.
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * 63.0, sy * 63.0
            obox('steel_grey', (x, y, 0.0), (x, y, 34.0), 0.55, 0.55)
            box('concrete_w', (x - 1.0, y - 1.0, 0.0), (x + 1.0, y + 1.0, 0.8))
            yaw = math.atan2(-y, -x)
            with xf(T(x, y, 34.0) @ RZ(yaw) @ RY(math.radians(28.0))):
                box('steel_dark', (-0.3, -2.4, -0.1), (0.25, 2.4, 1.9), skip='+x')
                for r in range(2):
                    for c in range(4):
                        quad('lamp', (0.26, -2.2 + 1.1 * c + 0.08, 0.05 + 0.9 * r), (0.26, -2.2 + 1.1 * c + 1.02, 0.05 + 0.9 * r),
                             (0.26, -2.2 + 1.1 * c + 1.02, 0.85 + 0.9 * r), (0.26, -2.2 + 1.1 * c + 0.08, 0.85 + 0.9 * r), [(1, 0, 0)] * 4)
    # Painted markings on the apron: a ring round the mount and the edge lines.
    disc('hazard', (0, 0), 0.365, 17.0, 17.6, 64)
    for s in (-1, 1):
        box('paint', (-69.0, s * 68.6 - 0.15, 0.35), (69.0, s * 68.6 + 0.15, 0.365), skip='-z')
        box('paint', (s * 68.6 - 0.15, -69.0, 0.35), (s * 68.6 + 0.15, 69.0, 0.365), skip='-z')


# ================================================================== the control house
F = 0.30                   # floor level above the ground
HX, HY = 3.5, 3.0          # half size (outside)
WT = 0.30                  # wall thickness
WH = 3.0                   # wall height above the floor
BUTTON = (2.55, 0.0, F + 1.09)       # top of the launch button (src/engine/vk_engine/rocket.cpp kButton)


def build_house():
    box('concrete_w', (-HX - 0.3, -HY - 0.3, -3.0), (HX + 0.3, HY + 0.3, F - 0.02), skip='-z')      # plinth
    box('floor', (-HX + WT, -HY + WT, F - 0.05), (HX - WT, HY - WT, F), skip='-z')
    # the walls' inside faces, painted light
    box('wall_in', (-HX + WT, -HY + WT, F), (-2.5, -HY + WT + 0.01, F + WH), skip='-y-z+z'); box('wall_in', (-1.5, -HY + WT, F), (HX - WT, -HY + WT + 0.01, F + WH), skip='-y-z+z'); box('wall_in', (-2.5, -HY + WT, F + 2.1), (-1.5, -HY + WT + 0.01, F + WH), skip='-y-z+z'); box('wall_in', (-HX + WT, HY - WT - 0.01, F), (HX - WT, HY - WT, F + WH), skip='+y-z+z')
    box('wall_in', (-HX + WT, -HY + WT, F), (-HX + WT + 0.01, HY - WT, F + WH), skip='-x-z+z')
    box('wall_in', (-HX + WT, -HY + WT, F + WH - 0.01), (HX - WT, HY - WT, F + WH), skip='+z')
    z0, z1 = F - 0.02, F + WH
    sill, head = F + 1.05, F + 2.15
    # east wall (+X, the pad's side) with the wide window
    box('concrete_w', (HX - WT, -HY, z0), (HX, HY, sill)); box('concrete_w', (HX - WT, -HY, head), (HX, HY, z1))
    box('concrete_w', (HX - WT, -HY, sill), (HX, -2.3, head)); box('concrete_w', (HX - WT, 2.3, sill), (HX, HY, head))
    for y in (-2.3, -0.77, 0.77, 2.3 - 0.06): box('steel_dark', (HX - 0.22, y, sill), (HX - 0.10, y + 0.06, head))
    box('steel_dark', (HX - 0.22, -2.3, sill), (HX - 0.10, 2.3, sill + 0.06)); box('steel_dark', (HX - 0.22, -2.3, head - 0.06), (HX - 0.10, 2.3, head))
    # west wall, north wall
    box('concrete_w', (-HX, -HY, z0), (-HX + WT, HY, z1))
    box('concrete_w', (-HX + WT, HY - WT, z0), (HX - WT, HY, z1))
    # south wall (-Y) with the door
    dx0, dx1, dh = -2.5, -1.5, F + 2.1
    box('concrete_w', (-HX + WT, -HY, z0), (dx0, -HY + WT, z1)); box('concrete_w', (dx1, -HY, z0), (HX - WT, -HY + WT, z1))
    box('concrete_w', (dx0, -HY, dh), (dx1, -HY + WT, z1))
    for x in (dx0 - 0.06, dx1): box('steel_dark', (x, -HY - 0.02, F), (x + 0.06, -HY + WT + 0.02, dh))
    box('steel_dark', (dx0 - 0.06, -HY - 0.02, dh), (dx1 + 0.06, -HY + WT + 0.02, dh + 0.06))
    # the door, standing open (inside, against the wall)
    with xf(T(dx0, -HY + WT, F) @ RZ(math.radians(96.0))): box('door', (0.0, -0.05, 0.0), (1.0, 0.0, 2.08))
    # steps
    box('concrete_w', (dx0 - 0.3, -HY - 0.9, -1.5), (dx1 + 0.3, -HY, F - 0.02), skip='-z')
    box('concrete_w', (dx0 - 0.3, -HY - 1.6, -1.5), (dx1 + 0.3, -HY - 0.9, F - 0.17), skip='-z')
    # roof slab with an overhang
    box('roof', (-HX - 0.5, -HY - 0.5, z1), (HX + 0.5, HY + 0.5, z1 + 0.28))
    box('steel_dark', (-HX - 0.55, -HY - 0.55, z1 + 0.28), (HX + 0.55, HY + 0.55, z1 + 0.34))
    # console under the window: desk, a raised panel with screens, the button on its box
    cx0, cx1 = 2.05, HX - WT
    box('console', (cx0, -2.2, F), (cx1, 2.2, F + 0.92))
    box('console_top', (cx0 - 0.06, -2.26, F + 0.92), (cx1, 2.26, F + 0.96))
    for y in (-1.45, 1.45):
        with xf(T(cx1 - 0.42, y, F + 0.96) @ RY(math.radians(-22.0))):
            box('console', (0.0, -0.55, 0.0), (0.10, 0.55, 0.62))
            quad('screen', (-0.005, -0.50, 0.05), (-0.005, 0.50, 0.05), (-0.005, 0.50, 0.57), (-0.005, -0.50, 0.57), [(-1, 0, 0)] * 4,
                 [(0, 0), (1, 0), (1, 1), (0, 1)])
    bx, by, bz = BUTTON
    box('hazard', (bx - 0.17, by - 0.17, F + 0.96), (bx + 0.17, by + 0.17, F + 1.03))
    lathe('steel_dark', [(0.085, F + 1.03), (0.085, F + 1.05), (0.0, F + 1.05)], 20, c=(bx, by))
    lathe('button', [(0.062, F + 1.04), (0.070, F + 1.075), (0.050, bz), (0.0, bz + 0.004)], 20, c=(bx, by))
    # a chair's worth of furniture: a locker and a small table
    box('console', (-HX + WT, HY - WT - 0.55, F), (-HX + WT + 1.1, HY - WT, F + 1.9))
    # roof: a dish on a mast, an aerial, and a lamp over the door
    obox('steel_grey', (-1.6, 1.2, z1 + 0.3), (-1.6, 1.2, z1 + 1.7), 0.12, 0.12)
    with xf(T(-1.6, 1.2, z1 + 1.9) @ RY(math.radians(-48.0))):
        prof = [(0.0, 0.0)] + [(1.0 * i / 6, 0.42 * (i / 6) ** 2) for i in range(1, 7)]
        lathe('white', prof, 24); lathe('white', prof, 24, inside=True)
        obox('steel_grey', (0.0, 0.0, 0.0), (0.0, 0.0, 0.7), 0.05, 0.05)
    obox('steel_grey', (2.6, -2.2, z1 + 0.3), (2.6, -2.2, z1 + 4.6), 0.06, 0.06)
    box('steel_dark', ((dx0 + dx1) / 2 - 0.18, -HY - 0.22, dh + 0.25), ((dx0 + dx1) / 2 + 0.18, -HY, dh + 0.37))
    quad('lamp', ((dx0 + dx1) / 2 - 0.15, -HY - 0.20, dh + 0.249), ((dx0 + dx1) / 2 + 0.15, -HY - 0.20, dh + 0.249),
         ((dx0 + dx1) / 2 + 0.15, -HY - 0.02, dh + 0.249), ((dx0 + dx1) / 2 - 0.15, -HY - 0.02, dh + 0.249), [(0, 0, -1)] * 4)
    # ceiling light
    quad('lamp_in', (-0.6, -0.15, z1 - 0.01), (0.6, -0.15, z1 - 0.01), (0.6, 0.15, z1 - 0.01), (-0.6, 0.15, z1 - 0.01), [(0, 0, -1)] * 4)


# ================================================================== textures
def vnoise(n, cells, rng):
    """Tileable value noise, n x n, `cells` lattice cells across."""
    g = rng.random((cells, cells)).astype(np.float32)
    x = np.arange(n, dtype=np.float32) * cells / n
    i0 = np.floor(x).astype(int); f = x - i0; f = f * f * (3 - 2 * f); i1 = (i0 + 1) % cells
    a = g[np.ix_(i0, i0)]; b = g[np.ix_(i0, i1)]; c = g[np.ix_(i1, i0)]; d = g[np.ix_(i1, i1)]
    fx, fy = f[None, :], f[:, None]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def fbm(n, cells, octs, rng):
    s, amp, tot = np.zeros((n, n), np.float32), 1.0, 0.0
    for o in range(octs):
        s += amp * vnoise(n, min(cells * 2 ** o, n), rng); tot += amp; amp *= 0.5
    return s / tot


def save_png(name, rgb):
    """8-bit RGB PNG, the values written as they are (the game reads them as sRGB)."""
    import struct, zlib
    n = rgb.shape[0]
    if rgb.ndim == 2: rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    b = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    raw = b''.join(b'\x00' + b[y].tobytes() for y in range(n))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    with open(os.path.join(OUT, name + '.png'), 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', n, n, 8, 2, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def line(coord, at, width):                              # soft line profile, wraps at 1
    d = np.abs(((coord - at + 0.5) % 1.0) - 0.5)
    return np.clip(1.0 - d / width, 0.0, 1.0)


def make_hull():
    # One tile = 3.6 m: two rings of 1.8 m, the upper one's vertical seams offset.
    n = 1024; rng = np.random.default_rng(7)
    v, u = np.mgrid[0:n, 0:n].astype(np.float32) / n
    streak = np.repeat(fbm(n, 64, 3, rng)[:1, :], n, axis=0) * 0.6 + fbm(n, 8, 4, rng) * 0.4     # brushed, running up the hull
    cloud = fbm(n, 4, 4, rng)
    ringw = line(v, 0.0, 0.006) + line(v, 0.5, 0.006)
    haz = np.clip(line(v, 0.0, 0.035) + line(v, 0.5, 0.035), 0, 1)                             # heat-affected zone
    low = (v < 0.5)
    vs = np.where(low, line(u, 0.12, 0.004) + line(u, 0.62, 0.004), line(u, 0.37, 0.004) + line(u, 0.87, 0.004))
    vh = np.where(low, line(u, 0.12, 0.022) + line(u, 0.62, 0.022), line(u, 0.37, 0.022) + line(u, 0.87, 0.022))
    base = 0.60 + 0.10 * (streak - 0.5) + 0.08 * (cloud - 0.5)
    col = np.stack([base * 1.00, base * 1.005, base * 1.02], axis=2)
    tint = np.stack([0.95 + 0 * u, 0.90 + 0 * u, 0.80 + 0 * u], axis=2)                          # straw tint by the welds
    hz = np.clip(haz + vh, 0, 1)[:, :, None] * 0.55
    col = col * (1 - hz) + col * tint * 0.93 * hz
    col *= (1.0 - 0.45 * np.clip(ringw + vs, 0, 1))[:, :, None]
    grime = np.clip(fbm(n, 6, 5, rng) - 0.55, 0, 1) * 1.6                                      # run-off streaks
    col *= (1.0 - 0.18 * grime * np.repeat(fbm(n, 96, 2, rng)[:1, :], n, axis=0))[:, :, None]
    save_png('hull', col)
    # Height: each panel bulges a little between its welds (the wavy reflections of thin steel
    # sheet), the weld beads stand up.
    pv = np.abs(np.sin(np.pi * v * 2.0)); pu = np.abs(np.sin(np.pi * (u - np.where(low, 0.12, 0.37)) * 2.0))
    h = 0.45 + 0.20 * (pv ** 0.6) * (pu ** 0.6) + 0.10 * (fbm(n, 5, 3, rng) - 0.5) + 0.30 * np.clip(ringw + vs, 0, 1)
    save_png('hull_h', h)


def make_concrete():
    # One tile = 8 m: slabs of 4 m, pores, stains, a few cracks.
    n = 1024; rng = np.random.default_rng(11)
    v, u = np.mgrid[0:n, 0:n].astype(np.float32) / n
    fine = fbm(n, 128, 3, rng); mid = fbm(n, 16, 4, rng); big = fbm(n, 3, 3, rng)
    joint = np.clip(line(u, 0.0, 0.0035) + line(u, 0.5, 0.0035) + line(v, 0.0, 0.0035) + line(v, 0.5, 0.0035), 0, 1)
    slab = (np.floor(u * 2) + 2 * np.floor(v * 2))
    tone = 1.0 + 0.05 * (np.sin(slab * 2.4) )                                                 # each slab its own pour
    base = (0.47 + 0.10 * (mid - 0.5) + 0.07 * (fine - 0.5) + 0.06 * (big - 0.5)) * tone
    crack = np.clip(1.0 - np.abs(fbm(n, 5, 5, rng) - 0.5) * 60.0, 0, 1) * (fbm(n, 4, 2, rng) > 0.56)
    stain = np.clip(fbm(n, 7, 4, rng) - 0.58, 0, 1) * 2.2
    col = np.stack([base * 1.0, base * 0.985, base * 0.95], axis=2)
    col *= (1.0 - 0.30 * stain)[:, :, None]
    col *= (1.0 - 0.55 * joint - 0.40 * crack)[:, :, None]
    save_png('concrete', col)
    h = 0.6 + 0.25 * (fine - 0.5) + 0.15 * (mid - 0.5) - 0.5 * joint - 0.3 * crack
    save_png('concrete_h', h)


# ================================================================== run
build_rocket(); write_mesh('rocket')
build_site(); write_mesh('site')
build_house(); write_mesh('house')
make_hull(); make_concrete()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'rocket.blend'))
print('ROCKET OK')
