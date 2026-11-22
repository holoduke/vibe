# Builds the launch operations pavilion - the office by the launch pad - in Blender and
# exports it for the game:
#   blender -b --python tools/office/make_office.py
# Output (assets/office/):
#   office.mesh   the building and everything in it, one block per material (the format of
#                 tools/rocket/make_rocket.py: "mesh <name> <triangles>", then position, normal,
#                 uv in metres per vertex). Origin: the middle of its footprint, at ground level.
#   office.col    what the player bumps into ("box x0 y0 z0 x1 y1 z1") and where its lights
#                 hang ("light x y z"), in the game's space, from the same origin.
#   office.blend  the same geometry as Blender objects, to look at.
#
# The design: a long steel-framed hall, 36 x 14 m, on a concrete plinth that runs out into a
# terrace toward the pad. Seven black portal frames at 6 m; between them the south side is
# glass from floor to eave, so the whole hall looks at the rocket. The roof is a sawtooth -
# six north-light teeth, their upright faces glazed - and the frames' beams run on 3.5 m over
# the terrace, carrying a timber brise-soleil. The north side is weathering steel with a
# clerestory; the east end is a board-marked concrete core. Inside: one double-height room
# with bench desks, a wall of mission screens, a lounge, and at the east a mezzanine over a
# glass meeting room and a kitchen, reached by a steel stair with timber treads.
#
# Blender space: X east, Y north, Z up. The front (to the pad) is -Y.
# Game space: x = X, y = Z, z = -Y.
import bpy, math, os, random
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'office')
os.makedirs(OUT, exist_ok=True)
TAU = 2.0 * math.pi

G = {}          # material -> [(p, n, uv)] three per triangle
COL = []        # (lo, hi) boxes the player collides with
LIGHTS = []     # points
RNG = random.Random(7)         # the small disorder of a used room: the same every run


def planar_uv(p, n):
    if abs(n.z) > 0.95: t, b = Vector((1, 0, 0)), Vector((0, 1, 0))
    else:
        t = Vector((0, 0, 1)).cross(n).normalized()
        b = n.cross(t).normalized()
    return (p.dot(t), -p.dot(b))


def tri(g, a, b, c, n=None):
    a, b, c = Vector(a), Vector(b), Vector(c)
    fn = (b - a).cross(c - a)
    if fn.length < 1e-10: return
    fn.normalize()
    N = [fn, fn, fn] if n is None else [Vector(x).normalized() for x in n]
    G.setdefault(g, []).extend([(a, N[0], planar_uv(a, fn)), (b, N[1], planar_uv(b, fn)), (c, N[2], planar_uv(c, fn))])


def quad(g, a, b, c, d):
    tri(g, a, b, c); tri(g, a, c, d)


SCREEN_N = [0]


def screen(a, b, c, d):
    """A display. Its corners as the viewer sees them: bottom left, bottom right, top right,
    top left. uv = (its number + 0..1 across, 0 at the top .. 1 at the bottom): the game shows
    one picture of an atlas on each, by its number (cube.frag, texture slot 36)."""
    i = SCREEN_N[0]; SCREEN_N[0] += 1
    a, b, c, d = Vector(a), Vector(b), Vector(c), Vector(d)
    n = (b - a).cross(d - a).normalized()
    uv = [(i + 0.0, 1.0), (i + 1.0, 1.0), (i + 1.0, 0.0), (i + 0.0, 0.0)]
    G.setdefault('o_screen', []).extend([(a, n, uv[0]), (b, n, uv[1]), (c, n, uv[2]), (a, n, uv[0]), (c, n, uv[2]), (d, n, uv[3])])


def quad2(g, a, b, c, d):
                  # seen from both sides (glass, thin sheets)
    quad(g, a, b, c, d); quad(g, d, c, b, a)


def box(g, lo, hi, col=False):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    if z1 < z0: z0, z1 = z1, z0
    quad(g, (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))     # top
    quad(g, (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0))     # bottom
    quad(g, (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))     # south
    quad(g, (x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1))     # north
    quad(g, (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))     # east
    quad(g, (x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1))     # west
    if col: COL.append(((x0, y0, z0), (x1, y1, z1)))


def collide(lo, hi):
    COL.append((tuple(min(a, b) for a, b in zip(lo, hi)), tuple(max(a, b) for a, b in zip(lo, hi))))


def cyl(g, p0, p1, r, seg=12, caps=True):
    p0, p1 = Vector(p0), Vector(p1)
    ax = (p1 - p0).normalized()
    u = ax.cross(Vector((0, 0, 1)))
    if u.length < 1e-4: u = ax.cross(Vector((1, 0, 0)))
    u.normalize(); v = ax.cross(u)
    ring = [(u * math.cos(TAU * i / seg) + v * math.sin(TAU * i / seg)) for i in range(seg)]
    for i in range(seg):
        a, b = ring[i], ring[(i + 1) % seg]
        tri(g, p0 + a * r, p0 + b * r, p1 + b * r, (a, b, b)); tri(g, p0 + a * r, p1 + b * r, p1 + a * r, (a, b, a))
        if caps:
            tri(g, p0, p0 + b * r, p0 + a * r); tri(g, p1, p1 + a * r, p1 + b * r)


def cone(g, p0, p1, r, seg=12):
    p0, p1 = Vector(p0), Vector(p1)
    ax = (p1 - p0).normalized()
    u = ax.cross(Vector((1, 0, 0)))
    if u.length < 1e-4: u = ax.cross(Vector((0, 1, 0)))
    u.normalize(); v = ax.cross(u)
    for i in range(seg):
        a = u * math.cos(TAU * i / seg) + v * math.sin(TAU * i / seg)
        b = u * math.cos(TAU * (i + 1) / seg) + v * math.sin(TAU * (i + 1) / seg)
        tri(g, p0 + a * r, p0 + b * r, p1)


def ball(g, c, r, sx=1.0, sz=1.0, seg=10, rings=6):
    c = Vector(c)
    def P(i, j):
        th = math.pi * j / rings; ph = TAU * i / seg
        return Vector((r * sx * math.sin(th) * math.cos(ph), r * sx * math.sin(th) * math.sin(ph), r * sz * math.cos(th)))
    for j in range(rings):
        for i in range(seg):
            a, b, c2, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
            tri(g, c + a, c + d, c + c2, (a, d, c2)); tri(g, c + a, c + c2, c + b, (a, c2, b))


# ------------------------------------------------------------------ the building's measures
F = 0.45                  # the floor: top of the plinth
EAVE = 7.05               # underside of the roof at the low side of a tooth
RIDGE = 9.25              # top of a tooth
X0, X1 = -18.0, 18.0      # the hall, west to east
YS, YN = -7.0, 7.0        # front (south, to the pad), back
BAY = 6.0
FRAMES = [X0 + BAY * i for i in range(7)]
CORE_X = 14.6             # the concrete core takes the east end from here
MEZ = 3.75                # mezzanine floor
MEZ_X0, MEZ_Y0 = 3.0, 0.6
STEEL, GLASS, WOOD, CONC = 'o_steel', 'o_glass', 'o_wood', 'o_concrete'


def glass_wall(a, b, z0, z1, n_mull, transoms=(), frame=0.07):
    """A glazed wall from a to b (x, y), z0..z1: one sheet, slim steel mullions and transoms."""
    ax, ay = a; bx, by = b
    quad2(GLASS, (ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1))
    d = Vector((bx - ax, by - ay, 0.0)); L = d.length; d.normalize()
    nx, ny = -d.y, d.x
    t = frame * 0.5
    def post(s, w):
        px, py = ax + d.x * s, ay + d.y * s
        box(STEEL, (px - abs(d.x) * w - abs(nx) * t * 1.4, py - abs(d.y) * w - abs(ny) * t * 1.4, z0),
                   (px + abs(d.x) * w + abs(nx) * t * 1.4, py + abs(d.y) * w + abs(ny) * t * 1.4, z1))
    for i in range(1, n_mull): post(L * i / n_mull, t)
    for z in list(transoms) + [z0 + t, z1 - t]:
        box(STEEL, (min(ax, bx) - abs(nx) * t * 1.4, min(ay, by) - abs(ny) * t * 1.4, z - t),
                   (max(ax, bx) + abs(nx) * t * 1.4, max(ay, by) + abs(ny) * t * 1.4, z + t))


def obox(g, x, y, ang, lo, hi):
    """A box turned by `ang` about the vertical through (x, y); lo / hi in its own frame (+Y = the way it faces)."""
    c, s = math.cos(ang), math.sin(ang)
    def W(px, py, pz): return (x + px * c - py * s, y + px * s + py * c, pz)
    x0, y0, z0 = [min(a, b) for a, b in zip(lo, hi)]; x1, y1, z1 = [max(a, b) for a, b in zip(lo, hi)]
    p = [W(x0, y0, z0), W(x1, y0, z0), W(x1, y1, z0), W(x0, y1, z0), W(x0, y0, z1), W(x1, y0, z1), W(x1, y1, z1), W(x0, y1, z1)]
    quad(g, p[4], p[5], p[6], p[7]); quad(g, p[3], p[2], p[1], p[0]); quad(g, p[0], p[1], p[5], p[4])
    quad(g, p[2], p[3], p[7], p[6]); quad(g, p[1], p[2], p[6], p[5]); quad(g, p[3], p[0], p[4], p[7])


def opt(x, y, ang, px, py, pz):
    c, s = math.cos(ang), math.sin(ang)
    return (x + px * c - py * s, y + px * s + py * c, pz)


def chair(x, y, face, g_seat='o_fabric', zf=None):
    """A task chair at (x, y), facing `face` (radians, 0 = +Y): five-star base on casters, gas
    lift, a shaped seat, a framed back on a spine, armrests."""
    z = F if zf is None else zf
    for k in range(5):
        a = face + TAU * k / 5 + 0.3
        ex, ey = x + 0.30 * math.cos(a), y + 0.30 * math.sin(a)
        cyl(STEEL, (x, y, z + 0.075), (ex, ey, z + 0.055), 0.018, 6, False)
        cyl('o_black', (ex, ey, z), (ex, ey, z + 0.05), 0.026, 8)                      # caster
    cyl(STEEL, (x, y, z + 0.05), (x, y, z + 0.11), 0.05, 10)
    cyl('o_metal', (x, y, z + 0.11), (x, y, z + 0.40), 0.026, 8, False)                # gas lift
    obox('o_black', x, y, face, (-0.10, -0.12, z + 0.40), (0.10, 0.12, z + 0.43))       # mechanism
    obox(g_seat, x, y, face, (-0.235, -0.21, z + 0.43), (0.235, 0.23, z + 0.475))
    obox(g_seat, x, y, face, (-0.215, -0.19, z + 0.475), (0.215, 0.21, z + 0.495))      # the cushion's crown
    obox('o_black', x, y, face, (-0.035, -0.285, z + 0.40), (0.035, -0.235, z + 0.66))  # the spine
    obox('o_black', x, y, face, (-0.225, -0.275, z + 0.58), (0.225, -0.245, z + 1.02))  # back frame
    obox(g_seat, x, y, face, (-0.20, -0.247, z + 0.61), (0.20, -0.225, z + 0.99))       # back pad
    for sx in (-1.0, 1.0):
        obox('o_black', x, y, face, (sx * 0.265 - 0.012, -0.03, z + 0.455), (sx * 0.265 + 0.012, 0.03, z + 0.655))
        obox('o_black', x, y, face, (sx * 0.265 - 0.03, -0.13, z + 0.655), (sx * 0.265 + 0.03, 0.12, z + 0.68))


def monitor(x, y, face, zf=None, extras=True):
    """A monitor on its stand at (x, y), its screen facing `face`; with a keyboard and a mouse
    in front of it."""
    z = (F + 0.75) if zf is None else zf
    cyl('o_black', (x, y, z), (x, y, z + 0.012), 0.105, 14)                               # foot
    obox('o_black', x, y, face, (-0.022, -0.05, z + 0.012), (0.022, -0.02, z + 0.30))      # neck
    obox('o_black', x, y, face, (-0.31, -0.02, z + 0.13), (0.31, 0.004, z + 0.50))         # the panel
    a, b, c, d = opt(x, y, face, -0.298, 0.016, z + 0.142), opt(x, y, face, 0.298, 0.016, z + 0.142), opt(x, y, face, 0.298, 0.016, z + 0.488), opt(x, y, face, -0.298, 0.016, z + 0.488)
    screen(b, a, d, c)
    if extras:
        obox('o_black', x, y, face, (-0.21, 0.24, z), (0.21, 0.37, z + 0.014))             # keyboard
        obox('o_white', x, y, face, (0.27, 0.26, z), (0.33, 0.36, z + 0.022))              # mouse


def mug(x, y, z, g='o_white'):
    cyl(g, (x, y, z), (x, y, z + 0.09), 0.038, 10)
    cyl('o_black', (x, y, z + 0.085), (x, y, z + 0.091), 0.031, 10)


def laptop(x, y, face, z):
    obox('o_metal', x, y, face, (-0.16, -0.11, z), (0.16, 0.11, z + 0.012))
    a, b, c, d = opt(x, y, face, -0.16, -0.11, z + 0.012), opt(x, y, face, 0.16, -0.11, z + 0.012), opt(x, y, face, 0.16, -0.17, z + 0.22), opt(x, y, face, -0.16, -0.17, z + 0.22)
    quad2('o_metal', a, b, c, d)
    e, f2, g2, h = opt(x, y, face, -0.15, -0.108, z + 0.022), opt(x, y, face, 0.15, -0.108, z + 0.022), opt(x, y, face, 0.15, -0.163, z + 0.212), opt(x, y, face, -0.15, -0.163, z + 0.212)
    screen(e, f2, g2, h)


def bench_desk(xa, xb, y, seats):
    """A long shared desk along X, people on both sides: timber top on steel trestles, a cable
    spine, a monitor, keyboard and mouse at every place."""
    box(WOOD, (xa, y - 0.8, F + 0.71), (xb, y + 0.8, F + 0.75))
    box(STEEL, (xa + 0.2, y - 0.03, F + 0.55), (xb - 0.2, y + 0.03, F + 0.71))          # the spine under it
    box('o_black', (xa + 0.3, y - 0.045, F + 0.75), (xb - 0.3, y + 0.045, F + 0.775))     # cable lid
    for x in (xa + 0.12, (xa + xb) * 0.5, xb - 0.12):
        box(STEEL, (x - 0.03, y - 0.74, F + 0.06), (x + 0.03, y - 0.68, F + 0.71)); box(STEEL, (x - 0.03, y + 0.68, F + 0.06), (x + 0.03, y + 0.74, F + 0.71))
        box(STEEL, (x - 0.03, y - 0.76, F), (x + 0.03, y + 0.76, F + 0.06))
    collide((xa, y - 0.8, F), (xb, y + 0.8, F + 0.78))
    for i in range(seats):
        x = xa + (xb - xa) * (i + 0.5) / seats
        monitor(x, y - 0.26, math.pi); monitor(x, y + 0.26, 0.0)
        if RNG.random() < 0.55: mug(x + 0.42, y - 0.5 - 0.1 * RNG.random(), F + 0.75, 'o_orange' if RNG.random() < 0.3 else 'o_white')
        if RNG.random() < 0.55: mug(x - 0.42, y + 0.5 + 0.1 * RNG.random(), F + 0.75)
        chair(x + RNG.uniform(-0.12, 0.12), y - 1.28 - RNG.uniform(0.0, 0.25), RNG.uniform(-0.35, 0.35), 'o_orange' if i % 4 == 1 else 'o_fabric')
        chair(x + RNG.uniform(-0.12, 0.12), y + 1.28 + RNG.uniform(0.0, 0.25), math.pi + RNG.uniform(-0.35, 0.35), 'o_orange' if i % 4 == 2 else 'o_fabric')


def leaf(g, base, yaw, pitch, length, width):
    """One broad leaf: a pointed oval, creased along its midrib, seen from both sides."""
    bx, by, bz = base
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    def P(along, across, lift=0.0):
        r = along * cp - lift * sp; zz = along * sp + lift * cp
        return (bx + r * cy - across * sy, by + r * sy + across * cy, bz + zz)
    tip = P(length, 0.0, -0.10 * length)
    m1, m2 = P(0.34 * length, 0.0, 0.03 * length), P(0.72 * length, 0.0, 0.0)
    l1, l2 = P(0.34 * length, 0.5 * width), P(0.72 * length, 0.38 * width)
    r1, r2 = P(0.34 * length, -0.5 * width), P(0.72 * length, -0.38 * width)
    b0 = P(0.0, 0.0)
    for a, b, c in ((b0, m1, l1), (b0, r1, m1), (m1, m2, l2), (m1, l2, l1), (m1, r2, m2), (m1, r1, r2), (m2, tip, l2), (m2, r2, tip)):
        tri(g, a, b, c); tri(g, c, b, a)


def planter(x, y, z, r=0.42, h=0.7, big=1.0):
    """A plant in a round concrete pot: a slim stem with broad leaves up its length and a crown."""
    cyl(CONC, (x, y, z), (x, y, z + h), r, 18)
    cyl(CONC, (x, y, z + h - 0.03), (x, y, z + h), r + 0.02, 18)                       # the rim
    cyl('o_soil', (x, y, z + h - 0.05), (x, y, z + h - 0.02 + 0.03), r * 0.88, 14)
    top = z + h + 1.25 * big
    cyl('o_bark', (x, y, z + h - 0.02), (x + 0.03 * big, y - 0.02 * big, top), 0.022 * max(big, 0.8), 6, False)
    n = int(26 * max(big, 0.8))
    for k in range(n):
        t = (k + 0.6) / n
        a = k * 2.39996 + RNG.uniform(-0.25, 0.25)
        zz = z + h + (0.20 + 1.05 * t) * big
        size = (0.56 - 0.16 * t) * big * RNG.uniform(0.85, 1.12)
        leaf('o_plant' if k % 3 else 'o_plant_d', (x + 0.03 * big * t, y - 0.02 * big * t, zz), a, 0.50 - 0.25 * t + RNG.uniform(-0.15, 0.15), size, size * 0.52)
    for k in range(7):
        leaf('o_plant', (x + 0.03 * big, y - 0.02 * big, top), k * TAU / 7 + 0.4, 0.75 + 0.25 * (k % 2), 0.40 * big, 0.20 * big)
    collide((x - r, y - r, z), (x + r, y + r, z + h + 0.9))


def sofa(xa, xb, ya, yb, back):
    """A sofa: a slim plinth on steel legs, separate seat and back cushions, low arms.
    back: 'N', 'S', 'E' or 'W' - the side its backrest is on."""
    along_x = back in ('N', 'S')
    box('o_black', (xa + 0.04, ya + 0.04, F + 0.12), (xb - 0.04, yb - 0.04, F + 0.22))
    for (lx, ly) in ((xa + 0.08, ya + 0.08), (xb - 0.08, ya + 0.08), (xa + 0.08, yb - 0.08), (xb - 0.08, yb - 0.08)):
        cyl(STEEL, (lx, ly, F), (lx, ly, F + 0.12), 0.018, 6)
    L = (xb - xa) if along_x else (yb - ya)
    n = max(1, int(round(L / 0.85)))
    arm = 0.14
    for i in range(n):
        u0 = arm + (L - 2 * arm) * i / n + 0.012; u1 = arm + (L - 2 * arm) * (i + 1) / n - 0.012
        if along_x:
            y0s, y1s = (ya + 0.02, yb - 0.24) if back == 'N' else (ya + 0.24, yb - 0.02)
            box('o_fabric', (xa + u0, y0s, F + 0.22), (xa + u1, y1s, F + 0.44))
            yb0, yb1 = (yb - 0.24, yb - 0.02) if back == 'N' else (ya + 0.02, ya + 0.24)
            box('o_fabric', (xa + u0, yb0, F + 0.30), (xa + u1, yb1, F + 0.84))
        else:
            x0s, x1s = (xa + 0.02, xb - 0.24) if back == 'E' else (xa + 0.24, xb - 0.02)
            box('o_fabric', (x0s, ya + u0, F + 0.22), (x1s, ya + u1, F + 0.44))
            xb0, xb1 = (xb - 0.24, xb - 0.02) if back == 'E' else (xa + 0.02, xa + 0.24)
            box('o_fabric', (xb0, ya + u0, F + 0.30), (xb1, ya + u1, F + 0.84))
    if along_x:
        box('o_fabric', (xa, ya, F + 0.20), (xa + arm, yb, F + 0.62)); box('o_fabric', (xb - arm, ya, F + 0.20), (xb, yb, F + 0.62))
        cx, cyy = xa + arm + 0.22, (ya + yb) * 0.5
    else:
        box('o_fabric', (xa, ya, F + 0.20), (xb, ya + arm, F + 0.62)); box('o_fabric', (xa, yb - arm, F + 0.20), (xb, yb, F + 0.62))
        cx, cyy = (xa + xb) * 0.5, ya + arm + 0.22
    obox('o_orange', cx, cyy, 0.5, (-0.19, -0.19, F + 0.44), (0.19, 0.19, F + 0.56))      # a loose cushion
    collide((xa, ya, F), (xb, yb, F + 0.86))


def books(xa, xb, y0, y1, z):
    """A shelf's worth of books and a few boxes, standing on height z."""
    x = xa + 0.05
    cols = ('o_white', 'o_orange', 'o_fabric', 'o_black', 'o_wall', 'o_corten', 'o_white', 'o_black')
    while x < xb - 0.08:
        if RNG.random() < 0.12:
            x += RNG.uniform(0.08, 0.3); continue
        if RNG.random() < 0.1:              # a storage box
            w = RNG.uniform(0.22, 0.3)
            if x + w < xb - 0.04: box('o_wall', (x, y0 + 0.02, z), (x + w, y1 - 0.03, z + RNG.uniform(0.16, 0.24)))
            x += w + 0.02; continue
        w = RNG.uniform(0.022, 0.055); hgt = RNG.uniform(0.19, 0.31)
        box(cols[int(RNG.random() * len(cols))], (x, y0 + 0.03 + RNG.uniform(0.0, 0.05), z), (x + w, y1 - 0.02, z + hgt))
        x += w + 0.003


def pendant_bar(xa, xb, y, z, drop_from):
    """A linear light: a slim black bar with a glowing underside, on two cables."""
    box('o_black', (xa, y - 0.045, z), (xb, y + 0.045, z + 0.07))
    quad('o_light', (xa + 0.03, y + 0.035, z - 0.004), (xb - 0.03, y + 0.035, z - 0.004), (xb - 0.03, y - 0.035, z - 0.004), (xa + 0.03, y - 0.035, z - 0.004))
    for x in (xa + 0.6, xb - 0.6):
        cyl('o_black', (x, y, z + 0.07), (x, y, drop_from), 0.006, 5, False)


def build():
    # ---------------------------------------------------------------- plinth and terrace
    box(CONC, (X0 - 1.5, YS - 6.0, -0.9), (X1 + 2.6, YN + 0.6, F), True)
    box('o_floor', (X0 + 0.02, YS + 0.02, F), (CORE_X, YN - 0.02, F + 0.012))                 # the hall's polished floor
    # timber deck let into the terrace, in front of the glass
    box(WOOD, (X0 + 1.0, YS - 4.6, F), (CORE_X - 1.0, YS - 0.5, F + 0.02))
    # steps down to the apron, across the middle
    for k in range(3):
        box(CONC, (-7.0, YS - 6.0 - 0.36 * (k + 1), -0.9), (7.0, YS - 6.0 - 0.36 * k, F - 0.15 * (k + 1)), True)
    # a low wall-bench along the terrace's edge either side of the steps
    for xa, xb in ((X0 - 1.5, -7.6), (7.6, X1 + 2.6)):
        box(CONC, (xa, YS - 6.0, F), (xb, YS - 5.55, F + 0.46), True)
        box(WOOD, (xa + 0.2, YS - 5.98, F + 0.46), (xb - 0.2, YS - 5.57, F + 0.50))

    # ---------------------------------------------------------------- the steel frame
    for x in FRAMES:
        for y in (YS + 0.16, YN - 0.16):
            box(STEEL, (x - 0.15, y - 0.15, F), (x + 0.15, y + 0.15, EAVE))
            box(STEEL, (x - 0.22, y - 0.22, F), (x + 0.22, y + 0.22, F + 0.03))               # base plate
        # the beam: across the hall and on over the terrace
        box(STEEL, (x - 0.13, YS - 3.6, EAVE - 0.42), (x + 0.13, YN, EAVE))
        box(STEEL, (x - 0.2, YS - 3.6, EAVE - 0.03), (x + 0.2, YN, EAVE))                      # its top flange
        box(STEEL, (x - 0.2, YS - 3.6, EAVE - 0.42), (x + 0.2, YN, EAVE - 0.39))               # ... and bottom
        collide((x - 0.16, YS, F), (x + 0.16, YS + 0.32, EAVE)); collide((x - 0.16, YN - 0.32, F), (x + 0.16, YN, EAVE))
    # eave beams along both long sides, and the fascia of the canopy
    for y in (YS + 0.16, YN - 0.16):
        box(STEEL, (X0, y - 0.12, EAVE - 0.36), (X1, y + 0.12, EAVE))
    box(STEEL, (X0 - 0.13, YS - 3.66, EAVE - 0.42), (X1 + 0.13, YS - 3.54, EAVE))
    # the brise-soleil: timber fins between the beams, over the terrace
    y = YS - 3.4
    while y < YS - 0.35:
        for i in range(6):
            box(WOOD, (FRAMES[i] + 0.21, y - 0.03, EAVE - 0.30), (FRAMES[i + 1] - 0.21, y + 0.03, EAVE - 0.04))
        y += 0.42

    # ---------------------------------------------------------------- the sawtooth roof
    for i in range(6):
        xa, xb = FRAMES[i], FRAMES[i + 1]
        # the slope (falls to the east), metal above, timber-lined below
        quad('o_roof', (xa, YS - 0.3, RIDGE), (xb, YS - 0.3, EAVE + 0.02), (xb, YN + 0.3, EAVE + 0.02), (xa, YN + 0.3, RIDGE))
        quad('o_ceiling', (xa + 0.05, YN, RIDGE - 0.16), (xb, YN, EAVE - 0.14 + 0.02), (xb, YS, EAVE - 0.14 + 0.02), (xa + 0.05, YS, RIDGE - 0.16))
        # the upright north-light face (looks west), glazed, with its frame
        quad2(GLASS, (xa + 0.04, YS + 0.3, EAVE + 0.1), (xa + 0.04, YN - 0.3, EAVE + 0.1), (xa + 0.04, YN - 0.3, RIDGE - 0.12), (xa + 0.04, YS + 0.3, RIDGE - 0.12))
        box(STEEL, (xa - 0.06, YS - 0.3, RIDGE - 0.12), (xa + 0.12, YN + 0.3, RIDGE + 0.04))     # ridge cap
        box(STEEL, (xa - 0.02, YS, EAVE - 0.0), (xa + 0.1, YN, EAVE + 0.1))
        for k in range(1, 7):
            yy = YS + (YN - YS) * k / 7
            box(STEEL, (xa, yy - 0.03, EAVE + 0.1), (xa + 0.08, yy + 0.03, RIDGE - 0.12))
        # the gable triangles at the two long sides
        for yy, g in ((YS - 0.02, 'o_corten'), (YN + 0.02, 'o_corten')):
            tri(g, (xa, yy, EAVE), (xb, yy, EAVE), (xa, yy, RIDGE)); tri(g, (xb, yy, EAVE), (xa, yy, EAVE), (xa, yy, RIDGE))
        # purlins under the slope
        for k in range(1, 4):
            t = k / 4.0
            xx = xa + (xb - xa) * t; zz = RIDGE - 0.2 + (EAVE - RIDGE) * t
            box(STEEL, (xx - 0.04, YS, zz - 0.12), (xx + 0.04, YN, zz))
    # a gutter line along the low side of every tooth
    for i in range(1, 7):
        box(STEEL, (FRAMES[i] - 0.16, YS - 0.3, EAVE - 0.02), (FRAMES[i] + 0.02, YN + 0.3, EAVE + 0.06))

    # ---------------------------------------------------------------- the envelope
    # South: glass from floor to eave in every bay; the middle bay has the doors.
    for i in range(6):
        xa, xb = FRAMES[i] + 0.15, FRAMES[i + 1] - 0.15
        if xb > CORE_X: continue
        if i == 2:                # the entrance: an open doorway 2.6 m wide in a steel portal
            mid = (xa + xb) * 0.5
            glass_wall((xa, YS + 0.16), (mid - 1.3, YS + 0.16), F, EAVE - 0.36, 1, (F + 2.95,))
            glass_wall((mid + 1.3, YS + 0.16), (xb, YS + 0.16), F, EAVE - 0.36, 1, (F + 2.95,))
            glass_wall((mid - 1.3, YS + 0.16), (mid + 1.3, YS + 0.16), F + 2.95, EAVE - 0.36, 2)
            box(STEEL, (mid - 1.42, YS - 0.10, F), (mid - 1.3, YS + 0.42, F + 3.07)); box(STEEL, (mid + 1.3, YS - 0.10, F), (mid + 1.42, YS + 0.42, F + 3.07))
            box(STEEL, (mid - 1.42, YS - 0.10, F + 2.95), (mid + 1.42, YS + 0.42, F + 3.07))
            box('o_orange', (mid - 1.42, YS - 0.9, F + 3.07), (mid + 1.42, YS + 0.16, F + 3.13))     # a thin canopy over the door
            collide((xa, YS + 0.08, F), (mid - 1.3, YS + 0.24, EAVE)); collide((mid + 1.3, YS + 0.08, F), (xb, YS + 0.24, EAVE))
            collide((mid - 1.3, YS + 0.08, F + 2.95), (mid + 1.3, YS + 0.24, EAVE))
        else:
            glass_wall((xa, YS + 0.16), (xb, YS + 0.16), F, EAVE - 0.36, 4, (F + 2.95,))
            collide((xa, YS + 0.08, F), (xb, YS + 0.24, EAVE))
    # West: glass too - the corner of the hall is all window.
    glass_wall((X0 + 0.16, YS + 0.3), (X0 + 0.16, YN - 0.3), F, EAVE - 0.36, 8, (F + 2.95,))
    collide((X0 + 0.08, YS, F), (X0 + 0.24, YN, EAVE))
    quad2('o_corten', (X0 - 0.02, YS, EAVE - 0.36), (X0 - 0.02, YN, EAVE - 0.36), (X0 - 0.02, YN, RIDGE), (X0 - 0.02, YS, RIDGE))     # the end of the first tooth
    # North: weathering steel in standing seams up to a clerestory.
    box('o_corten', (X0, YN - 0.06, F), (CORE_X, YN + 0.16, 5.3))
    x = X0 + 0.3
    while x < CORE_X:
        box('o_corten', (x - 0.02, YN + 0.16, F), (x + 0.02, YN + 0.22, 5.3)); x += 0.6
    box('o_wall', (X0 + 0.3, YN - 0.10, F), (CORE_X, YN - 0.06, 5.3))                            # its inner lining
    glass_wall((X0 + 0.3, YN + 0.05), (CORE_X, YN + 0.05), 5.3, EAVE - 0.36, 16)
    collide((X0, YN - 0.12, F), (CORE_X, YN + 0.2, EAVE))

    # ---------------------------------------------------------------- the concrete core (east end)
    box(CONC, (CORE_X, YS - 0.5, F), (X1 + 1.4, YN + 0.4, RIDGE + 0.5), True)
    # board marks: shallow horizontal shadow joints
    z = F + 0.6
    while z < RIDGE + 0.4:
        box('o_joint', (CORE_X - 0.012, YS - 0.512, z), (X1 + 1.412, YN + 0.412, z + 0.025)); z += 0.6
    # a recessed door on its hall side, and the name on its front
    box('o_black', (CORE_X - 0.03, -1.6, F), (CORE_X + 0.02, -0.5, F + 2.2))
    box(STEEL, (CORE_X - 0.05, -1.7, F), (CORE_X + 0.0, -1.6, F + 2.3)); box(STEEL, (CORE_X - 0.05, -0.5, F), (CORE_X + 0.0, -0.4, F + 2.3)); box(STEEL, (CORE_X - 0.05, -1.7, F + 2.2), (CORE_X + 0.0, -0.4, F + 2.3))

    # ---------------------------------------------------------------- the mezzanine
    box(CONC, (MEZ_X0, MEZ_Y0, MEZ - 0.22), (CORE_X, YN - 0.1, MEZ), True)
    box(WOOD, (MEZ_X0 + 0.02, MEZ_Y0 + 0.02, MEZ), (CORE_X, YN - 0.12, MEZ + 0.02))
    box('o_soffit', (MEZ_X0 + 0.12, MEZ_Y0 + 0.12, MEZ - 0.27), (CORE_X, YN - 0.12, MEZ - 0.22))          # a white soffit under it
    box(STEEL, (MEZ_X0 - 0.1, MEZ_Y0 - 0.1, MEZ - 0.42), (CORE_X, MEZ_Y0 + 0.1, MEZ - 0.02))     # edge beams
    box(STEEL, (MEZ_X0 - 0.1, MEZ_Y0 - 0.1, MEZ - 0.42), (MEZ_X0 + 0.1, YN - 0.1, MEZ - 0.02))
    for x in (MEZ_X0, 9.0):
        box(STEEL, (x - 0.1, MEZ_Y0 - 0.1, F), (x + 0.1, MEZ_Y0 + 0.1, MEZ - 0.42))
        collide((x - 0.12, MEZ_Y0 - 0.12, F), (x + 0.12, MEZ_Y0 + 0.12, MEZ))
    # glass balustrades with a steel rail (open where the stair arrives: x 3.0 .. 4.3 at the back)
    def balustrade(a, b):
        quad2(GLASS, (a[0], a[1], MEZ + 0.02), (b[0], b[1], MEZ + 0.02), (b[0], b[1], MEZ + 1.08), (a[0], a[1], MEZ + 1.08))
        box(STEEL, (min(a[0], b[0]) - 0.025, min(a[1], b[1]) - 0.025, MEZ + 1.08), (max(a[0], b[0]) + 0.025, max(a[1], b[1]) + 0.025, MEZ + 1.12))
        collide((min(a[0], b[0]) - 0.05, min(a[1], b[1]) - 0.05, MEZ), (max(a[0], b[0]) + 0.05, max(a[1], b[1]) + 0.05, MEZ + 1.12))
    balustrade((MEZ_X0, MEZ_Y0), (CORE_X, MEZ_Y0))
    balustrade((MEZ_X0, MEZ_Y0), (MEZ_X0, 5.05))

    # ---------------------------------------------------------------- the stair (along the back wall, rising east)
    n_steps = 18
    sx0, sx1 = -4.6, MEZ_X0
    rise = (MEZ - F) / n_steps; going = (sx1 - sx0) / n_steps
    for k in range(n_steps):
        xa = sx0 + going * k
        ztop = F + rise * (k + 1)
        box(WOOD, (xa, 5.2, ztop - 0.05), (xa + going + 0.03, 6.6, ztop))
        collide((xa, 5.2, F if k < 4 else ztop - 0.35), (xa + going, 6.6, ztop))
    for y in (5.15, 6.65):          # the two stringers
        quad2(STEEL, (sx0 - 0.1, y, F), (sx0 + 0.3, y, F), (sx1 + 0.05, y, MEZ - 0.02), (sx1 - 0.35, y, MEZ - 0.02))
        quad2(STEEL, (sx0 - 0.1, y, F), (sx1 - 0.35, y, MEZ - 0.02), (sx1 - 0.35, y, MEZ - 0.3), (sx0 - 0.1, y, F - 0.0))
    # its balustrade on the open side: glass under a rail
    quad2(GLASS, (sx0, 5.15, F + rise + 0.05), (sx1, 5.15, MEZ + 0.05), (sx1, 5.15, MEZ + 1.05), (sx0, 5.15, F + rise + 1.05))
    cyl(STEEL, (sx0, 5.15, F + rise + 1.07), (sx1, 5.15, MEZ + 1.07), 0.025, 8)
    cyl(STEEL, (sx0, 5.15, F), (sx0, 5.15, F + rise + 1.07), 0.025, 8)
    collide((sx0, 5.07, F), (sx1, 5.19, F + 1.2)); collide((sx0 + 2.5, 5.07, F + 1.0), (sx1, 5.19, MEZ + 1.1))

    # ---------------------------------------------------------------- under the mezzanine: meeting room and kitchen
    # the meeting room: a glass box, x 9 .. core, y 1 .. back
    glass_wall((9.0, 1.2), (CORE_X, 1.2), F, MEZ - 0.27, 4)
    collide((9.0, 1.14, F), (CORE_X, 1.26, MEZ))
    glass_wall((9.0, 2.5), (9.0, YN - 0.1), F, MEZ - 0.27, 3)
    collide((8.94, 2.5, F), (9.06, YN, MEZ))
    box(STEEL, (8.95, 1.2, F), (9.05, 1.3, MEZ - 0.27)); box(STEEL, (8.95, 2.4, F), (9.05, 2.5, MEZ - 0.27)); box(STEEL, (8.95, 1.3, F + 2.2), (9.05, 2.4, F + 2.26))    # the door's posts
    box(WOOD, (10.0, 3.3, F + 0.72), (13.6, 4.7, F + 0.76))                                     # the table
    for x in (10.5, 13.1):
        box(STEEL, (x - 0.04, 3.6, F), (x + 0.04, 4.4, F + 0.72))
    collide((10.0, 3.3, F), (13.6, 4.7, F + 0.78))
    for i in range(4):
        x = 10.45 + 0.9 * i
        chair(x + RNG.uniform(-0.08, 0.08), 2.78, RNG.uniform(-0.3, 0.3)); chair(x + RNG.uniform(-0.08, 0.08), 5.22, math.pi + RNG.uniform(-0.3, 0.3))
    laptop(11.0, 3.75, 0.1, F + 0.76); laptop(12.7, 4.3, math.pi - 0.15, F + 0.76); mug(11.5, 3.6, F + 0.76); mug(12.2, 4.45, F + 0.76, 'o_orange')
    cyl('o_black', (11.8, 4.0, F + 0.76), (11.8, 4.0, F + 0.79), 0.09, 14)                           # the conference speaker
    box('o_black', (CORE_X - 0.06, 2.6, F + 1.1), (CORE_X - 0.01, 5.4, F + 2.55))                    # the wall screen
    screen((CORE_X - 0.078, 5.35, F + 1.15), (CORE_X - 0.078, 2.65, F + 1.15), (CORE_X - 0.078, 2.65, F + 2.5), (CORE_X - 0.078, 5.35, F + 2.5))
    box('o_rug', (9.6, 2.4, F + 0.012), (14.0, 5.6, F + 0.022))
    pendant_bar(10.4, 13.2, 4.0, F + 2.2, MEZ - 0.22)
    # the kitchen: a run of units along the back wall up to the frame's column (x = 6), the
    # fridge and a tall cupboard beyond it, and an island
    box('o_white', (3.6, YN - 0.72, F + 0.1), (5.78, YN - 0.12, F + 0.9)); box('o_black', (3.62, YN - 0.66, F), (5.76, YN - 0.12, F + 0.1))
    box(WOOD, (3.55, YN - 0.76, F + 0.9), (5.8, YN - 0.1, F + 0.94))
    for k in range(4):          # the drawer fronts' joints and handles
        xk = 3.6 + 0.545 * (k + 1)
        if k < 3: box('o_black', (xk - 0.004, YN - 0.724, F + 0.1), (xk + 0.004, YN - 0.72, F + 0.9))
        box(STEEL, (xk - 0.40, YN - 0.745, F + 0.78), (xk - 0.14, YN - 0.72, F + 0.80))
    box('o_black', (3.85, YN - 0.62, F + 0.90), (4.5, YN - 0.22, F + 0.945)); box('o_metal', (3.89, YN - 0.58, F + 0.86), (4.46, YN - 0.26, F + 0.944))      # the sink
    cyl('o_metal', (4.17, YN - 0.2, F + 0.94), (4.17, YN - 0.2, F + 1.2), 0.014, 8); cyl('o_metal', (4.17, YN - 0.2, F + 1.2), (4.17, YN - 0.4, F + 1.2), 0.014, 8)
    box('o_black', (4.85, YN - 0.6, F + 0.94), (5.6, YN - 0.2, F + 0.95))                           # the hob
    box('o_white', (3.6, YN - 0.42, F + 1.5), (5.78, YN - 0.12, F + 2.2))                           # wall cupboards
    for k in range(1, 4):
        box('o_black', (3.6 + 0.545 * k - 0.004, YN - 0.424, F + 1.5), (3.6 + 0.545 * k + 0.004, YN - 0.42, F + 2.2))
    box('o_light', (3.7, YN - 0.40, F + 1.49), (5.7, YN - 0.34, F + 1.5))                            # a strip of light under them
    collide((3.55, YN - 0.78, F), (5.82, YN, F + 2.2)); mug(4.7, YN - 0.5, F + 0.94); mug(5.7, YN - 0.3, F + 0.94, 'o_orange')
    box('o_metal', (6.3, YN - 0.74, F + 0.03), (7.2, YN - 0.12, F + 2.0)); box('o_black', (6.32, YN - 0.72, F), (7.18, YN - 0.14, F + 0.03))        # the fridge
    box('o_black', (6.3, YN - 0.744, F + 0.72), (7.2, YN - 0.74, F + 0.73)); box(STEEL, (6.36, YN - 0.77, F + 0.9), (6.385, YN - 0.74, F + 1.6))
    box('o_white', (7.22, YN - 0.72, F + 0.1), (8.5, YN - 0.12, F + 2.2)); box('o_black', (7.24, YN - 0.66, F), (8.48, YN - 0.12, F + 0.1))         # a tall cupboard
    box('o_black', (7.856, YN - 0.724, F + 0.1), (7.864, YN - 0.72, F + 2.2)); box(STEEL, (7.78, YN - 0.745, F + 0.9), (7.80, YN - 0.72, F + 1.4)); box(STEEL, (7.92, YN - 0.745, F + 0.9), (7.94, YN - 0.72, F + 1.4))
    collide((6.3, YN - 0.78, F), (8.5, YN, F + 2.2))
    box('o_white', (4.2, 3.5, F + 0.1), (7.6, 4.3, F + 0.9)); box('o_black', (4.25, 3.55, F), (7.55, 4.25, F + 0.1)); box(WOOD, (4.1, 3.4, F + 0.9), (7.7, 4.4, F + 0.95))
    laptop(6.6, 3.85, 0.2, F + 0.95); mug(7.1, 3.7, F + 0.95)
    collide((4.1, 3.4, F), (7.7, 4.4, F + 0.96))
    for i in range(4):          # bar stools
        x = 4.6 + 0.85 * i
        cyl(STEEL, (x, 3.0, F), (x, 3.0, F + 0.68), 0.025, 8, False); cyl('o_orange', (x, 3.0, F + 0.68), (x, 3.0, F + 0.73), 0.19, 16)
        for q in range(12):
            a0, a1 = TAU * q / 12, TAU * (q + 1) / 12
            cyl(STEEL, (x + 0.15 * math.cos(a0), 3.0 + 0.15 * math.sin(a0), F + 0.28), (x + 0.15 * math.cos(a1), 3.0 + 0.15 * math.sin(a1), F + 0.28), 0.01, 5, False)
        cyl(STEEL, (x, 3.0, F), (x, 3.0, F + 0.02), 0.2, 12)
    for i in range(3):          # three cone pendants over the island
        x = 4.9 + 1.0 * i
        cyl('o_black', (x, 3.9, F + 1.95), (x, 3.9, MEZ - 0.22), 0.006, 5, False)
        cone('o_black', (x, 3.9, F + 1.72), (x, 3.9, F + 1.98), 0.16, 12)
        cyl('o_light', (x, 3.9, F + 1.715), (x, 3.9, F + 1.725), 0.13, 12)

    # ---------------------------------------------------------------- the hall
    for y in (-3.3, 0.6):
        bench_desk(-15.6, -6.4, y, 6)
        pendant_bar(-15.2, -6.8, y, F + 2.35, EAVE - 0.42)
    # the mission wall: twelve screens on the back wall, six in each bay between the frame's columns
    for (ga, gb) in ((-17.45, -12.3), (-11.7, -6.3)):
        box('o_black', (ga, YN - 0.2, F + 1.5), (gb, YN - 0.11, F + 4.6))
        wcell = (gb - ga - 0.1) / 3.0
        for i in range(3):
            for j in range(2):
                xa = ga + 0.05 + wcell * i + 0.035; za = F + 1.6 + 1.5 * j
                amber = (i + j + (1 if ga > -15.0 else 0)) % 4 == 0
                screen((xa, YN - 0.218, za), (xa + wcell - 0.07, YN - 0.218, za), (xa + wcell - 0.07, YN - 0.218, za + 1.42), (xa, YN - 0.218, za + 1.42))
        box(WOOD, (ga, YN - 0.55, F + 0.08), (gb, YN - 0.12, F + 0.5)); box('o_black', (ga + 0.04, YN - 0.5, F), (gb - 0.04, YN - 0.14, F + 0.08))    # a credenza under each
        for k in range(1, 4):
            xk = ga + (gb - ga) * k / 4.0
            box('o_black', (xk - 0.004, YN - 0.554, F + 0.08), (xk + 0.004, YN - 0.55, F + 0.5))
        collide((ga, YN - 0.55, F), (gb, YN, F + 0.5))
    # the lounge by the west glass
    box('o_rug', (-0.6, -6.0, F + 0.012), (4.9, -2.4, F + 0.022))
    sofa(-0.1, 2.7, -3.35, -2.45, 'N'); sofa(3.65, 4.55, -5.7, -3.3, 'E')
    box(WOOD, (0.6, -5.0, F + 0.34), (2.3, -4.1, F + 0.38))
    for (lx, ly) in ((0.68, -4.92), (2.22, -4.92), (0.68, -4.18), (2.22, -4.18)):
        box(STEEL, (lx - 0.02, ly - 0.02, F), (lx + 0.02, ly + 0.02, F + 0.34))
    box(STEEL, (0.66, -4.94, F + 0.30), (2.24, -4.16, F + 0.34)); mug(1.0, -4.4, F + 0.38); mug(1.9, -4.7, F + 0.38, 'o_orange')
    obox('o_white', 1.45, -4.55, 0.3, (-0.15, -0.11, F + 0.38), (0.15, 0.11, F + 0.40)); obox('o_fabric', 1.47, -4.56, 0.42, (-0.13, -0.10, F + 0.40), (0.13, 0.10, F + 0.415))
    collide((0.6, -5.0, F), (2.3, -4.1, F + 0.4))
    # a ring light over the lounge
    for k in range(24):
        a0, a1 = TAU * k / 24, TAU * (k + 1) / 24
        p0 = (1.5 + 1.3 * math.cos(a0), -4.2 + 1.3 * math.sin(a0)); p1 = (1.5 + 1.3 * math.cos(a1), -4.2 + 1.3 * math.sin(a1))
        cyl('o_light', (p0[0], p0[1], F + 2.9), (p1[0], p1[1], F + 2.9), 0.03, 6, False)
    for a in (0.4, 2.5, 4.6):
        cyl('o_black', (1.5 + 1.3 * math.cos(a), -4.2 + 1.3 * math.sin(a), F + 2.9), (1.5 + 0.3 * math.cos(a), -4.2 + 0.3 * math.sin(a), EAVE - 0.42), 0.006, 5, False)
    # reception: a concrete and timber desk facing the door
    box(CONC, (5.2, -4.9, F), (5.6, -2.3, F + 1.05)); box(WOOD, (5.1, -5.0, F + 1.05), (6.3, -2.2, F + 1.09)); box('o_white', (5.6, -4.9, F), (6.3, -2.3, F + 0.74))
    collide((5.1, -5.0, F), (6.3, -2.2, F + 1.1))
    monitor(5.92, -3.6, -math.pi * 0.5, F + 0.74); chair(6.95, -3.55, math.pi * 0.5 + 0.2); mug(5.9, -2.9, F + 0.74)
    # a model of the rocket on a plinth, by the entrance
    box(CONC, (-5.7, -6.3, F), (-4.7, -5.3, F + 0.5), True)
    cyl('o_white', (-5.2, -5.8, F + 0.56), (-5.2, -5.8, F + 2.5), 0.16, 20)
    cone('o_white', (-5.2, -5.8, F + 2.5), (-5.2, -5.8, F + 2.95), 0.16, 20)
    cyl('o_black', (-5.2, -5.8, F + 0.5), (-5.2, -5.8, F + 0.56), 0.13, 16)
    for k in range(4):
        a = TAU * k / 4 + 0.4
        tri('o_black', (-5.2 + 0.16 * math.cos(a), -5.8 + 0.16 * math.sin(a), F + 0.56), (-5.2 + 0.36 * math.cos(a), -5.8 + 0.36 * math.sin(a), F + 0.56), (-5.2 + 0.16 * math.cos(a), -5.8 + 0.16 * math.sin(a), F + 1.0))
        tri('o_black', (-5.2 + 0.36 * math.cos(a), -5.8 + 0.36 * math.sin(a), F + 0.56), (-5.2 + 0.16 * math.cos(a), -5.8 + 0.16 * math.sin(a), F + 0.56), (-5.2 + 0.16 * math.cos(a), -5.8 + 0.16 * math.sin(a), F + 1.0))
    # exposed services: a round duct down the hall with outlets, and a cable tray
    cyl('o_metal', (X0 + 0.8, -2.2, 6.2), (CORE_X - 0.2, -2.2, 6.2), 0.3, 18)
    for x in (-15.0, -9.0, -3.0, 3.0, 9.0):
        cyl('o_metal', (x, -2.2, 5.7), (x, -2.2, 5.95), 0.2, 12); cyl('o_metal', (x, -2.2 - 0.33, 6.2), (x, -2.2 + 0.33, 6.2), 0.312, 14)
        cyl('o_black', (x, -2.2, 6.5), (x, -2.2, EAVE - 0.42), 0.008, 5, False)
    box('o_metal', (X0 + 0.8, -4.4, 6.0), (CORE_X - 0.2, -4.0, 6.04))
    # plants
    for (x, y) in ((-16.9, -6.1), (-16.9, 5.9), (-0.9, -5.9), (2.2, -1.3), (8.2, -6.0), (13.6, -6.0)):
        planter(x, y, F, 0.36, 0.62, 0.85)
    # ---------------------------------------------------------------- on the mezzanine
    box(WOOD, (5.0, 2.4, MEZ + 0.71), (9.0, 3.8, MEZ + 0.75))
    for x in (5.1, 8.9):
        box(STEEL, (x - 0.03, 2.5, MEZ), (x + 0.03, 3.7, MEZ + 0.71))
    collide((5.0, 2.4, MEZ), (9.0, 3.8, MEZ + 0.78))
    for i in range(3):
        x = 5.7 + 1.3 * i
        monitor(x, 2.84, math.pi, MEZ + 0.75); monitor(x, 3.36, 0.0, MEZ + 0.75)
        chair(x + RNG.uniform(-0.1, 0.1), 1.80, RNG.uniform(-0.3, 0.3), 'o_fabric', MEZ); chair(x + RNG.uniform(-0.1, 0.1), 4.42, math.pi + RNG.uniform(-0.3, 0.3), 'o_orange' if i == 1 else 'o_fabric', MEZ)
    pendant_bar(5.2, 8.8, 3.1, MEZ + 2.3, EAVE - 0.42)
    # shelving on the back wall, and two armchairs looking out over the hall
    for (sa, sb) in ((9.7, 11.72), (12.28, 14.3)):          # (two runs, either side of the frame's column)
        for k in range(4):
            box(WOOD, (sa, YN - 0.5, MEZ + 0.4 + 0.5 * k), (sb, YN - 0.12, MEZ + 0.44 + 0.5 * k))
            books(sa + 0.04, sb - 0.04, YN - 0.48, YN - 0.12, MEZ + 0.44 + 0.5 * k)
        for x in (sa, sb):
            box(STEEL, (x - 0.02, YN - 0.5, MEZ), (x + 0.02, YN - 0.12, MEZ + 1.96))
        collide((sa, YN - 0.5, MEZ), (sb, YN, MEZ + 2.0))
    for x in (11.0, 12.6):      # two armchairs looking out over the hall
        box('o_black', (x - 0.36, 1.34, MEZ + 0.12), (x + 0.36, 2.06, MEZ + 0.2))
        for (lx, ly) in ((x - 0.33, 1.37), (x + 0.33, 1.37), (x - 0.33, 2.03), (x + 0.33, 2.03)):
            cyl(STEEL, (lx, ly, MEZ + 0.02), (lx, ly, MEZ + 0.12), 0.016, 6)
        box('o_fabric', (x - 0.28, 1.32, MEZ + 0.2), (x + 0.28, 1.92, MEZ + 0.42)); box('o_fabric', (x - 0.28, 1.92, MEZ + 0.28), (x + 0.28, 2.1, MEZ + 0.84))
        box('o_fabric', (x - 0.4, 1.3, MEZ + 0.2), (x - 0.28, 2.1, MEZ + 0.58)); box('o_fabric', (x + 0.28, 1.3, MEZ + 0.2), (x + 0.4, 2.1, MEZ + 0.58))
        collide((x - 0.4, 1.3, MEZ), (x + 0.4, 2.14, MEZ + 0.85))
    cyl(STEEL, (11.8, 1.7, MEZ + 0.02), (11.8, 1.7, MEZ + 0.44), 0.02, 8); cyl(WOOD, (11.8, 1.7, MEZ + 0.44), (11.8, 1.7, MEZ + 0.47), 0.22, 18); mug(11.84, 1.66, MEZ + 0.47)
    planter(13.8, 1.4, MEZ, 0.3, 0.5, 0.7)

    # ---------------------------------------------------------------- outside
    for x in (-14.0, -9.5, 9.5):
        box(CONC, (x - 1.1, YS - 3.1, F), (x - 0.8, YS - 2.5, F + 0.42)); box(CONC, (x + 0.8, YS - 3.1, F), (x + 1.1, YS - 2.5, F + 0.42))
        box(WOOD, (x - 1.2, YS - 3.15, F + 0.42), (x + 1.2, YS - 2.45, F + 0.48))
        collide((x - 1.2, YS - 3.15, F), (x + 1.2, YS - 2.45, F + 0.48))
    for x in (-17.6, -11.6, 5.0, 16.5):
        planter(x, YS - 4.9, F, 0.55, 0.6, 1.25)
    # bollard lights along the terrace's edge
    for x in (-14.6, -8.6, 8.2, 13.6):
        cyl(STEEL, (x, YS - 5.2, F), (x, YS - 5.2, F + 0.8), 0.07, 10); cyl('o_light', (x, YS - 5.2, F + 0.8), (x, YS - 5.2, F + 0.86), 0.075, 10)
    # ---------------------------------------------------------------- where the light comes from at night
    for x in (-14.0, -8.0, 1.5):
        LIGHTS.append((x, -1.3, F + 3.4))
    LIGHTS.extend([(6.0, 3.8, F + 2.3), (11.8, 4.0, F + 2.4), (8.0, 3.2, MEZ + 2.6), (0.0, YS - 2.0, F + 3.4)])


def add_name():
    """The name on the concrete core, as raised letters (Blender's own font)."""
    try:
        cu = bpy.data.curves.new('name', 'FONT')
        cu.body = 'VIBE  AEROSPACE'
        cu.size = 0.72; cu.extrude = 0.03; cu.align_x = 'LEFT'
        ob = bpy.data.objects.new('name', cu)
        bpy.context.scene.collection.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        me.calc_loop_triangles()
        ox, oz, y = CORE_X + 0.25, F + 5.6, YS - 0.53
        for t in me.loop_triangles:
            p = [me.vertices[i].co for i in t.vertices]
            tri('o_black', *[(ox + q.x * 0.62, y - q.z, oz + q.y) for q in p])
        bpy.data.objects.remove(ob)
        # a thin steel line under it
        box(STEEL, (CORE_X + 0.25, YS - 0.53, F + 5.42), (X1 + 1.1, YS - 0.5, F + 5.46))
    except Exception as e:
        print('NAME: skipped (%s)' % e)


build()
add_name()

with open(os.path.join(OUT, 'office.mesh'), 'w') as f:
    for g, v in G.items():
        f.write('mesh %s %d\n' % (g, len(v) // 3))
        for p, n, uv in v:
            f.write('%.4f %.4f %.4f %.3f %.3f %.3f %.4f %.4f\n' % (p.x, p.z, -p.y, n.x, n.z, -n.y, uv[0], uv[1]))
with open(os.path.join(OUT, 'office.col'), 'w') as f:
    f.write('# office.mesh: colliders and lights, game space (x east, y up, z south), from its origin\n')
    for lo, hi in COL:
        f.write('box %.3f %.3f %.3f %.3f %.3f %.3f\n' % (lo[0], lo[2], -hi[1], hi[0], hi[2], -lo[1]))
    for p in LIGHTS:
        f.write('light %.3f %.3f %.3f\n' % (p[0], p[2], -p[1]))

col = bpy.data.collections.new('office'); bpy.context.scene.collection.children.link(col)
for g, v in G.items():
    me = bpy.data.meshes.new('office_' + g)
    me.from_pydata([tuple(p) for p, _, _ in v], [], [(i, i + 1, i + 2) for i in range(0, len(v), 3)])
    me.update()
    col.objects.link(bpy.data.objects.new('office_' + g, me))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'office.blend'))
print('OFFICE', {g: len(v) // 3 for g, v in G.items()}, 'colliders', len(COL), 'lights', len(LIGHTS))
