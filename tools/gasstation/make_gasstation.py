# Builds the highway service station - fuel forecourt, food mart and diner - in Blender and
# exports it for the game:
#   blender -b --python tools/gasstation/make_gasstation.py
# Output (assets/gasstation/):
#   gasstation.mesh   everything, one block per material ("mesh <name> <triangles>", then
#                     position, normal, uv in metres per vertex - the format of tools/office).
#   gasstation.col    what the player bumps into ("box"), where its lights hang ("light"),
#                     what is roofed ("shelter x0 z0 x1 z1 top") and where no grass grows
#                     ("nograss x0 z0 x1 z1") - game space, from the same origin.
#   gasstation.blend  the same geometry as Blender objects, to look at.
#
# The site (54 x 72 m, level, beside the coast highway which runs along its east side):
#   * east half: an asphalt forecourt under a steel canopy on four columns, two pump islands
#     with two dispensers each, a sloped apron out to the road, a price pylon at the corner;
#   * west half: one long flat-roofed building - brick piers, ribbed cladding, a red fascia -
#     with a food mart (shelves, fridges, counter, coffee bar) in its south part and a diner
#     (booths at the windows, counter with stools, kitchen behind a pass) in its north part;
#   * round it: parking bays, an air and water post, vacuum, ice and propane, picnic terrace,
#     a bin store, light masts, roof plant and a rooftop sign.
#
# Blender space: X east (to the road), Y north, Z up. Game space: x = X, y = Z, z = -Y.
# Origin: the middle of the plot, at the level of the terrace it stands on.
import bpy, math, os, random
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'gasstation')
os.makedirs(OUT, exist_ok=True)
TAU = 2.0 * math.pi

G = {}          # material -> [(p, n, uv)] three per triangle
COL = []        # (lo, hi) boxes the player collides with
LIGHTS = []     # points
SHELTERS = []   # (x0, y0, x1, y1, top) in Blender space
NOGRASS = []    # (x0, y0, x1, y1)
PADS = []       # (x0, y0, x1, y1, h at x0, h at x1): paved ground a car drives on
# What the people and cars that live here go by (gasstation.nav; the NPC engine, src/npc):
SEATS = []      # (kind, x, y, z of the seat's top, facing x, facing y)
NODES = []      # (name, x, y, z): where a walker can stand
LINKS = []      # (name a, name b): he can walk straight from one to the other
SPOTS = []      # (kind, x, y, z, facing x, facing y): a place to stand and do something
BAYS = []       # (x, y, z, facing x, facing y): a parked car's place
STOPS = []      # (name, x, y, z, facing x, facing y): where a car stops at a pump
PATHS = []      # (name, [(x, y, z)]): a car's way, in order
RNG = random.Random(11)


# ------------------------------------------------------------------ geometry helpers
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


def quad2(g, a, b, c, d):
    quad(g, a, b, c, d); quad(g, d, c, b, a)


def box(g, lo, hi, col=False):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    if z1 < z0: z0, z1 = z1, z0
    quad(g, (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    quad(g, (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0))
    quad(g, (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))
    quad(g, (x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1))
    quad(g, (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))
    quad(g, (x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1))
    if col: COL.append(((x0, y0, z0), (x1, y1, z1)))


def collide(lo, hi):
    COL.append((tuple(min(a, b) for a, b in zip(lo, hi)), tuple(max(a, b) for a, b in zip(lo, hi))))


def cyl(g, p0, p1, r, seg=12, caps=True, r1=None):
    p0, p1 = Vector(p0), Vector(p1)
    if r1 is None: r1 = r
    ax = (p1 - p0).normalized()
    u = ax.cross(Vector((0, 0, 1)))
    if u.length < 1e-4: u = ax.cross(Vector((1, 0, 0)))
    u.normalize(); v = ax.cross(u)
    ring = [(u * math.cos(TAU * i / seg) + v * math.sin(TAU * i / seg)) for i in range(seg)]
    for i in range(seg):
        a, b = ring[i], ring[(i + 1) % seg]
        tri(g, p0 + a * r, p0 + b * r, p1 + b * r1, (a, b, b)); tri(g, p0 + a * r, p1 + b * r1, p1 + a * r1, (a, b, a))
        if caps:
            tri(g, p0, p0 + b * r, p0 + a * r); tri(g, p1, p1 + a * r1, p1 + b * r1)


def ball(g, c, r, sx=1.0, sz=1.0, seg=10, rings=6):
    c = Vector(c)
    def P(i, j):
        th = math.pi * j / rings; ph = TAU * i / seg
        return Vector((r * sx * math.sin(th) * math.cos(ph), r * sx * math.sin(th) * math.sin(ph), r * sz * math.cos(th)))
    for j in range(rings):
        for i in range(seg):
            a, b, c2, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
            tri(g, c + a, c + d, c + c2, (a, d, c2)); tri(g, c + a, c + c2, c + b, (a, c2, b))


def obox(g, x, y, ang, lo, hi):
    """A box turned by `ang` about the vertical through (x, y); lo / hi in its own frame."""
    c, s = math.cos(ang), math.sin(ang)
    def W(px, py, pz): return (x + px * c - py * s, y + px * s + py * c, pz)
    x0, y0, z0 = [min(a, b) for a, b in zip(lo, hi)]; x1, y1, z1 = [max(a, b) for a, b in zip(lo, hi)]
    p = [W(x0, y0, z0), W(x1, y0, z0), W(x1, y1, z0), W(x0, y1, z0), W(x0, y0, z1), W(x1, y0, z1), W(x1, y1, z1), W(x0, y1, z1)]
    quad(g, p[4], p[5], p[6], p[7]); quad(g, p[3], p[2], p[1], p[0]); quad(g, p[0], p[1], p[5], p[4])
    quad(g, p[2], p[3], p[7], p[6]); quad(g, p[1], p[2], p[6], p[5]); quad(g, p[3], p[0], p[4], p[7])


def tube(g, pts, r, seg=6):
    """A hose or pipe through the points."""
    for a, b in zip(pts[:-1], pts[1:]):
        cyl(g, a, b, r, seg, caps=False)
    for p in pts[1:-1]: ball(g, p, r, seg=seg, rings=3)


def text(g, body, size, pos, face, depth=0.03, align='CENTER', squeeze=1.0):
    """Raised letters (Blender's own font) on an upright plane. `face`: the way the letters
    look - 'E', 'W', 'N', 'S' - or 'U' for letters lying on the ground, read from the south.
    pos: the middle of the baseline (CENTER) or its left end (LEFT)."""
    try:
        cu = bpy.data.curves.new('t', 'FONT')
        cu.body = body; cu.size = size; cu.extrude = depth; cu.align_x = align
        ob = bpy.data.objects.new('t', cu)
        bpy.context.scene.collection.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        me.calc_loop_triangles()
        ox, oy, oz = pos
        def W(q):
            a, u, d = q.x * squeeze, q.y, q.z          # along the line, up, out of the plane
            if face == 'E': return (ox + d, oy + a, oz + u)
            if face == 'W': return (ox - d, oy - a, oz + u)
            if face == 'S': return (ox + a, oy - d, oz + u)
            if face == 'N': return (ox - a, oy + d, oz + u)
            return (ox + a, oy + u, oz + d)
        for t in me.loop_triangles:
            tri(g, *[W(me.vertices[i].co) for i in t.vertices])
        bpy.data.objects.remove(ob)
    except Exception as e:
        print('TEXT "%s": skipped (%s)' % (body, e))


# ------------------------------------------------------------------ the measures
ASPH = 0.06               # top of the asphalt over the terrace
WALK = 0.20               # top of the pavers and of the building's floor slab
FL = 0.22                 # finished floor inside
PX0, PX1, PY0, PY1 = -28.0, 26.0, -36.0, 36.0        # the plot
BX0, BX1, BY0, BY1 = -26.0, -8.0, -22.0, 22.0        # the building
PARTY = -2.0              # the wall between the food mart (south) and the diner (north)
HEAD = 3.2                # top of the glazing
CEIL = 3.5
ROOF = 4.6                # top of the roof deck
PARA = 5.2                # top of the parapet
CAN_X0, CAN_X1, CAN_Y0, CAN_Y1 = 3.0, 23.0, -13.0, 13.0
CAN_Z0, CAN_Z1 = 5.3, 6.25
ISL_X = (9.0, 17.0)       # the two pump islands (they run north-south)
# The highway, in the plot's own coordinates (the engine logs it: VIBE_LOG_LOOK). Its centre
# line runs 6 degrees off the plot's long axis; it climbs to the north.
def road_x(y): return 42.7 - 0.104 * y            # the centre line
def road_h(y): return 0.0879 * y                  # its height over the terrace
ROAD_PAVED, ROAD_VERGE = 5.5, 8.0                 # engine/highway.h kPaved, kVerge
WALL_DEEP = -45.0                                 # the retaining wall's foot: below the river's bed at the plot's south-west (9 m left it open underneath)
EDGE = 0.6                                        # the ground starts to leave the terrace's level this far east of the plot (terrain.h kStationEdge)


def ground_east(x, y):
    """The landscape between the plot's east edge and the road, as the engine makes it
    (highway.cpp ground(), terrain_rm_noise_road.glsl rm_highway - keep in step): the road's
    shelf out to its verge, and from there a straight ramp down (or up) to the terrace."""
    d = (road_x(y) - x) * 0.9946
    fall = 0.02 * min(d, ROAD_PAVED) + 0.05 * min(max(d - ROAD_PAVED, 0.0), ROAD_VERGE - ROAD_PAVED)
    if d <= ROAD_VERGE: return road_h(y) - fall
    a, b = x - (PX1 + EDGE), d - ROAD_VERGE
    if a <= 0.0: return 0.0
    k = a / (a + b); k = k * k * (3.0 - 2.0 * k)
    return (road_h(y) - fall) * k

BRICK, CLAD, GLASS, STEEL, METAL = 'g_brick', 'g_clad', 'g_glass', 'g_steel', 'g_metal'
WHITE, BLACK, RED, YELLOW = 'g_white', 'g_black', 'g_red', 'g_yellow'
WALL, CEILM, LIGHT = 'g_wall', 'g_ceiling', 'g_light'
PRODUCTS = ['g_p_red', 'g_p_yellow', 'g_p_blue', 'g_p_green', 'g_p_orange', 'g_white', 'g_p_purple']


# ------------------------------------------------------------------ small things, used many times
def bollard(x, y, z=ASPH, h=1.0, g=YELLOW):
    cyl(g, (x, y, z), (x, y, z + h), 0.085, 10); ball(g, (x, y, z + h), 0.085, seg=10, rings=4)
    cyl(BLACK, (x, y, z + h * 0.62), (x, y, z + h * 0.72), 0.088, 10, caps=False)
    collide((x - 0.09, y - 0.09, z), (x + 0.09, y + 0.09, z + h))


def bin_(x, y, z=ASPH):
    cyl(STEEL, (x, y, z), (x, y, z + 0.85), 0.27, 12)
    cyl(BLACK, (x, y, z + 0.85), (x, y, z + 0.92), 0.29, 12)
    cyl('g_p_green', (x, y, z + 0.3), (x, y, z + 0.5), 0.275, 12, caps=False)
    collide((x - 0.28, y - 0.28, z), (x + 0.28, y + 0.28, z + 0.92))


def planter_box(x0, y0, x1, y1, z, h=0.55):
    box('g_conc', (x0, y0, z), (x1, y1, z + h), col=True)
    box('g_soil', (x0 + 0.08, y0 + 0.08, z + h - 0.04), (x1 - 0.08, y1 - 0.08, z + h + 0.01))
    n = max(2, int((x1 - x0 + y1 - y0) / 0.5))
    for i in range(n):
        t = (i + 0.5) / n
        cx = x0 + 0.25 + (x1 - x0 - 0.5) * (t if (x1 - x0) > (y1 - y0) else RNG.random())
        cy = y0 + 0.25 + (y1 - y0 - 0.5) * (t if (y1 - y0) >= (x1 - x0) else RNG.random())
        r = RNG.uniform(0.26, 0.4)
        ball('g_plant' if i % 2 else 'g_plant_d', (cx, cy, z + h + r * 0.75), r, sz=0.9, seg=8, rings=5)


def chair(x, y, face, z=FL, seat='g_seat'):
    """A diner chair: chrome legs, a padded seat and back."""
    SEATS.append(('chair', x, y, z + 0.5, -math.sin(face), math.cos(face)))
    for (lx, ly) in ((-0.19, -0.19), (0.19, -0.19), (-0.19, 0.19), (0.19, 0.19)):
        px, py = x + lx * math.cos(face) - ly * math.sin(face), y + lx * math.sin(face) + ly * math.cos(face)
        cyl('g_chrome', (px, py, z), (px, py, z + 0.44), 0.014, 6, caps=False)
    obox(seat, x, y, face, (-0.21, -0.21, z + 0.44), (0.21, 0.21, z + 0.5))
    obox(seat, x, y, face, (-0.2, -0.235, z + 0.62), (0.2, -0.195, z + 0.92))
    for sx in (-0.18, 0.18):
        px, py = x + sx * math.cos(face) + 0.215 * math.sin(face), y + sx * math.sin(face) - 0.215 * math.cos(face)
        cyl('g_chrome', (px, py, z + 0.44), (px, py, z + 0.9), 0.012, 6, caps=False)


def stool(x, y, z=FL):
    SEATS.append(('stool', x, y, z + 0.78, -1.0, 0.0))
    cyl('g_chrome', (x, y, z), (x, y, z + 0.03), 0.2, 14); cyl('g_chrome', (x, y, z + 0.03), (x, y, z + 0.66), 0.035, 8, caps=False)
    cyl('g_chrome', (x, y, z + 0.22), (x, y, z + 0.24), 0.16, 12, caps=False)
    cyl('g_chrome', (x, y, z + 0.66), (x, y, z + 0.7), 0.185, 14); cyl('g_seat', (x, y, z + 0.7), (x, y, z + 0.78), 0.19, 14)
    collide((x - 0.19, y - 0.19, z), (x + 0.19, y + 0.19, z + 0.78))


def table_setting(x, y, z):
    """What stands on a diner table: ketchup, mustard, a napkin box, salt and pepper."""
    cyl('g_p_red', (x - 0.09, y, z), (x - 0.09, y, z + 0.16), 0.028, 8); cyl(WHITE, (x - 0.09, y, z + 0.16), (x - 0.09, y, z + 0.19), 0.014, 6)
    cyl('g_p_yellow', (x - 0.02, y + 0.03, z), (x - 0.02, y + 0.03, z + 0.15), 0.026, 8); cyl('g_p_yellow', (x - 0.02, y + 0.03, z + 0.15), (x - 0.02, y + 0.03, z + 0.19), 0.01, 6)
    box('g_chrome', (x + 0.04, y - 0.06, z), (x + 0.1, y + 0.06, z + 0.11)); box(WHITE, (x + 0.045, y - 0.05, z + 0.11), (x + 0.095, y + 0.05, z + 0.125))
    for dy in (-0.11, -0.16):
        cyl(GLASS, (x - 0.03, y + dy, z), (x - 0.03, y + dy, z + 0.07), 0.018, 6); cyl('g_chrome', (x - 0.03, y + dy, z + 0.07), (x - 0.03, y + dy, z + 0.085), 0.019, 6)


def plate(x, y, z, food=True):
    cyl(WHITE, (x, y, z), (x, y, z + 0.015), 0.13, 14)
    if food:
        k = RNG.randrange(3)
        if k == 0:
            cyl('g_bun', (x, y, z + 0.015), (x, y, z + 0.05), 0.06, 10); cyl('g_p_brown', (x, y, z + 0.05), (x, y, z + 0.068), 0.063, 10)
            cyl('g_p_green', (x, y, z + 0.068), (x, y, z + 0.076), 0.066, 10); ball('g_bun', (x, y, z + 0.085), 0.062, sz=0.6, seg=10, rings=4)
        elif k == 1:
            cyl('g_p_yellow', (x, y, z + 0.015), (x, y, z + 0.035), 0.085, 12); ball('g_p_yellow', (x + 0.02, y, z + 0.04), 0.022, seg=6, rings=3)
        else:
            for i in range(7):
                a = RNG.uniform(0, TAU); obox('g_p_yellow', x + RNG.uniform(-0.05, 0.05), y + RNG.uniform(-0.05, 0.05), a, (-0.035, -0.006, z + 0.015 + 0.006 * i), (0.035, 0.006, z + 0.027 + 0.006 * i))


def cup(x, y, z, g=WHITE):
    cyl(g, (x, y, z), (x, y, z + 0.09), 0.036, 8, r1=0.042)


def pendant(x, y, z, top=CEIL):
    cyl(BLACK, (x, y, z + 0.16), (x, y, top), 0.006, 4, caps=False)
    cyl(RED, (x, y, z + 0.02), (x, y, z + 0.2), 0.2, 14, caps=False, r1=0.05)
    ball(LIGHT, (x, y, z + 0.06), 0.07, seg=8, rings=4)


def ceiling_panel(x, y, z=CEIL, w=1.2, d=0.6):
    box(LIGHT, (x - w * 0.5, y - d * 0.5, z - 0.03), (x + w * 0.5, y + d * 0.5, z + 0.005))
    box(WHITE, (x - w * 0.5 - 0.03, y - d * 0.5 - 0.03, z - 0.02), (x + w * 0.5 + 0.03, y + d * 0.5 + 0.03, z + 0.004))


def wall(lo, hi, ext, inner=None, col=True):
    """A wall piece (its faces axis-aligned). `inner`: which side is a room ('W', 'E', 'N',
    'S') - that face gets the room's plaster as a thin lining; the rest is `ext`."""
    x0, y0, z0 = [min(a, b) for a, b in zip(lo, hi)]; x1, y1, z1 = [max(a, b) for a, b in zip(lo, hi)]
    box(ext, (x0, y0, z0), (x1, y1, z1), col)
    t = 0.012
    if inner == 'W': box(WALL, (x0 - t, y0, z0), (x0, y1, min(z1, CEIL)))
    if inner == 'E': box(WALL, (x1, y0, z0), (x1 + t, y1, min(z1, CEIL)))
    if inner == 'S': box(WALL, (x0, y0 - t, z0), (x1, y0, min(z1, CEIL)))
    if inner == 'N': box(WALL, (x0, y1, z0), (x1, y1 + t, min(z1, CEIL)))


def glazing_x(x, y0, y1, z0, z1, n, col=True):
    """A glazed front in the plane X = x from y0 to y1: one sheet and slim dark mullions."""
    quad2(GLASS, (x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1))
    for i in range(n + 1):
        y = y0 + (y1 - y0) * i / n
        box(STEEL, (x - 0.05, y - 0.03, z0), (x + 0.05, y + 0.03, z1))
    for z in (z0, z1):
        box(STEEL, (x - 0.05, y0, z - 0.03), (x + 0.05, y1, z + 0.03))
    if col: collide((x - 0.06, y0, z0 - 0.3), (x + 0.06, y1, z1))


def glazing_y(y, x0, x1, z0, z1, n, col=True):
    quad2(GLASS, (x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        box(STEEL, (x - 0.03, y - 0.05, z0), (x + 0.03, y + 0.05, z1))
    for z in (z0, z1):
        box(STEEL, (x0, y - 0.05, z - 0.03), (x1, y + 0.05, z + 0.03))
    if col: collide((x0, y - 0.06, z0 - 0.3), (x1, y + 0.06, z1))


# ------------------------------------------------------------------ the ground
def ground():
    # asphalt forecourt and the pavers the building stands on (deep slabs: the terrace under
    # them is the landscape, level but not to the millimetre)
    box('g_asphalt', (-4.0, PY0, -1.2), (PX1, PY1, ASPH))
    box('g_paver', (PX0, PY0, -1.2), (-4.0, PY1, WALK))
    box('g_kerb', (-4.16, PY0, -0.2), (-3.98, PY1, WALK + 0.02))
    collide((PX0, PY0, -1.2), (PX1, PY1, -0.04))      # (under the ground: the level ground itself is walked on. With its top AT the ground its east face stopped a walker coming in from the road.)
    collide((PX0, PY0, -1.2), (-4.0, PY1, WALK))
    NOGRASS.append((PX0 - 1.5, PY0 - 1.5, PX1 + 1.5, PY1 + 1.5))
    # The terrace is level, but the river's bank cuts into it at the south-west and the road's
    # bank at the east: a concrete retaining wall goes down round the whole plot.
    for (a, b, top) in (((PX0 - 0.55, PY0 - 0.55), (PX1 + 0.55, PY0 - 0.18), WALK), ((PX0 - 0.55, PY1 + 0.18), (PX1 + 0.55, PY1 + 0.55), WALK),
                        ((PX0 - 0.55, PY0 - 0.55), (PX0 - 0.18, PY1 + 0.55), WALK), ((PX1 + 0.18, PY0 - 0.55), (PX1 + 0.55, -9.2), ASPH + 0.1), ((PX1 + 0.18, 9.2), (PX1 + 0.55, PY1 + 0.55), ASPH + 0.1)):
        box('o_concrete', (a[0], a[1], WALL_DEEP), (b[0], b[1], top - 0.02))
        collide((a[0], a[1], WALL_DEEP), (b[0], b[1], top - 0.02))
    PADS.append((-4.0, PY0, PX1, PY1, ASPH, ASPH)); PADS.append((PX0, PY0, -4.0, PY1, WALK, WALK))
    # a kerb round the asphalt, open at the apron
    for (a, b) in (((-4.0, PY0 - 0.18), (PX1 + 0.18, PY0)), ((-4.0, PY1), (PX1 + 0.18, PY1 + 0.18)),
                   ((PX1, PY0), (PX1 + 0.18, -9.0)), ((PX1, 9.0), (PX1 + 0.18, PY1))):
        box('g_kerb', (a[0], a[1], -0.6), (b[0], b[1], ASPH + 0.14), col=True)
    for (a, b) in (((PX0 - 0.18, PY0 - 0.18), (-4.0, PY0)), ((PX0 - 0.18, PY1), (-4.0, PY1 + 0.18)), ((PX0 - 0.18, PY0), (PX0, PY1))):
        box('g_kerb', (a[0], a[1], -0.6), (b[0], b[1], WALK + 0.02))
    # The way in and out: a flared mouth from the forecourt's edge to the road's asphalt. It lies
    # on the ground the engine makes between the two (ground_east), a hand above it, and runs
    # out to nothing on the road's own surface. White edge lines, a give-way line at the road,
    # kerbs round the two flares.
    NU, NV = 14, 12
    HALF0, HALF1 = 9.0, 14.0
    def mouth(u, v):
        half = HALF0 + (HALF1 - HALF0) * u * u
        y = v * half
        x = PX1 + u * (road_x(y) - ROAD_PAVED + 0.35 - PX1)
        # (a hand above the ground at the plot, more over the ramp where the road's real line and this
        #  script's straight one differ by centimetres, down to nearly nothing on the road itself)
        lift = 0.03 + 0.03 * (1.0 - max(0.0, min(1.0, (u - 0.8) / 0.2))) + 0.09 * math.sin(math.pi * min(u / 0.85, 1.0))
        return Vector((x, y, max(ground_east(x, y), ASPH - 0.02 if u < 0.05 else -99.0) + lift if u > 0.0 else ASPH))
    for i in range(NU):
        for j in range(NV):
            u0, u1 = i / NU, (i + 1) / NU; v0, v1 = -1.0 + 2.0 * j / NV, -1.0 + 2.0 * (j + 1) / NV
            quad('g_asphalt', mouth(u0, v0), mouth(u1, v0), mouth(u1, v1), mouth(u0, v1))
    up = Vector((0, 0, 1))
    for s in (-1.0, 1.0):
        for i in range(NU):
            u0, u1 = i / NU, (i + 1) / NU
            # the edge line, just inside the flare
            a, b, c, d = mouth(u0, s * 0.955), mouth(u1, s * 0.955), mouth(u1, s * 0.975), mouth(u0, s * 0.975)
            if s > 0: quad('g_line', a + up * 0.02, b + up * 0.02, c + up * 0.02, d + up * 0.02)
            else: quad('g_line', d + up * 0.02, c + up * 0.02, b + up * 0.02, a + up * 0.02)
            if u1 > 0.86: continue
            # the kerb round the flare: a raised strip outside the asphalt
            p0, p1 = mouth(u0, s), mouth(u1, s)
            q0, q1 = p0 + Vector((0, s * 0.2, 0)), p1 + Vector((0, s * 0.2, 0))
            h = Vector((0, 0, 0.13))
            if s > 0:
                quad('g_kerb', p0 + h, p1 + h, q1 + h, q0 + h); quad('g_kerb', p0 - h * 3, p1 - h * 3, p1 + h, p0 + h); quad('g_kerb', q1 - h * 6, q0 - h * 6, q0 + h, q1 + h)
            else:
                quad('g_kerb', q0 + h, q1 + h, p1 + h, p0 + h); quad('g_kerb', p1 - h * 3, p0 - h * 3, p0 + h, p1 + h); quad('g_kerb', q0 - h * 6, q1 - h * 6, q1 + h, q0 + h)
    for j in range(NV):                             # the give-way line: dashes across the mouth at the road
        if j % 2: continue
        v0, v1 = -0.94 + 1.88 * j / NV, -0.94 + 1.88 * (j + 0.6) / NV
        a, b, c, d = mouth(0.86, v0), mouth(0.9, v0), mouth(0.9, v1), mouth(0.86, v1)
        quad('g_line', a + up * 0.02, b + up * 0.02, c + up * 0.02, d + up * 0.02)
    for k in range(5):                              # a short centre line down the middle of the throat
        a, b, c, d = mouth(0.1 + 0.15 * k, -0.01), mouth(0.19 + 0.15 * k, -0.01), mouth(0.19 + 0.15 * k, 0.01), mouth(0.1 + 0.15 * k, 0.01)
        quad('g_line', a + up * 0.02, b + up * 0.02, c + up * 0.02, d + up * 0.02)
    for j in range(8):                              # what a car drives on: strips across the mouth
        ya, yb = -HALF1 + 2.0 * HALF1 * j / 8, -HALF1 + 2.0 * HALF1 * (j + 1) / 8
        ym = 0.5 * (ya + yb); xe = road_x(ym) - ROAD_PAVED
        PADS.append((PX1, ya, xe, yb, ASPH, ground_east(xe, ym) + 0.03))
    NOGRASS.append((PX1, -HALF1 - 2.0, road_x(0.0) - ROAD_PAVED + 2.0, HALF1 + 2.0))
    # paint: parking bays in front of the building, an accessible bay, arrows, stop bars
    def line(x0, y0, x1, y1, g='g_line'):
        box(g, (x0, y0, ASPH), (x1, y1, ASPH + 0.05))      # (2 cm over the asphalt flickered against it from a distance, and against a coloured bay)
    for i in range(16):
        y = -20.0 + 2.6 * i
        if -2.9 < y < 2.0: continue                 # (clear in front of the diner's door)
        line(-3.9, y - 0.06, 1.2, y + 0.06)
    line(-3.9, -20.0 - 0.06, 1.2, -20.0 + 0.06); line(1.14, -20.0, 1.26, 21.6)
    box('g_p_blue', (-3.8, -19.9, ASPH), (1.1, -17.5, ASPH + 0.03))
    text('g_line', 'P', 1.3, (-1.4, -19.3, ASPH + 0.03), 'U', depth=0.03)
    for i in range(15):                             # wheel stops
        y = -18.7 + 2.6 * i
        if -2.9 < y - 1.3 < 0.8 or y > 20.5: continue
        box('g_conc', (-3.7, y - 0.85, ASPH), (-3.45, y + 0.85, ASPH + 0.11), col=True)
        for dy in (-0.6, 0.6): box(YELLOW, (-3.712, y + dy - 0.1, ASPH + 0.02), (-3.438, y + dy + 0.1, ASPH + 0.122))
    # lanes under the canopy and the way in and out
    for x in (5.0, 13.0, 21.0):
        text('g_line', '>', 1.6, (x, -17.5, ASPH), 'U', depth=0.05)
    for y in (-8.0, 8.0):
        line(24.6, y - 1.6, 25.0, y + 1.6)
    text('g_line', 'STOP', 1.0, (22.4, 4.6, ASPH), 'U', depth=0.05)
    for (x, y) in ((6.5, -22.0), (13.0, 22.5), (20.0, -24.0), (2.0, 27.0)):      # tank fills and manholes
        cyl('g_steel', (x, y, ASPH), (x, y, ASPH + 0.02), 0.42, 16); cyl(YELLOW if x > 6 else 'g_steel', (x, y, ASPH + 0.02), (x, y, ASPH + 0.035), 0.2, 12)
    for y in (-28.0, 0.0, 28.0):                    # drain gratings
        box('g_steel', (25.2, y - 1.0, ASPH - 0.01), (25.6, y + 1.0, ASPH + 0.02))
        for k in range(9): box(BLACK, (25.24, y - 0.9 + 0.2 * k, ASPH + 0.02), (25.56, y - 0.82 + 0.2 * k, ASPH + 0.032))
    # the tanks' vent pipes, at the south edge
    for i in range(3):
        x = 22.0 + 0.45 * i
        cyl(METAL, (x, -34.6, ASPH), (x, -34.6, 4.2), 0.04, 8); cyl(METAL, (x, -34.6, 4.2), (x, -34.6, 4.32), 0.07, 8)
    box('g_steel', (21.7, -34.75, 0.9), (23.2, -34.68, 1.0)); box('g_steel', (21.7, -34.75, 2.6), (23.2, -34.68, 2.7))


# ------------------------------------------------------------------ the canopy and the pumps
def dispenser(x, y, number):
    """A fuel dispenser on the island at (x, y): two faces (east and west), three grades."""
    z = ASPH + 0.16
    box('g_steel', (x - 0.3, y - 0.52, z), (x + 0.3, y + 0.52, z + 0.12))                    # plinth
    box(WHITE, (x - 0.27, y - 0.48, z + 0.12), (x + 0.27, y + 0.48, z + 1.25))                # lower cabinet
    box(RED, (x - 0.285, y - 0.495, z + 0.12), (x + 0.285, y + 0.495, z + 0.5))
    box('g_steel', (x - 0.24, y - 0.44, z + 1.25), (x + 0.24, y + 0.44, z + 1.34))
    box(WHITE, (x - 0.27, y - 0.48, z + 1.34), (x + 0.27, y + 0.48, z + 2.02))                # head
    box(RED, (x - 0.3, y - 0.52, z + 2.02), (x + 0.3, y + 0.52, z + 2.26))                    # crown
    collide((x - 0.3, y - 0.52, z), (x + 0.3, y + 0.52, z + 2.3))
    for s, face in ((1.0, 'E'), (-1.0, 'W')):
        fx = x + s * 0.272
        box(BLACK, (fx, y - 0.4, z + 1.42), (fx + s * 0.012, y + 0.12, z + 1.94))             # display surround
        box('g_lcd', (fx + s * 0.012, y - 0.36, z + 1.66), (fx + s * 0.024, y + 0.08, z + 1.9))
        box('g_lcd', (fx + s * 0.012, y - 0.36, z + 1.46), (fx + s * 0.024, y - 0.14, z + 1.6))
        box('g_lcd', (fx + s * 0.012, y - 0.1, z + 1.46), (fx + s * 0.024, y + 0.08, z + 1.6))
        box('g_steel', (fx, y + 0.18, z + 1.5), (fx + s * 0.03, y + 0.4, z + 1.86))            # card reader
        box('g_lcd', (fx + s * 0.03, y + 0.2, z + 1.72), (fx + s * 0.034, y + 0.38, z + 1.84))
        for k in range(12): box('g_chrome', (fx + s * 0.03, y + 0.205 + 0.045 * (k % 4), z + 1.54 + 0.05 * (k // 4)), (fx + s * 0.036, y + 0.235 + 0.045 * (k % 4), z + 1.575 + 0.05 * (k // 4)))
        text(WHITE, str(number), 0.2, (x + s * 0.302, y, z + 2.05), face, depth=0.01)
        # three nozzles in their holsters, hoses from the head
        for k, col in enumerate(('g_p_green', BLACK, YELLOW)):
            ny = y - 0.3 + 0.3 * k
            box('g_steel', (fx, ny - 0.08, z + 0.72), (fx + s * 0.07, ny + 0.08, z + 1.14))
            box(col, (fx + s * 0.07, ny - 0.045, z + 0.86), (fx + s * 0.17, ny + 0.045, z + 1.1))  # the handle
            cyl('g_chrome', (fx + s * 0.12, ny, z + 0.86), (fx + s * 0.1, ny, z + 0.66), 0.018, 6)
            tube(BLACK, [(fx + s * 0.13, ny, z + 1.1), (fx + s * 0.3, ny, z + 1.3), (fx + s * 0.36, ny, z + 0.75), (fx + s * 0.24, ny, z + 0.42),
                         (fx + s * 0.1, ny, z + 0.7), (fx + s * 0.02, ny, z + 1.28)], 0.017, 6)
        box(YELLOW, (fx, y - 0.46, z + 0.52), (fx + s * 0.025, y + 0.46, z + 0.62))


def island(x):
    z = ASPH
    y0, y1 = -5.6, 5.6
    box('g_kerb', (x - 0.75, y0, z - 0.3), (x + 0.75, y1, z + 0.16), col=True)
    for yy in (y0, y1):                             # rounded ends
        cyl('g_kerb', (x, yy, z - 0.3), (x, yy, z + 0.16), 0.75, 16)
    box(YELLOW, (x - 0.76, y0, z + 0.1), (x - 0.74, y1, z + 0.16)); box(YELLOW, (x + 0.74, y0, z + 0.1), (x + 0.76, y1, z + 0.16))
    dispenser(x, -2.4, 1 if x < 12 else 3); dispenser(x, 2.4, 2 if x < 12 else 4)
    # the canopy's columns stand on the island: square, clad white, a red base, a steel shoe
    for yy in (-4.7, 4.7):
        box(WHITE, (x - 0.22, yy - 0.22, z + 0.16), (x + 0.22, yy + 0.22, CAN_Z0), col=True)
        box(RED, (x - 0.235, yy - 0.235, z + 0.16), (x + 0.235, yy + 0.235, z + 1.2))
        box('g_steel', (x - 0.3, yy - 0.3, z + 0.16), (x + 0.3, yy + 0.3, z + 0.24))
        box('g_steel', (x - 0.32, yy - 0.32, CAN_Z0 - 0.25), (x + 0.32, yy + 0.32, CAN_Z0))
        # extinguisher and emergency stop on the column
        s = 1.0 if yy < 0 else -1.0
        box(RED, (x - 0.09, yy + s * 0.22, z + 1.3), (x + 0.09, yy + s * 0.34, z + 1.8))
        cyl('g_p_red', (x, yy + s * 0.28, z + 1.34), (x, yy + s * 0.28, z + 1.72), 0.055, 8)
        box(YELLOW, (x - 0.235, yy - 0.08, z + 1.5), (x - 0.225, yy + 0.08, z + 1.66)); cyl(RED, (x - 0.235, yy, z + 1.58), (x - 0.26, yy, z + 1.58), 0.04, 10)
    # between the pumps: bin, squeegee bucket and paper towels on a post
    bin_(x, 0.0, z + 0.16)
    box('g_p_blue', (x - 0.14, 0.42, z + 0.16), (x + 0.14, 0.72, z + 0.42)); cyl('g_chrome', (x, 0.5, z + 0.3), (x + 0.1, 0.62, z + 0.95), 0.008, 5); box(BLACK, (x + 0.06, 0.58, z + 0.92), (x + 0.2, 0.66, z + 0.96))
    cyl('g_steel', (x, -0.62, z + 0.16), (x, -0.62, z + 1.3), 0.025, 6); box(WHITE, (x - 0.14, -0.74, z + 1.05), (x + 0.14, -0.5, z + 1.36))
    for (bx, by) in ((x - 0.55, y0 - 0.55), (x + 0.55, y0 - 0.55), (x - 0.55, y1 + 0.55), (x + 0.55, y1 + 0.55)):
        bollard(bx, by, z)


def canopy():
    # deck: a deep fascia in the station's red with a white band, a white soffit
    box(WHITE, (CAN_X0 + 0.25, CAN_Y0 + 0.25, CAN_Z0), (CAN_X1 - 0.25, CAN_Y1 - 0.25, CAN_Z0 + 0.12))
    box('g_roof', (CAN_X0 + 0.25, CAN_Y0 + 0.25, CAN_Z1 - 0.12), (CAN_X1 - 0.25, CAN_Y1 - 0.25, CAN_Z1 - 0.03))
    for (a, b) in (((CAN_X0, CAN_Y0), (CAN_X1, CAN_Y0 + 0.25)), ((CAN_X0, CAN_Y1 - 0.25), (CAN_X1, CAN_Y1)),
                   ((CAN_X0, CAN_Y0), (CAN_X0 + 0.25, CAN_Y1)), ((CAN_X1 - 0.25, CAN_Y0), (CAN_X1, CAN_Y1))):
        box(RED, (a[0], a[1], CAN_Z0 - 0.05), (b[0], b[1], CAN_Z1))
    e = 0.03
    for (a, b) in (((CAN_X0 - e, CAN_Y0 - e), (CAN_X1 + e, CAN_Y0)), ((CAN_X0 - e, CAN_Y1), (CAN_X1 + e, CAN_Y1 + e)), ((CAN_X0 - e, CAN_Y0), (CAN_X0, CAN_Y1)), ((CAN_X1, CAN_Y0), (CAN_X1 + e, CAN_Y1))):
        box(WHITE, (a[0], a[1], CAN_Z0 + 0.12), (b[0], b[1], CAN_Z0 + 0.3))
    for (a, b) in (((CAN_X0, CAN_Y0), (CAN_X1, CAN_Y0 + 0.2)), ((CAN_X0, CAN_Y1 - 0.2), (CAN_X1, CAN_Y1)), ((CAN_X0, CAN_Y0), (CAN_X0 + 0.2, CAN_Y1)), ((CAN_X1 - 0.2, CAN_Y0), (CAN_X1, CAN_Y1))):
        box('g_neon_w', (a[0], a[1], CAN_Z0 - 0.07), (b[0], b[1], CAN_Z0 - 0.05))
    SHELTERS.append((CAN_X0, CAN_Y0, CAN_X1, CAN_Y1, CAN_Z1))
    # the frame under the soffit: two beams over the islands and purlins
    for x in ISL_X:
        box('g_steel', (x - 0.16, CAN_Y0 + 0.4, CAN_Z0 - 0.02), (x + 0.16, CAN_Y1 - 0.4, CAN_Z0 + 0.02))
    # recessed lights, a grid of them
    for x in (5.5, 9.0, 13.0, 17.0, 20.5):
        for y in (-10.0, -5.0, 0.0, 5.0, 10.0):
            box(LIGHT, (x - 0.38, y - 0.38, CAN_Z0 - 0.025), (x + 0.38, y + 0.38, CAN_Z0 + 0.004))
            box('g_steel', (x - 0.43, y - 0.43, CAN_Z0 - 0.015), (x + 0.43, y + 0.43, CAN_Z0 + 0.002))
    for x in (9.0, 17.0):
        for y in (-7.0, 0.0, 7.0): LIGHTS.append((x, y, CAN_Z0 - 0.5))
    # the name, on three sides
    text(WHITE, 'VIBE FUEL', 0.55, (CAN_X1 + 0.02, 0.0, CAN_Z0 + 0.42), 'E', depth=0.05)
    text(WHITE, 'VIBE FUEL', 0.55, (13.0, CAN_Y0 - 0.02, CAN_Z0 + 0.42), 'S', depth=0.05)
    text(WHITE, 'VIBE FUEL', 0.55, (13.0, CAN_Y1 + 0.02, CAN_Z0 + 0.42), 'N', depth=0.05)
    text(WHITE, 'VIBE FUEL', 0.55, (CAN_X0 - 0.02, 0.0, CAN_Z0 + 0.42), 'W', depth=0.05)
    # clearance bar, cameras, a speaker
    box(YELLOW, (CAN_X0 + 0.3, CAN_Y0 - 0.02, CAN_Z0 - 0.3), (CAN_X1 - 0.3, CAN_Y0 + 0.02, CAN_Z0 - 0.12))
    text(BLACK, 'CLEARANCE 4.9 m', 0.13, (13.0, CAN_Y0 - 0.025, CAN_Z0 - 0.26), 'S', depth=0.006)
    for (x, y) in ((CAN_X0 + 0.6, CAN_Y0 + 0.6), (CAN_X1 - 0.6, CAN_Y1 - 0.6)):
        cyl(WHITE, (x, y, CAN_Z0), (x, y, CAN_Z0 - 0.14), 0.035, 6); ball(BLACK, (x, y, CAN_Z0 - 0.2), 0.09, seg=10, rings=5)
    for x in ISL_X: island(x)


# ------------------------------------------------------------------ the building's shell
def shell():
    t = 0.3
    # floor slabs: checker in the diner, polished concrete in the mart, tile in the wet rooms
    box('g_checker', (BX0 + t, PARTY + 0.15, WALK - 0.1), (BX1 - t, BY1 - t, FL))
    box('o_floor', (BX0 + t, BY0 + t, WALK - 0.1), (BX1 - t, PARTY - 0.15, FL))
    collide((BX0, BY0, 0.0), (BX1, BY1, FL))
    SHELTERS.append((BX0, BY0, BX1, BY1, PARA))
    # ---- east front (to the forecourt)
    E = BX1
    def pier(y0, y1): wall((E - t, y0, WALK), (E, y1, HEAD), BRICK, 'W')
    pier(BY0 + t, -20.5); pier(-4.0, -1.0); pier(1.2, 2.0); pier(20.5, BY1 - t)
    # the mart: a storefront from the kerb up, a doorway in it
    for (a, b, n) in ((-20.5, -13.2, 5), (-10.8, -4.0, 5)):
        box('g_steel', (E - 0.2, a, WALK), (E - 0.05, b, 0.5), col=False)
        glazing_x(E - 0.12, a, b, 0.5, HEAD, n)
    box('g_steel', (E - 0.2, -13.26, WALK), (E - 0.04, -13.14, HEAD)); box('g_steel', (E - 0.2, -10.86, WALK), (E - 0.04, -10.74, HEAD))
    box('g_steel', (E - 0.2, -13.2, 2.45), (E - 0.04, -10.8, 2.55)); quad2(GLASS, (E - 0.12, -13.2, 2.55), (E - 0.12, -10.8, 2.55), (E - 0.12, -10.8, HEAD), (E - 0.12, -13.2, HEAD))
    # (its two door leaves, slid open behind the glass on either side)
    for (a, b) in ((-14.35, -13.25), (-10.75, -9.65)):
        quad2(GLASS, (E - 0.2, a, WALK + 0.05), (E - 0.2, b, WALK + 0.05), (E - 0.2, b, 2.42), (E - 0.2, a, 2.42))
        box('g_chrome', (E - 0.215, a, 1.1), (E - 0.185, b, 1.16))
    # the diner: a doorway, then four windows over a brick sill between piers
    box('g_steel', (E - 0.2, -1.0, 2.45), (E - 0.04, 1.2, 2.55)); quad2(GLASS, (E - 0.12, -1.0, 2.55), (E - 0.12, 1.2, 2.55), (E - 0.12, 1.2, HEAD), (E - 0.12, -1.0, HEAD))
    obox('g_steel', E - 0.1, -0.98, 1.05, (0.0, -0.03, WALK + 0.02), (1.05, 0.03, 2.43))                 # the door, standing open
    obox(GLASS, E - 0.1, -0.98, 1.05, (0.1, -0.035, WALK + 0.9), (0.95, 0.035, 2.3)); obox('g_chrome', E - 0.1, -0.98, 1.05, (0.75, -0.07, 1.1), (0.95, 0.07, 1.16))
    bays = [(2.0, 6.3), (6.7, 11.0), (11.4, 15.7), (16.1, 20.5)]
    for i, (a, b) in enumerate(bays):
        wall((E - t, a, WALK), (E, b, 0.95), BRICK, 'W')
        box('g_conc', (E - t - 0.04, a, 0.95), (E + 0.06, b, 1.02))
        glazing_x(E - 0.14, a, b, 1.02, HEAD, 3)
        if i < 3: wall((E - t, b, WALK), (E, bays[i + 1][0], HEAD), BRICK, 'W')
    # over every opening: the cladding band, then the fascia and the parapet
    wall((E - t, BY0, HEAD), (E, BY1, ROOF), CLAD, None)
    box(WALL, (E - t - 0.012, BY0 + t, HEAD), (E - t, BY1 - t, CEIL))
    # ---- north end (the diner looks up the coast): windows over a sill
    N = BY1
    wall((BX0, N - t, WALK), (BX0 + 1.2, N, HEAD), BRICK, 'S'); wall((BX1 - 1.2, N - t, WALK), (BX1, N, HEAD), BRICK, 'S')
    nb = [(BX0 + 1.2, -19.2), (-18.8, -13.2), (-12.8, BX1 - 1.2)]
    for i, (a, b) in enumerate(nb):
        wall((a, N - t, WALK), (b, N, 0.95), BRICK, 'S')
        box('g_conc', (a, N - t - 0.04, 0.95), (b, N + 0.06, 1.02))
        glazing_y(N - 0.14, a, b, 1.02, HEAD, 3)
        if i < 2: wall((b, N - t, WALK), (nb[i + 1][0], N, HEAD), BRICK, 'S')
    wall((BX0, N - t, HEAD), (BX1, N, ROOF), CLAD, None); box(WALL, (BX0 + t, N - t - 0.012, HEAD), (BX1 - t, N - t, CEIL))
    # ---- south end and the back: cladding over a brick plinth, service doors
    wall((BX0, BY0, WALK), (BX1, BY0 + t, 1.1), BRICK, 'N'); wall((BX0, BY0, 1.1), (BX1, BY0 + t, ROOF), CLAD, 'N')
    for (a, b) in ((BY0 + t, -9.0), (-7.8, 9.0), (10.2, BY1 - t)):
        wall((BX0, a, WALK), (BX0 + t, b, 1.1), BRICK, 'E'); wall((BX0, a, 1.1), (BX0 + t, b, ROOF), CLAD, 'E')
    for y in (-8.4, 9.6):                           # two steel doors in the back wall
        wall((BX0, y - 0.6, 2.3), (BX0 + t, y + 0.6, ROOF), CLAD, 'E')
        box('g_door', (BX0 + 0.1, y - 0.6, WALK), (BX0 + 0.16, y + 0.6, 2.3), col=True)
        box('g_chrome', (BX0 + 0.05, y + 0.38, 1.15), (BX0 + 0.1, y + 0.5, 1.2))
        box(LIGHT, (BX0 - 0.1, y - 0.2, 2.5), (BX0, y + 0.2, 2.62)); box('g_steel', (BX0 - 0.12, y - 0.22, 2.62), (BX0, y + 0.22, 2.66))
    # ---- parapet, fascia, roof deck
    for (a, b) in (((BX0, BY0), (BX1, BY0 + 0.25)), ((BX0, BY1 - 0.25), (BX1, BY1)), ((BX0, BY0), (BX0 + 0.25, BY1)), ((BX1 - 0.25, BY0), (BX1, BY1))):
        box(CLAD, (a[0], a[1], ROOF), (b[0], b[1], PARA))
    box('g_roof', (BX0 + 0.25, BY0 + 0.25, 4.4), (BX1 - 0.25, BY1 - 0.25, ROOF))
    for (a, b) in (((BX0 - 0.05, BY0 - 0.05), (BX1 + 0.05, BY0)), ((BX0 - 0.05, BY1), (BX1 + 0.05, BY1 + 0.05)), ((BX0 - 0.05, BY0), (BX0, BY1)), ((BX1, BY0), (BX1 + 0.05, BY1))):
        box(RED, (a[0], a[1], 4.55), (b[0], b[1], 5.0))
    for (a, b) in (((BX0 - 0.08, BY0 - 0.08), (BX1 + 0.08, BY0 + 0.27)), ((BX0 - 0.08, BY1 - 0.27), (BX1 + 0.08, BY1 + 0.08)), ((BX0 - 0.08, BY0), (BX0 + 0.27, BY1)), ((BX1 - 0.27, BY0), (BX1 + 0.08, BY1))):
        box(WHITE, (a[0], a[1], PARA), (b[0], b[1], PARA + 0.06))
    box('g_neon_r', (BX1 + 0.05, BY0, 4.5), (BX1 + 0.085, BY1, 4.54))                                   # a red line of light under the fascia
    box('g_neon_r', (BX0, BY1 + 0.05, 4.5), (BX1, BY1 + 0.085, 4.54))
    # a flat steel awning over the whole front walk, on tie rods
    box('g_roof', (BX1, BY0 + 0.5, 3.42), (BX1 + 2.2, BY1 - 0.5, 3.5)); box(RED, (BX1 + 2.2, BY0 + 0.5, 3.36), (BX1 + 2.26, BY1 - 0.5, 3.56))
    box(WHITE, (BX1 + 0.02, BY0 + 0.52, 3.4), (BX1 + 2.18, BY1 - 0.52, 3.42))
    for y in range(-20, 22, 4):
        cyl('g_steel', (BX1 + 2.0, y, 3.5), (BX1 + 0.02, y, 4.4), 0.015, 5, caps=False)
        box(LIGHT, (BX1 + 0.9, y + 1.8, 3.385), (BX1 + 1.3, y + 2.2, 3.402))
    SHELTERS.append((BX1, BY0 + 0.5, BX1 + 2.2, BY1 - 0.5, 3.5))
    # names over the doors
    text(WHITE, 'FOOD MART', 0.5, (BX1 + 0.07, -12.0, 4.6), 'E', depth=0.05)
    text(WHITE, 'OPEN 24 HOURS', 0.3, (BX1 + 0.07, 11.0, 4.63), 'E', depth=0.04)
    # ---- the ceiling and the wall between the two halves
    box(CEILM, (BX0 + t, BY0 + t, CEIL), (BX1 - t, BY1 - t, CEIL + 0.05))
    wall((BX0 + t, PARTY - 0.12, FL), (-14.2, PARTY + 0.12, CEIL), WALL); wall((-12.6, PARTY - 0.12, FL), (BX1 - t, PARTY + 0.12, CEIL), WALL)
    wall((-14.2, PARTY - 0.12, 2.3), (-12.6, PARTY + 0.12, CEIL), WALL)
    # ---- on the roof
    for (x, y) in ((-22.0, -14.0), (-22.0, 6.0), (-13.0, 14.0)):
        box(METAL, (x - 1.0, y - 0.7, ROOF + 0.15), (x + 1.0, y + 0.7, ROOF + 1.1)); box('g_steel', (x - 1.05, y - 0.75, ROOF), (x + 1.05, y + 0.75, ROOF + 0.15))
        cyl(BLACK, (x - 0.4, y, ROOF + 1.1), (x - 0.4, y, ROOF + 1.13), 0.38, 14); cyl(BLACK, (x + 0.45, y, ROOF + 1.1), (x + 0.45, y, ROOF + 1.13), 0.38, 14)
        for k in range(9): box('g_steel', (x - 0.95, y - 0.72, ROOF + 0.25 + 0.09 * k), (x + 0.95, y - 0.7, ROOF + 0.29 + 0.09 * k))
    cyl(METAL, (-24.2, 16.0, ROOF), (-24.2, 16.0, ROOF + 1.5), 0.3, 12); cyl(METAL, (-24.2, 16.0, ROOF + 1.5), (-24.2, 16.0, ROOF + 1.62), 0.42, 12)   # the kitchen's flue
    for (x, y) in ((-18.0, 18.0), (-16.0, -6.0), (-11.0, -18.0)):
        cyl(METAL, (x, y, ROOF), (x, y, ROOF + 0.5), 0.11, 8); ball(METAL, (x, y, ROOF + 0.55), 0.16, sz=0.6, seg=8, rings=4)
    cyl('g_steel', (-10.5, 19.0, ROOF), (-10.5, 19.0, ROOF + 1.1), 0.03, 6); ball(WHITE, (-10.5, 19.3, ROOF + 1.2), 0.45, sx=1.0, sz=0.35, seg=12, rings=5)      # a dish
    # the diner's sign: big letters on a steel frame over the east parapet, lit red
    for y in (4.0, 8.0, 12.0, 16.0, 18.0):
        cyl('g_steel', (BX1 - 0.6, y, ROOF), (BX1 - 0.6, y, PARA + 2.3), 0.035, 6)
        cyl('g_steel', (BX1 - 0.6, y, PARA + 1.6), (BX1 - 2.2, y, ROOF), 0.025, 5)
    for z in (PARA + 0.35, PARA + 2.15): box('g_steel', (BX1 - 0.64, 3.6, z - 0.03), (BX1 - 0.56, 18.4, z + 0.03))
    text('g_neon_r', 'DINER', 1.7, (BX1 - 0.54, 11.0, PARA + 0.5), 'E', depth=0.12, squeeze=1.15)
    text('g_neon_w', 'EAT', 0.55, (BX1 - 0.54, 4.8, PARA + 0.55), 'E', depth=0.08)


# ------------------------------------------------------------------ the diner
def diner():
    E, N = BX1 - 0.3, BY1 - 0.3
    W = BX0 + 0.3
    KX = -21.6                                       # the kitchen's wall
    # ---- booths at the east windows
    def booth(yc):
        z = FL
        box('g_table', (E - 2.25, yc - 0.36, z + 0.72), (E - 0.12, yc + 0.36, z + 0.76)); box('g_chrome', (E - 2.27, yc - 0.38, z + 0.7), (E - 0.1, yc + 0.38, z + 0.72))
        cyl('g_chrome', (E - 1.5, yc, z), (E - 1.5, yc, z + 0.7), 0.045, 8); cyl('g_chrome', (E - 1.5, yc, z), (E - 1.5, yc, z + 0.03), 0.26, 12)
        collide((E - 2.27, yc - 0.38, z), (E - 0.1, yc + 0.38, z + 0.76))
        table_setting(E - 0.32, yc, z + 0.76)
        for s in (-1.0, 1.0):
            if RNG.random() < 0.7: plate(E - RNG.uniform(0.9, 1.9), yc + s * 0.18, z + 0.76)
            if RNG.random() < 0.7: cup(E - RNG.uniform(0.7, 2.0), yc + s * 0.08, z + 0.76)
    def bench(y0, y1, back_at):
        """A booth bench from y0 to y1; its back stands at `back_at` (y0 or y1)."""
        z = FL
        box('g_steel', (E - 2.25, y0, z), (E - 0.12, y1, z + 0.12)); box('g_seat', (E - 2.25, y0, z + 0.12), (E - 0.12, y1, z + 0.46))
        b0, b1 = (y0, y0 + 0.16) if back_at == y0 else (y1 - 0.16, y1)
        box('g_seat', (E - 2.25, b0, z + 0.46), (E - 0.12, b1, z + 1.12)); box('g_chrome', (E - 2.27, b0 - 0.01, z + 1.12), (E - 0.1, b1 + 0.01, z + 1.15))
        for k in range(4):                           # the piping of the upholstery
            x = E - 2.25 + 2.13 * (k + 0.5) / 4
            box(WHITE, (x - 0.008, b0 - 0.003, z + 0.5), (x + 0.008, b1 + 0.003, z + 1.1))
        collide((E - 2.25, y0, z), (E - 0.12, y1, z + 1.15))
    ycs = [3.6 + 2.5 * k for k in range(7)]
    for i, yc in enumerate(ycs):
        booth(yc)
        bench(yc - 1.02, yc - 0.48, yc - 1.02)
        bench(yc + 0.48, yc + 1.02, yc + 1.02)
        pendant(E - 1.2, yc, 2.25)
    # ---- the counter: a long bar with stools, the back bar against the kitchen wall
    cx0, cx1 = -19.9, -19.2
    box('g_clad', (cx0, 5.5, FL + 0.1), (cx1, 18.5, FL + 1.02)); box('g_steel', (cx0 + 0.04, 5.5, FL), (cx1 - 0.04, 18.5, FL + 0.1))
    box('g_table', (cx0 - 0.1, 5.4, FL + 1.02), (cx1 + 0.28, 18.6, FL + 1.07)); box('g_chrome', (cx0 - 0.12, 5.38, FL + 0.99), (cx1 + 0.3, 18.62, FL + 1.02))
    box(RED, (cx1, 5.5, FL + 0.55), (cx1 + 0.012, 18.5, FL + 0.72))
    cyl('g_chrome', (cx1 + 0.22, 5.6, FL + 0.18), (cx1 + 0.22, 18.4, FL + 0.18), 0.025, 8)                 # the foot rail
    collide((cx0 - 0.1, 5.4, FL), (cx1 + 0.28, 18.6, FL + 1.07))
    for k in range(10):
        y = 6.3 + 1.26 * k
        stool(cx1 + 0.75, y)
        if k % 2 == 0: plate(cx1 - 0.1, y, FL + 1.07, food=RNG.random() < 0.6)
        if k % 3 == 1: cup(cx1 + 0.05, y + 0.2, FL + 1.07)
        if k % 3 == 0: table_setting(cx0 + 0.2, y + 0.5, FL + 1.07)
    # till, pie stand and a coffee machine on the counter
    box('g_steel', (cx0 + 0.1, 5.7, FL + 1.07), (cx1 - 0.1, 6.15, FL + 1.3)); box('g_lcd', (cx0 + 0.16, 5.74, FL + 1.3), (cx1 - 0.16, 5.78, FL + 1.52))
    cyl('g_chrome', (cx0 + 0.35, 12.0, FL + 1.07), (cx0 + 0.35, 12.0, FL + 1.2), 0.03, 6); cyl('g_chrome', (cx0 + 0.35, 12.0, FL + 1.2), (cx0 + 0.35, 12.0, FL + 1.22), 0.2, 14)
    cyl('g_bun', (cx0 + 0.35, 12.0, FL + 1.22), (cx0 + 0.35, 12.0, FL + 1.28), 0.16, 12); ball(GLASS, (cx0 + 0.35, 12.0, FL + 1.26), 0.21, sz=0.75, seg=12, rings=5)
    # back bar
    bx0, bx1 = KX + 0.12, KX + 0.75
    box('g_steel', (bx0, 5.5, FL), (bx1, 18.5, FL + 0.9), col=True); box('g_chrome', (bx0, 5.5, FL + 0.9), (bx1 + 0.03, 18.5, FL + 0.94))
    for k, y in enumerate((6.4, 9.0, 15.2, 17.6)):
        if k % 2 == 0:                               # coffee machines
            box('g_chrome', (bx0 + 0.08, y - 0.3, FL + 0.94), (bx1 - 0.08, y + 0.3, FL + 1.4)); box(BLACK, (bx1 - 0.08, y - 0.22, FL + 1.0), (bx1 - 0.06, y + 0.22, FL + 1.2))
            for dy in (-0.14, 0.14): cyl(GLASS, (bx1 + 0.02, y + dy, FL + 0.94), (bx1 + 0.02, y + dy, FL + 1.1), 0.07, 8); cyl('g_p_brown', (bx1 + 0.02, y + dy, FL + 0.95), (bx1 + 0.02, y + dy, FL + 1.04), 0.065, 8)
        else:                                        # a milkshake mixer and stacked cups
            box('g_p_green', (bx0 + 0.15, y - 0.12, FL + 0.94), (bx1 - 0.2, y + 0.12, FL + 1.45)); cyl('g_chrome', (bx1 - 0.15, y, FL + 0.98), (bx1 - 0.15, y, FL + 1.18), 0.045, 8)
            for j in range(3): cyl(WHITE, (bx0 + 0.2, y + 0.3 + 0.1 * j, FL + 0.94), (bx0 + 0.2, y + 0.3 + 0.1 * j, FL + 1.2), 0.04, 8)
    # ---- the kitchen behind its wall: a pass, a door gap, a range and a hood to look at
    wall((KX - 0.1, 4.0, 2.3), (KX + 0.1, 5.2, CEIL), WALL, col=False)                                  # the doorway's head
    wall((KX - 0.1, 5.2, FL), (KX + 0.1, N, 1.15), 'g_subway'); wall((KX - 0.1, 5.2, 2.05), (KX + 0.1, N, CEIL), WALL)
    wall((KX - 0.1, 5.2, 1.15), (KX + 0.1, 7.6, 2.05), WALL); wall((KX - 0.1, 16.4, 1.15), (KX + 0.1, N, 2.05), WALL)
    wall((KX - 0.1, 2.9, FL), (KX + 0.1, 4.0, CEIL), WALL)
    box('g_chrome', (KX - 0.2, 7.6, 1.13), (KX + 0.3, 16.4, 1.17))                                       # the pass shelf
    for y in (9.0, 11.5, 14.0): plate(KX + 0.1, y, 1.17); box(WHITE, (KX + 0.12, y + 0.4, 1.17), (KX + 0.13, y + 0.55, 1.3))
    for y in (8.5, 12.0, 15.5): box('g_neon_r', (KX + 0.02, y - 0.3, 1.96), (KX + 0.06, y + 0.3, 2.0))   # heat lamps over the pass
    text(BLACK, 'BURGERS   FRIES   SHAKES   PIE   COFFEE', 0.2, (KX + 0.115, 12.0, 2.75), 'E', depth=0.012)
    text(RED, 'ALL DAY BREAKFAST', 0.26, (KX + 0.115, 12.0, 2.35), 'E', depth=0.012)
    box('g_subway', (W + 0.012, 5.2, FL), (W + 0.026, N - 0.012, CEIL)); box('g_subway', (W + 0.012, N - 0.026, FL), (KX - 0.1, N - 0.012, CEIL))
    box('g_chrome', (W + 0.05, 7.0, FL), (W + 0.85, 16.5, FL + 0.9), col=True)                             # the cook line
    for y in (8.0, 9.2):
        box(BLACK, (W + 0.1, y - 0.5, FL + 0.9), (W + 0.8, y + 0.5, FL + 0.93))
        for (dx, dy) in ((0.28, -0.25), (0.28, 0.25), (0.62, -0.25), (0.62, 0.25)): cyl('g_steel', (W + dx, y + dy, FL + 0.93), (W + dx, y + dy, FL + 0.95), 0.1, 10)
    box('g_steel', (W + 0.1, 10.4, FL + 0.9), (W + 0.8, 12.6, FL + 0.94))                                  # the griddle
    for k in range(5): cyl('g_p_brown', (W + 0.3 + 0.1 * (k % 2), 10.7 + 0.4 * k, FL + 0.94), (W + 0.3 + 0.1 * (k % 2), 10.7 + 0.4 * k, FL + 0.96), 0.055, 8)
    for y in (13.4, 14.4): box('g_chrome', (W + 0.15, y - 0.3, FL + 0.9), (W + 0.75, y + 0.3, FL + 1.12)); cyl(BLACK, (W + 0.45, y, FL + 1.12), (W + 0.78, y, FL + 1.2), 0.012, 5)      # fryers
    box('g_chrome', (W + 0.02, 6.8, 2.05), (W + 1.1, 16.7, 2.5)); box('g_steel', (W + 0.02, 6.8, 2.5), (W + 0.7, 16.7, CEIL))                               # the hood
    cyl('g_p_brown', (W + 0.45, 8.0, FL + 0.95), (W + 0.45, 8.0, FL + 1.1), 0.12, 10); cyl('g_steel', (W + 0.45, 9.45, FL + 0.95), (W + 0.45, 9.45, FL + 1.02), 0.14, 10)
    box('g_chrome', (KX - 0.9, 17.0, FL), (KX - 0.15, 21.3, FL + 2.0), col=True)                             # fridges
    for y in (18.0, 20.2): box('g_steel', (KX - 0.92, y - 0.02, FL + 0.1), (KX - 0.9, y + 0.02, FL + 1.9))
    box('g_chrome', (W + 1.6, 19.2, FL), (KX - 1.2, 21.4, FL + 0.9), col=True); box('g_steel', (W + 1.9, 19.6, FL + 0.9), (W + 2.7, 21.0, FL + 0.92))            # a sink
    cyl('g_chrome', (W + 2.3, 21.2, FL + 0.9), (W + 2.3, 21.2, FL + 1.25), 0.02, 6)
    LIGHTS.append((W + 2.0, 12.0, 3.0))
    # ---- the wet rooms in the south-west corner: two doors, signs
    wall((W, 2.9, FL), (KX + 0.1, 3.1, CEIL), WALL)
    wall((KX - 0.1, PARTY + 0.12, FL), (KX + 0.1, -0.9, CEIL), WALL); wall((KX - 0.1, 0.1, FL), (KX + 0.1, 0.6, CEIL), WALL); wall((KX - 0.1, 1.6, FL), (KX + 0.1, 2.9, CEIL), WALL)
    for (a, b, lab) in ((-0.9, 0.1, 'W'), (0.6, 1.6, 'M')):
        wall((KX - 0.1, a, 2.15), (KX + 0.1, b, CEIL), WALL)
        box('g_door', (KX - 0.03, a, FL), (KX + 0.03, b, 2.15), col=True); box('g_chrome', (KX + 0.03, b - 0.2, 1.1), (KX + 0.08, b - 0.08, 1.16))
        box('g_p_blue', (KX + 0.03, 0.5 * (a + b) - 0.12, 1.6), (KX + 0.04, 0.5 * (a + b) + 0.12, 1.84)); text(WHITE, lab, 0.16, (KX + 0.04, 0.5 * (a + b), 1.64), 'E', depth=0.006)
    # ---- the floor of the room: round tables, a jukebox, a gumball machine, plants
    for (x, y) in ((-14.4, 5.8), (-14.9, 10.6), (-14.4, 15.4), (-15.2, 19.6)):
        cyl('g_chrome', (x, y, FL), (x, y, FL + 0.03), 0.28, 14); cyl('g_chrome', (x, y, FL + 0.03), (x, y, FL + 0.72), 0.045, 8)
        cyl('g_chrome', (x, y, FL + 0.72), (x, y, FL + 0.74), 0.5, 20); cyl('g_table', (x, y, FL + 0.74), (x, y, FL + 0.77), 0.49, 20)
        collide((x - 0.42, y - 0.42, FL), (x + 0.42, y + 0.42, FL + 0.77))
        table_setting(x, y, FL + 0.77)
        for k in range(4):
            a = TAU * k / 4 + 0.6 + RNG.uniform(-0.2, 0.2)
            chair(x + 0.78 * math.sin(a), y - 0.78 * math.cos(a), a + math.pi + RNG.uniform(-0.15, 0.15))
            if RNG.random() < 0.5: plate(x + 0.3 * math.sin(a), y - 0.3 * math.cos(a), FL + 0.77)
    jx, jy = -16.5, PARTY + 0.5                                                                        # the jukebox, against the party wall
    box('g_p_brown', (jx - 0.42, jy - 0.3, FL), (jx + 0.42, jy + 0.3, FL + 1.0)); cyl('g_p_brown', (jx, jy - 0.3, FL + 1.0), (jx, jy + 0.3, FL + 1.0), 0.42, 16)
    cyl('g_neon_r', (jx, jy + 0.3, FL + 1.0), (jx, jy + 0.33, FL + 1.0), 0.36, 16); cyl('g_neon_w', (jx, jy + 0.33, FL + 1.0), (jx, jy + 0.34, FL + 1.0), 0.24, 16)
    box('g_chrome', (jx - 0.34, jy + 0.3, FL + 0.25), (jx + 0.34, jy + 0.32, FL + 0.6)); box('g_p_yellow', (jx - 0.3, jy + 0.32, FL + 0.66), (jx + 0.3, jy + 0.33, FL + 0.8))
    collide((jx - 0.42, jy - 0.3, FL), (jx + 0.42, jy + 0.34, FL + 1.42))
    gx, gy = -9.3, 1.7                                                                                 # gumballs by the door
    cyl(RED, (gx, gy, FL), (gx, gy, FL + 0.05), 0.16, 10); cyl('g_chrome', (gx, gy, FL + 0.05), (gx, gy, FL + 0.8), 0.025, 6); box(RED, (gx - 0.1, gy - 0.1, FL + 0.8), (gx + 0.1, gy + 0.1, FL + 0.95))
    ball(GLASS, (gx, gy, FL + 1.1), 0.16, seg=10, rings=6)
    for k in range(16): ball(PRODUCTS[k % 5], (gx + RNG.uniform(-0.09, 0.09), gy + RNG.uniform(-0.09, 0.09), FL + 1.02 + RNG.uniform(0.0, 0.1)), 0.025, seg=5, rings=3)
    planter_box(-19.0, 3.4, -17.2, 4.0, FL, 0.5); planter_box(-11.6, PARTY + 0.2, -9.6, PARTY + 0.7, FL, 0.5)
    # ---- on the walls: frames, a clock, a neon in the window, the specials
    for k, x in enumerate((-20.0, -17.8, -15.6, -11.2)):
        box(BLACK, (x - 0.38, PARTY + 0.12, 1.5), (x + 0.38, PARTY + 0.14, 2.1)); box(PRODUCTS[(k * 2) % 7], (x - 0.33, PARTY + 0.14, 1.55), (x + 0.33, PARTY + 0.145, 2.05))
        box(WHITE, (x - 0.2, PARTY + 0.145, 1.7), (x + 0.2, PARTY + 0.15, 1.9))
    cyl(WHITE, (-13.4, PARTY + 0.12, 2.7), (-13.4, PARTY + 0.15, 2.7), 0.24, 20); cyl(BLACK, (-13.4, PARTY + 0.115, 2.7), (-13.4, PARTY + 0.135, 2.7), 0.26, 20)
    box(BLACK, (-13.41, PARTY + 0.15, 2.7), (-13.39, PARTY + 0.155, 2.88)); box(BLACK, (-13.4, PARTY + 0.15, 2.69), (-13.28, PARTY + 0.155, 2.71))
    text('g_neon_r', 'OPEN', 0.34, (E + 0.05, 4.1, 2.45), 'E', depth=0.03)
    text('g_neon_b', 'COFFEE', 0.26, (E + 0.05, 13.5, 2.6), 'E', depth=0.03)
    # ceiling: light panels in rows, two fans
    for x in (-18.2, -14.6, -11.0):
        for y in (0.5, 5.0, 9.5, 14.0, 18.5): ceiling_panel(x, y)
    for (x, y) in ((-16.4, 7.5), (-16.4, 15.5)):
        cyl('g_steel', (x, y, CEIL), (x, y, CEIL - 0.3), 0.03, 6); cyl('g_steel', (x, y, CEIL - 0.3), (x, y, CEIL - 0.38), 0.1, 10)
        for k in range(4): obox('g_table', x, y, TAU * k / 4 + 0.3, (0.1, -0.07, CEIL - 0.36), (0.75, 0.07, CEIL - 0.345))
    for (x, y) in ((-12.0, 4.0), (-12.0, 11.0), (-12.0, 18.0), (-17.5, 12.0)): LIGHTS.append((x, y, 2.9))


# ------------------------------------------------------------------ the food mart
def gondola(y, x0, x1):
    """A shelving run along X at y: a base, a spine, four shelves each side, full of goods."""
    box('g_steel', (x0, y - 0.45, FL), (x1, y + 0.45, FL + 0.14)); box(WHITE, (x0, y - 0.03, FL + 0.14), (x1, y + 0.03, FL + 1.62))
    for x in (x0, x1): box(WHITE, (x - 0.03, y - 0.44, FL + 0.14), (x + 0.03, y + 0.44, FL + 1.62))
    box(RED, (x0 - 0.035, y - 0.45, FL + 1.5), (x0 - 0.03, y + 0.45, FL + 1.62))
    collide((x0 - 0.04, y - 0.46, FL), (x1 + 0.04, y + 0.46, FL + 1.62))
    for lvl in range(4):
        z = FL + 0.16 + 0.37 * lvl
        for s in (-1.0, 1.0):
            d = 0.42 - 0.04 * lvl
            box(WHITE, (x0 + 0.03, y + s * 0.03, z - 0.02), (x1 - 0.03, y + s * d, z))
            box(YELLOW if lvl == 1 else WHITE, (x0 + 0.03, y + s * d, z - 0.035), (x1 - 0.03, y + s * (d + 0.006), z + 0.005))     # the price strip
            x = x0 + 0.06
            while x < x1 - 0.2:
                w = RNG.uniform(0.09, 0.2); h = RNG.uniform(0.12, 0.3); g = RNG.choice(PRODUCTS)
                n = RNG.randrange(2, 5)
                for k in range(n):
                    if x + w > x1 - 0.05: break
                    if RNG.random() < 0.93:
                        if RNG.random() < 0.3: cyl(g, (x + w * 0.5, y + s * (d - 0.09), z), (x + w * 0.5, y + s * (d - 0.09), z + h), w * 0.42, 6)
                        else: box(g, (x, y + s * (d - 0.17), z), (x + w - 0.012, y + s * (d - 0.03), z + h))
                    x += w
                x += RNG.uniform(0.0, 0.03)


def fridge(y0, y1):
    """A glass-doored drinks fridge on the west wall, from y0 to y1."""
    W = BX0 + 0.3
    box(BLACK, (W, y0, FL), (W + 0.72, y1, FL + 2.2)); box('g_fridge', (W + 0.06, y0 + 0.05, FL + 0.12), (W + 0.12, y1 - 0.05, FL + 2.08))
    for yy in (y0 + 0.05, y1 - 0.07): box('g_fridge', (W + 0.12, yy, FL + 0.12), (W + 0.7, yy + 0.02, FL + 2.08))
    box('g_light_c', (W + 0.62, y0 + 0.05, FL + 2.02), (W + 0.7, y1 - 0.05, FL + 2.08))
    quad2(GLASS, (W + 0.73, y0 + 0.04, FL + 0.14), (W + 0.73, y1 - 0.04, FL + 0.14), (W + 0.73, y1 - 0.04, FL + 2.06), (W + 0.73, y0 + 0.04, FL + 2.06))
    box('g_chrome', (W + 0.73, y1 - 0.12, FL + 0.9), (W + 0.77, y1 - 0.08, FL + 1.4)); box(BLACK, (W + 0.72, y0, FL + 0.1), (W + 0.745, y1, FL + 0.14))
    for lvl in range(5):
        z = FL + 0.16 + 0.37 * lvl
        box('g_chrome', (W + 0.08, y0 + 0.05, z - 0.012), (W + 0.68, y1 - 0.05, z))
        g = PRODUCTS[(lvl + int(y0 * 3)) % 7]
        y = y0 + 0.13
        while y < y1 - 0.1:
            for dx in (0.56, 0.42):
                cyl(g, (W + dx, y, z), (W + dx, y, z + 0.2), 0.034, 6); cyl(WHITE if lvl % 2 else g, (W + dx, y, z + 0.2), (W + dx, y, z + 0.27), 0.014, 5)
            y += 0.085
            if RNG.random() < 0.15: g = RNG.choice(PRODUCTS)
    collide((W, y0, FL), (W + 0.75, y1, FL + 2.2))


def mart():
    E = BX1 - 0.3; S = BY0 + 0.3; W = BX0 + 0.3
    for y in (-18.4, -14.8, -11.2): gondola(y, -22.2, -13.2)
    for k in range(6): fridge(-21.4 + 2.05 * k, -21.4 + 2.05 * (k + 1) - 0.05)
    text('g_p_blue', 'COLD DRINKS', 0.3, (W + 0.02, -15.2, 2.85), 'E', depth=0.012)
    # ---- the counter by the door, with everything a counter has
    box(CLAD, (-11.9, -8.2, FL), (-11.3, -3.2, FL + 1.0)); box(CLAD, (-11.9, -8.2, FL), (-9.2, -7.6, FL + 1.0))
    box('g_table', (-12.0, -8.3, FL + 1.0), (-11.2, -3.1, FL + 1.04)); box('g_table', (-12.0, -8.3, FL + 1.0), (-9.1, -7.5, FL + 1.04))
    box(RED, (-11.91, -8.21, FL + 0.6), (-11.9, -3.2, FL + 0.75))
    collide((-12.0, -8.3, FL), (-11.2, -3.1, FL + 1.04)); collide((-12.0, -8.3, FL), (-9.1, -7.5, FL + 1.04))
    box('g_steel', (-11.8, -5.9, FL + 1.04), (-11.35, -5.3, FL + 1.22)); box('g_lcd', (-11.75, -5.85, FL + 1.22), (-11.7, -5.35, FL + 1.5))                       # the till
    box(BLACK, (-11.7, -6.6, FL + 1.04), (-11.5, -6.4, FL + 1.12)); box('g_lcd', (-11.68, -6.58, FL + 1.12), (-11.52, -6.42, FL + 1.125))                         # card terminal
    box(GLASS, (-11.85, -4.9, FL + 1.04), (-11.4, -3.5, FL + 1.38))                                                                                              # a glass case of pastries
    for k in range(8): cyl('g_bun', (-11.7 + 0.13 * (k % 2), -4.75 + 0.17 * (k // 2) * 2, FL + 1.05), (-11.7 + 0.13 * (k % 2), -4.75 + 0.17 * (k // 2) * 2, FL + 1.1), 0.05, 8)
    for k in range(10):                             # sweets on the counter front, in a rack
        box(PRODUCTS[k % 7], (-12.08, -7.9 + 0.36 * k, FL + 0.3 + 0.0), (-12.0, -7.6 + 0.36 * k, FL + 0.42)); box(PRODUCTS[(k + 3) % 7], (-12.08, -7.9 + 0.36 * k, FL + 0.5), (-12.0, -7.6 + 0.36 * k, FL + 0.62))
    box('g_steel', (-12.09, -8.0, FL + 0.26), (-12.07, -4.3, FL + 0.66))
    # the wall of small things behind the counter
    box('g_steel', (-11.0, PARTY - 0.5, FL), (-8.6, PARTY - 0.12, FL + 2.4), col=True)
    for lvl in range(6):
        z = FL + 1.05 + 0.22 * lvl
        box(WHITE, (-10.95, PARTY - 0.52, z - 0.01), (-8.65, PARTY - 0.5, z))
        x = -10.9
        while x < -8.7:
            g = RNG.choice(PRODUCTS)
            for k in range(RNG.randrange(2, 6)):
                if x > -8.75: break
                box(g, (x, PARTY - 0.56, z), (x + 0.058, PARTY - 0.5, z + 0.1)); x += 0.062
            x += 0.01
    text(RED, 'PAY HERE', 0.22, (-10.0, PARTY - 0.53, FL + 2.48), 'S', depth=0.012)
    # ---- the coffee bar on the south wall
    box(WHITE, (-21.5, S, FL), (-12.5, S + 0.7, FL + 0.92), col=True); box('g_table', (-21.55, S, FL + 0.92), (-12.45, S + 0.75, FL + 0.96))
    box('g_subway', (-21.6, S + 0.012, FL + 0.96), (-12.4, S + 0.026, FL + 2.4))
    for k, x in enumerate((-20.6, -19.2)):          # bean-to-cup machines
        box(BLACK, (x - 0.3, S + 0.08, FL + 0.96), (x + 0.3, S + 0.6, FL + 1.7)); box('g_lcd', (x - 0.2, S + 0.6, FL + 1.4), (x + 0.2, S + 0.605, FL + 1.62))
        box('g_chrome', (x - 0.24, S + 0.6, FL + 0.98), (x + 0.24, S + 0.68, FL + 1.0)); cup(x, S + 0.63, FL + 1.0)
    for j in range(4):                              # stacked cups and lids
        for k in range(6): cyl(WHITE, (-18.3 + 0.13 * j, S + 0.3, FL + 0.96 + 0.035 * k), (-18.3 + 0.13 * j, S + 0.3, FL + 1.0 + 0.035 * k), 0.045, 8, r1=0.05)
    for k, g in enumerate(('g_p_red', 'g_p_blue', 'g_p_green')):     # slush machine
        x = -17.0 + 0.36 * k
        box(BLACK, (x - 0.16, S + 0.12, FL + 0.96), (x + 0.16, S + 0.55, FL + 1.15)); cyl(g, (x, S + 0.34, FL + 1.15), (x, S + 0.34, FL + 1.5), 0.14, 10); cyl(BLACK, (x, S + 0.34, FL + 1.5), (x, S + 0.34, FL + 1.54), 0.15, 10)
    box('g_chrome', (-15.6, S + 0.1, FL + 0.96), (-14.6, S + 0.6, FL + 1.12))                           # hot dog roller
    for k in range(6): cyl('g_p_brown', (-15.55, S + 0.16 + 0.075 * k, FL + 1.14), (-14.65, S + 0.16 + 0.075 * k, FL + 1.14), 0.022, 6)
    quad2(GLASS, (-15.65, S + 0.62, FL + 0.96), (-14.55, S + 0.62, FL + 0.96), (-14.55, S + 0.4, FL + 1.4), (-15.65, S + 0.4, FL + 1.4))
    box(WHITE, (-14.2, S + 0.1, FL + 0.96), (-13.5, S + 0.55, FL + 1.3)); box(BLACK, (-14.15, S + 0.55, FL + 1.0), (-13.75, S + 0.555, FL + 1.26))     # microwave
    text('g_p_brown', 'FRESH COFFEE', 0.28, (-17.0, S + 0.005, FL + 2.5), 'N', depth=0.012)
    # ---- by the entrance: a cash machine, papers, an ice-cream chest, a rack of maps
    box('g_p_blue', (E - 0.75, -20.6, FL), (E - 0.1, -19.8, FL + 1.75), col=True); box('g_lcd', (E - 0.76, -20.45, FL + 1.05), (E - 0.75, -19.95, FL + 1.4)); box('g_chrome', (E - 0.78, -20.4, FL + 0.82), (E - 0.75, -20.0, FL + 0.98))
    text(WHITE, 'ATM', 0.16, (E - 0.755, -20.2, FL + 1.5), 'W', depth=0.006)
    box(WHITE, (E - 1.3, -18.8, FL), (E - 0.15, -17.0, FL + 0.85), col=True); quad2(GLASS, (E - 1.28, -18.75, FL + 0.86), (E - 0.17, -18.75, FL + 0.86), (E - 0.17, -17.05, FL + 0.86), (E - 1.28, -17.05, FL + 0.86))
    for k in range(18): box(PRODUCTS[k % 7], (E - 1.2 + 0.26 * (k % 4), -18.65 + 0.33 * (k // 4), FL + 0.6), (E - 1.0 + 0.26 * (k % 4), -18.4 + 0.33 * (k // 4), FL + 0.72))
    for k in range(4):                              # newspapers, in a wire rack inside the door
        box('g_steel', (E - 0.5, -9.6 + 0.0, FL + 0.2 + 0.32 * k), (E - 0.12, -8.6, FL + 0.22 + 0.32 * k)); box(WHITE, (E - 0.48, -9.55, FL + 0.22 + 0.32 * k), (E - 0.14, -8.65, FL + 0.3 + 0.32 * k))
        box(BLACK, (E - 0.47, -9.5, FL + 0.3 + 0.32 * k), (E - 0.3, -8.7, FL + 0.302 + 0.32 * k))
    collide((E - 0.5, -9.6, FL), (E - 0.12, -8.6, FL + 1.3))
    # a rotating rack of sunglasses and a stack of baskets by the door
    cyl('g_chrome', (-10.2, -14.6, FL), (-10.2, -14.6, FL + 1.7), 0.02, 6); cyl('g_chrome', (-10.2, -14.6, FL), (-10.2, -14.6, FL + 0.04), 0.22, 10)
    for k in range(24):
        a = TAU * (k % 6) / 6; z = FL + 0.6 + 0.26 * (k // 6)
        obox(BLACK, -10.2 + 0.2 * math.cos(a), -14.6 + 0.2 * math.sin(a), a + math.pi / 2, (-0.07, -0.01, z), (0.07, 0.01, z + 0.05))
    collide((-10.4, -14.8, FL), (-10.0, -14.4, FL + 1.7))
    for k in range(5): box(RED, (-9.4, -10.4, FL + 0.02 + 0.06 * k), (-8.9, -10.05, FL + 0.07 + 0.06 * k))
    # ---- an open chiller of sandwiches and salads against the party wall
    cx0, cx1, cy0, cy1 = -21.0, -15.2, PARTY - 0.95, PARTY - 0.14
    box(WHITE, (cx0, cy0, FL), (cx1, cy1, FL + 0.55), col=False); box(BLACK, (cx0, cy0, FL), (cx1, cy1, FL + 0.1)); box(WHITE, (cx0, cy1 - 0.08, FL + 0.55), (cx1, cy1, FL + 1.9))
    box(RED, (cx0 - 0.005, cy0 - 0.005, FL + 0.38), (cx1 + 0.005, cy0, FL + 0.52))
    for lvl in range(4):
        z = FL + 0.56 + 0.34 * lvl; d = 0.72 - 0.13 * lvl
        box('g_fridge', (cx0 + 0.03, cy1 - 0.08 - d, z - 0.012), (cx1 - 0.03, cy1 - 0.08, z))
        x = cx0 + 0.08
        while x < cx1 - 0.2:
            g = RNG.choice(PRODUCTS); w = RNG.uniform(0.11, 0.17)
            for k in range(RNG.randrange(2, 5)):
                if x + w > cx1 - 0.05: break
                if lvl < 2: obox(g, x + w * 0.5, cy1 - 0.08 - d + 0.1, 0.0, (-w * 0.45, -0.07, z), (w * 0.45, 0.07, z + 0.05)); obox(WHITE, x + w * 0.5, cy1 - 0.08 - d + 0.1, 0.0, (-w * 0.3, -0.05, z + 0.05), (w * 0.3, 0.05, z + 0.055))
                else: cyl(g, (x + w * 0.5, cy1 - 0.08 - d + 0.09, z), (x + w * 0.5, cy1 - 0.08 - d + 0.09, z + 0.2), 0.035, 6)
                x += w
            x += 0.02
    box('g_light_c', (cx0 + 0.03, cy1 - 0.3, FL + 1.86), (cx1 - 0.03, cy1 - 0.1, FL + 1.9))
    text('g_p_green', 'FRESH  TO  GO', 0.24, (0.5 * (cx0 + cx1), cy1 - 0.085, FL + 2.05), 'S', depth=0.012)
    collide((cx0, cy0, FL), (cx1, cy1, FL + 1.9))
    # ---- pallet displays of drinks near the front, each with a price on a pole
    for (px, py, ga, gb) in ((-10.0, -17.2, 'g_p_red', 'g_p_blue'), (-10.0, -12.4, 'g_p_green', 'g_p_orange'), (-13.0, -5.2, 'g_p_blue', 'g_p_yellow')):
        for j in range(3): box('g_wood', (px - 0.6, py - 0.5 + 0.42 * j, FL), (px + 0.6, py - 0.38 + 0.42 * j, FL + 0.1))
        for j in range(5): box('g_wood', (px - 0.6 + 0.27 * j, py - 0.5, FL + 0.1), (px - 0.48 + 0.27 * j, py + 0.46, FL + 0.125))
        for lvl in range(4):
            for i in range(3):
                for j in range(2):
                    if lvl == 3 and RNG.random() < 0.45: continue
                    g = ga if (i + j + lvl) % 2 == 0 else gb
                    x0 = px - 0.58 + 0.39 * i; y0 = py - 0.48 + 0.48 * j; z0 = FL + 0.125 + 0.26 * lvl
                    box(g, (x0, y0, z0), (x0 + 0.37, y0 + 0.45, z0 + 0.25)); box(WHITE, (x0 + 0.06, y0 - 0.002, z0 + 0.07), (x0 + 0.31, y0, z0 + 0.18))
        cyl('g_steel', (px + 0.62, py + 0.5, FL), (px + 0.62, py + 0.5, FL + 1.75), 0.012, 5); box(YELLOW, (px + 0.3, py + 0.49, FL + 1.4), (px + 0.94, py + 0.51, FL + 1.8))
        text(RED, '2 FOR 5', 0.13, (px + 0.62, py + 0.488, FL + 1.53), 'S', depth=0.006)
        collide((px - 0.6, py - 0.5, FL), (px + 0.6, py + 0.46, FL + 1.2))
    # ---- end caps on the shelving runs, facing the front
    for k, y in enumerate((-18.4, -14.8, -11.2)):
        x = -13.12
        box(WHITE, (x, y - 0.44, FL + 0.14), (x + 0.04, y + 0.44, FL + 1.62))
        for lvl in range(4):
            z = FL + 0.2 + 0.36 * lvl
            box(WHITE, (x + 0.04, y - 0.42, z - 0.02), (x + 0.3, y + 0.42, z))
            g = PRODUCTS[(k * 2 + lvl) % 7]
            for j in range(6): box(g, (x + 0.06, y - 0.4 + 0.135 * j, z), (x + 0.26, y - 0.29 + 0.135 * j, z + RNG.uniform(0.18, 0.28)))
        collide((x, y - 0.44, FL), (x + 0.3, y + 0.44, FL + 1.62))
    # ---- a door mat, a wet-floor sign, a stand of crisps by the counter
    box(BLACK, (E - 1.9, -13.1, FL), (E - 0.3, -10.9, FL + 0.012))
    quad2(YELLOW, (-12.75, -9.75, FL), (-12.45, -9.75, FL), (-12.6, -9.6, FL + 0.62), (-12.6, -9.6, FL + 0.62)); quad2(YELLOW, (-12.75, -9.45, FL), (-12.45, -9.45, FL), (-12.6, -9.6, FL + 0.62), (-12.6, -9.6, FL + 0.62))
    cyl('g_steel', (-12.4, -6.6, FL), (-12.4, -6.6, FL + 1.6), 0.014, 5)
    for lvl in range(5):
        for k in range(6):
            a = TAU * k / 6 + 0.3 * lvl
            obox(PRODUCTS[(k + lvl) % 7], -12.4 + 0.16 * math.cos(a), -6.6 + 0.16 * math.sin(a), a + math.pi / 2, (-0.07, -0.02, FL + 0.3 + 0.27 * lvl), (0.07, 0.02, FL + 0.5 + 0.27 * lvl))
    collide((-12.6, -6.8, FL), (-12.2, -6.4, FL + 1.6))
    # ceiling lights and signs over the aisles
    for x in (-22.5, -19.0, -15.5, -12.0):
        for y in (-19.8, -16.6, -13.0, -9.4, -5.6): ceiling_panel(x, y)
    for k, (y, lab) in enumerate(((-18.4, 'SNACKS'), (-14.8, 'GROCERY'), (-11.2, 'AUTO'))):
        box(WHITE, (-18.4, y - 0.02, 2.75), (-17.0, y + 0.02, 3.1)); text(RED, lab, 0.17, (-17.7, y - 0.025, 2.86), 'S', depth=0.006); text(RED, lab, 0.17, (-17.7, y + 0.025, 2.86), 'N', depth=0.006)
        for x in (-18.3, -17.1): cyl('g_steel', (x, y, 3.1), (x, y, CEIL), 0.006, 4, caps=False)
    for (x, y) in ((-20.0, -16.5), (-14.5, -16.5), (-17.0, -7.0), (-10.5, -6.0)): LIGHTS.append((x, y, 2.9))


# ------------------------------------------------------------------ round the site
def price_pylon(x, y):
    z = ASPH
    for dy in (-1.25, 1.25):
        box('g_steel', (x - 0.2, y + dy - 0.2, z), (x + 0.2, y + dy + 0.2, z + 9.2), col=True)
    box('g_conc', (x - 0.5, y - 1.8, z), (x + 0.5, y + 1.8, z + 0.35), col=True)
    box(RED, (x - 0.3, y - 1.7, z + 6.6), (x + 0.3, y + 1.7, z + 9.4))
    rows = (('UNLEADED', '1.59'), ('PREMIUM', '1.79'), ('DIESEL', '1.49'))
    for face, s in (('E', 1.0), ('W', -1.0), ):
        fx = x + s * 0.3
        text('g_neon_w', 'VIBE', 0.95, (fx + s * 0.005, y, z + 8.15), face, depth=0.05)
        text('g_neon_w', 'FUEL', 0.62, (fx + s * 0.005, y, z + 7.1), face, depth=0.05)
    for k, (lab, price) in enumerate(rows):
        z0 = z + 5.5 - 1.05 * k
        box(BLACK, (x - 0.26, y - 1.6, z0), (x + 0.26, y + 1.6, z0 + 0.95))
        for face, s in (('E', 1.0), ('W', -1.0)):
            fx = x + s * 0.262
            ya = y - s * 1.5
            text(WHITE, lab, 0.26, (fx, ya, z0 + 0.36), face, depth=0.02, align='LEFT')
            text('g_neon_g' if k < 2 else 'g_neon_y', price, 0.5, (fx, y + s * 0.85, z0 + 0.24), face, depth=0.03)
    box('g_p_blue', (x - 0.26, y - 1.6, z + 2.25), (x + 0.26, y + 1.6, z + 3.25))
    for face, s in (('E', 1.0), ('W', -1.0)):
        text(WHITE, 'DINER  SHOP', 0.34, (x + s * 0.262, y, z + 2.6), face, depth=0.02)
    LIGHTS.append((x + 1.5, y, z + 6.0))


def light_mast(x, y, arm=(-1.0, 0.0)):
    z = ASPH
    box('g_conc', (x - 0.3, y - 0.3, z), (x + 0.3, y + 0.3, z + 0.5), col=True)
    cyl('g_steel', (x, y, z + 0.5), (x, y, z + 8.0), 0.09, 10, r1=0.06)
    ax, ay = arm
    cyl('g_steel', (x, y, z + 7.9), (x + ax * 1.4, y + ay * 1.4, z + 8.1), 0.04, 6)
    obox('g_steel', x + ax * 1.6, y + ay * 1.6, math.atan2(ay, ax), (-0.45, -0.22, z + 8.02), (0.45, 0.22, z + 8.14))
    obox(LIGHT, x + ax * 1.6, y + ay * 1.6, math.atan2(ay, ax), (-0.4, -0.18, z + 8.0), (0.4, 0.18, z + 8.02))
    LIGHTS.append((x + ax * 1.6, y + ay * 1.6, z + 7.4))


def picnic_table(x, y, ang=0.0, z=WALK):
    obox('g_wood', x, y, ang, (-0.9, -0.38, z + 0.72), (0.9, 0.38, z + 0.77))
    for s in (-1.0, 1.0):
        obox('g_wood', x, y, ang, (-0.9, s * 0.62 - 0.14, z + 0.42), (0.9, s * 0.62 + 0.14, z + 0.46))
    for ex in (-0.7, 0.7):
        obox('g_steel', x, y, ang, (ex - 0.03, -0.76, z + 0.38), (ex + 0.03, 0.76, z + 0.42))
        obox('g_steel', x, y, ang, (ex - 0.03, -0.3, z), (ex + 0.03, 0.3, z + 0.04))
        obox('g_steel', x, y, ang, (ex - 0.03, -0.03, z), (ex + 0.03, 0.03, z + 0.72))
    c, s_ = abs(math.cos(ang)), abs(math.sin(ang))
    collide((x - 0.9 * c - 0.76 * s_, y - 0.9 * s_ - 0.76 * c, z), (x + 0.9 * c + 0.76 * s_, y + 0.9 * s_ + 0.76 * c, z + 0.77))


def parasol(x, y, g, z=WALK):
    cyl('g_conc', (x, y, z), (x, y, z + 0.1), 0.25, 10); cyl('g_steel', (x, y, z + 0.1), (x, y, z + 2.5), 0.022, 6)
    seg = 8
    for i in range(seg):
        a, b = TAU * i / seg, TAU * (i + 1) / seg
        pa = (x + 1.5 * math.cos(a), y + 1.5 * math.sin(a), z + 2.05); pb = (x + 1.5 * math.cos(b), y + 1.5 * math.sin(b), z + 2.05)
        gg = g if i % 2 == 0 else WHITE
        tri(gg, pa, pb, (x, y, z + 2.5)); tri(gg, pb, pa, (x, y, z + 2.5))


def site():
    # the price pylon at the north-east corner, by the road
    price_pylon(23.5, 31.0)
    # light masts
    light_mast(24.6, -30.0, (-1.0, 0.3)); light_mast(24.6, 20.0, (-1.0, 0.0)); light_mast(1.5, -33.0, (0.3, 1.0)); light_mast(1.5, 33.0, (0.3, -1.0))
    # ---- air, water and vacuum, at the south end of the forecourt
    ax, ay = 9.0, -29.5
    box('g_conc', (ax - 1.6, ay - 0.5, ASPH), (ax + 1.6, ay + 0.5, ASPH + 0.14), col=True)
    box('g_p_blue', (ax - 1.2, ay - 0.25, ASPH + 0.14), (ax - 0.5, ay + 0.25, ASPH + 1.6), col=True); box('g_lcd', (ax - 1.05, ay + 0.25, ASPH + 1.15), (ax - 0.65, ay + 0.255, ASPH + 1.4))
    text(WHITE, 'AIR', 0.2, (ax - 0.85, ay + 0.256, ASPH + 0.75), 'N', depth=0.008)
    tube(BLACK, [(ax - 0.5, ay, ASPH + 1.3), (ax - 0.3, ay + 0.1, ASPH + 1.5), (ax - 0.2, ay + 0.2, ASPH + 0.8), (ax - 0.35, ay + 0.3, ASPH + 0.3), (ax - 0.5, ay + 0.2, ASPH + 0.9)], 0.014, 5)
    cyl(YELLOW, (ax + 0.7, ay, ASPH + 0.14), (ax + 0.7, ay, ASPH + 1.5), 0.33, 14); cyl('g_steel', (ax + 0.7, ay, ASPH + 1.5), (ax + 0.7, ay, ASPH + 1.62), 0.35, 14)
    text(BLACK, 'VAC', 0.18, (ax + 0.7, ay + 0.335, ASPH + 0.9), 'N', depth=0.006)
    tube(BLACK, [(ax + 1.0, ay, ASPH + 1.3), (ax + 1.35, ay + 0.1, ASPH + 1.45), (ax + 1.45, ay + 0.2, ASPH + 0.6), (ax + 1.2, ay + 0.3, ASPH + 0.25)], 0.03, 6)
    collide((ax + 0.36, ay - 0.34, ASPH), (ax + 1.04, ay + 0.34, ASPH + 1.62))
    for bx in (ax - 1.9, ax + 1.9): bollard(bx, ay + 0.9)
    # ---- two car chargers at the south parking bays
    for y in (-23.3, -25.9):
        x = -3.3
        box(WHITE, (x - 0.18, y - 0.28, WALK), (x + 0.18, y + 0.28, WALK + 1.75), col=True); box('g_p_green', (x + 0.18, y - 0.28, WALK + 1.35), (x + 0.185, y + 0.28, WALK + 1.75))
        box('g_lcd', (x + 0.18, y - 0.16, WALK + 1.0), (x + 0.186, y + 0.16, WALK + 1.28))
        tube(BLACK, [(x + 0.18, y + 0.2, WALK + 0.9), (x + 0.4, y + 0.25, WALK + 0.7), (x + 0.42, y + 0.1, WALK + 0.3), (x + 0.25, y - 0.1, WALK + 0.45), (x + 0.19, y - 0.2, WALK + 0.85)], 0.018, 5)
    for i in range(3): box('g_line', (-3.9, -22.6 - 2.6 * i - 0.06, ASPH), (1.2, -22.6 - 2.6 * i + 0.06, ASPH + 0.05))
    box('g_p_green', (-3.8, -27.7, ASPH), (1.1, -22.7, ASPH + 0.03))
    # ---- along the front walk: ice, propane, papers, a bench, bins, planters
    ix, iy = BX1 + 0.05, -17.5                       # the ice freezer, against the mart's front
    box(WHITE, (ix, iy - 1.0, WALK), (ix + 0.85, iy + 1.0, WALK + 1.9), col=True); box('g_p_blue', (ix + 0.85, iy - 0.95, WALK + 1.35), (ix + 0.865, iy + 0.95, WALK + 1.85))
    text(WHITE, 'ICE', 0.36, (ix + 0.865, iy, WALK + 1.42), 'E', depth=0.015)
    for dy in (-0.5, 0.5): box('g_chrome', (ix + 0.85, iy + dy - 0.42, WALK + 0.2), (ix + 0.87, iy + dy + 0.42, WALK + 1.25)); box(BLACK, (ix + 0.87, iy + dy + 0.3, WALK + 0.7), (ix + 0.9, iy + dy + 0.34, WALK + 0.95))
    px, py = BX1 + 0.05, -7.2                        # propane cage
    box('g_steel', (px, py - 0.9, WALK), (px + 0.7, py + 0.9, WALK + 0.1), col=False)
    for k in range(13): box('g_steel', (px + 0.68, py - 0.9 + 0.15 * k, WALK + 0.1), (px + 0.7, py - 0.88 + 0.15 * k, WALK + 1.8))
    for z in (0.1, 0.95, 1.8): box('g_steel', (px, py - 0.9, WALK + z - 0.02), (px + 0.7, py + 0.9, WALK + z + 0.02))
    for lvl in range(2):
        for k in range(4):
            cy = py - 0.66 + 0.44 * k; cz = WALK + 0.12 + 0.85 * lvl
            cyl(WHITE, (px + 0.34, cy, cz), (px + 0.34, cy, cz + 0.42), 0.16, 10); ball(WHITE, (px + 0.34, cy, cz + 0.42), 0.16, sz=0.6, seg=10, rings=4); cyl('g_steel', (px + 0.34, cy, cz + 0.5), (px + 0.34, cy, cz + 0.58), 0.1, 8, caps=False)
    collide((px, py - 0.9, WALK), (px + 0.72, py + 0.9, WALK + 1.82))
    text(RED, 'PROPANE', 0.16, (px + 0.705, py, WALK + 1.84), 'E', depth=0.006)
    for k, g in enumerate(('g_p_blue', 'g_p_red', YELLOW)):        # newspaper boxes
        y = -3.4 + 0.62 * k
        box(g, (BX1 + 0.1, y - 0.26, WALK + 0.25), (BX1 + 0.58, y + 0.26, WALK + 1.1), col=True); cyl('g_steel', (BX1 + 0.34, y, WALK), (BX1 + 0.34, y, WALK + 0.25), 0.04, 6)
        quad2(GLASS, (BX1 + 0.585, y - 0.2, WALK + 0.6), (BX1 + 0.585, y + 0.2, WALK + 0.6), (BX1 + 0.585, y + 0.2, WALK + 1.02), (BX1 + 0.585, y - 0.2, WALK + 1.02))
    for y in (4.2, 13.6):                            # benches under the diner's windows
        box('g_wood', (BX1 + 0.15, y - 0.9, WALK + 0.42), (BX1 + 0.6, y + 0.9, WALK + 0.47)); box('g_wood', (BX1 + 0.1, y - 0.9, WALK + 0.55), (BX1 + 0.15, y + 0.9, WALK + 0.9))
        for dy in (-0.75, 0.75): box('g_steel', (BX1 + 0.12, y + dy - 0.03, WALK), (BX1 + 0.58, y + dy + 0.03, WALK + 0.42))
        collide((BX1 + 0.1, y - 0.9, WALK), (BX1 + 0.6, y + 0.9, WALK + 0.9))
    bin_(BX1 + 0.5, -14.6, WALK); bin_(BX1 + 0.5, 1.9, WALK); bin_(BX1 + 0.5, 21.0, WALK)
    planter_box(BX1 + 0.1, 7.0, BX1 + 0.75, 10.6, WALK); planter_box(BX1 + 0.1, 16.3, BX1 + 0.75, 19.9, WALK)
    planter_box(-5.2, -21.6, -4.3, -20.4, WALK, 0.5); planter_box(-5.2, 20.6, -4.3, 21.8, WALK, 0.5)
    # a rack with windscreen fluid and firewood bundles
    box('g_steel', (BX1 + 0.1, -9.9, WALK), (BX1 + 0.6, -8.5, WALK + 0.06), col=False)
    for k in range(8): box('g_p_blue', (BX1 + 0.14 + 0.24 * (k % 2), -9.85 + 0.34 * (k // 2), WALK + 0.06), (BX1 + 0.34 + 0.24 * (k % 2), -9.55 + 0.34 * (k // 2), WALK + 0.4)); cyl(WHITE, (BX1 + 0.24 + 0.24 * (k % 2), -9.7 + 0.34 * (k // 2), WALK + 0.4), (BX1 + 0.24 + 0.24 * (k % 2), -9.7 + 0.34 * (k // 2), WALK + 0.46), 0.03, 6)
    collide((BX1 + 0.1, -9.9, WALK), (BX1 + 0.6, -8.5, WALK + 0.46))
    # ---- the picnic terrace north of the diner
    for (x, y, a, g) in ((-22.0, 27.5, 0.1, RED), (-16.0, 30.5, -0.2, 'g_p_blue'), (-10.5, 27.0, 0.3, RED)):
        picnic_table(x, y, a); parasol(x + 1.4 * math.cos(a + 1.2), y + 1.4 * math.sin(a + 1.2), g)
    planter_box(PX0 + 0.3, 24.0, PX0 + 1.0, 35.0, WALK, 0.6); planter_box(-26.5, 34.9, -8.0, 35.6, WALK, 0.6)
    bin_(-7.0, 25.0, WALK)
    # a bike rack
    for k in range(4):
        x = -7.0; y = 29.0 + 0.6 * k
        tube('g_chrome', [(x - 0.35, y, WALK), (x - 0.35, y, WALK + 0.8), (x + 0.35, y, WALK + 0.8), (x + 0.35, y, WALK)], 0.02, 6)
    # ---- the bin store at the south-west: a slatted fence, two skips, crates
    sx0, sx1, sy0, sy1 = -27.0, -19.5, -35.0, -27.0
    for (a, b) in (((sx0, sy0), (sx1, sy0 + 0.08)), ((sx0, sy0), (sx0 + 0.08, sy1)), ((sx0, sy1 - 0.08), (sx0 + 3.0, sy1))):
        collide((a[0], a[1], WALK), (b[0], b[1], WALK + 2.0))
        horiz = (b[0] - a[0]) > (b[1] - a[1])
        n = int(((b[0] - a[0]) if horiz else (b[1] - a[1])) / 0.16)
        for k in range(n):
            if horiz: box('g_wood', (a[0] + 0.16 * k, a[1], WALK + 0.05), (a[0] + 0.16 * k + 0.13, b[1], WALK + 2.0))
            else: box('g_wood', (a[0], a[1] + 0.16 * k, WALK + 0.05), (b[0], a[1] + 0.16 * k + 0.13, WALK + 2.0))
    for k, g in enumerate(('g_p_green', 'g_p_blue')):
        x = sx0 + 1.4 + 2.6 * k
        box(g, (x - 1.0, sy0 + 0.6, WALK + 0.15), (x + 1.0, sy0 + 2.0, WALK + 1.25), col=True); box(BLACK, (x - 1.05, sy0 + 0.55, WALK + 1.25), (x + 1.05, sy0 + 2.05, WALK + 1.32))
        for (dx, dy) in ((-0.8, 0.8), (0.8, 0.8), (-0.8, 1.8), (0.8, 1.8)): cyl(BLACK, (x + dx, sy0 + dy - 0.05, WALK + 0.08), (x + dx, sy0 + dy + 0.05, WALK + 0.08), 0.08, 8)
    for k in range(7):                               # crates and a stack of pallets
        box(PRODUCTS[k % 4 + 1], (sx0 + 0.5 + 0.5 * (k % 3), sy1 - 1.6, WALK + 0.3 * (k // 3)), (sx0 + 0.95 + 0.5 * (k % 3), sy1 - 1.2, WALK + 0.28 + 0.3 * (k // 3)))
    for k in range(5):
        z = WALK + 0.15 * k
        for j in range(5): box('g_wood', (sx1 - 2.2, sy0 + 3.2 + 0.24 * j, z + 0.1), (sx1 - 1.0, sy0 + 3.34 + 0.24 * j, z + 0.125))
        for j in range(3): box('g_wood', (sx1 - 2.2 + 0.55 * j, sy0 + 3.2, z), (sx1 - 2.1 + 0.55 * j, sy0 + 4.3, z + 0.1))
    collide((sx1 - 2.2, sy0 + 3.2, WALK), (sx1 - 1.0, sy0 + 4.3, WALK + 0.75))
    # ---- on the back wall: meters, a gas pipe, a ladder to the roof, two gas bottles
    box('g_steel', (BX0 - 0.22, -2.0, 1.0), (BX0, -1.0, 2.2)); box('g_steel', (BX0 - 0.16, -0.6, 1.2), (BX0, 0.0, 1.9))
    tube(YELLOW, [(BX0 - 0.1, 2.0, WALK), (BX0 - 0.1, 2.0, 3.0), (BX0 - 0.1, 14.0, 3.0), (BX0 - 0.1, 14.0, 4.0)], 0.03, 6)
    for dy in (-0.25, 0.25): cyl('g_steel', (BX0 - 0.3, -16.0 + dy, WALK), (BX0 - 0.3, -16.0 + dy, PARA + 0.9), 0.02, 5)
    for k in range(17): cyl('g_steel', (BX0 - 0.3, -16.25, WALK + 0.35 + 0.33 * k), (BX0 - 0.3, -15.75, WALK + 0.35 + 0.33 * k), 0.015, 5)
    for dy in (0.0, 0.5): cyl('g_p_orange', (BX0 - 0.35, 4.0 + dy, WALK), (BX0 - 0.35, 4.0 + dy, WALK + 1.3), 0.17, 10); ball('g_p_orange', (BX0 - 0.35, 4.0 + dy, WALK + 1.3), 0.17, sz=0.5, seg=10, rings=4)
    collide((BX0 - 0.55, 3.8, WALK), (BX0, 4.75, WALK + 1.4))
    # ---- road signs at the apron: the way in, the way out
    for (y, lab) in ((11.5, 'IN'), (-11.5, 'OUT')):
        x = 24.8
        cyl('g_steel', (x, y, ASPH), (x, y, ASPH + 2.4), 0.04, 6); box('g_p_blue', (x - 0.04, y - 0.45, ASPH + 1.8), (x + 0.04, y + 0.45, ASPH + 2.4))
        text(WHITE, lab, 0.3, (x + 0.042, y, ASPH + 1.96), 'E', depth=0.01); text(WHITE, lab, 0.3, (x - 0.042, y, ASPH + 1.96), 'W', depth=0.01)
        collide((x - 0.05, y - 0.05, ASPH), (x + 0.05, y + 0.05, ASPH + 2.4))



# ------------------------------------------------------------------ for those who live here
def nav():
    E = BX1 - 0.3
    def node(name, x, y, z=FL): NODES.append((name, x, y, z))
    def link(*names):
        for a, b in zip(names[:-1], names[1:]): LINKS.append((a, b))
    def spot(kind, x, y, z, fx, fy): SPOTS.append((kind, x, y, z, fx, fy))
    # ---- seats: the booths (two a bench), the benches outside, the picnic tables
    for k in range(7):
        yc = 3.6 + 2.5 * k
        for sx in (E - 0.65, E - 1.7):
            SEATS.append(('booth', sx, yc - 0.74, FL + 0.46, 0.0, 1.0)); SEATS.append(('booth', sx, yc + 0.74, FL + 0.46, 0.0, -1.0))
    for y in (4.2, 13.6):
        for dy in (-0.5, 0.5): SEATS.append(('bench', BX1 + 0.4, y + dy, WALK + 0.47, 1.0, 0.0))
    for (x, y, a) in ((-22.0, 27.5, 0.1), (-16.0, 30.5, -0.2), (-10.5, 27.0, 0.3)):
        c, s = math.cos(a), math.sin(a)
        for side in (-1.0, 1.0):
            for along in (-0.45, 0.45):
                px, py = x + along * c - side * 0.62 * s, y + along * s + side * 0.62 * c
                SEATS.append(('picnic', px, py, WALK + 0.46, side * s, -side * c))
    # ---- the walk outside: along the front, to the terrace, across the forecourt at the diner's door
    for (nm, y) in (('w_s', -21.0), ('w_mart', -12.0), ('w_mid', -5.0), ('w_diner', 0.1), ('w_n', 12.0), ('w_ne', 23.0)): node(nm, -6.2, y, WALK)
    link('w_s', 'w_mart', 'w_mid', 'w_diner', 'w_n', 'w_ne')
    for (nm, x, y) in (('t_a', -10.5, 24.6), ('t_b', -16.0, 27.6), ('t_c', -22.0, 24.8), ('t_d', -13.0, 31.5)): node(nm, x, y, WALK)
    link('w_ne', 't_a', 't_b', 't_c'); link('t_b', 't_d'); link('t_a', 't_d')
    node('gap_w', -4.5, -0.4, WALK); node('gap_e', 2.2, -0.4, ASPH); link('w_diner', 'gap_w', 'gap_e')
    node('fc_n', 2.4, 8.0, ASPH); node('fc_s', 2.4, -8.0, ASPH); link('fc_n', 'gap_e', 'fc_s')
    for (nm, x) in (('l5', 5.6), ('l13', 13.0), ('l20', 20.6)):
        node(nm + '_n', x, 8.0, ASPH); node(nm + '_s', x, -8.0, ASPH); node(nm + '_c', x, 0.0, ASPH)
        link(nm + '_n', nm + '_c', nm + '_s')
    link('fc_n', 'l5_n', 'l13_n', 'l20_n'); link('fc_s', 'l5_s', 'l13_s', 'l20_s')
    node('air', 9.0, -28.2, ASPH); link('l13_s', 'air'); node('smoke', -6.6, -23.5, WALK); link('w_s', 'smoke')
    # where a driver stands to fill up, by the car stop of the same name
    for (nm, x, y, fx) in (('pump13a', 10.0, 2.4, -1.0), ('pump13b', 10.0, -2.4, -1.0), ('pump20a', 18.0, 2.4, -1.0), ('pump20b', 18.0, -2.4, -1.0)):
        node(nm, x, y + 0.9, ASPH); link(nm, nm[4:6].join(['l', '_c']))
        spot(nm, x, y + 0.9, ASPH, fx, -0.4)
    # ---- the mart
    node('m_door', -8.15, -12.0); node('m_in', -8.95, -12.0); node('m_x', -11.7, -11.0); link('w_mart', 'm_door', 'm_in', 'm_x')
    for (i, y) in enumerate((-20.0, -16.6, -13.0, -9.6)):
        node('a%dw' % i, -23.3, y); node('a%de' % i, -12.3, y); link('a%dw' % i, 'a%de' % i)
        for k in range(4):
            x = -21.0 + 2.4 * k + RNG.uniform(-0.3, 0.3)
            if i > 0: spot('browse', x, y + 0.55, FL, 0.0, 1.0)
            if i < 3: spot('browse', x + 0.9, y - 0.55, FL, 0.0, -1.0)
    link('a0w', 'a1w', 'a2w', 'a3w'); link('a0e', 'a1e', 'a2e', 'a3e'); link('m_x', 'a2e'); link('m_x', 'a3e')
    for k in range(5): spot('browse', -24.6, -20.4 + 2.05 * k, FL, -1.0, 0.0)             # at the fridges
    spot('browse', -19.9, -20.25, FL, 0.0, -1.0); spot('browse', -17.0, -20.25, FL, 0.0, -1.0)   # the coffee bar
    node('m_n', -17.0, -4.6); link('a3w', 'm_n'); link('a3e', 'm_n')
    spot('browse', -18.5, -3.3, FL, 0.0, 1.0); spot('browse', -16.2, -3.3, FL, 0.0, 1.0)          # the chiller
    node('till_c', -12.85, -7.5); link('a3e', 'till_c'); spot('pay', -12.85, -7.5, FL, 1.0, 0.25)
    spot('cashier', -10.5, -5.7, FL, -1.0, 0.0)
    node('pd_s', -13.4, -3.0); node('pd_n', -13.4, -1.0); link('m_n', 'pd_s', 'pd_n')
    # ---- the diner
    node('d_door', -8.15, 0.1); node('d_in', -9.5, 0.1); link('w_diner', 'd_door', 'd_in')
    node('d_a', -12.0, 1.1); link('d_in', 'd_a'); link('pd_n', 'd_a')
    prev = 'd_a'
    for k in range(7):
        nm = 'b%d' % k; node(nm, -11.5, 3.6 + 2.5 * k); link(prev, nm); prev = nm
    node('d_w', -16.9, 1.6); link('pd_n', 'd_w'); link('d_a', 'd_w')
    prev = 'd_w'
    for k in range(5):
        nm = 'c%d' % k; node(nm, -17.1, 5.0 + 3.4 * k); link(prev, nm); prev = nm
    node('k_out', -20.45, 4.6); link('c0', 'k_out'); node('k_in', -22.7, 4.6); link('k_out', 'k_in')
    prev = 'k_out'
    for k in range(4):
        nm = 'bar%d' % k; node(nm, -20.42, 6.2 + 3.7 * k); link(prev, nm); prev = nm
        spot('serve', -20.42, 6.2 + 3.7 * k, FL, 1.0, 0.0)
    node('k_a', -23.6, 7.5); node('k_b', -24.25, 11.5); link('k_in', 'k_a', 'k_b')
    spot('cook', -24.25, 9.0, FL, -1.0, 0.0); spot('cook', -24.25, 11.5, FL, -1.0, 0.0); spot('cook', -24.25, 14.0, FL, -1.0, 0.0)
    spot('host', -10.6, 1.4, FL, 1.0, -0.3)
    # ---- cars: the parking bays in front (nose in), the stops at the pumps, the ways in and out
    for i in range(16):
        y = -18.7 + 2.6 * i
        if -2.9 < y - 1.3 < 0.8 or y > 20.5 or i == 0: continue
        BAYS.append((-1.25, y, ASPH, -1.0, 0.0)); node('bay%d' % len(BAYS), 2.0, y, ASPH)
    for i in range(1, len(BAYS) + 1):
        by = [n for n in NODES if n[0] == 'bay%d' % i][0][2]
        link('bay%d' % i, 'fc_n' if by > 0 else 'fc_s')
    # (the ground east of the plot as the engine makes it - it lies BELOW the terrace south of the
    #  mouth: clamped at the terrace's level the cars drove out through the air - and the mouth's
    #  asphalt a hand above it, down to the road's own surface at the lane)
    def onroad(x, y):
        if x <= PX1: return (x, y, ASPH)
        d = road_x(y) - x
        return (x, y, ground_east(x, y) + (0.02 if d < ROAD_PAVED - 0.5 else 0.10))
    def way(name, pts): PATHS.append((name, [onroad(x, y) for (x, y) in pts]))
    rin = (road_x(16.0) - 1.9, 16.0); rout = (road_x(-17.0) - 1.9, -17.0)                  # on the road's near lane (it runs south)
    mouth_in = [rin, (road_x(9.0) - 4.6, 9.0), (31.5, 6.2), (26.5, 5.4)]
    mouth_out = [(26.5, -5.2), (31.5, -6.0), (road_x(-10.0) - 4.6, -10.0), rout]
    for (nm, x, turn) in (('13', 13.0, [(23.0, 8.4), (19.0, 10.8), (16.4, 10.8), (14.2, 9.8), (13.05, 7.2)]), ('20', 20.5, [(24.2, 8.4), (22.6, 9.3), (21.1, 8.4), (20.55, 6.4)])):
        way('in' + nm, mouth_in + turn + [(x, 2.4)])
        way('mid' + nm, [(x, 2.4), (x, -2.4)])
        back = [(x, -7.0), (14.6, -10.2), (18.0, -11.2), (22.6, -8.6)] if nm == '13' else [(x, -6.6), (21.3, -8.8), (23.4, -8.4)]
        way('out' + nm, [(x, -2.4)] + back + mouth_out)
        STOPS.append(('pump' + nm + 'a', x, 2.4, ASPH, 0.0, -1.0)); STOPS.append(('pump' + nm + 'b', x, -2.4, ASPH, 0.0, -1.0))
    way('inpark', mouth_in + [(23.0, 9.6), (19.0, 14.6), (8.5, 15.6), (4.6, 14.6), (3.3, 11.5)])
    way('aisle', [(3.3, 11.5), (3.3, -11.5)])
    way('outpark', [(3.3, -11.5), (4.2, -14.2), (7.0, -15.4), (18.0, -14.2), (23.2, -9.6)] + mouth_out)


ground()
canopy()
shell()
diner()
mart()
site()
nav()

tris = sum(len(v) // 3 for v in G.values())
with open(os.path.join(OUT, 'gasstation.mesh'), 'w') as f:
    for g, v in G.items():
        if not v: continue
        f.write('mesh %s %d\n' % (g, len(v) // 3))
        for p, n, uv in v:
            f.write('%.4f %.4f %.4f %.3f %.3f %.3f %.4f %.4f\n' % (p.x, p.z, -p.y, n.x, n.z, -n.y, uv[0], uv[1]))
with open(os.path.join(OUT, 'gasstation.col'), 'w') as f:
    f.write('# gasstation.mesh: colliders, lights, shelters, no-grass rectangles - game space (x east, y up, z south), from its origin\n')
    for lo, hi in COL:
        f.write('box %.3f %.3f %.3f %.3f %.3f %.3f\n' % (lo[0], lo[2], -hi[1], hi[0], hi[2], -lo[1]))
    for p in LIGHTS:
        f.write('light %.3f %.3f %.3f\n' % (p[0], p[2], -p[1]))
    for (x0, y0, x1, y1, top) in SHELTERS:
        f.write('shelter %.3f %.3f %.3f %.3f %.3f\n' % (x0, -y1, x1, -y0, top))
    for (x0, y0, x1, y1) in NOGRASS:
        f.write('nograss %.3f %.3f %.3f %.3f\n' % (x0, -y1, x1, -y0))
    for (x0, y0, x1, y1, h0, h1) in PADS:
        f.write('pad %.3f %.3f %.3f %.3f %.3f %.3f\n' % (x0, -y1, x1, -y0, h0, h1))
def yaw(fx, fy): return math.atan2(fx, fy)        # the game's yaw of a way of facing (0 = north, toward -z)
with open(os.path.join(OUT, 'gasstation.nav'), 'w') as f:
    f.write('# gasstation: where people sit, stand and walk and where cars park, stop and drive - game space (x east, y up, z south), from its origin\n')
    for (k, x, y, z, fx, fy) in SEATS: f.write('seat %s %.3f %.3f %.3f %.4f\n' % (k, x, z, -y, yaw(fx, fy)))
    for (nm, x, y, z) in NODES: f.write('node %s %.3f %.3f %.3f\n' % (nm, x, z, -y))
    for (a, b) in LINKS: f.write('link %s %s\n' % (a, b))
    for (k, x, y, z, fx, fy) in SPOTS: f.write('spot %s %.3f %.3f %.3f %.4f\n' % (k, x, z, -y, yaw(fx, fy)))
    for (x, y, z, fx, fy) in BAYS: f.write('bay %.3f %.3f %.3f %.4f\n' % (x, z, -y, yaw(fx, fy)))
    for (nm, x, y, z, fx, fy) in STOPS: f.write('stop %s %.3f %.3f %.3f %.4f\n' % (nm, x, z, -y, yaw(fx, fy)))
    for (nm, pts) in PATHS:
        for (x, y, z) in pts: f.write('path %s %.3f %.3f %.3f\n' % (nm, x, z, -y))

col = bpy.data.collections.new('gasstation'); bpy.context.scene.collection.children.link(col)
for g, v in G.items():
    if not v: continue
    me = bpy.data.meshes.new('gs_' + g)
    me.from_pydata([tuple(p) for p, _, _ in v], [], [(i, i + 1, i + 2) for i in range(0, len(v), 3)])
    me.update()
    col.objects.link(bpy.data.objects.new('gs_' + g, me))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'gasstation.blend'))
print('GASSTATION', tris, 'triangles;', {g: len(v) // 3 for g, v in G.items()}, 'colliders', len(COL), 'lights', len(LIGHTS))
