# Builds the little mountain chapel in Blender and exports it for the game.
#   blender -b --python tools/church/make_church.py
# Output: assets/church/church.mesh  (text: one block per material)
#
# Shape (after a small stone chapel by a mountain lake): a rectangular nave
# under a steep shingle roof, a half-octagon apse with its own hipped roof, a
# bell turret on the ridge with a pointed spire and a cross, arched windows
# and an arched plank door. Walls are solids with the window / door recesses
# cut in by boolean modifiers and the edges bevelled.
#
# Blender space: X = length (door end at -X, apse at +X), Y = across, Z = up.
# Game space: x = X, y = Z, z = -Y. UVs are in METRES along the surface (the
# game divides by its tile size).
import bpy, bmesh, math, os
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)

L, W = 9.0, 6.0            # nave length, width (outside)
WALL_H = 3.7               # wall height above the plinth
PLINTH = 0.45              # plinth height (floor level)
BASE = -1.2                # the plinth goes this deep into the ground
T = 0.55                   # wall thickness
ROOF_RISE = 3.3            # ridge above the wall top
EAVE = 0.45                # roof overhang
APSE_R = 2.55              # apse radius (half octagon)
APSE_H = 3.2               # apse wall height above the plinth


def obj_from(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([Vector(v) for v in verts], [], faces)
    me.update()
    # Every part is one closed shell: make its faces point outward here
    # (never on the joined mesh, where overlapping shells confuse it).
    bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def box(name, x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return obj_from(name, v, f)


def prism(name, outline, z0, z1):
    """Vertical prism over a CCW outline [(x, y)]."""
    n = len(outline)
    v = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    f = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        f.append((i, j, n + j, n + i))
    return obj_from(name, v, f)


def arch_cutter(name, centre, width, height, depth, axis):
    """A round-arched opening as a solid, extruded `depth` along `axis` ('X' or 'Y'),
    centred on `centre` (centre of the sill)."""
    r = width * 0.5
    prof = [(-r, 0.0), (r, 0.0)]
    for k in range(0, 13):
        a = math.pi * k / 12.0
        prof.append((r * math.cos(a), height - r + r * math.sin(a)))
    n = len(prof)
    verts, faces = [], []
    for s in (-0.5, 0.5):
        for (u, z) in prof:
            if axis == 'X':
                verts.append((centre[0] + s * depth, centre[1] + u, centre[2] + z))
            else:
                verts.append((centre[0] + u, centre[1] + s * depth, centre[2] + z))
    faces.append(tuple(range(n)))
    faces.append(tuple(reversed(range(n, 2 * n))))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, n + i, n + j, j))
    ob = obj_from(name, verts, faces)
    bm = bmesh.new(); bm.from_mesh(ob.data); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
    return ob


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = me


def boolean(ob, cutter, op='DIFFERENCE'):
    m = ob.modifiers.new('bool', 'BOOLEAN')
    m.operation = op
    m.object = cutter
    m.solver = 'EXACT'
    apply_modifiers(ob)
    bpy.data.objects.remove(cutter)


def bevel(ob, width, angle_deg=40.0):
    m = ob.modifiers.new('bevel', 'BEVEL')
    m.width = width
    m.segments = 2
    m.limit_method = 'ANGLE'
    m.angle_limit = math.radians(angle_deg)
    apply_modifiers(ob)


def join(objs, name):
    bm = bmesh.new()
    for o in objs:
        bm.from_mesh(o.data)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    for o in objs:
        bpy.data.objects.remove(o)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


hx, hy = L * 0.5, W * 0.5
top = PLINTH + WALL_H

# ---------------------------------------------------------------- stone
def gable_wall(name, x):
    """End wall with the gable triangle, T thick, centred on x."""
    prof = [(-hy, PLINTH), (hy, PLINTH), (hy, top), (0.0, top + ROOF_RISE - 0.12), (-hy, top)]
    v = [(x - T * 0.5, y, z) for y, z in prof] + [(x + T * 0.5, y, z) for y, z in prof]
    n = len(prof)
    f = [tuple(range(n)), tuple(reversed(range(n, 2 * n)))] + [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    ob = obj_from(name, v, f)
    bm = bmesh.new(); bm.from_mesh(ob.data); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
    return ob

REC = 0.22   # how deep windows / the door sit in the wall
win_x = (-2.6, 0.0, 2.6)
stone = []
stone.append(box('plinth', -hx - 0.12, hx + 0.12, -hy - 0.12, hy + 0.12, BASE, PLINTH))
# Each wall is cut on its own (a clean solid), then everything is joined.
for sy, (y0, y1) in ((-1, (-hy, -hy + T)), (1, (hy - T, hy))):
    wall = box('wall', -hx, hx, y0, y1, PLINTH, top)
    for x in win_x:
        boolean(wall, arch_cutter('cut', (x, sy * (hy - REC * 0.5 + 0.02), PLINTH + 1.15), 0.80, 1.95, REC + 0.04, 'Y'))
    stone.append(wall)
gw = gable_wall('gable_w', -hx + T * 0.5)
boolean(gw, arch_cutter('cut', (-hx + REC * 0.5 - 0.02, 0.0, PLINTH), 1.35, 2.55, REC + 0.04, 'X'))
boolean(gw, arch_cutter('cut', (-hx + REC * 0.5 - 0.02, 0.0, top + 0.55), 0.70, 0.95, REC + 0.04, 'X'))
stone.append(gw)
stone.append(gable_wall('gable_e', hx - T * 0.5))
# Apse: half octagon on the east end.
ap = [(hx - 0.05, -APSE_R), (hx - 0.05, APSE_R)]
for k in range(1, 4):
    a = math.pi * 0.5 - math.pi * k / 4.0
    ap.insert(1 + (k - 1), (hx + APSE_R * math.cos(a), APSE_R * math.sin(a)))
ap = [(hx - 0.05, APSE_R)] + [(hx + APSE_R * math.cos(math.pi * 0.5 - math.pi * k / 4.0), APSE_R * math.sin(math.pi * 0.5 - math.pi * k / 4.0)) for k in range(1, 4)] + [(hx - 0.05, -APSE_R)]
ap_ccw = list(reversed(ap))
apse = prism('apse', ap_ccw, PLINTH, PLINTH + APSE_H)
boolean(apse, arch_cutter('cut', (hx + APSE_R - REC * 0.5 + 0.02, 0.0, PLINTH + 1.05), 0.70, 1.70, REC + 0.04, 'X'))
stone.append(apse)
grow = 0.12
ap_pl = [(x + (grow if x > hx else 0.0) * (1 if x > hx else 0), y * (1.0 + grow / APSE_R)) for x, y in ap_ccw]
stone.append(prism('apse_plinth', ap_pl, BASE, PLINTH))
# Buttresses at the four nave corners.
for sx in (-1, 1):
    for sy in (-1, 1):
        stone.append(box('buttress', sx * hx - 0.30, sx * hx + 0.30, sy * hy - 0.30, sy * hy + 0.30, BASE, PLINTH + 2.6))
# Door step.
stone.append(box('step', -hx - 0.85, -hx - 0.10, -1.05, 1.05, BASE, PLINTH - 0.22))
stone.append(box('step2', -hx - 0.50, -hx - 0.10, -0.90, 0.90, BASE, PLINTH - 0.02))
walls = join(stone, 'stone')
bm = bmesh.new(); bm.from_mesh(walls.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5); bm.to_mesh(walls.data); bm.free()

bevel(walls, 0.035)

# ---------------------------------------------------------------- glass + door (sit in the recesses)
glass, wood, iron = [], [], []
for sy in (-1, 1):
    for x in win_x:
        o = arch_cutter('g', (x, sy * (hy - REC - 0.01), PLINTH + 1.15), 0.80, 1.95, 0.03, 'Y'); glass.append(o)
        # Stone-coloured mullion + transom are part of the iron set (dark lead).
        iron.append(box('m', x - 0.02, x + 0.02, sy * (hy - REC - 0.04), sy * (hy - REC + 0.02), PLINTH + 1.15, PLINTH + 3.05))
        iron.append(box('m', x - 0.40, x + 0.40, sy * (hy - REC - 0.04), sy * (hy - REC + 0.02), PLINTH + 2.05, PLINTH + 2.09))
glass.append(arch_cutter('g', (-hx + REC + 0.01, 0.0, top + 0.55), 0.70, 0.95, 0.03, 'X'))
glass.append(arch_cutter('g', (hx + APSE_R - REC - 0.01, 0.0, PLINTH + 1.05), 0.70, 1.70, 0.03, 'X'))
door = arch_cutter('door', (-hx + REC - 0.02, 0.0, PLINTH), 1.35, 2.55, 0.08, 'X'); wood.append(door)
# Plank grooves + iron straps and a ring handle on the door.
for k in range(-2, 3):
    iron.append(box('groove', -hx + REC - 0.075, -hx + REC - 0.060, k * 0.27 - 0.006, k * 0.27 + 0.006, PLINTH + 0.02, PLINTH + 2.0))
for z in (0.45, 1.25, 2.0):
    iron.append(box('strap', -hx + REC - 0.085, -hx + REC - 0.060, -0.62, 0.62, PLINTH + z - 0.035, PLINTH + z + 0.035))
iron.append(box('handle', -hx + REC - 0.12, -hx + REC - 0.06, 0.40, 0.46, PLINTH + 1.05, PLINTH + 1.20))

# ---------------------------------------------------------------- roof
def slab(name, a, b, c, d, thick):
    """A roof plane a-b-c-d (CCW seen from outside) with thickness."""
    n = (Vector(b) - Vector(a)).cross(Vector(d) - Vector(a)).normalized()
    lo = [tuple(Vector(p) - n * thick) for p in (a, b, c, d)]
    v = [a, b, c, d] + lo
    f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return obj_from(name, v, f)

roof = []
ridge_z = top + ROOF_RISE
ex = hx + EAVE * 0.7
slope = ROOF_RISE / hy
eave_y = hy + EAVE
eave_z = top - EAVE * slope
roof.append(slab('roof_s', (-ex, -eave_y, eave_z), (ex, -eave_y, eave_z), (ex, 0.0, ridge_z), (-ex, 0.0, ridge_z), 0.14))
roof.append(slab('roof_n', (ex, eave_y, eave_z), (-ex, eave_y, eave_z), (-ex, 0.0, ridge_z), (ex, 0.0, ridge_z), 0.14))
# Ridge cap.
roof.append(box('ridge', -ex, ex, -0.10, 0.10, ridge_z - 0.06, ridge_z + 0.07))
# Apse roof: a half cone of flat facets up to a point on the east gable.
apex = (hx + 0.05, 0.0, PLINTH + APSE_H + 2.5)
rim = [(x + (0.38 if x > hx else 0.0), y * (1.0 + 0.38 / APSE_R), PLINTH + APSE_H - 0.12) for x, y in ap]
nr = len(rim)
av = list(rim) + [apex, (hx - 0.3, 0.0, PLINTH + APSE_H - 0.12)]
af = [(i, i + 1, nr) for i in range(nr - 1)] + [(i + 1, i, nr + 1) for i in range(nr - 1)] + [(0, nr, nr + 1), (nr, nr - 1, nr + 1)]
roof.append(obj_from('apse_roof', av, af))
# Spire on the turret.
TX = -hx + 1.55            # turret centre along the ridge
TW = 0.62                  # half width
tz0 = ridge_z - 0.55
tz1 = ridge_z + 1.45       # belfry top
sp = ridge_z + 4.05        # spire tip
flare = TW + 0.26
sv = [(TX - flare, -flare, tz1), (TX + flare, -flare, tz1), (TX + flare, flare, tz1), (TX - flare, flare, tz1),
      (TX - TW * 0.55, -TW * 0.55, tz1 + 0.55), (TX + TW * 0.55, -TW * 0.55, tz1 + 0.55),
      (TX + TW * 0.55, TW * 0.55, tz1 + 0.55), (TX - TW * 0.55, TW * 0.55, tz1 + 0.55), (TX, 0.0, sp)]
sf = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 8), (5, 6, 8), (6, 7, 8), (7, 4, 8)]
roof.append(obj_from('spire', sv, sf))
roof_ob = join(roof, 'roof')

# ---------------------------------------------------------------- timber turret (posts, rails), cross
for sx in (-1, 1):
    for sy in (-1, 1):
        wood.append(box('post', TX + sx * TW - 0.07, TX + sx * TW + 0.07, sy * TW - 0.07, sy * TW + 0.07, tz0, tz1))
wood.append(box('turret_base', TX - TW - 0.05, TX + TW + 0.05, -TW - 0.05, TW + 0.05, tz0, tz0 + 0.95))
wood.append(box('turret_top', TX - TW - 0.05, TX + TW + 0.05, -TW - 0.05, TW + 0.05, tz1 - 0.14, tz1))
# Barge boards along the gable edges.
for sx in (-1, 1):
    x = sx * ex
    for sy in (-1, 1):
        wood.append(slab('barge', (x - 0.05, sy * eave_y, eave_z - 0.02), (x + 0.05, sy * eave_y, eave_z - 0.02),
                         (x + 0.05, 0.0, ridge_z - 0.02), (x - 0.05, 0.0, ridge_z - 0.02), 0.20))
# Bell + cross.
bell = []
for k in range(8):
    pass
iron.append(box('bell_yoke', TX - 0.30, TX + 0.30, -0.035, 0.035, tz1 - 0.30, tz1 - 0.22))
bv, bf = [], []
prof = [(0.05, 0.0), (0.12, -0.05), (0.15, -0.22), (0.21, -0.36), (0.24, -0.40)]
for i, (r, z) in enumerate(prof):
    for k in range(12):
        a = 2 * math.pi * k / 12
        bv.append((TX + r * math.cos(a), r * math.sin(a), tz1 - 0.26 + z))
for i in range(len(prof) - 1):
    for k in range(12):
        a0, a1 = i * 12 + k, i * 12 + (k + 1) % 12
        bf.append((a0, a1, a1 + 12, a0 + 12))
        bf.append((a0 + 12, a1 + 12, a1, a0))      # inside of the bell
iron.append(obj_from('bell', bv, bf))
iron.append(box('cross_v', TX - 0.035, TX + 0.035, -0.035, 0.035, sp - 0.15, sp + 1.05))
iron.append(box('cross_h', TX - 0.035, TX + 0.035, -0.30, 0.30, sp + 0.62, sp + 0.69))

wood_ob = join(wood, 'wood'); bevel(wood_ob, 0.012)
iron_ob = join(iron, 'iron')
glass_ob = join(glass, 'glass')
bevel(roof_ob, 0.02)

# ---------------------------------------------------------------- export
def export(ob, f):
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.normal_update()
    f.write('mesh %s %d\n' % (ob.name, len(bm.faces)))
    for face in bm.faces:
        n = face.normal
        # Tangent frame on the face: u runs horizontally, v up the surface.
        if abs(n.z) > 0.95:
            t, b = Vector((1, 0, 0)), Vector((0, 1, 0))
        else:
            t = Vector((0, 0, 1)).cross(n).normalized()
            b = n.cross(t).normalized()
        for v in face.verts:
            p = v.co
            # Smooth only across shallow angles: use the face normal, softened
            # toward the vertex normal where they nearly agree (bevels).
            vn = v.normal if v.normal.dot(n) > 0.75 else n
            vn = (n + vn).normalized()
            u_m, v_m = p.dot(t), -p.dot(b)
            f.write('%.4f %.4f %.4f %.3f %.3f %.3f %.4f %.4f\n' % (p.x, p.z, -p.y, vn.x, vn.z, -vn.y, u_m, v_m))
    bm.free()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'church')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'church.mesh'), 'w') as f:
    for ob in (walls, roof_ob, wood_ob, iron_ob, glass_ob):
        export(ob, f)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, 'church.blend'))
print('CHURCH OK', [(o.name, len(o.data.polygons)) for o in (walls, roof_ob, wood_ob, iron_ob, glass_ob)])
