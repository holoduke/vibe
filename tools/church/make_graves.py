# Old gravestones for the chapel yard, built in Blender.
#   blender -b --python tools/church/make_graves.py
# Output: assets/church/graves.mesh (same text format as church.mesh: one
# block per stone, game axes, origin at the ground under the stone, +x =
# the stone's face width, the carved face looks toward -z in game space).
#
# Every stone is a real solid: a silhouette extruded to a slab, subdivided,
# its carving (cross, border groove, lines of worn lettering) cut in with
# booleans, corners knocked off with booleans, then weathered by two scales
# of displacement so no edge is straight any more.
import bpy, bmesh, math, os, random
from mathutils import Vector, noise

bpy.ops.wm.read_factory_settings(use_empty=True)
random.seed(7)


def obj_from(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([Vector(v) for v in verts], [], faces)
    me.update()
    bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def slab(name, outline, thick, y0=None):
    """Extrude a CCW outline [(x, z)] (front view) to a slab `thick` deep (along y)."""
    n = len(outline)
    y0 = -thick * 0.5 if y0 is None else y0
    v = [(x, y0, z) for x, z in outline] + [(x, y0 + thick, z) for x, z in outline]
    f = [tuple(range(n)), tuple(reversed(range(n, 2 * n)))] + [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    return obj_from(name, v, f)


def box(name, x0, x1, y0, y1, z0, z1):
    return slab(name, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)], y1 - y0, y0)


def apply_mods(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear(); ob.data = me


def boolean(ob, cutter, op='DIFFERENCE'):
    m = ob.modifiers.new('b', 'BOOLEAN'); m.operation = op; m.object = cutter; m.solver = 'EXACT'
    apply_mods(ob); bpy.data.objects.remove(cutter)


def ico(name, centre, r, squash=(1, 1, 1)):
    bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r)
    for v in bm.verts:
        v.co = Vector((v.co.x * squash[0] + centre[0], v.co.y * squash[1] + centre[1], v.co.z * squash[2] + centre[2]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    return ob


def chip(ob, pts, r=(0.05, 0.11)):
    """Knock pieces out at the given points (corners, edges)."""
    for p in pts:
        rr = random.uniform(*r)
        boolean(ob, ico('chip', (p[0] + random.uniform(-0.02, 0.02), p[1], p[2] + random.uniform(-0.02, 0.02)), rr,
                        (1.0, random.uniform(1.2, 2.0), random.uniform(0.8, 1.3))))


def weather(ob, seed, amount=1.0, max_edge=0.055):
    """Remesh to an even surface and erode it: big soft dents + fine pitting."""
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    for _ in range(6):
        long_edges = [e for e in bm.edges if e.calc_length() > max_edge]
        if not long_edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1)
        bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.normal_update()
    off = Vector((seed * 13.7, seed * 5.1, seed * 9.3))
    for v in bm.verts:
        p = v.co
        ground = min(1.0, max(0.0, (p.z + 0.02) / 0.10))                 # keep the buried foot plain
        big = noise.noise(p * 2.3 + off) * 0.022
        mid = noise.noise(p * 9.0 + off * 2.0) * 0.007
        pit = max(0.0, noise.noise(p * 34.0 + off * 3.0) - 0.25) * -0.012   # pits only go inward
        # Tops and upper edges are eaten away more (rain).
        top = 1.0 + 0.8 * min(1.0, max(0.0, v.normal.z))
        v.co = p + v.normal * ((big + mid) * top + pit) * amount * ground
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data); bm.free()


def arch_outline(w, h, kind):
    """Front silhouette, CCW from bottom-left; (x, z)."""
    hw = w * 0.5
    pts = [(-hw, -0.35), (hw, -0.35)]
    if kind == 'round':
        for k in range(0, 17):
            a = math.pi * k / 16.0
            pts.append((hw * math.cos(a), h - hw + hw * math.sin(a)))
    elif kind == 'pointed':
        pts.append((hw, h * 0.62))
        for k in range(1, 9):
            t = k / 8.0
            pts.append((hw * (1 - t) * (1 + 0.25 * (1 - t) * t * 4 * 0.3), h * 0.62 + (h * 0.38) * (t ** 0.8)))
        for k in range(7, 0, -1):
            t = k / 8.0
            pts.append((-hw * (1 - t) * (1 + 0.25 * (1 - t) * t * 4 * 0.3), h * 0.62 + (h * 0.38) * (t ** 0.8)))
        pts.append((-hw, h * 0.62))
    elif kind == 'shoulder':
        s = hw * 0.62
        pts += [(hw, h * 0.74), (s, h * 0.74)]
        for k in range(0, 13):
            a = math.pi * k / 12.0
            pts.append((s * math.cos(a), h - s + s * math.sin(a)))
        pts += [(-s, h * 0.74), (-hw, h * 0.74)]
    else:   # square
        pts += [(hw, h), (-hw, h)]
    return pts


def carve_cross(ob, cx, cz, size, face_y, depth=0.016):
    boolean(ob, box('c', cx - size * 0.085, cx + size * 0.085, face_y - depth, face_y + 0.05, cz - size * 0.55, cz + size * 0.45))
    boolean(ob, box('c', cx - size * 0.32, cx + size * 0.32, face_y - depth, face_y + 0.05, cz + size * 0.08, cz + size * 0.25))


def carve_lines(ob, w, z_top, n, face_y, depth=0.008):
    """Worn lines of lettering: short grooves of uneven length."""
    for i in range(n):
        z = z_top - i * 0.075
        x = -w * 0.34
        while x < w * 0.30:
            ln = random.uniform(0.03, 0.10)
            if random.random() > 0.25:                       # some letters are worn away entirely
                boolean(ob, box('l', x, min(x + ln, w * 0.34), face_y - depth, face_y + 0.05, z, z + 0.028))
            x += ln + random.uniform(0.012, 0.03)


stones = []

# 1. Round-top headstone, cross + lettering, chipped shoulder.
o = slab('round', arch_outline(0.62, 1.05, 'round'), 0.13)
carve_cross(o, 0.0, 0.78, 0.26, -0.065)
carve_lines(o, 0.62, 0.52, 4, -0.065)
chip(o, [(0.28, 0.0, 0.92), (-0.31, 0.0, 0.35)])
weather(o, 1); stones.append(o)

# 2. Pointed (gothic) headstone, tall and narrow.
o = slab('pointed', arch_outline(0.50, 1.30, 'pointed'), 0.12)
carve_cross(o, 0.0, 0.86, 0.22, -0.06)
carve_lines(o, 0.50, 0.60, 5, -0.06)
chip(o, [(0.0, 0.0, 1.30), (0.25, 0.0, 0.55)], (0.04, 0.07))
weather(o, 2); stones.append(o)

# 3. Shouldered headstone, wide, with a border groove.
o = slab('shoulder', arch_outline(0.74, 0.92, 'shoulder'), 0.15)
boolean(o, box('g', -0.30, 0.30, -0.075 - 0.01, -0.075 + 0.012, 0.10, 0.125))
boolean(o, box('g', -0.30, 0.30, -0.075 - 0.01, -0.075 + 0.012, 0.60, 0.625))
carve_lines(o, 0.70, 0.50, 4, -0.075)
chip(o, [(0.37, 0.0, 0.70), (-0.36, 0.0, 0.10), (0.05, 0.0, 0.92)])
weather(o, 3); stones.append(o)

# 4. Plain Latin cross on a stepped base.
o = box('cross', -0.11, 0.11, -0.09, 0.09, -0.35, 1.45)
boolean(o, box('arm', -0.42, 0.42, -0.085, 0.085, 0.95, 1.16), 'UNION')
boolean(o, box('base', -0.36, 0.36, -0.24, 0.24, -0.35, 0.16), 'UNION')
boolean(o, box('base2', -0.25, 0.25, -0.17, 0.17, 0.16, 0.30), 'UNION')
chip(o, [(0.42, 0.0, 1.16), (-0.36, -0.2, 0.16), (0.0, 0.0, 1.45)], (0.05, 0.08))
weather(o, 4); stones.append(o)

# 5. Celtic cross: cross with a ring, on a tapered shaft.
o = slab('celtic', [(-0.17, -0.35), (0.17, -0.35), (0.12, 1.02), (-0.12, 1.02)], 0.14)
boolean(o, box('up', -0.10, 0.10, -0.065, 0.065, 1.0, 1.62), 'UNION')
boolean(o, box('arm', -0.34, 0.34, -0.065, 0.065, 1.18, 1.36), 'UNION')
ring_out = [(0.27 * math.cos(2 * math.pi * k / 28), 1.27 + 0.27 * math.sin(2 * math.pi * k / 28)) for k in range(28)]
ring = slab('ring', ring_out, 0.09)
hole = slab('hole', [(0.185 * math.cos(2 * math.pi * k / 28), 1.27 + 0.185 * math.sin(2 * math.pi * k / 28)) for k in range(28)], 0.3)
boolean(ring, hole)
boolean(o, ring, 'UNION')
chip(o, [(0.34, 0.0, 1.30), (0.0, 0.0, 1.62)], (0.04, 0.07))
weather(o, 5, 0.8, 0.04); stones.append(o)

# 6. Broken headstone: the stump (snapped off on a slant) ...
full = slab('stump', arch_outline(0.66, 1.10, 'round'), 0.14)
carve_lines(full, 0.66, 0.40, 3, -0.07)
cut = slab('cut', [(-0.6, 0.48), (0.6, 0.66), (0.6, 2.0), (-0.6, 2.0)], 0.6)
boolean(full, cut)
chip(full, [(0.20, 0.0, 0.60), (-0.15, 0.0, 0.50), (-0.33, 0.0, 0.20)], (0.04, 0.08))
weather(full, 6); stones.append(full)

# 7. ... and its top, lying in the grass in front of it.
top = slab('fallen', arch_outline(0.66, 1.10, 'round'), 0.14)
carve_cross(top, 0.0, 0.84, 0.24, -0.07)
keep = slab('keep', [(-0.6, 0.48), (0.6, 0.66), (0.6, 2.0), (-0.6, 2.0)], 0.6)
boolean(top, keep, 'INTERSECT')
chip(top, [(0.25, 0.0, 0.70), (-0.30, 0.0, 0.62)], (0.04, 0.08))
weather(top, 7)
# Lay it flat on its back, slightly tilted, half sunk.
for v in top.data.vertices:
    p = v.co
    v.co = Vector((p.x, (p.z - 0.80), -p.y + 0.02 + 0.10 * (p.z - 0.8) * 0.35))
bm = bmesh.new(); bm.from_mesh(top.data); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(top.data); bm.free()
stones.append(top)

# 8. Ledger slab: a long flat stone on the grave, cracked across and sunk at one end.
o = box('ledger', -0.42, 0.42, -0.95, 0.95, -0.10, 0.11)
boolean(o, box('gr', -0.36, 0.36, -0.88, 0.88, 0.098, 0.2))
boolean(o, box('in', -0.345, 0.345, -0.865, 0.865, 0.09, 0.2), 'UNION')
boolean(o, slab('crack', [(-0.6, 0.0), (0.6, 0.0), (0.6, 0.012), (-0.6, 0.012)], 0.4, -0.2))
for v in o.data.vertices:                # the crack runs across; rotate the cut look by sinking one end
    pass
chip(o, [(0.42, 0.95, 0.11), (-0.42, -0.30, 0.11)], (0.06, 0.10))
weather(o, 8, 0.9, 0.09)
for v in o.data.vertices:
    v.co.z -= 0.06 * max(0.0, v.co.y) + 0.015 * v.co.x
stones.append(o)

# 9. Small pillar with a pyramid cap.
o = slab('pillar', [(-0.19, -0.35), (0.19, -0.35), (0.16, 1.05), (-0.16, 1.05)], 0.32)
boolean(o, box('plinth', -0.27, 0.27, -0.27, 0.27, -0.35, 0.14), 'UNION')
cap = obj_from('cap', [(-0.21, -0.21, 1.05), (0.21, -0.21, 1.05), (0.21, 0.21, 1.05), (-0.21, 0.21, 1.05),
                       (-0.21, -0.21, 1.13), (0.21, -0.21, 1.13), (0.21, 0.21, 1.13), (-0.21, 0.21, 1.13), (0.0, 0.0, 1.36)],
               [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 8), (5, 6, 8), (6, 7, 8), (7, 4, 8)])
boolean(o, cap, 'UNION')
carve_lines(o, 0.30, 0.80, 5, -0.16)
chip(o, [(0.21, -0.2, 1.13), (0.0, 0.0, 1.36), (-0.27, 0.0, 0.14)], (0.04, 0.07))
weather(o, 9, 0.9); stones.append(o)

# 10. Low, squat, very worn marker: barely a stone any more.
o = slab('stub', arch_outline(0.48, 0.46, 'round'), 0.16)
chip(o, [(0.20, 0.0, 0.40), (-0.22, 0.0, 0.28), (0.0, 0.0, 0.46)], (0.07, 0.12))
weather(o, 10, 1.6); stones.append(o)


def export(ob, f):
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.normal_update()
    f.write('mesh %s %d\n' % (ob.name, len(bm.faces)))
    for face in bm.faces:
        n = face.normal
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        for v in face.verts:
            p = v.co
            vn = v.normal if v.normal.dot(n) > 0.5 else n            # smooth except across real edges
            # Box projection, metres.
            if ay >= ax and ay >= az:   u_m, v_m = p.x, -p.z
            elif ax >= az:              u_m, v_m = p.y, -p.z
            else:                       u_m, v_m = p.x, p.y
            f.write('%.4f %.4f %.4f %.3f %.3f %.3f %.4f %.4f\n' % (p.x, p.z, -p.y, vn.x, vn.z, -vn.y, u_m, v_m))
    bm.free()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'church')
with open(os.path.join(out_dir, 'graves.mesh'), 'w') as f:
    for ob in stones:
        export(ob, f)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, 'graves.blend'))
print('GRAVES OK', [(o.name, len(o.data.polygons)) for o in stones])
