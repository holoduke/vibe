# Height maps for the screen-space displacement (ssdm), made from the photo scans' own heights.
#
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --factory-startup --python tools/textures/make_ssdm_heights.py
#
# The scans' height maps (ambientCG "Displacement") have their joints at mid grey and a shallow,
# noisy relief: a displacement that moves the picture by them barely lifts the stones. The
# reference game's height maps are made for its displacement: joint 0, stone top 1, every stone a
# rounded body. This makes such a map for each scan, from the scan itself, so it matches the
# photographed colour and normal map exactly:
#   1. the scan's height, stretched (2nd..98th percentile -> 0..1);
#   2. measured against its surroundings (the scan drifts across the picture), so that the joints
#      are 0 and the stone tops 1 everywhere - the stones' own shapes and surface relief kept;
#   3. a power under 1, so each stone rises quickly out of its joint and rounds off.
# Output: <scan folder>/<name>_SSDM.png (8-bit grey), read by world_textures.cpp.
import os, struct, zlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (scan, output, size, the width of a stone in pixels at that size / 4, the share of the picture
#  that is joint (as it looks in the photo), the body's fullness)
SETS = [
    ('tex/Bricks078/Bricks078_8K-JPG_Displacement.jpg', 'tex/Bricks078/Bricks078_SSDM.png', 4096, 30, 0.22, 0.70),   # castle walls: field stones in lime mortar
    ('tex/Tiles130/Tiles130_8K-JPG_Displacement.jpg',   'tex/Tiles130/Tiles130_SSDM.png',   4096, 40, 0.14, 0.55),   # courtyard: flags, flat tops
    ('terrain_pbr/PavingStones131/PavingStones131_2K-JPG_Displacement.jpg', 'terrain_pbr/PavingStones131/PavingStones131_SSDM.png', 2048, 45, 0.26, 0.70),   # the paved roads: setts
]


def load_grey(rel, size):
    import bpy
    img = bpy.data.images.load(os.path.join(ROOT, 'assets', rel))
    if img.size[0] != size: img.scale(size, size)
    w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, 0]
    bpy.data.images.remove(img)
    return a


def png_grey(rel, a):
    a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = a.shape
    raw = b''.join(b'\x00' + a[y].tobytes() for y in range(h))
    def chunk(tag, data): return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    with open(os.path.join(ROOT, 'assets', rel), 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    print('WROTE', rel, a.shape, 'mean %.3f' % (a.mean() / 255.0))


def box(a, r, axis):
    """Box blur of radius r along one axis, wrapping (the scans tile), by running sums."""
    if r <= 0: return a
    n = a.shape[axis]
    pad = np.concatenate([np.take(a, range(n - r - 1, n), axis=axis), a, np.take(a, range(0, r), axis=axis)], axis=axis)
    c = np.cumsum(pad, axis=axis, dtype=np.float64)
    hi = np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, range(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a, r, passes=3):
    """Near-Gaussian blur (three box passes each way)."""
    for _ in range(passes):
        a = box(box(a, r, 0), r, 1)
    return a


def otsu(a):
    hist, edges = np.histogram(a, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / hist.sum()
    w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    between = (mt * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return edges[int(np.argmax(between))]


for src, out, size, R, joint_share, gamma in SETS:
    h = load_grey(src, size)
    s = blur(h, 1)                                       # (the scan's pixel noise off)
    # The scan's own shapes, its height measured against its surroundings: the scan drifts across
    # the picture (one stretch for all lost whole patches of stones), so each pixel is taken as
    # standard deviations above the local mean over a few stones' width.
    m = blur(s, R * 4)
    sd = np.sqrt(np.maximum(blur((s - m) ** 2, R * 4), 1e-8))
    z = (s - m) / sd
    # the lowest `joint_share` of it is the joints (0), the top half-percent the stone tops (1)
    zl = float(np.percentile(z, 100.0 * joint_share)); zh = float(np.percentile(z, 99.5))
    t = np.clip((z - zl) / max(zh - zl, 1e-6), 0.0, 1.0)
    # (a power under 1 fattens the bodies: stones rise quickly out of the joint, then round off)
    out_h = t ** gamma
    print('SET', src, 'joints below z %.2f' % zl, 'mean %.3f' % float(out_h.mean()))
    png_grey(out, out_h)
print('DONE')
