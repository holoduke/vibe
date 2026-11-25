# Writes a height map beside each of the service station's textures whose relief is NOT its
# light and dark:
#   blender -b --python tools/gasstation/make_maps.py
# The game derives a surface's normal map and height from the picture's own brightness
# (world_textures.cpp) - right for asphalt, pavers and ribbed cladding, wrong for:
#   brick     the mortar is lighter than the brick but lies deeper: height from how red a pixel is;
#   checker   black and white tiles are level, only the grout between them is lower;
#   subway    white tiles, grey grout: the grout is lower, the tile's glaze is flat.
# Output: assets/gasstation/tex/<name>_h.png (grey: white = high), which the loader takes in
# place of the brightness when it is there.
import bpy, os, struct, zlib
import numpy as np

TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'gasstation', 'tex')


def load(name):
    img = bpy.data.images.load(os.path.join(TEX, name + '.png'))
    w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]     # top row first
    bpy.data.images.remove(img)
    return a


def blur(a, r):
    """A box blur, r pixels each way, wrapping (the textures tile)."""
    if r <= 0: return a
    out = np.zeros_like(a)
    for d in range(-r, r + 1): out += np.roll(a, d, axis=0)
    a = out / (2 * r + 1); out = np.zeros_like(a)
    for d in range(-r, r + 1): out += np.roll(a, d, axis=1)
    return out / (2 * r + 1)


def save(name, hgt):
    """An 8-bit grey PNG, written here (Blender's own image save went out black from a script)."""
    a = (np.clip(hgt, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = a.shape
    raw = b''.join(bytes([0]) + a[y].tobytes() for y in range(h))
    def chunk(tag, data): return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    with open(os.path.join(TEX, name + '_h.png'), 'wb') as fp:
        fp.write(sig + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    print('MAP', name, 'mean %.3f' % float(a.mean() / 255.0))


def norm(a):
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / max(hi - lo, 1e-5), 0.0, 1.0)


LUM = np.array([0.30, 0.59, 0.11], dtype=np.float32)

# brick: red-and-saturated = brick face (high), grey = mortar (low); the brick's own grain on top
c = load('brick')
lum = c @ LUM
red = norm(c[:, :, 0] - 0.5 * (c[:, :, 1] + c[:, :, 2]))
face = blur(np.clip((red - 0.25) / 0.3, 0.0, 1.0), 2)
save('brick', 0.15 + 0.62 * face + 0.23 * face * norm(lum))

# checker: grout = neither black nor white (mid grey) -> low; tiles level, a touch of glaze wear
c = load('checker')
lum = c @ LUM
grout = np.clip(1.0 - np.abs(lum - 0.5) / 0.22, 0.0, 1.0)
# (only along the lines between tiles: keep what survives a small blur)
grout = np.clip(blur(grout, 1) * 1.6 - 0.25, 0.0, 1.0)
wear = norm(lum - blur(lum, 3))
save('checker', 0.9 - 0.75 * grout + 0.06 * (wear - 0.5))

# subway tile: grout darker than the glaze -> low, with a rounded edge to each tile
c = load('subway')
lum = c @ LUM
g = np.clip((np.percentile(lum, 60) - lum) / 0.12, 0.0, 1.0)
edge = blur(g, 3)
save('subway', 0.92 - 0.7 * np.maximum(g, edge * 0.8))
