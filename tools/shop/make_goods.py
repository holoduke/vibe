# The goods on the service station's shelves: a KIT of shop products (SKUs) - cereal and
# snack boxes, cans, jars, bottles (water, soda, cooking oil, wine; glass and plastic), milk
# and juice cartons, bags of crisps, sweets, books and magazines, cleaning products, motor
# oil, small boxed goods, small generic tobacco-style packs, ice-cream tubs, drink cases and
# plain shipping cartons - every one invented (no real brand, logo or product), each at
# three levels of detail, all printed from ONE texture atlas painted here in numpy.
#
#   "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b --python tools/shop/make_goods.py
#
# Output (assets/shop/):
#   goods.kit    the SKUs (binary, little endian):
#                "GDK1", u32 count, u32 atlas size; per SKU: char name[24], char category[16],
#                f32 w, h, d (its box, m), u32 levels (3); per level: u32 vertices, u32 indices,
#                vertices (f32 pos[3], normal[3], u, v), indices u32.
#                Axes: x its width, y up (0 = what it stands on), z toward its FRONT (the label).
#                uv: 0..1 over the atlas, v = 0 at the TOP row of the picture (as it is loaded).
#   goods.png    the atlas, 4096 x 4096 (texture slot kTexGoods; world_textures.cpp)
#   goods_h.png  its height (paper grain, creases, embossing) for the normal map
# Placed by src/engine/vk_engine/shop_goods.cpp from assets/gasstation/gasstation.shelf.
# Deterministic: every random number comes from the SKU's own seed.
#
# THE ATLAS: blocks of 512 x 512 (an 8 x 8 grid); a big SKU (cereal, cases, shipping
# cartons) has a block, a small one a quarter of one (256 x 256). Inside its cell an SKU's
# faces are laid out as a net (boxes: left side, front, right side, back in a row, the top
# above the front and the bottom under it; turned shapes: one band round the side - its
# rows follow the profile from the top down - and two discs, top and bottom), uniform in
# scale, 4 px apart. Small parts (caps, triggers, handles) read one solid swatch in a corner.
import bpy, math, os, struct
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(ROOT, 'assets', 'shop')
os.makedirs(OUT, exist_ok=True)
A = 4096
C = np.zeros((A, A, 3), np.float32) + np.float32(0.5)       # the atlas, row 0 = top
HM = np.zeros((A, A), np.float32) + np.float32(0.5)         # its height
CLIP = [0, 0, A, A]                                         # x0, y0, x1, y1: painting stays inside
TAU = 2.0 * math.pi


def log(*a): print('[goods]', *a, flush=True)


# ------------------------------------------------------------------ painting
def _blend(x0, y0, alpha, col):
    """alpha: (h, w) at the integer corner (x0, y0); clipped to CLIP."""
    h, w = alpha.shape
    cx0, cy0, cx1, cy1 = CLIP
    ax0, ay0 = max(x0, cx0), max(y0, cy0)
    ax1, ay1 = min(x0 + w, cx1), min(y0 + h, cy1)
    if ax1 <= ax0 or ay1 <= ay0: return
    a = alpha[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None]
    col = np.asarray(col, np.float32)
    if col.ndim == 1: C[ay0:ay1, ax0:ax1] = C[ay0:ay1, ax0:ax1] * (1.0 - a) + col * a
    else: C[ay0:ay1, ax0:ax1] = C[ay0:ay1, ax0:ax1] * (1.0 - a) + col[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * a


def rect(x0, y0, x1, y1, col, a=1.0):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    if x1 <= x0 or y1 <= y0: return
    _blend(x0, y0, np.full((y1 - y0, x1 - x0), a, np.float32), col)


def grad(x0, y0, x1, y1, c0, c1, vertical=True):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    if x1 <= x0 or y1 <= y0: return
    n = (y1 - y0) if vertical else (x1 - x0)
    t = np.linspace(0.0, 1.0, max(n, 1), dtype=np.float32)
    c = np.asarray(c0, np.float32)[None, :] * (1 - t[:, None]) + np.asarray(c1, np.float32)[None, :] * t[:, None]
    img = np.broadcast_to(c[:, None, :], (y1 - y0, x1 - x0, 3)) if vertical else np.broadcast_to(c[None, :, :], (y1 - y0, x1 - x0, 3))
    _blend(x0, y0, np.ones((y1 - y0, x1 - x0), np.float32), np.ascontiguousarray(img))


def ellipse(cx, cy, rx, ry, col, a=1.0, ring=0.0, shade=0.0):
    x0, y0 = int(math.floor(cx - rx - 1)), int(math.floor(cy - ry - 1))
    x1, y1 = int(math.ceil(cx + rx + 1)), int(math.ceil(cy + ry + 1))
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = (xs - cx) / max(rx, 0.5), (ys - cy) / max(ry, 0.5)
    d = np.sqrt(dx * dx + dy * dy)
    m = min(rx, ry)
    al = np.clip((1.0 - d) * m + 0.5, 0.0, 1.0)
    if ring > 0: al = al * np.clip((d - (1.0 - ring / m)) * m + 0.5, 0.0, 1.0)
    col = np.asarray(col, np.float32)
    if shade != 0.0:
        lit = np.clip(1.0 - shade * (0.35 * (dx + dy) + 0.5 * d * d), 0.2, 1.4)
        img = col[None, None, :] * lit[:, :, None]
        _blend(x0, y0, al * a, np.clip(img, 0, 1))
    else:
        _blend(x0, y0, al * a, col)


def star(cx, cy, r, n, col, depth=0.18):
    x0, y0 = int(cx - r * 1.3), int(cy - r * 1.3)
    s = int(r * 2.6) + 2
    ys, xs = np.mgrid[y0:y0 + s, x0:x0 + s].astype(np.float32) + 0.5
    a = np.arctan2(ys - cy, xs - cx); d = np.hypot(xs - cx, ys - cy)
    rr = r * (1.0 + depth * np.sign(np.sin(n * a)) * 0.5 + depth * 0.5 * np.cos(n * a))
    _blend(x0, y0, np.clip(rr - d + 0.5, 0, 1), col)


def wave(x0, y0, x1, y1, col, amp, per, phase, up=True):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 <= x0 or y1 <= y0: return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    edge = y0 + amp + amp * np.sin(xs / max(per, 1) * TAU + phase)
    al = np.clip((ys - edge) + 0.5, 0, 1) if up else np.clip((edge - ys) + 0.5, 0, 1)
    _blend(x0, y0, al, col)


def diag(x0, y0, x1, y1, col, k, off):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    al = np.clip((ys - (y0 + off + k * (xs - x0))) + 0.5, 0, 1)
    _blend(x0, y0, al, col)


def stripes(x0, y0, x1, y1, col, period, width, k=0.0):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    t = np.mod(xs + k * ys, period)
    al = np.clip(np.minimum(t, width - t) + 0.5, 0, 1) * (t < width + 1)
    _blend(x0, y0, al.astype(np.float32), col)


def emboss(x0, y0, x1, y1, dh):
    x0, y0, x1, y1 = max(int(x0), CLIP[0]), max(int(y0), CLIP[1]), min(int(x1), CLIP[2]), min(int(y1), CLIP[3])
    if x1 > x0 and y1 > y0: HM[y0:y1, x0:x1] += dh


# ---- letters: strokes in a unit box (x right, y down), rendered by distance, cached
L = {
    'A': [[(0, 1), (0.5, 0), (1, 1)], [(0.22, 0.62), (0.78, 0.62)]],
    'B': [[(0, 0), (0, 1)], [(0, 0), (0.68, 0), (0.9, 0.12), (0.9, 0.38), (0.68, 0.5), (0, 0.5)], [(0.68, 0.5), (0.97, 0.62), (0.97, 0.88), (0.72, 1), (0, 1)]],
    'C': [[(1, 0.16), (0.75, 0), (0.28, 0), (0, 0.26), (0, 0.74), (0.28, 1), (0.75, 1), (1, 0.84)]],
    'D': [[(0, 0), (0, 1), (0.58, 1), (1, 0.7), (1, 0.3), (0.58, 0), (0, 0)]],
    'E': [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.75, 0.5)]],
    'F': [[(1, 0), (0, 0), (0, 1)], [(0, 0.5), (0.75, 0.5)]],
    'G': [[(1, 0.16), (0.75, 0), (0.28, 0), (0, 0.26), (0, 0.74), (0.28, 1), (0.75, 1), (1, 0.84), (1, 0.56), (0.56, 0.56)]],
    'H': [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.5), (1, 0.5)]],
    'I': [[(0.5, 0), (0.5, 1)], [(0.2, 0), (0.8, 0)], [(0.2, 1), (0.8, 1)]],
    'J': [[(1, 0), (1, 0.75), (0.75, 1), (0.25, 1), (0, 0.78)]],
    'K': [[(0, 0), (0, 1)], [(1, 0), (0, 0.58)], [(0.32, 0.4), (1, 1)]],
    'L': [[(0, 0), (0, 1), (1, 1)]],
    'M': [[(0, 1), (0, 0), (0.5, 0.62), (1, 0), (1, 1)]],
    'N': [[(0, 1), (0, 0), (1, 1), (1, 0)]],
    'O': [[(0.3, 0), (0.7, 0), (1, 0.3), (1, 0.7), (0.7, 1), (0.3, 1), (0, 0.7), (0, 0.3), (0.3, 0)]],
    'P': [[(0, 1), (0, 0), (0.75, 0), (1, 0.15), (1, 0.4), (0.75, 0.55), (0, 0.55)]],
    'Q': [[(0.3, 0), (0.7, 0), (1, 0.3), (1, 0.7), (0.7, 1), (0.3, 1), (0, 0.7), (0, 0.3), (0.3, 0)], [(0.62, 0.72), (1.02, 1.05)]],
    'R': [[(0, 1), (0, 0), (0.75, 0), (1, 0.15), (1, 0.4), (0.75, 0.55), (0, 0.55)], [(0.48, 0.55), (1, 1)]],
    'S': [[(1, 0.13), (0.8, 0), (0.2, 0), (0, 0.15), (0, 0.38), (0.2, 0.5), (0.8, 0.5), (1, 0.62), (1, 0.87), (0.8, 1), (0.2, 1), (0, 0.87)]],
    'T': [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]],
    'U': [[(0, 0), (0, 0.75), (0.25, 1), (0.75, 1), (1, 0.75), (1, 0)]],
    'V': [[(0, 0), (0.5, 1), (1, 0)]],
    'W': [[(0, 0), (0.25, 1), (0.5, 0.42), (0.75, 1), (1, 0)]],
    'X': [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
    'Y': [[(0, 0), (0.5, 0.5), (1, 0)], [(0.5, 0.5), (0.5, 1)]],
    'Z': [[(0, 0), (1, 0), (0, 1), (1, 1)]],
    '0': [[(0.3, 0), (0.7, 0), (0.95, 0.25), (0.95, 0.75), (0.7, 1), (0.3, 1), (0.05, 0.75), (0.05, 0.25), (0.3, 0)]],
    '1': [[(0.25, 0.22), (0.6, 0), (0.6, 1)]],
    '2': [[(0, 0.2), (0.25, 0), (0.75, 0), (1, 0.2), (1, 0.4), (0, 1), (1, 1)]],
    '3': [[(0, 0), (1, 0), (0.45, 0.42), (0.8, 0.5), (1, 0.72), (0.78, 1), (0.2, 1), (0, 0.86)]],
    '4': [[(0.75, 1), (0.75, 0), (0, 0.68), (1, 0.68)]],
    '5': [[(1, 0), (0.1, 0), (0.02, 0.45), (0.7, 0.42), (1, 0.65), (0.9, 0.94), (0.6, 1), (0, 0.92)]],
    '6': [[(0.9, 0), (0.35, 0.15), (0, 0.65), (0.2, 1), (0.8, 1), (1, 0.75), (0.8, 0.5), (0.2, 0.5), (0.02, 0.68)]],
    '7': [[(0, 0), (1, 0), (0.4, 1)]],
    '8': [[(0.5, 0.48), (0.12, 0.36), (0.12, 0.1), (0.35, 0), (0.65, 0), (0.88, 0.1), (0.88, 0.36), (0.5, 0.48), (0.05, 0.64), (0.05, 0.9), (0.3, 1), (0.7, 1), (0.95, 0.9), (0.95, 0.64), (0.5, 0.48)]],
    '9': [[(0.1, 1), (0.65, 0.85), (1, 0.35), (0.8, 0), (0.2, 0), (0, 0.25), (0.2, 0.5), (0.8, 0.5), (0.98, 0.32)]],
    '-': [[(0.15, 0.55), (0.85, 0.55)]],
    '.': [[(0.45, 0.95), (0.55, 0.95)]],
    '%': [[(0, 1), (1, 0)], [(0.1, 0.1), (0.25, 0.1)], [(0.75, 0.9), (0.9, 0.9)]],
    '/': [[(0, 1), (1, 0)]],
    "'": [[(0.5, 0), (0.5, 0.25)]],
    '+': [[(0.1, 0.5), (0.9, 0.5)], [(0.5, 0.15), (0.5, 0.85)]],
}
_GC = {}


def glyph(ch, h, wt):
    key = (ch, h, wt)
    if key in _GC: return _GC[key]
    w = max(2, int(round(h * (0.34 if ch in "I1.'" else 0.62))))
    S = 3
    ys, xs = np.mgrid[0:h * S, 0:w * S].astype(np.float32)
    pad = wt * 0.5 + 0.06
    u = (xs + 0.5) / (w * S) * (1 + 2 * pad * h / w) - pad * h / w
    v = (ys + 0.5) / (h * S) * (1 + 2 * pad) - pad
    u = u * (w / h)                                   # (distances in units of the height)
    d = np.full(u.shape, 9.0, np.float32)
    for poly in L.get(ch, []):
        for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
            ax2, bx2 = ax * (w / h), bx * (w / h)
            ex, ey = bx2 - ax2, by - ay
            ll = ex * ex + ey * ey + 1e-9
            t = np.clip(((u - ax2) * ex + (v - ay) * ey) / ll, 0, 1)
            d = np.minimum(d, np.hypot(u - ax2 - t * ex, v - ay - t * ey))
    al = (d < wt * 0.5).astype(np.float32)
    al = al.reshape(h, S, w, S).mean((1, 3))
    _GC[key] = al
    return al


def text(x, y, h, s, col, wt=0.13, align='l', maxw=None, track=0.14):
    """Letters `s`, cap height h px, top-left (or centre: align 'c', right: 'r') at (x, y)."""
    h = int(round(h))
    if h < 3: return
    def width(hh): return sum(int(round(hh * (0.34 if c in "I1.'" else 0.62))) + int(hh * track) if c != ' ' else int(hh * 0.45) for c in s)
    if maxw is not None:
        while h > 3 and width(h) > maxw: h -= 1
    if h < 8:
        greek_line(x if align == 'l' else (x - width(h) / 2 if align == 'c' else x - width(h)), y, width(h), h, col)
        return
    W = width(h)
    cx = x if align == 'l' else (x - W / 2 if align == 'c' else x - W)
    cx = int(round(cx)); y = int(round(y))
    for c in s:
        if c == ' ': cx += int(h * 0.45); continue
        g = glyph(c, h, wt)
        _blend(cx, y, g, col)
        cx += g.shape[1] + int(h * track)


def greek_line(x, y, w, h, col, rng=None):
    """A line of text too small to read: words as soft bars."""
    x, y, w, h = int(x), int(y), int(w), max(1, int(round(h)))
    r = rng or np.random.RandomState(int(x * 7 + y * 13) & 0xffff)
    cx = x
    while cx < x + w:
        ww = int(r.randint(2, 7) * max(h * 0.55, 1.2))
        ww = min(ww, x + w - cx)
        if ww > 0: rect(cx, y + h * 0.15, cx + ww, y + h * 0.85, col, 0.85)
        cx += ww + max(1, int(h * 0.5))


def para(x0, y0, x1, y1, lh, col, rng):
    y = y0
    while y + lh <= y1:
        w = (x1 - x0) * (1.0 if rng.rand() > 0.2 else rng.uniform(0.3, 0.9))
        greek_line(x0, y, w, lh * 0.62, col, rng)
        y += lh


def barcode(x, y, w, h, rng):
    rect(x, y, x + w, y + h, (0.97, 0.97, 0.96))
    bx = x + w * 0.08
    end = x + w * 0.92
    while bx < end:
        bw = rng.choice([1.0, 1.0, 2.0, 3.0]) * max(w / 90.0, 0.7)
        rect(bx, y + h * 0.08, min(bx + bw, end), y + h * 0.78, (0.05, 0.05, 0.05))
        bx += bw + rng.choice([1.0, 2.0]) * max(w / 90.0, 0.7)
    greek_line(x + w * 0.1, y + h * 0.82, w * 0.8, h * 0.13, (0.1, 0.1, 0.1), rng)


def panel(x0, y0, x1, y1, rng, ink=(0.08, 0.08, 0.08), bg=(0.97, 0.97, 0.95)):
    """A table of numbers (what is in it): a box, a heading, rows with ruled lines."""
    rect(x0, y0, x1, y1, bg)
    w = x1 - x0; h = y1 - y0
    lw = max(1, w / 120)
    for (a, b, c, d) in ((x0, y0, x1, y0 + lw), (x0, y1 - lw, x1, y1), (x0, y0, x0 + lw, y1), (x1 - lw, y0, x1, y1)):
        rect(a, b, c, d, ink)
    hh = h * 0.12
    text(x0 + w * 0.06, y0 + h * 0.03, hh * 0.7, 'NUTRI VALUE', ink, wt=0.2, maxw=w * 0.88)
    rect(x0 + lw, y0 + hh * 1.1, x1 - lw, y0 + hh * 1.1 + 2 * lw, ink)
    y = y0 + hh * 1.4
    rows = int((h - hh * 1.5) / max(h * 0.075, 3))
    for k in range(rows):
        rh = (y1 - y - lw) / max(rows - k, 1)
        greek_line(x0 + w * 0.06, y + rh * 0.25, w * rng.uniform(0.3, 0.5), rh * 0.5, ink, rng)
        greek_line(x0 + w * 0.7, y + rh * 0.25, w * 0.22, rh * 0.5, ink, rng)
        rect(x0 + lw, y + rh - lw * 0.5, x1 - lw, y + rh + lw * 0.5, ink, 0.6)
        y += rh


# ------------------------------------------------------------------ names, colours
NAMES = ['QUENDO', 'VASTRI', 'PELUMA', 'TORVIK', 'ZAMBELO', 'NORVA', 'KITRAL', 'SOLMIRA', 'FENNOK', 'ODRIM',
         'YUKORA', 'TELBAN', 'GRIMSA', 'VOLPENA', 'HAZRO', 'PRAVETO', 'SKORRA', 'MUNDELA', 'ORVASK', 'TEMBRO',
         'CALUVI', 'DRESKO', 'NIBOLA', 'FRANZOL', 'WEXLO', 'JOVARI', 'PIMBRA', 'KOLSTE', 'ZUPRA', 'ELVONO',
         'TROVEL', 'GALDRI', 'MOXITA', 'BELNOR', 'SIRVA', 'KUMBO', 'FLEXIRA', 'BRONDA', 'QUAMI', 'ULVENTO',
         'HOPLA', 'ZEFRI', 'MARVIDO', 'NUSKA', 'OBRELL', 'TAZINO', 'VELKRA', 'DUMORA']
PAL = [(0.80, 0.10, 0.08), (0.95, 0.62, 0.05), (0.08, 0.30, 0.70), (0.10, 0.52, 0.20), (0.92, 0.40, 0.06),
       (0.45, 0.12, 0.55), (0.06, 0.55, 0.65), (0.88, 0.80, 0.15), (0.55, 0.30, 0.12), (0.90, 0.30, 0.45),
       (0.15, 0.15, 0.20), (0.96, 0.95, 0.90), (0.20, 0.65, 0.85), (0.60, 0.75, 0.15), (0.70, 0.05, 0.25),
       (0.30, 0.20, 0.55)]
WHITE, BLACK, CREAM = (0.96, 0.96, 0.94), (0.06, 0.06, 0.07), (0.93, 0.89, 0.78)


def dark(c, k=0.55): return tuple(float(x) * k for x in c)
def light(c, k=0.5): return tuple(float(x) + (1 - float(x)) * k for x in c)
def lum(c): return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
def ink_on(c): return BLACK if lum(c) > 0.55 else WHITE


# ------------------------------------------------------------------ meshes
class Mesh:
    def __init__(self): self.v = []; self.i = []

    def tri(self, a, b, c, n, ta, tb, tc, na=None, nb=None, nc=None):
        a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
        fn = np.cross(b - a, c - a)
        if np.linalg.norm(fn) < 1e-12: return
        if np.dot(fn, n) < 0: b, c, tb, tc, nb, nc = c, b, tc, tb, nc, nb
        k = len(self.v)
        for p, t, nn in ((a, ta, na), (b, tb, nb), (c, tc, nc)):
            nn = n if nn is None else nn
            self.v.append((p[0], p[1], p[2], nn[0], nn[1], nn[2], t[0] / A, t[1] / A))
        self.i += [k, k + 1, k + 2]

    def quad(self, p, n, t, nn=None):
        """p: 4 corners in order round the quad; t: their pixel coords in the atlas."""
        nn = nn or [None] * 4
        self.tri(p[0], p[1], p[2], n, t[0], t[1], t[2], nn[0], nn[1], nn[2])
        self.tri(p[0], p[2], p[3], n, t[0], t[2], t[3], nn[0], nn[2], nn[3])

    def arrays(self):
        v = np.array(self.v, np.float32).reshape(-1, 8)
        i = np.array(self.i, np.uint32)
        # weld identical vertices (same place, normal, uv)
        key = np.round(v * np.array([1e4] * 3 + [1e3] * 3 + [1e5] * 2, np.float32)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        return v[first], inv.reshape(-1)[i].astype(np.uint32)


def box_net(cell, W, H, D, pad=5):
    """Pixel rects of a box's six faces in its cell: (x0, y0, x1, y1) and the scale (px per m)."""
    x0, y0, S = cell
    s = min((S - 5 * pad) / (2 * W + 2 * D), (S - 4 * pad) / (H + 2 * D))
    gx = x0 + (S - s * (2 * W + 2 * D) - 3 * pad) / 2.0
    gy = y0 + pad
    fl = (gx, gy + s * D + pad, gx + s * D, gy + s * D + pad + s * H)
    fr = (fl[2] + pad, fl[1], fl[2] + pad + s * W, fl[3])
    rs = (fr[2] + pad, fl[1], fr[2] + pad + s * D, fl[3])
    bk = (rs[2] + pad, fl[1], rs[2] + pad + s * W, fl[3])
    tp = (fr[0], gy, fr[2], gy + s * D)
    bt = (fr[0], fl[3] + pad, fr[2], fl[3] + pad + s * D)
    sw = (x0 + S - 18, y0 + S - 18, x0 + S - 6, y0 + S - 6)        # the swatch
    return dict(front=fr, left=fl, right=rs, back=bk, top=tp, bottom=bt, swatch=sw), s


def add_box(m, R, W, H, D, x=0.0, y=0.0, z=0.0, faces=None):
    """A box W x H x D standing on y with its front at z + D/2, its faces textured by the rects R."""
    a, b = x - W / 2, x + W / 2
    f, k = z + D / 2, z - D / 2
    t = y + H
    def uv(r, flip=False): return [(r[0], r[3]), (r[2], r[3]), (r[2], r[1]), (r[0], r[1])]
    F = faces or R
    m.quad([(a, y, f), (b, y, f), (b, t, f), (a, t, f)], (0, 0, 1), uv(F['front']))
    m.quad([(b, y, k), (a, y, k), (a, t, k), (b, t, k)], (0, 0, -1), uv(F['back']))
    m.quad([(a, y, k), (a, y, f), (a, t, f), (a, t, k)], (-1, 0, 0), uv(F['left']))
    m.quad([(b, y, f), (b, y, k), (b, t, k), (b, t, f)], (1, 0, 0), uv(F['right']))
    r = F['top']                                   # (its bottom edge on the front's top)
    m.quad([(a, t, f), (b, t, f), (b, t, k), (a, t, k)], (0, 1, 0), [(r[0], r[3]), (r[2], r[3]), (r[2], r[1]), (r[0], r[1])])
    r = F['bottom']                                # (its top edge on the front's bottom)
    m.quad([(a, y, f), (b, y, f), (b, y, k), (a, y, k)], (0, -1, 0), [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])])


def sw_faces(R):
    s = R['swatch']; c = ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)
    r = (c[0] - 0.01, c[1] - 0.01, c[0] + 0.01, c[1] + 0.01)
    return {k: r for k in ('front', 'back', 'left', 'right', 'top', 'bottom')}


def lathe_layout(cell, prof, pad=5):
    """The band (all of the side) and two discs. prof: [(r, y)] from the bottom up."""
    x0, y0, S = cell
    L_ = sum(math.hypot(prof[k + 1][0] - prof[k][0], prof[k + 1][1] - prof[k][1]) for k in range(len(prof) - 1))
    rmax = max(p[0] for p in prof)
    bh = (S - 3 * pad) * 0.72
    bw = min(S - 2 * pad, max(0.35 * S, bh * TAU * rmax / max(L_, 1e-3)))
    band = (x0 + (S - bw) / 2, y0 + pad, x0 + (S + bw) / 2, y0 + pad + bh)
    dr = min((S - 3 * pad - bh) * 0.5 - pad, S * 0.25)
    cy = band[3] + pad + dr + 1
    top = (x0 + S * 0.3, cy, dr); bot = (x0 + S * 0.7, cy, dr)
    sw = (x0 + S - 18, y0 + S - 18, x0 + S - 6, y0 + S - 6)
    return dict(band=band, top=top, bottom=bot, swatch=sw, length=L_)


def band_row(R, prof, s):
    b = R['band']
    return b[1] + (1.0 - s / R['length']) * (b[3] - b[1])


def add_lathe(m, R, prof, seg, x=0.0, z=0.0, caps=True):
    b = R['band']
    s = [0.0]
    for k in range(len(prof) - 1): s.append(s[-1] + math.hypot(prof[k + 1][0] - prof[k][0], prof[k + 1][1] - prof[k][1]))
    full = R['length']
    sc = s[-1] / full if full > 0 else 1.0
    for k in range(len(prof) - 1):
        (r0, y0), (r1, y1) = prof[k], prof[k + 1]
        dr, dy = r1 - r0, y1 - y0
        ln = math.hypot(dr, dy)
        if ln < 1e-6: continue
        nr, ny = dy / ln, -dr / ln
        v0 = band_row(R, prof, s[k] / sc if sc else s[k]); v1 = band_row(R, prof, s[k + 1] / sc if sc else s[k + 1])
        for j in range(seg):
            u0, u1 = j / seg, (j + 1) / seg
            P = []; N = []
            for (r, yy, u) in ((r0, y0, u0), (r0, y0, u1), (r1, y1, u1), (r1, y1, u0)):
                ph = TAU * u
                P.append((x - r * math.sin(ph), yy, z - r * math.cos(ph)))
                N.append((-nr * math.sin(ph), ny, -nr * math.cos(ph)))
            um = TAU * (j + 0.5) / seg
            n = (-nr * math.sin(um), ny, -nr * math.cos(um))
            T = [(b[0] + u0 * (b[2] - b[0]), v0), (b[0] + u1 * (b[2] - b[0]), v0), (b[0] + u1 * (b[2] - b[0]), v1), (b[0] + u0 * (b[2] - b[0]), v1)]
            m.quad(P, n, T, N)
    if not caps: return
    for which, (r, yy), sgn in (('bottom', prof[0], -1), ('top', prof[-1], 1)):
        if r < 1e-4: continue
        cx, cy, dr = R[which]
        c = (x, yy, z)
        for j in range(seg):
            p = []; t = []
            for u in (j / seg, (j + 1) / seg):
                ph = TAU * u
                px, pz = -math.sin(ph), -math.cos(ph)
                p.append((x + r * px, yy, z + r * pz))
                t.append((cx + dr * px, cy + dr * pz * (1 if sgn > 0 else -1)))
            m.tri(c, p[0], p[1], (0, sgn, 0), (cx, cy), t[0], t[1])


def simplify(prof, keep):
    if len(prof) <= keep: return prof
    idx = sorted(set([0, len(prof) - 1] + [int(round(k * (len(prof) - 1) / (keep - 1))) for k in range(keep)]))
    return [prof[k] for k in idx]


# ------------------------------------------------------------------ the SKUs
SKUS = []        # (name, category, (w, h, d), [3 meshes])


def finish(name, cat, size, meshes):
    SKUS.append((name, cat, size, [mm.arrays() for mm in meshes]))


def set_clip(r): CLIP[:] = [int(math.floor(r[0])), int(math.floor(r[1])), int(math.ceil(r[2])), int(math.ceil(r[3]))]


def paint_front(r, rng, name, sub, c1, c2, style, pic=None, weight='500 G'):
    x0, y0, x1, y1 = r; w = x1 - x0; h = y1 - y0
    set_clip(r)
    grad(x0, y0, x1, y1, light(c1, 0.15), dark(c1, 0.8))
    if style == 0: wave(x0, y0 + h * 0.55, x1, y1, c2, h * 0.03, w * 0.6, rng.uniform(0, 6))
    elif style == 1: diag(x0, y0 + h * 0.4, x1, y1, c2, -0.35, 0)
    elif style == 2: stripes(x0, y0, x1, y1, light(c1, 0.25), w * 0.18, w * 0.05, 0.6)
    else: ellipse(x0 + w * 0.5, y0 + h * 0.9, w * 0.75, h * 0.38, c2)
    # the name in a band
    bandc = WHITE if style % 2 else dark(c1, 0.35)
    by = y0 + h * rng.uniform(0.07, 0.14)
    rect(x0 + w * 0.05, by, x1 - w * 0.05, by + h * 0.2, bandc, 0.92)
    text(x0 + w * 0.5, by + h * 0.035, h * 0.13, name, c1 if style % 2 else WHITE, wt=0.2, align='c', maxw=w * 0.84)
    text(x0 + w * 0.5, by + h * 0.24, h * 0.07, sub, ink_on(c1), wt=0.16, align='c', maxw=w * 0.86)
    if pic: pic(x0 + w * 0.5, y0 + h * 0.66, w * 0.36, h * 0.2)
    if rng.rand() < 0.5: star(x0 + w * 0.82, y0 + h * 0.47, min(w, h) * 0.1, 14, (0.98, 0.85, 0.1)); text(x0 + w * 0.82, y0 + h * 0.455, min(w, h) * 0.05, rng.choice(['NEW', '20%', 'XL', '+1']), (0.7, 0.05, 0.05), wt=0.22, align='c')
    text(x0 + w * 0.06, y1 - h * 0.08, h * 0.045, weight, ink_on(dark(c1, 0.8)), wt=0.16)
    emboss(x0 + w * 0.05, by, x1 - w * 0.05, by + h * 0.2, 0.06)


def paint_side(r, rng, c1, panel_on=True, code=False):
    x0, y0, x1, y1 = r; w = x1 - x0; h = y1 - y0
    set_clip(r)
    grad(x0, y0, x1, y1, light(c1, 0.1), dark(c1, 0.85))
    ink = ink_on(c1)
    if panel_on and w > 12 and h > 30:
        panel(x0 + w * 0.08, y0 + h * 0.08, x1 - w * 0.08, y0 + h * 0.55, rng)
        para(x0 + w * 0.08, y0 + h * 0.6, x1 - w * 0.08, y1 - h * 0.05, max(h * 0.035, 2.5), ink, rng)
    else:
        para(x0 + w * 0.1, y0 + h * 0.1, x1 - w * 0.1, y1 - h * 0.1, max(min(w, h) * 0.08, 2.5), ink, rng)
    if code and w > 10:
        bw = min(w * 0.7, h * 0.5)
        barcode(x0 + (w - bw) / 2, y1 - h * 0.05 - bw * 0.55, bw, bw * 0.5, rng)


def paint_flat(r, col, rng=None, txt=None, code=False):
    x0, y0, x1, y1 = r; w = x1 - x0; h = y1 - y0
    set_clip(r)
    rect(x0, y0, x1, y1, col)
    if txt: text(x0 + w / 2, y0 + h * 0.3, h * 0.3, txt, ink_on(col), wt=0.18, align='c', maxw=w * 0.9)
    if code and rng is not None and w > 16: barcode(x0 + w * 0.3, y0 + h * 0.15, w * 0.4, h * 0.7, rng)


def pic_flakes(rng, base):
    def f(cx, cy, rx, ry):
        ellipse(cx, cy + ry * 0.4, rx * 1.1, ry * 0.55, WHITE, shade=0.4)                         # the bowl
        for k in range(40):
            a = rng.uniform(0, TAU); d = math.sqrt(rng.uniform(0, 1))
            ellipse(cx + math.cos(a) * d * rx * 0.95, cy + math.sin(a) * d * ry * 0.5, rx * 0.13, ry * 0.09, base, shade=0.6)
    return f


def pic_balls(rng, cols, n=12):
    def f(cx, cy, rx, ry):
        for k in range(n):
            a = rng.uniform(0, TAU); d = math.sqrt(rng.uniform(0, 1))
            c = cols[k % len(cols)]
            rr = rng.uniform(0.18, 0.3) * min(rx, ry) * 1.6
            ellipse(cx + math.cos(a) * d * rx * 0.8, cy + math.sin(a) * d * ry * 0.7, rr, rr * 0.9, c, shade=0.7)
    return f


def new_cell(big):
    """The next free cell: (x0, y0, size)."""
    global BLOCK, QUART
    if big:
        b = BLOCK; BLOCK += 1
        return ((b % 8) * 512, (b // 8) * 512, 512)
    if QUART is None or QUART[1] == 4:
        b = BLOCK; BLOCK += 1
        QUART = [b, 0]
    b, q = QUART; QUART[1] += 1
    return ((b % 8) * 512 + (q % 2) * 256, (b // 8) * 512 + (q // 2) * 256, 256)


BLOCK = 0
QUART = None


def make_box_sku(name, cat, W, H, D, seed, kind):
    rng = np.random.RandomState(seed)
    big = max(W, H) > 0.24 or kind in ('ship', 'case')
    cell = new_cell(big)
    R, s = box_net(cell, W, H, D)
    nm = NAMES[seed % len(NAMES)]
    c1 = PAL[rng.randint(len(PAL))]; c2 = PAL[rng.randint(len(PAL))]
    if c2 == c1: c2 = PAL[(PAL.index(c1) + 5) % len(PAL)]
    style = rng.randint(4)
    set_clip((cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2]))
    rect(cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2], dark(c1, 0.6))
    if kind == 'ship':
        kraft = (0.62 + rng.uniform(-0.04, 0.04), 0.46, 0.30)
        for f in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            r = R[f]; set_clip(r)
            rect(*r, kraft)
            HM[int(r[1]):int(r[3]), int(r[0]):int(r[2])] += np.float32(0.04) * np.sin(np.arange(int(r[0]), int(r[2]), dtype=np.float32) * 1.3)[None, :]
            if f in ('top', 'bottom'):
                cx = (r[0] + r[2]) / 2; rect(r[0], (r[1] + r[3]) / 2 - 1, r[2], (r[1] + r[3]) / 2 + 1, dark(kraft, 0.6))
                rect(cx - (r[2] - r[0]) * 0.0 - 1, r[1], cx + 1, r[3], dark(kraft, 0.7))
                rect(r[0], (r[1] + r[3]) / 2 - (r[3] - r[1]) * 0.12, r[2], (r[1] + r[3]) / 2 + (r[3] - r[1]) * 0.12, (0.78, 0.66, 0.42) if rng.rand() < 0.6 else (0.82, 0.82, 0.8), 0.85)
            else:
                w = r[2] - r[0]; h = r[3] - r[1]
                rect(r[0], r[1], r[2], r[1] + h * 0.08, (0.78, 0.66, 0.42), 0.85)
                if f in ('front', 'back'):
                    text(r[0] + w * 0.08, r[1] + h * 0.18, h * 0.1, '%d' % rng.randint(10, 99) + 'X' + '%d' % rng.randint(10, 99), (0.12, 0.1, 0.1), wt=0.2)
                    text(r[0] + w * 0.08, r[1] + h * 0.33, h * 0.07, nm, (0.15, 0.12, 0.1), wt=0.18, maxw=w * 0.6)
                    for k in range(2):     # the arrows "this way up"
                        ax = r[0] + w * (0.72 + 0.12 * k); ay = r[1] + h * 0.2
                        rect(ax - w * 0.012, ay + h * 0.06, ax + w * 0.012, ay + h * 0.22, (0.12, 0.1, 0.1))
                        for j in range(8): rect(ax - w * 0.005 * j, ay + h * 0.008 * j, ax + w * 0.005 * j, ay + h * 0.008 * j + 2, (0.12, 0.1, 0.1))
                    rect(r[0] + w * 0.55, r[1] + h * 0.55, r[0] + w * 0.92, r[1] + h * 0.88, WHITE)
                    barcode(r[0] + w * 0.58, r[1] + h * 0.6, w * 0.3, h * 0.2, rng)
                else:
                    text(r[0] + w * 0.5, r[1] + h * 0.4, h * 0.12, nm[:3], (0.12, 0.1, 0.1), wt=0.2, align='c', maxw=w * 0.8)
    elif kind in ('book', 'mag'):
        sub = rng.choice(['A NOVEL', 'ROAD ATLAS', 'PUZZLES', 'COOKING', 'THE COAST', 'STARS', 'MOTOR', 'GARDEN', 'TRAVEL', 'HOME']) if kind == 'mag' else rng.choice(['A NOVEL', 'THE LONG ROAD', 'NIGHT SHIFT', 'NORTH SEA', 'GHOST TOWN', 'THE ROCKET'])
        r = R['front']; set_clip(r); w = r[2] - r[0]; h = r[3] - r[1]
        grad(*r, light(c1, 0.2), dark(c1, 0.7))
        if kind == 'mag':
            rect(r[0], r[1], r[2], r[1] + h * 0.17, c2)
            text(r[0] + w * 0.5, r[1] + h * 0.025, h * 0.12, nm, WHITE, wt=0.22, align='c', maxw=w * 0.92)
            ellipse(r[0] + w * rng.uniform(0.35, 0.65), r[1] + h * 0.62, w * 0.34, h * 0.3, light(c2, 0.4), shade=0.6)
            ellipse(r[0] + w * 0.5, r[1] + h * 0.45, w * 0.12, h * 0.1, (0.85, 0.66, 0.5), shade=0.5)
            for k in range(4): greek_line(r[0] + w * 0.06, r[1] + h * (0.24 + 0.07 * k), w * rng.uniform(0.25, 0.45), h * 0.035, WHITE, rng)
            text(r[0] + w * 0.06, r[1] + h * 0.86, h * 0.06, sub, WHITE, wt=0.2, maxw=w * 0.6)
            barcode(r[0] + w * 0.72, r[1] + h * 0.86, w * 0.22, h * 0.1, rng)
        else:
            ellipse(r[0] + w * 0.5, r[1] + h * 0.55, w * 0.38, w * 0.38, light(c2, 0.2), shade=0.3)
            text(r[0] + w * 0.5, r[1] + h * 0.1, h * 0.09, sub, WHITE, wt=0.2, align='c', maxw=w * 0.9)
            text(r[0] + w * 0.5, r[1] + h * 0.85, h * 0.06, nm, light(c2, 0.5), wt=0.16, align='c', maxw=w * 0.8)
        r = R['back']; set_clip(r); w = r[2] - r[0]; h = r[3] - r[1]
        grad(*r, light(c1, 0.2), dark(c1, 0.7))
        para(r[0] + w * 0.1, r[1] + h * 0.15, r[2] - w * 0.1, r[1] + h * 0.6, max(h * 0.035, 2.5), WHITE, rng)
        barcode(r[0] + w * 0.6, r[1] + h * 0.8, w * 0.3, h * 0.12, rng)
        r = R['left']; set_clip(r)            # the spine
        rect(*r, dark(c1, 0.8)); w = r[2] - r[0]; h = r[3] - r[1]
        if w > 6:
            for k in range(min(len(nm), 7)): text(r[0] + w / 2, r[1] + h * 0.1 + k * w * 0.8, w * 0.6, nm[k], WHITE, wt=0.2, align='c')
        for f in ('right', 'top', 'bottom'):  # the pages
            r = R[f]; set_clip(r); rect(*r, CREAM)
            if f == 'right': stripes(r[0], r[1], r[2], r[3], (0.78, 0.74, 0.64), 2.5, 0.6)
            else:
                ys = np.arange(int(r[1]), int(r[3]))
                for yy in ys[::3]: rect(r[0], yy, r[2], yy + 1, (0.8, 0.76, 0.66), 0.6)
    elif kind == 'cig':
        c = PAL[rng.randint(len(PAL))]
        for f in ('front', 'back'):
            r = R[f]; set_clip(r); w = r[2] - r[0]; h = r[3] - r[1]
            rect(*r, WHITE)
            rect(r[0], r[1], r[2], r[1] + h * 0.45, c)
            diag(r[0], r[1] + h * 0.3, r[2], r[1] + h * 0.45, WHITE, -0.3, 0)
            text(r[0] + w * 0.5, r[1] + h * 0.08, h * 0.12, nm, WHITE, wt=0.2, align='c', maxw=w * 0.9)
            rect(r[0] + w * 0.06, r[1] + h * 0.55, r[2] - w * 0.06, r[3] - h * 0.06, BLACK)
            rect(r[0] + w * 0.1, r[1] + h * 0.59, r[2] - w * 0.1, r[3] - h * 0.1, WHITE)
            para(r[0] + w * 0.14, r[1] + h * 0.63, r[2] - w * 0.14, r[3] - h * 0.12, max(h * 0.06, 2.5), BLACK, rng)
            rect(r[0], r[1] + h * 0.2, r[2], r[1] + h * 0.2 + 1, dark(c, 0.6))         # the lid's line
        for f in ('left', 'right', 'top', 'bottom'):
            r = R[f]; set_clip(r); rect(*r, c)
            rect(r[0], (r[1] + r[3]) / 2, r[2], r[3], WHITE)
            if f == 'left': barcode(r[0] + 1, r[1] + (r[3] - r[1]) * 0.55, r[2] - r[0] - 2, (r[3] - r[1]) * 0.3, rng)
    elif kind == 'candy':
        r = R['front']; w = r[2] - r[0]; h = r[3] - r[1]
        for f in ('front', 'back', 'top', 'bottom'):
            r = R[f]; set_clip(r); w = r[2] - r[0]; h = r[3] - r[1]
            grad(*r, light(c1, 0.15), dark(c1, 0.75), vertical=False)
            rect(r[0] + w * 0.15, r[1], r[0] + w * 0.2, r[3], c2)
            if f == 'front':
                text(r[0] + w * 0.55, r[1] + h * 0.15, h * 0.45, nm, WHITE, wt=0.22, align='c', maxw=w * 0.6)
                text(r[0] + w * 0.55, r[1] + h * 0.68, h * 0.18, rng.choice(['CARAMEL', 'NUT', 'MINT', 'COCOA', 'CRISP']), light(c2, 0.4), wt=0.18, align='c', maxw=w * 0.6)
                ellipse(r[0] + w * 0.9, r[1] + h * 0.5, h * 0.3, h * 0.3, (0.35, 0.18, 0.08), shade=0.6)
            elif f == 'back':
                para(r[0] + w * 0.25, r[1] + h * 0.1, r[0] + w * 0.65, r[3] - h * 0.1, max(h * 0.15, 2.5), WHITE, rng)
                barcode(r[0] + w * 0.7, r[1] + h * 0.15, w * 0.22, h * 0.7, rng)
        for f in ('left', 'right'):
            r = R[f]; set_clip(r); rect(*r, dark(c1, 0.6)); stripes(r[0], r[1], r[2], r[3], light(c1, 0.3), 2.0, 0.7, 0)
    else:   # printed cartons: cereal, snack, small boxed goods, drink cases
        subs = dict(cereal=['CRUNCHY FLAKES', 'OAT RINGS', 'HONEY PUFFS', 'MUESLI', 'CORN FLAKES', 'CHOCO BITES', 'BRAN', 'FRUIT CLUSTERS'],
                    snack=['CRACKERS', 'COOKIES', 'WAFERS', 'SALT STICKS', 'OAT BARS', 'RICE CAKES'],
                    small=['BLACK TEA', 'GREEN TEA', 'COFFEE PADS', 'RICE', 'PASTA', 'CAKE MIX', 'TISSUES', 'MATCHES', 'SUGAR', 'SALT', 'SOUP MIX', 'BATTERIES', 'FUSES', 'BULBS', 'PLASTERS'],
                    case=['COLA 24', 'LEMON 24', 'WATER 12', 'ORANGE 24', 'ENERGY 24'])[kind]
        sub = subs[seed % len(subs)]
        if kind == 'cereal': pic = pic_flakes(rng, PAL[rng.choice([1, 4, 8, 7])])
        elif kind == 'case': pic = pic_balls(rng, [c2, light(c2, 0.4), WHITE], 9)
        else: pic = pic_balls(rng, [PAL[rng.randint(len(PAL))], (0.85, 0.65, 0.35), (0.5, 0.3, 0.15)], 10)
        wt = '%d G' % rng.choice([125, 250, 375, 500, 750, 1000]) if kind != 'case' else '24 X 330 ML'
        paint_front(R['front'], rng, nm, sub, c1, c2, style, pic, wt)
        paint_front(R['back'], rng, nm, sub, c1, c2, (style + 1) % 4, pic if kind == 'case' else None, wt)
        if kind != 'case':
            r = R['back']; w = r[2] - r[0]; h = r[3] - r[1]; set_clip(r)
            rect(r[0] + w * 0.08, r[1] + h * 0.45, r[2] - w * 0.08, r[3] - h * 0.15, light(c1, 0.75))
            para(r[0] + w * 0.12, r[1] + h * 0.5, r[2] - w * 0.12, r[3] - h * 0.2, max(h * 0.03, 2.5), dark(c1, 0.4), rng)
        paint_side(R['left'], rng, c1, panel_on=kind != 'case')
        paint_side(R['right'], rng, c1, panel_on=False, code=True)
        paint_flat(R['top'], c1, rng, nm)
        paint_flat(R['bottom'], light(c1, 0.6), rng, None, code=True)
    # the swatch (cap, handle) - the second colour
    set_clip(R['swatch']); rect(*R['swatch'], c2)
    meshes = []
    for lod in range(3):
        m = Mesh(); add_box(m, R, W, H, D, z=0.0); meshes.append(m)
    finish(name, cat, (W, H, D), meshes)


def make_lathe_sku(name, cat, prof, zones, seed, kind, extra=None):
    """zones: [(s0, s1, what)] over the profile's height fraction from the bottom, painted onto the band."""
    rng = np.random.RandomState(seed)
    cell = new_cell(False)
    R = lathe_layout(cell, prof)
    nm = NAMES[seed % len(NAMES)]
    c1 = PAL[rng.randint(len(PAL))]; c2 = PAL[rng.randint(len(PAL))]
    if c2 == c1: c2 = PAL[(PAL.index(c1) + 7) % len(PAL)]
    b = R['band']
    set_clip((cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2]))
    rect(cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2], (0.4, 0.4, 0.42))
    H = prof[-1][1]
    # arc length at each profile point, as a fraction; zones go by height: map height -> row
    s = [0.0]
    for k in range(len(prof) - 1): s.append(s[-1] + math.hypot(prof[k + 1][0] - prof[k][0], prof[k + 1][1] - prof[k][1]))
    def row_at(yh):
        yy = yh * H
        for k in range(len(prof) - 1):
            if prof[k + 1][1] >= yy - 1e-9:
                t = 0.0 if prof[k + 1][1] == prof[k][1] else (yy - prof[k][1]) / (prof[k + 1][1] - prof[k][1])
                return band_row(R, prof, s[k] + t * (s[k + 1] - s[k]))
        return b[1]
    caps = {}
    for (z0, z1, what, col) in zones:
        ra, rb = row_at(z1), row_at(z0)
        r = (b[0], ra, b[2], rb)
        set_clip((b[0], min(ra, rb) - 0.5, b[2], max(ra, rb) + 0.5))
        w = b[2] - b[0]; h = abs(rb - ra)
        if what == 'solid':
            grad(b[0], ra, b[2], rb, light(col, 0.12), dark(col, 0.85))
            stripes(b[0], ra, b[2], rb, light(col, 0.35), w / 6.0, w * 0.015)       # (a highlight round it)
        elif what == 'ribs':
            rect(b[0], ra, b[2], rb, col)
            for yy in np.arange(ra, rb, 3.0): rect(b[0], yy, b[2], yy + 1, dark(col, 0.7))
        elif what == 'label':
            lc = col
            grad(b[0], ra, b[2], rb, light(lc, 0.15), dark(lc, 0.8))
            # the front (u = 0.5) carries the name; round the back: the table, the code, text
            fx = b[0] + w * 0.5
            fw = w * 0.42
            st = rng.randint(3)
            if st == 0: wave(b[0], ra + h * 0.55, b[2], rb, c2, h * 0.05, w * 0.25, rng.uniform(0, 6))
            elif st == 1: rect(b[0], ra + h * 0.08, b[2], ra + h * 0.16, c2); rect(b[0], rb - h * 0.16, b[2], rb - h * 0.08, c2)
            else: ellipse(fx, ra + h * 0.55, fw * 0.55, h * 0.4, c2)
            sub = extra.get('sub', '') if extra else ''
            text(fx, ra + h * 0.22, h * 0.22, nm, ink_on(lc) if st != 2 else WHITE, wt=0.2, align='c', maxw=fw)
            text(fx, ra + h * 0.55, h * 0.12, sub, ink_on(c2) if st != 1 else ink_on(lc), wt=0.16, align='c', maxw=fw)
            text(fx, ra + h * 0.78, h * 0.09, extra.get('vol', '') if extra else '', ink_on(lc), wt=0.16, align='c', maxw=fw)
            if extra and extra.get('pic'): extra['pic'](rng)(b[0] + w * 0.2, ra + h * 0.5, w * 0.07, h * 0.28)
            if w > 60 and h > 40: panel(b[0] + w * 0.03, ra + h * 0.1, b[0] + w * 0.03 + min(w * 0.16, h * 1.1), rb - h * 0.1, rng)
            bw = min(w * 0.12, h * 0.9)
            barcode(b[2] - w * 0.05 - bw, ra + h * 0.15, bw, min(h * 0.6, bw * 0.6), rng)
            para(b[0] + w * 0.75, ra + h * 0.15, b[0] + w * 0.8, rb - h * 0.1, max(h * 0.08, 2.5), ink_on(lc), rng)
            emboss(b[0], ra, b[2], ra + 2, 0.05); emboss(b[0], rb - 2, b[2], rb, 0.05)
        elif what == 'glass':
            grad(b[0], ra, b[2], rb, light(col, 0.05), dark(col, 0.7), vertical=False)
            for k in range(6):                       # the gleam of the glass, twice round
                xx = b[0] + w * (k / 6.0 + 0.08)
                rect(xx, ra, xx + w * 0.012, rb, light(col, 0.75), 0.7)
        elif what == 'cap':
            rect(b[0], ra, b[2], rb, col)
            stripes(b[0], ra, b[2], rb, dark(col, 0.7), 4.0, 1.2)
            caps['cap'] = col
        elif what == 'foil':
            grad(b[0], ra, b[2], rb, light(col, 0.3), dark(col, 0.6))
            caps['cap'] = col
    # the discs
    for which in ('top', 'bottom'):
        cx, cy, dr = R[which]
        set_clip((cx - dr - 2, cy - dr - 2, cx + dr + 2, cy + dr + 2))
        top_col = caps.get('cap', zones[-1][3]) if which == 'top' else zones[0][3]
        ellipse(cx, cy, dr + 1.5, dr + 1.5, dark(top_col, 0.9))
        ellipse(cx, cy, dr * 0.85, dr * 0.85, top_col, ring=max(dr * 0.06, 1.0))
        if which == 'top' and kind in ('can', 'tallcan'):       # the ring pull
            ellipse(cx, cy, dr, dr, (0.78, 0.79, 0.8))
            ellipse(cx, cy, dr * 0.88, dr * 0.88, (0.6, 0.61, 0.63), ring=max(dr * 0.05, 1))
            ellipse(cx, cy - dr * 0.35, dr * 0.22, dr * 0.32, (0.7, 0.71, 0.73), ring=max(dr * 0.06, 1))
            ellipse(cx, cy + dr * 0.3, dr * 0.3, dr * 0.18, (0.35, 0.35, 0.37))
        if which == 'bottom' and kind not in ('can', 'tallcan'):
            text(cx, cy - dr * 0.15, dr * 0.25, '%d' % rng.randint(1, 9), dark(top_col, 0.5), wt=0.2, align='c')
    set_clip(R['swatch']); rect(*R['swatch'], caps.get('cap', c2))
    meshes = []
    for lod, (seg, keep) in enumerate(((14, 99), (8, 6), (6, 2))):
        m = Mesh()
        p = prof if lod == 0 else simplify(prof, keep)
        if lod == 2: p = [(max(q[0] for q in prof) * 0.97, 0.0), (max(q[0] for q in prof) * 0.97, H)]
        # (the simplified profiles keep the full band: the rows still go by height)
        add_lathe_by_height(m, R, prof, p, seg)
        if extra and extra.get('trigger') and lod < 2:
            sw = sw_faces(R)
            add_box(m, sw, 0.05, 0.035, 0.07, x=0.0, y=H, z=0.012, faces=sw)
            add_box(m, sw, 0.012, 0.05, 0.015, x=0.0, y=H - 0.035, z=0.04, faces=sw)
        meshes.append(m)
    r = max(q[0] for q in prof)
    finish(name, cat, (2 * r, H + (0.035 if extra and extra.get('trigger') else 0.0), 2 * r), meshes)


def add_lathe_by_height(m, R, full, prof, seg):
    """A lathe of the profile `prof` whose band rows are taken from where its points lie on the
    FULL profile (so a coarse level wears the same picture)."""
    s = [0.0]
    for k in range(len(full) - 1): s.append(s[-1] + math.hypot(full[k + 1][0] - full[k][0], full[k + 1][1] - full[k][1]))
    def s_at(yy):
        for k in range(len(full) - 1):
            if full[k + 1][1] >= yy - 1e-9:
                t = 0.0 if full[k + 1][1] == full[k][1] else (yy - full[k][1]) / (full[k + 1][1] - full[k][1])
                return s[k] + t * (s[k + 1] - s[k])
        return s[-1]
    b = R['band']
    for k in range(len(prof) - 1):
        (r0, y0), (r1, y1) = prof[k], prof[k + 1]
        dr, dy = r1 - r0, y1 - y0
        ln = math.hypot(dr, dy)
        if ln < 1e-6: continue
        nr, ny = dy / ln, -dr / ln
        if prof is full: v0, v1 = band_row(R, full, s[k]), band_row(R, full, s[k + 1])
        else: v0, v1 = band_row(R, full, s_at(y0)), band_row(R, full, s_at(y1))
        for j in range(seg):
            u0, u1 = j / seg, (j + 1) / seg
            P = []; N = []
            for (r, yy, u) in ((r0, y0, u0), (r0, y0, u1), (r1, y1, u1), (r1, y1, u0)):
                ph = TAU * u
                P.append((-r * math.sin(ph), yy, -r * math.cos(ph)))
                N.append((-nr * math.sin(ph), ny, -nr * math.cos(ph)))
            um = TAU * (j + 0.5) / seg
            n = (-nr * math.sin(um), ny, -nr * math.cos(um))
            T = [(b[0] + u0 * (b[2] - b[0]), v0), (b[0] + u1 * (b[2] - b[0]), v0), (b[0] + u1 * (b[2] - b[0]), v1), (b[0] + u0 * (b[2] - b[0]), v1)]
            m.quad(P, n, T, N)
    for which, (r, yy), sgn in (('bottom', prof[0], -1), ('top', prof[-1], 1)):
        if r < 1e-4: continue
        cx, cy, dd = R[which]
        if which == 'top' and prof[-1][0] < full[-1][0] - 1e-6: pass
        for j in range(seg):
            p = []; t = []
            for u in (j / seg, (j + 1) / seg):
                ph = TAU * u
                px, pz = -math.sin(ph), -math.cos(ph)
                p.append((r * px, yy, r * pz))
                t.append((cx + dd * px, cy + dd * pz * (1 if sgn > 0 else -1)))
            m.tri((0, yy, 0), p[0], p[1], (0, sgn, 0), (cx, cy), t[0], t[1])


def make_bag_sku(name, cat, W, H, D, seed):
    """A bag of crisps: a pillow - front and back bulged, crimped flat at top and bottom."""
    rng = np.random.RandomState(seed)
    cell = new_cell(False)
    x0, y0, S = cell
    s = min((S - 15) / (2 * W), (S - 10) / H)
    gx = x0 + (S - 2 * W * s - 5) / 2
    fr = (gx, y0 + 5, gx + W * s, y0 + 5 + H * s)
    bk = (fr[2] + 5, fr[1], fr[2] + 5 + W * s, fr[3])
    nm = NAMES[seed % len(NAMES)]
    c1 = PAL[rng.randint(len(PAL))]; c2 = PAL[rng.randint(len(PAL))]
    sub = ['SEA SALT', 'PAPRIKA', 'CHEESE', 'SOUR CREAM', 'BBQ', 'CHILLI', 'SALTED', 'ONION'][seed % 8]
    set_clip((x0, y0, x0 + S, y0 + S)); rect(x0, y0, x0 + S, y0 + S, c1)
    paint_front(fr, rng, nm, sub, c1, c2, rng.randint(4), pic_balls(rng, [(0.92, 0.78, 0.35), (0.85, 0.65, 0.25)], 14), '%d G' % rng.choice([75, 150, 200]))
    paint_side(bk, rng, c1, panel_on=True, code=True)
    for r in (fr, bk):                   # the crimped seals and the creases
        set_clip(r); h = r[3] - r[1]
        for (a, bb) in ((r[1], r[1] + h * 0.07), (r[3] - h * 0.07, r[3])):
            rect(r[0], a, r[2], bb, dark(c1, 0.7))
            stripes(r[0], a, r[2], bb, light(c1, 0.3), 3.0, 1.0)
            HM[int(a):int(bb), int(r[0]):int(r[2])] += np.float32(0.15) * np.sin(np.arange(int(r[0]), int(r[2]), dtype=np.float32) * 2.1)[None, :]
        ys, xs = np.mgrid[int(r[1]):int(r[3]), int(r[0]):int(r[2])].astype(np.float32)
        HM[int(r[1]):int(r[3]), int(r[0]):int(r[2])] += np.float32(0.06) * np.sin(xs * 0.21 + ys * 0.05) * np.sin(ys * 0.13 - xs * 0.03)
    meshes = []
    for lod, (nu, nv) in enumerate(((6, 7), (3, 4), (0, 0))):
        m = Mesh()
        if nu == 0:
            R = dict(front=fr, back=bk, left=(fr[0], fr[1], fr[0] + 1, fr[3]), right=(fr[2] - 1, fr[1], fr[2], fr[3]),
                     top=(fr[0], fr[1], fr[2], fr[1] + 1), bottom=(fr[0], fr[3] - 1, fr[2], fr[3]))
            add_box(m, R, W * 0.92, H, D * 0.75); meshes.append(m); continue
        for side, r in ((1, fr), (-1, bk)):
            P = np.zeros((nv + 1, nu + 1, 3)); T = np.zeros((nv + 1, nu + 1, 2))
            for j in range(nv + 1):
                v = j / nv
                body = min(max((v - 0.07) / 0.86, 0.0), 1.0)
                for i in range(nu + 1):
                    u = i / nu
                    pinch = 1.0 - 0.06 * math.sin(math.pi * body) ** 0.5
                    xx = (u - 0.5) * W * (pinch if 0 < body < 1 else 1.0)
                    bulge = (math.sin(math.pi * u) ** 0.7) * (math.sin(math.pi * body) ** 0.6)
                    P[j, i] = (xx * side, v * H, side * 0.5 * D * bulge)
                    uu = u if side > 0 else 1.0 - u
                    T[j, i] = (r[0] + uu * (r[2] - r[0]), r[3] - v * (r[3] - r[1]))
            Nn = np.zeros_like(P)
            for j in range(nv):
                for i in range(nu):
                    n = np.cross(P[j, i + 1] - P[j, i], P[j + 1, i] - P[j, i])
                    if np.dot(n, (0, 0, side)) < 0: n = -n
                    for (a, bb) in ((j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)): Nn[a, bb] += n
            Nn /= np.maximum(np.linalg.norm(Nn, axis=2, keepdims=True), 1e-9)
            for j in range(nv):
                for i in range(nu):
                    q = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
                    n = Nn[j, i] + Nn[j + 1, i + 1]
                    m.quad([P[a] for a in q], n, [T[a] for a in q], [Nn[a] for a in q])
        meshes.append(m)
    finish(name, cat, (W, H, D), meshes)


def make_carton_sku(name, cat, W, H, seed, juice):
    """A gable-top carton (milk, juice): a square box with a pitched top and a fin."""
    rng = np.random.RandomState(seed)
    cell = new_cell(False)
    G = W * 0.42                                     # the gable's rise
    R, s = box_net(cell, W, H, W)
    nm = NAMES[seed % len(NAMES)]
    c1 = PAL[rng.randint(len(PAL))] if juice else (0.92, 0.94, 0.96)
    c2 = PAL[rng.choice([1, 4, 9, 13])] if juice else PAL[rng.choice([2, 0, 3, 12])]
    sub = ['ORANGE JUICE', 'APPLE JUICE', 'MULTIFRUIT', 'PINEAPPLE'][seed % 4] if juice else ['WHOLE MILK', 'SEMI SKIMMED', 'CHOCOLATE MILK', 'OAT DRINK'][seed % 4]
    set_clip((cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2])); rect(cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2], c1)
    pic = pic_balls(rng, [c2, light(c2, 0.3)], 6) if juice else None
    paint_front(R['front'], rng, nm, sub, c1, c2, rng.randint(4), pic, '1 L')
    paint_front(R['back'], rng, nm, sub, c1, c2, 0, None, '1 L')
    paint_side(R['left'], rng, c1, True)
    paint_side(R['right'], rng, c1, False, code=True)
    paint_flat(R['top'], c2, rng, nm[:4])
    paint_flat(R['bottom'], light(c1, 0.3))
    set_clip(R['swatch']); rect(*R['swatch'], WHITE)
    meshes = []
    for lod in range(3):
        m = Mesh()
        add_box(m, R, W, H, W)
        if lod < 2:
            a, b, f, k, t = -W / 2, W / 2, W / 2, -W / 2, H
            r = R['top']; ym = (r[1] + r[3]) / 2
            # two roof panels (front half and back half of the top's picture), gable triangles, the fin
            m.quad([(a, t, f), (b, t, f), (b, t + G, 0.0), (a, t + G, 0.0)], (0, 0.75, 0.66), [(r[0], r[3]), (r[2], r[3]), (r[2], ym), (r[0], ym)])
            m.quad([(b, t, k), (a, t, k), (a, t + G, 0.0), (b, t + G, 0.0)], (0, 0.75, -0.66), [(r[2], r[1]), (r[0], r[1]), (r[0], ym), (r[2], ym)])
            for xx, sg, rr in ((a, -1, R['left']), (b, 1, R['right'])):
                m.tri((xx, t, f), (xx, t, k), (xx, t + G, 0.0), (sg, 0, 0), (rr[2] if sg < 0 else rr[0], rr[1]), (rr[0] if sg < 0 else rr[2], rr[1]), ((rr[0] + rr[2]) / 2, rr[1] + 1))
            sw = R['swatch']; c = ((sw[0] + sw[2]) / 2, (sw[1] + sw[3]) / 2)
            m.quad([(a, t + G, 0.0), (b, t + G, 0.0), (b, t + G + 0.012, 0.0), (a, t + G + 0.012, 0.0)], (0, 0, 1), [c, c, c, c])
            m.quad([(b, t + G, 0.0), (a, t + G, 0.0), (a, t + G + 0.012, 0.0), (b, t + G + 0.012, 0.0)], (0, 0, -1), [c, c, c, c])
        meshes.append(m)
    finish(name, cat, (W, H + G + 0.012, W), meshes)


def make_jug_sku(name, cat, W, H, D, seed, kind):
    """A jug (motor oil, detergent): a box body with a handle on top at the back and a cap at the front."""
    rng = np.random.RandomState(seed)
    cell = new_cell(False)
    R, s = box_net(cell, W, H, D)
    nm = NAMES[seed % len(NAMES)]
    if kind == 'oil':
        c1 = [(0.08, 0.08, 0.09), (0.05, 0.2, 0.55), (0.85, 0.65, 0.1), (0.6, 0.05, 0.05), (0.85, 0.85, 0.85)][seed % 5]
        c2 = PAL[rng.choice([0, 1, 7, 12])]
        sub = ['5W-30', '10W-40', '0W-20', '15W-40', 'COOLANT', 'WASHER FLUID'][seed % 6]
        vol = '%d L' % rng.choice([1, 4, 5])
    else:
        c1 = PAL[rng.choice([2, 3, 6, 12, 13, 11])]; c2 = PAL[rng.choice([0, 1, 7, 4])]
        sub = ['DISH SOAP', 'FLOOR CLEAN', 'BLEACH', 'LAUNDRY', 'ALL PURPOSE'][seed % 5]
        vol = '%d ML' % rng.choice([750, 1000, 1500])
    set_clip((cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2])); rect(cell[0], cell[1], cell[0] + cell[2], cell[1] + cell[2], c1)
    for f in ('front', 'back'):
        r = R[f]; set_clip(r); w = r[2] - r[0]; h = r[3] - r[1]
        rect(*r, c1)
        rr = (r[0] + w * 0.08, r[1] + h * 0.18, r[2] - w * 0.08, r[3] - h * 0.1)
        if f == 'front':
            paint_front(rr, rng, nm, sub, c2 if kind == 'oil' else WHITE, c1, rng.randint(4), None, vol)
        else:
            set_clip(rr); rect(*rr, WHITE); panel(rr[0] + 3, rr[1] + 3, rr[2] - 3, rr[1] + (rr[3] - rr[1]) * 0.5, rng)
            para(rr[0] + 4, rr[1] + (rr[3] - rr[1]) * 0.55, rr[2] - 4, rr[3] - (rr[3] - rr[1]) * 0.25, max(h * 0.03, 2.5), BLACK, rng)
            barcode(rr[0] + (rr[2] - rr[0]) * 0.55, rr[3] - (rr[3] - rr[1]) * 0.2, (rr[2] - rr[0]) * 0.35, (rr[3] - rr[1]) * 0.16, rng)
        # the level window down one side
        set_clip(r); rect(r[0] + w * 0.02, r[1] + h * 0.2, r[0] + w * 0.05, r[3] - h * 0.1, light(c1, 0.5), 0.7)
    for f in ('left', 'right', 'top', 'bottom'):
        r = R[f]; set_clip(r); rect(*r, c1)
        HM[int(r[1]):int(r[3]), int(r[0]):int(r[2])] += np.float32(0.02)
    set_clip(R['swatch']); rect(*R['swatch'], c2 if kind == 'oil' else WHITE)
    sw = sw_faces(R)
    meshes = []
    for lod in range(3):
        m = Mesh()
        add_box(m, R, W, H, D)
        if lod < 2:
            hz = -D / 2 + D * 0.3
            add_box(m, sw, W * 0.3, H * 0.13, 0.02, y=H, z=hz - 0.04, faces=sw)                 # the handle's posts
            add_box(m, sw, W * 0.3, 0.02, D * 0.32, y=H + H * 0.13, z=hz, faces=sw)              # its grip
            add_box(m, sw, W * 0.3, H * 0.13, 0.02, y=H, z=hz + 0.04 + D * 0.05, faces=sw)
        if lod == 0:
            prof = [(0.017, 0.0), (0.017, 0.03)]
            cm = Mesh()
            RR = dict(band=(sw['front'][0], sw['front'][1], sw['front'][2], sw['front'][3]), top=(sw['front'][0] + 0.01, sw['front'][1] + 0.01, 0.005), bottom=(sw['front'][0] + 0.01, sw['front'][1] + 0.01, 0.005), length=0.03)
            for k in range(8):
                a0, a1 = TAU * k / 8, TAU * (k + 1) / 8
                c = sw['front'][0] + 0.01, sw['front'][1] + 0.01
                zc = D / 2 - 0.035
                P = [(0.017 * math.sin(a0), H, zc + 0.017 * math.cos(a0)), (0.017 * math.sin(a1), H, zc + 0.017 * math.cos(a1)),
                     (0.017 * math.sin(a1), H + 0.03, zc + 0.017 * math.cos(a1)), (0.017 * math.sin(a0), H + 0.03, zc + 0.017 * math.cos(a0))]
                n = (math.sin((a0 + a1) / 2), 0, math.cos((a0 + a1) / 2))
                m.quad(P, n, [c, c, c, c])
                m.tri((0, H + 0.03, zc), P[2], P[3], (0, 1, 0), c, c, c)
        meshes.append(m)
    finish(name, cat, (W, H + H * 0.13 + 0.02, D), meshes)


# ------------------------------------------------------------------ the catalogue
def build():
    seed = 1000
    def nxt():
        nonlocal seed
        seed += 7
        return seed
    # boxes: cereal (three sizes), snacks, small boxed goods, drink cases, shipping cartons
    for k, (W, H, D) in enumerate([(0.19, 0.28, 0.065)] * 4 + [(0.22, 0.32, 0.075)] * 4 + [(0.15, 0.22, 0.055)] * 4):
        make_box_sku('cereal_%02d' % k, 'cereal', W, H, D, nxt(), 'cereal')
    for k, (W, H, D) in enumerate([(0.16, 0.12, 0.06), (0.21, 0.08, 0.06), (0.12, 0.2, 0.05), (0.18, 0.14, 0.07), (0.1, 0.16, 0.04), (0.2, 0.1, 0.1)]):
        make_box_sku('snack_%02d' % k, 'snack', W, H, D, nxt(), 'snack')
    for k, (W, H, D) in enumerate([(0.12, 0.07, 0.07), (0.09, 0.14, 0.05), (0.16, 0.06, 0.09), (0.1, 0.1, 0.1), (0.14, 0.09, 0.04),
                                   (0.08, 0.12, 0.03), (0.07, 0.05, 0.02), (0.11, 0.17, 0.06), (0.13, 0.05, 0.13), (0.06, 0.1, 0.06)]):
        make_box_sku('boxed_%02d' % k, 'small', W, H, D, nxt(), 'small')
    for k in range(4): make_box_sku('case_%02d' % k, 'case', 0.40, 0.13, 0.27, nxt(), 'case')
    for k, (W, H, D) in enumerate([(0.6, 0.4, 0.4), (0.5, 0.3, 0.35), (0.4, 0.4, 0.4), (0.7, 0.35, 0.45)]):
        make_box_sku('ship_%02d' % k, 'ship', W, H, D, nxt(), 'ship')
    for k in range(8): make_box_sku('candy_%02d' % k, 'candy', 0.15 if k % 2 else 0.11, 0.028, 0.055 if k < 6 else 0.07, nxt(), 'candy')
    for k in range(6): make_box_sku('pack_%02d' % k, 'pack', 0.055, 0.087, 0.022, nxt(), 'cig')
    for k in range(5): make_box_sku('book_%02d' % k, 'book', 0.13, 0.2, 0.025 + 0.006 * k, nxt(), 'book')
    for k in range(6): make_box_sku('mag_%02d' % k, 'mag', 0.21, 0.28, 0.006, nxt(), 'mag')
    # turned shapes: (r, y) from the bottom up
    def can_prof(r, h): return [(r * 0.8, 0.0), (r, 0.008), (r, h - 0.012), (r * 0.86, h - 0.002), (r * 0.86, h)]
    for k in range(8):
        lc = PAL[[0, 2, 3, 4, 5, 6, 7, 14][k]]
        make_lathe_sku('can_%02d' % k, 'can', can_prof(0.033, 0.122), [(0.0, 0.08, 'solid', (0.75, 0.76, 0.78)), (0.08, 0.92, 'label', lc), (0.92, 1.0, 'solid', (0.75, 0.76, 0.78))], nxt(), 'can',
                       dict(sub=['COLA', 'LEMON', 'ORANGE', 'GINGER', 'CHERRY', 'TONIC', 'LIME', 'ICED TEA'][k], vol='330 ML'))
    for k in range(4):
        make_lathe_sku('tallcan_%02d' % k, 'can', can_prof(0.029, 0.168), [(0.0, 0.06, 'solid', (0.75, 0.76, 0.78)), (0.06, 0.94, 'label', PAL[[10, 2, 14, 3][k]]), (0.94, 1.0, 'solid', (0.75, 0.76, 0.78))], nxt(), 'tallcan',
                       dict(sub=['ENERGY', 'ZERO', 'BERRY', 'MANGO'][k], vol='500 ML'))
    for k in range(6):
        make_lathe_sku('tin_%02d' % k, 'tin', [(0.036, 0.0), (0.038, 0.004), (0.038, 0.106), (0.036, 0.11)], [(0.0, 0.05, 'ribs', (0.7, 0.7, 0.72)), (0.05, 0.95, 'label', PAL[[0, 3, 1, 4, 8, 13][k]]), (0.95, 1.0, 'ribs', (0.7, 0.7, 0.72))], nxt(), 'tin',
                       dict(sub=['TOMATOES', 'BEANS', 'SWEETCORN', 'SOUP', 'PEACHES', 'PEAS'][k], vol='400 G', pic=lambda rr: pic_balls(rr, [(0.8, 0.2, 0.1), (0.9, 0.7, 0.2)], 6)))
    for k in range(6):
        r = [0.04, 0.035, 0.045, 0.03, 0.038, 0.042][k]; h = [0.11, 0.09, 0.13, 0.08, 0.1, 0.12][k]
        food = [(0.6, 0.1, 0.1), (0.85, 0.6, 0.1), (0.35, 0.45, 0.12), (0.5, 0.25, 0.08), (0.85, 0.3, 0.2), (0.45, 0.15, 0.3)][k]
        make_lathe_sku('jar_%02d' % k, 'jar', [(r * 0.85, 0.0), (r, 0.006), (r, h * 0.78), (r * 0.9, h * 0.84), (r * 0.92, h * 0.86), (r * 0.92, h), (r * 0.6, h + 0.002)],
                       [(0.0, 0.2, 'glass', food), (0.2, 0.7, 'label', PAL[[11, 7, 3, 8, 11, 9][k]]), (0.7, 0.84, 'glass', food), (0.84, 1.0, 'cap', PAL[[0, 3, 10, 1, 2, 14][k]])], nxt(), 'jar',
                       dict(sub=['JAM', 'HONEY', 'PICKLES', 'PEANUT', 'SALSA', 'BERRIES'][k], vol='%d G' % [340, 250, 500, 200, 300, 400][k]))
    def bottle_prof(r, h, neck=0.38, nr=0.33):
        return [(r * 0.8, 0.0), (r, 0.01), (r, h * (1 - neck)), (r * 0.8, h * (1 - neck * 0.55)), (r * nr * 1.2, h * 0.9), (r * nr, h * 0.92), (r * nr, h)]
    for k in range(4):
        make_lathe_sku('water_%02d' % k, 'water', bottle_prof(0.033, 0.21 if k < 2 else 0.3), [(0.0, 0.3, 'glass', (0.72, 0.84, 0.92)), (0.3, 0.58, 'label', PAL[[12, 2, 6, 11][k]]), (0.58, 0.9, 'glass', (0.72, 0.84, 0.92)), (0.9, 1.0, 'cap', PAL[[2, 12, 3, 0][k]])], nxt(), 'bottle',
                       dict(sub=['STILL WATER', 'SPARKLING', 'MINERAL', 'SPRING WATER'][k], vol='500 ML' if k < 2 else '1.5 L'))
    for k in range(6):
        liq = [(0.2, 0.08, 0.03), (0.95, 0.6, 0.1), (0.85, 0.82, 0.3), (0.65, 0.1, 0.15), (0.2, 0.08, 0.03), (0.4, 0.7, 0.2)][k]
        make_lathe_sku('soda_%02d' % k, 'soda', bottle_prof(0.034 if k < 3 else 0.045, 0.23 if k < 3 else 0.32), [(0.0, 0.32, 'glass', liq), (0.32, 0.6, 'label', PAL[[0, 4, 7, 14, 0, 13][k]]), (0.6, 0.9, 'glass', liq), (0.9, 1.0, 'cap', PAL[[0, 1, 3, 2, 10, 3][k]])], nxt(), 'bottle',
                       dict(sub=['COLA', 'ORANGE', 'LEMONADE', 'CHERRY', 'COLA ZERO', 'APPLE'][k], vol='500 ML' if k < 3 else '1.5 L'))
    for k in range(3):
        make_lathe_sku('oil_%02d' % k, 'oil', bottle_prof(0.038, 0.29, 0.3, 0.38), [(0.0, 0.25, 'glass', (0.85, 0.7, 0.15)), (0.25, 0.62, 'label', PAL[[3, 7, 11][k]]), (0.62, 0.9, 'glass', (0.85, 0.7, 0.15)), (0.9, 1.0, 'cap', PAL[[3, 0, 1][k]])], nxt(), 'bottle',
                       dict(sub=['OLIVE OIL', 'SUNFLOWER OIL', 'VINEGAR'][k], vol='750 ML'))
    for k in range(4):
        gl = [(0.12, 0.22, 0.1), (0.1, 0.16, 0.08), (0.75, 0.7, 0.45), (0.25, 0.1, 0.06)][k]
        make_lathe_sku('wine_%02d' % k, 'wine', [(0.034, 0.0), (0.037, 0.008), (0.037, 0.2), (0.03, 0.235), (0.014, 0.26), (0.014, 0.3), (0.015, 0.302), (0.015, 0.31)],
                       [(0.0, 0.3, 'glass', gl), (0.3, 0.58, 'label', [CREAM, WHITE, (0.1, 0.1, 0.12), CREAM][k]), (0.58, 0.8, 'glass', gl), (0.8, 1.0, 'foil', [(0.55, 0.08, 0.1), (0.75, 0.62, 0.2), (0.1, 0.1, 0.1), (0.85, 0.85, 0.85)][k])], nxt(), 'wine',
                       dict(sub=['RED WINE', 'WHITE WINE', 'ROSE', 'RESERVE'][k], vol='750 ML'))
    for k in range(4):
        make_lathe_sku('tub_%02d' % k, 'tub', [(0.055, 0.0), (0.065, 0.075), (0.068, 0.078), (0.068, 0.088), (0.055, 0.09)], [(0.0, 0.85, 'label', PAL[[9, 8, 11, 12][k]]), (0.85, 1.0, 'cap', PAL[[9, 8, 1, 2][k]])], nxt(), 'tub',
                       dict(sub=['VANILLA', 'COCOA', 'STRAWBERRY', 'MINT'][k], vol='500 ML'))
    for k in range(3):
        make_lathe_sku('spray_%02d' % k, 'spray', [(0.032, 0.0), (0.036, 0.01), (0.036, 0.17), (0.022, 0.2), (0.014, 0.21), (0.014, 0.225)],
                       [(0.0, 0.08, 'solid', PAL[[12, 3, 2][k]]), (0.08, 0.72, 'label', PAL[[12, 3, 2][k]]), (0.72, 0.92, 'solid', PAL[[12, 3, 2][k]]), (0.92, 1.0, 'cap', WHITE)], nxt(), 'spray',
                       dict(sub=['GLASS CLEAN', 'KITCHEN', 'BATHROOM'][k], vol='500 ML', trigger=True))
    for k in range(6): make_carton_sku('carton_%02d' % k, 'carton', 0.07, 0.19, nxt(), k >= 3)
    for k in range(8): make_bag_sku('chips_%02d' % k, 'chips', 0.18 if k < 4 else 0.24, 0.26 if k < 4 else 0.34, 0.07 if k < 4 else 0.09, nxt())
    for k in range(4): make_jug_sku('motoroil_%02d' % k, 'motoroil', 0.14 if k % 2 else 0.2, 0.2 if k % 2 else 0.28, 0.07 if k % 2 else 0.1, nxt(), 'oil')
    for k in range(3): make_jug_sku('cleaner_%02d' % k, 'cleaner', 0.12, 0.24, 0.07, nxt(), 'clean')


def save():
    # the atlas and its height (rows flipped: Blender's pictures start at the bottom)
    for nm, arr in (('goods.png', np.concatenate([np.clip(C, 0, 1), np.ones((A, A, 1), np.float32)], 2)),
                    ('goods_h.png', np.repeat(np.concatenate([np.clip(HM, 0, 1)[:, :, None], np.ones((A, A, 1), np.float32)], 2)[:, :, [0, 0, 0, 1]], 1, 2))):
        img = bpy.data.images.new(nm, width=A, height=A, alpha=False)
        img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).reshape(-1))
        img.filepath_raw = os.path.join(OUT, nm); img.file_format = 'PNG'; img.save()
        bpy.data.images.remove(img)
    blob = bytearray(b'GDK1' + struct.pack('<II', len(SKUS), A))
    tri = [0, 0, 0]
    for name, cat, size, levels in SKUS:
        blob += struct.pack('<24s16s3fI', name.encode(), cat.encode(), *size, len(levels))
        for k, (v, i) in enumerate(levels):
            blob += struct.pack('<II', len(v), len(i)) + v.astype('<f4').tobytes() + i.astype('<u4').tobytes()
            tri[k] += len(i) // 3
    with open(os.path.join(OUT, 'goods.kit'), 'wb') as f: f.write(blob)
    cats = {}
    for s in SKUS: cats[s[1]] = cats.get(s[1], 0) + 1
    log('%d SKUs in %d blocks of 512; triangles per level (all SKUs): %s; kit %.2f MB' % (len(SKUS), BLOCK, tri, len(blob) / 1e6))
    log('categories:', cats)


bpy.ops.wm.read_factory_settings(use_empty=True)
HM[:] = 0.5 + 0.01 * np.sin(np.arange(A, dtype=np.float32) * 1.7)[None, :] * np.sin(np.arange(A, dtype=np.float32) * 2.3)[:, None]
build()
save()
