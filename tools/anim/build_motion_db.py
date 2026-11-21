# Builds the motion-matching database of the soldier (assets/character/motion.qmm) from the
# 100STYLE motion capture set (Ian Mason et al., CC BY 4.0, https://www.ianxmason.com/100style/).
#
#   python build_motion_db.py <100STYLE dir> <soldier_fp.qchar> <out.qmm> [Style ...]
#
# (any Python with numpy: Blender's will do). Per style: the forward / backward / sideways
# walks and runs, the idle and the transitions, cut as Frame_Cuts.csv says, at 30 fps.
#
# Retargeting: the capture skeleton stands in a T pose, the soldier in an A pose with other
# proportions. Per bone the capture's world rotation is taken relative to the pose in which
# its bones point the way the soldier's do at rest - so "the soldier's rest pose" maps to
# identity, and what is stored per bone is the rotation that takes it from rest to the pose,
# in the space of the character's root (x right, y up, -z forward). The root is the hips'
# path on the ground, smoothed, facing the smoothed hip direction.
#
# File (little endian):
#   "QMM1", u32 frames, u32 bones, u32 features, f32 fps, u32 ranges
#   bones x char[32]                      soldier bone names
#   ranges x (u32 first, u32 end, u32 tag)   end exclusive; tag = index of the style given
#   features x f32 offset, features x f32 scale     stored = (raw - offset) / scale
#   frames x features x f32               matching features, normalised and weighted
#   frames x 8 f32                        root velocity x,z (m/s, root space), root turn rate (rad/s),
#                                         hip position x,y,z (root space, soldier file units),
#                                         contact left, right (0/1)
#   frames x bones x 4 i16                rotation from rest (x,y,z,w) * 32767
import csv
import struct
import sys
import numpy as np

FPS = 30.0
STEP = 2                                    # the capture is 60 fps
C = np.diag([-1.0, 1.0, -1.0])              # capture (+z forward, +x left) -> soldier (-z forward, +x right)

# soldier bone <- capture joint, and the child that gives the bone its direction (soldier, capture)
MAP = [
    ("Hips", "Hips", None, None), ("Spine", "Chest", None, None), ("Spine1", "Chest2", None, None),
    ("Spine2", "Chest4", None, None), ("Neck", "Neck", None, None), ("Head", "Head", None, None),
]
for S in ("Left", "Right"):
    MAP += [
        (S + "Shoulder", S + "Collar", S + "Arm", S + "Shoulder"),
        (S + "Arm", S + "Shoulder", S + "ForeArm", S + "Elbow"),
        (S + "ForeArm", S + "Elbow", S + "Hand", S + "Wrist"),
        (S + "Hand", S + "Wrist", S + "HandMiddle1", "END"),
        (S + "UpLeg", S + "Hip", S + "Leg", S + "Knee"),
        (S + "Leg", S + "Knee", S + "Foot", S + "Ankle"),
        (S + "Foot", S + "Ankle", S + "ToeBase", S + "Toe"),
        (S + "ToeBase", S + "Toe", "TAIL", "END"),
    ]


def read_qchar(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"QCH1"
        nb, _, _ = struct.unpack("<3I", f.read(12))
        bones = []
        for _ in range(nb):
            name = f.read(32).split(b"\0")[0].decode()
            if ":" in name:
                name = name.split(":", 1)[1]
            parent, = struct.unpack("<i", f.read(4))
            head = np.array(struct.unpack("<3f", f.read(12)), dtype=np.float64)
            tail = np.array(struct.unpack("<3f", f.read(12)), dtype=np.float64)
            bones.append((name, parent, head, tail))
    return bones


def read_bvh(path):
    names, parents, offsets, chans, ends = [], [], [], [], {}
    stack = []
    with open(path, "r") as f:
        lines = f.read().split("\n")
    i = 0
    in_end = False
    while True:
        t = lines[i].split()
        i += 1
        if not t:
            continue
        if t[0] in ("ROOT", "JOINT"):
            names.append(t[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            chans.append(None)
            stack.append(len(names) - 1)
        elif t[0] == "End":
            in_end = True
        elif t[0] == "OFFSET":
            v = np.array([float(x) for x in t[1:4]])
            if in_end:
                ends[stack[-1]] = v
            else:
                offsets[stack[-1]] = v
        elif t[0] == "CHANNELS":
            chans[stack[-1]] = t[2:]
        elif t[0] == "}":
            if in_end:
                in_end = False
            else:
                stack.pop()
        elif t[0] == "MOTION":
            break
    nframes = int(lines[i].split()[1])
    i += 2
    data = np.array([[float(x) for x in ln.split()] for ln in lines[i:i + nframes] if ln.strip()])
    return names, parents, offsets, chans, ends, data


def axis_rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    if axis == "X":
        R[..., 0, 0] = 1; R[..., 1, 1] = c; R[..., 1, 2] = -s; R[..., 2, 1] = s; R[..., 2, 2] = c
    elif axis == "Y":
        R[..., 1, 1] = 1; R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 2, 0] = -s; R[..., 2, 2] = c
    else:
        R[..., 2, 2] = 1; R[..., 0, 0] = c; R[..., 0, 1] = -s; R[..., 1, 0] = s; R[..., 1, 1] = c
    return R


def rot_from_to(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-3:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def gauss(x, sigma):
    r = int(sigma * 3)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.zeros_like(x)
    for j, w in enumerate(k):
        out += w * pad[j:j + len(x)]
    return out


def mat_to_quat(R):
    # (N,3,3) -> (N,4) x,y,z,w
    m = R
    q = np.zeros(m.shape[:-2] + (4,))
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = np.sqrt(np.maximum(0.0, 1.0 + tr)) / 2
    x = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q[..., 0], q[..., 1], q[..., 2], q[..., 3] = x, y, z, w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def main():
    src_dir, qchar, out = sys.argv[1], sys.argv[2], sys.argv[3]
    styles = sys.argv[4:] or ["Neutral", "Crouched"]
    bones = read_qchar(qchar)
    bidx = {b[0]: i for i, b in enumerate(bones)}
    cuts = {}
    with open(src_dir + "/Frame_Cuts.csv", newline="") as f:
        for row in csv.DictReader(f):
            cuts[row["STYLE_NAME"]] = row

    tgt_leg = (np.linalg.norm(bones[bidx["LeftLeg"]][2] - bones[bidx["LeftUpLeg"]][2]) +
               np.linalg.norm(bones[bidx["LeftFoot"]][2] - bones[bidx["LeftLeg"]][2]))
    rest_ankle = bones[bidx["LeftFoot"]][2][1]
    rest_hips = bones[bidx["Hips"]][2]

    all_feat, all_hdr, all_quat, ranges = [], [], [], []
    total = 0
    for tag, style in enumerate(styles):
        for kind in ("FW", "FR", "BW", "BR", "SW", "SR", "ID", "TR1", "TR2", "TR3"):
            c0, c1 = cuts[style].get(kind + "_START", "N/A"), cuts[style].get(kind + "_STOP", "N/A")
            if c0 == "N/A" or c1 == "N/A":
                continue
            try:
                names, parents, offsets, chans, ends, data = read_bvh("%s/%s/%s_%s.bvh" % (src_dir, style, style, kind))
            except FileNotFoundError:
                continue
            data = data[int(c0):int(c1):STEP]
            n = len(data)
            jidx = {nm: i for i, nm in enumerate(names)}
            # capture FK
            G = [None] * len(names)
            P = [None] * len(names)
            col = 0
            for j, nm in enumerate(names):
                ch = chans[j]
                R = np.tile(np.eye(3), (n, 1, 1))
                pos = None
                for k, cname in enumerate(ch):
                    v = data[:, col + k]
                    if cname.endswith("position"):
                        if pos is None:
                            pos = np.zeros((n, 3))
                        pos[:, "XYZ".index(cname[0])] = v
                    else:
                        R = R @ axis_rot(cname[0], v)
                col += len(ch)
                if parents[j] < 0:
                    G[j] = R
                    P[j] = pos
                else:
                    G[j] = G[parents[j]] @ R
                    P[j] = P[parents[j]] + np.einsum("nij,j->ni", G[parents[j]], offsets[j])
            src_leg = np.linalg.norm(offsets[jidx["LeftKnee"]]) + np.linalg.norm(offsets[jidx["LeftAnkle"]])
            scale = tgt_leg / (src_leg * 0.01) * 0.01           # cm -> soldier units, leg for leg
            # rotations from rest, soldier frame, world
            D = {}
            for tb, sj, tchild, schild in MAP:
                j = jidx[sj]
                Gs = C @ G[j] @ C
                if tchild is None:
                    Gref = np.eye(3)
                else:
                    b = bones[bidx[tb]]
                    tdir = (b[3] - b[2]) if tchild == "TAIL" else (bones[bidx[tchild]][2] - b[2])
                    sdir = C @ (ends[j] if schild == "END" else offsets[jidx[schild]])
                    Gref = rot_from_to(sdir, tdir)
                D[tb] = Gs @ Gref.T
            hips_w = (P[jidx["Hips"]] @ C.T) * scale

            def soldier_fk(D, hips_w):
                Rb = [None] * len(bones)
                Pb = [None] * len(bones)
                for i, (nm, par, head, tail) in enumerate(bones):
                    if par < 0:
                        Rb[i] = np.tile(np.eye(3), (n, 1, 1))
                        Pb[i] = np.zeros((n, 3))
                        continue
                    Rb[i] = D[nm] if nm in D else Rb[par]
                    if nm == "Hips":
                        Pb[i] = hips_w.copy()
                    else:
                        Pb[i] = Pb[par] + np.einsum("nij,j->ni", Rb[par], head - bones[par][2])
                return Rb, Pb

            Rb, Pb = soldier_fk(D, hips_w)
            la, ra = bidx["LeftFoot"], bidx["RightFoot"]
            # the standing foot's ankle at its rest height
            low = np.minimum(Pb[la][:, 1], Pb[ra][:, 1])
            hips_w[:, 1] += rest_ankle - np.median(low)
            Rb, Pb = soldier_fk(D, hips_w)

            # the root: hips' path on the ground, smoothed; facing the smoothed hip direction
            sig = 0.20 * FPS
            rp = gauss(hips_w * np.array([1.0, 0.0, 1.0]), sig)
            fw = np.einsum("nij,j->ni", D["Hips"], np.array([0.0, 0.0, -1.0])) * np.array([1.0, 0.0, 1.0])
            fw = gauss(fw, sig)
            fw /= np.linalg.norm(fw, axis=1, keepdims=True)
            # root rotation: -z -> fw, y up.  columns: right, up, back
            Rr = np.zeros((n, 3, 3))
            Rr[:, :, 2] = -fw
            Rr[:, 1, 1] = 1.0
            Rr[:, :, 0] = np.cross(Rr[:, :, 1], Rr[:, :, 2])
            RrT = np.transpose(Rr, (0, 2, 1))
            to_root = lambda v: np.einsum("nij,nj->ni", RrT, v)
            ddt = lambda x: np.gradient(x, 1.0 / FPS, axis=0)
            yaw = np.unwrap(np.arctan2(fw[:, 0], -fw[:, 2]))
            hdr = np.zeros((n, 8), dtype=np.float32)
            rv = to_root(ddt(rp))
            hdr[:, 0], hdr[:, 1] = rv[:, 0], rv[:, 2]
            hdr[:, 2] = ddt(yaw)
            hl = to_root(hips_w - rp)
            hdr[:, 3:6] = hl - np.array([rest_hips[0], 0.0, rest_hips[2]])       # x,z as offsets from rest; y absolute
            fvl, fvr = ddt(Pb[la]), ddt(Pb[ra])
            for k, (fv, a) in enumerate(((fvl, la), (fvr, ra))):
                con = (np.linalg.norm(fv, axis=1) < 0.35) & (Pb[a][:, 1] < rest_ankle + 0.07)
                con = gauss(con.astype(np.float64), 1.2) > 0.5
                hdr[:, 6 + k] = con
            feat = np.zeros((n, 27), dtype=np.float64)
            feat[:, 0:3] = to_root(Pb[la] - rp)
            feat[:, 3:6] = to_root(Pb[ra] - rp)
            feat[:, 6:9] = to_root(fvl)
            feat[:, 9:12] = to_root(fvr)
            feat[:, 12:15] = to_root(ddt(hips_w))
            for k, dfr in enumerate((10, 20, 30)):
                ix = np.minimum(np.arange(n) + dfr, n - 1)
                tp = to_root(rp[ix] - rp)
                td = to_root(fw[ix])
                feat[:, 15 + 2 * k], feat[:, 16 + 2 * k] = tp[:, 0], tp[:, 2]
                feat[:, 21 + 2 * k], feat[:, 22 + 2 * k] = td[:, 0], td[:, 2]
            quat = np.zeros((n, len(MAP), 4))
            for bi, (tb, _, _, _) in enumerate(MAP):
                q = mat_to_quat(RrT @ D[tb])
                for f in range(1, n):                              # keep to one hemisphere through time
                    if np.dot(q[f], q[f - 1]) < 0.0:
                        q[f] = -q[f]
                quat[:, bi] = q
            all_feat.append(feat)
            all_hdr.append(hdr)
            all_quat.append(quat)
            ranges.append((total, total + n, tag))
            total += n
            sp = np.linalg.norm(rv[:, [0, 2]], axis=1)
            print("%-10s %-4s %6d frames  speed mean %.2f max %.2f m/s  contact %.0f%%/%.0f%%" %
                  (style, kind, n, sp.mean(), sp.max(), 100 * hdr[:, 6].mean(), 100 * hdr[:, 7].mean()))

    feat = np.concatenate(all_feat)
    hdr = np.concatenate(all_hdr)
    quat = np.concatenate(all_quat)
    groups = [(0, 6, 0.75), (6, 12, 1.0), (12, 15, 1.0), (15, 21, 1.0), (21, 27, 1.5)]
    offset = feat.mean(axis=0)
    scale = np.zeros(27)
    for a, b, w in groups:
        scale[a:b] = feat[:, a:b].std(axis=0).mean() / w
    featn = ((feat - offset) / scale).astype(np.float32)
    with open(out, "wb") as f:
        f.write(b"QMM1")
        f.write(struct.pack("<IIIfI", total, len(MAP), 27, FPS, len(ranges)))
        for tb, _, _, _ in MAP:
            f.write(tb.encode().ljust(32, b"\0"))
        for r in ranges:
            f.write(struct.pack("<III", *r))
        f.write(offset.astype(np.float32).tobytes())
        f.write(scale.astype(np.float32).tobytes())
        f.write(featn.tobytes())
        f.write(hdr.astype(np.float32).tobytes())
        f.write(np.round(quat * 32767.0).astype(np.int16).tobytes())
    print("wrote %s: %d frames, %d bones, %d ranges" % (out, total, len(MAP), len(ranges)))


main()
