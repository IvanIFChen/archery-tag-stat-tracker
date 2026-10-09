"""Lens correction for the two stitched GoPros.

Two kinds of parameters:

* EQUIPMENT (fixed while the cameras stay the same): the GoPro lens model,
  radial term k1, focal lengths and lens-center height.
* A per-video STITCH profile (data/stitch/<video>.json): where the stitcher cut
  the two cameras (the seam, auto-detected with `find_seam`), and the
  alignment measured on that video: far-baseline points and the pillar warp.

Each half of the frame gets:
  1. radial correction (k1) with the lens center placed ON the seam, so the
     seam column stays a straight vertical line in both halves and they still
     meet with no gap;
  2. a zoom so the seam column fills the full height (no black in the middle);
  3. a projective tidy-up (shears + a vertical stretch pivoting on the seam)
     so the far baseline, two kinked segments in the source, becomes one
     straight line across the seam;
  4. a local "seam warp" on the right half near the seam (fading out over
     `fade_px`) that lines up the far center pillar, which the two cameras see
     with parallax.

Values were chosen by eye from rendered candidates (see README).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EQUIPMENT = {
    "k1": 0.2,  # radial term (positive: the stitcher over-dewarped each half)
    "k2": 0.0,  # 4th-order radial term: acts mostly far from the lens center (bottom corners)
    "k3": 0.0,  # 6th-order radial term
    "f": {"left": 1822.0, "right": 1668.0},  # px, at the 3490x1400 stitched scale
    "cy": 190.0,  # lens-center height (on the far baseline row)
    "zoom": 1.0,  # extra zoom on top of "seam column fills the height"
}
# Stitch-profile keys and their defaults (see data/stitch/*.json). A profile may
# also carry an "equipment" dict that overrides EQUIPMENT (set by the lens tool).
STITCH_DEFAULTS = {
    "shear": 0.6,  # fraction of each half's far-baseline slope removed
    "align": True,  # stretch halves so the baselines meet at the seam
    "collinear": True,  # shear halves so both baseline segments lie on one line
    "camera": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},  # deg; mirrored on the right half
    # Corner pin: move a half's output corners by (dx, dy) px; the half is warped
    # by the perspective transform that takes its rectangle to the moved corners.
    "corner_pin": {s: {c: [0.0, 0.0] for c in ("top_left", "top_right", "bottom_left", "bottom_right")}
                   for s in ("left", "right")},
    # Parallelogram: per half, 0 = off .. 1 = make the half-court outline (far
    # baseline / side wall / near wall / zone line) have parallel opposite sides,
    # by sending both of its vanishing points to infinity. Needs profile "lines".
    "parallelogram": {"left": 0.0, "right": 0.0},
    # Quad pin: per half, where the 4 half-court corners (far x side, far x zone,
    # near x zone, near x side; full-frame output px) should end up. null = leave.
    "quad": {"left": None, "right": None},
}


def _tls_line(p):
    """Best-fit line through points as (a, b, c) with a x + b y + c = 0, (a, b) unit."""
    p = np.asarray(p, float)
    m = p.mean(0)
    d = np.linalg.svd(p - m, full_matrices=False)[2][0]
    n = np.array([-d[1], d[0]])
    return np.array([n[0], n[1], -n @ m])


def quad_corners(st: dict, h, side: str):
    """Half-local corrected corners of the half-court outline, in the order
    far x side, far x zone, near x zone, near x side."""
    loc = lambda pts: h.points(np.asarray(pts, float) - [h.x0, 0])
    L = st["lines"]
    far, wall = _tls_line(loc(st["baseline"][side])), _tls_line(loc(L["side_wall"][side]))
    near, zone = _tls_line(loc(L["near_wall"][side])), _tls_line(loc(L["zone_line"][side]))
    out = []
    for a, b in ((far, wall), (far, zone), (near, zone), (near, wall)):
        x = np.cross(a, b)
        out.append(x[:2] / x[2])
    return out


def _affine_from(src, dst):
    """3 point pairs -> 3x3 affine."""
    A = np.column_stack([np.asarray(src, float), np.ones(3)])
    X = np.linalg.solve(A, np.asarray(dst, float))
    return np.vstack([X.T, [0, 0, 1]])


def undistort(p, K, D, P, iters=50):
    """Distorted pixels -> undistorted pixels (radial k1, k2, k3 only). Same
    fixed-point iteration as OpenCV's undistortPoints, run longer; lens_tool.html
    uses the identical loop so the browser and the pipeline agree."""
    p = np.asarray(p, float).reshape(-1, 2)
    k1, k2, k3 = D[0], D[1], (D[4] if len(D) > 4 else 0.0)
    xd = (p - K[:2, 2]) / K[0, 0]
    x = xd.copy()
    for _ in range(iters):
        r2 = (x ** 2).sum(1, keepdims=True)
        x = xd / (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    return x * P[0, 0] + P[:2, 2]


def find_seam(video: Path, n: int = 30, search=(0.35, 0.65)) -> int:
    """The stitch is a hard vertical cut: find the column with the strongest
    brightness/texture step, averaged over `n` frames and all rows."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = None
    for t in np.linspace(0.02, 0.98, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * total))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dx = np.abs(np.diff(g, axis=1))
        # count rows where this column's step stands out from its neighbourhood
        local = cv2.blur(dx, (41, 1))
        score = ((dx > 2 * local + 2).mean(axis=0))
        acc = score if acc is None else acc + score
    w = acc.shape[0] + 1
    lo, hi = int(search[0] * w), int(search[1] * w)
    return lo + int(np.argmax(acc[lo:hi])) + 1


def _apply(Hm, p):
    q = np.column_stack([p, np.ones(len(p))]) @ Hm.T
    return q[:, :2] / q[:, 2:]


class Half:
    def __init__(self, x0, w, h, K, D, P, Hm):
        self.x0, self.w, self.h = x0, w, h
        self.K, self.D, self.P, self.Hm = (np.asarray(a, float) for a in (K, D, P, Hm))

    def points(self, p):
        """Source (half-local) -> corrected (half-local)."""
        u = undistort(p, self.K, self.D, self.P)
        return _apply(self.Hm, u)

    def maps(self):
        """remap() maps: corrected (half-local) pixel -> source (half-local) pixel."""
        m1, m2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.P, (self.w, self.h), cv2.CV_32FC1)
        Hi = np.linalg.inv(self.Hm)
        gx, gy = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
        den = Hi[2, 0] * gx + Hi[2, 1] * gy + Hi[2, 2]
        sx = ((Hi[0, 0] * gx + Hi[0, 1] * gy + Hi[0, 2]) / den).astype(np.float32)
        sy = ((Hi[1, 0] * gx + Hi[1, 1] * gy + Hi[1, 2]) / den).astype(np.float32)
        kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        return cv2.remap(m1, sx, sy, **kw), cv2.remap(m2, sx, sy, **kw)


def _seam_offsets(x, y, seam, sw):
    """Pillar seam warp as output -> sample offsets (dx, dy), right half only."""
    (lt, rt), (lb, rb) = sw["pad_top"], sw["base"]
    sc = (lb - lt) / (rb - rt)
    sy = rt + (y - lt) / sc
    (y0, d0), (y1, d1) = sw["dx"]
    dx = d0 + (d1 - d0) * np.clip((y - y0) / (y1 - y0), 0, 1)
    wx = np.clip(1 - (x - seam) / sw["fade_px"], 0, 1) * (x >= seam)
    r0, r1 = sw["rows"]
    w = wx * np.clip((r1 - y) / (r1 - r0), 0, 1)
    return -w * dx, w * (sy - y)


class Lens:
    def __init__(self, seam: int, size, halves: dict, seam_warp: dict | None = None):
        self.seam, (self.W, self.H) = seam, size
        self.halves, self.seam_warp = halves, seam_warp
        self._maps = None

    @classmethod
    def load(cls, path=Path("data/lens.json")):
        d = json.loads(Path(path).read_text())
        return cls(d["seam"], d["size"], {s: Half(**v) for s, v in d["halves"].items()}, d.get("seam_warp"))

    def save(self, path):
        d = {"seam": self.seam, "size": [self.W, self.H], "equipment": EQUIPMENT, "seam_warp": self.seam_warp,
             "halves": {s: {"x0": h.x0, "w": h.w, "h": h.h, "K": h.K.tolist(), "D": h.D.tolist(),
                            "P": h.P.tolist(), "Hm": h.Hm.tolist()} for s, h in self.halves.items()}}
        Path(path).write_text(json.dumps(d, indent=1))

    def points(self, p):
        """Full-frame source points (N, 2) -> full-frame corrected points."""
        p = np.asarray(p, float).reshape(-1, 2)
        out = np.empty_like(p)
        for s, h in self.halves.items():
            m = (p[:, 0] < self.seam) if s == "left" else (p[:, 0] >= self.seam)
            if m.any():
                out[m] = h.points(p[m] - [h.x0, 0]) + [h.x0, 0]
        if self.seam_warp:
            # invert the output->sample seam warp by fixed-point iteration (it is small and smooth)
            o = out.copy()
            for _ in range(20):
                dx, dy = _seam_offsets(o[:, 0], o[:, 1], self.seam, self.seam_warp)
                o = out - np.column_stack([dx, dy])
            out = o
        return out

    def frame(self, img):
        if self._maps is None:
            parts = [(h.maps(), h) for h in self.halves.values()]
            mx = np.hstack([m[0] + np.where(m[0] >= 0, h.x0, 0) for m, h in parts]).astype(np.float32)
            my = np.hstack([m[1] for m, _ in parts]).astype(np.float32)
            if self.seam_warp:
                gy, gx = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
                dx, dy = _seam_offsets(gx, gy, self.seam, self.seam_warp)
                sx = np.maximum(gx + dx, np.where(gx >= self.seam, self.seam, 0)).astype(np.float32)
                kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                mx, my = cv2.remap(mx, sx, (gy + dy).astype(np.float32), **kw), \
                    cv2.remap(my, sx, (gy + dy).astype(np.float32), **kw)
            self._maps = (mx, my)
        return cv2.remap(img, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _rotation(f, xs, cy, yaw, pitch, roll):
    """Turn the virtual camera about its lens center (kept fixed, so halves still meet)."""
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    Kp = np.array([[f, 0, xs], [0, f, cy], [0, 0, 1]])
    Hr = Kp @ Rz @ Rx @ Ry @ np.linalg.inv(Kp)
    c = Hr @ np.array([xs, cy, 1.0])
    return np.array([[1, 0, xs - c[0] / c[2]], [0, 1, cy - c[1] / c[2]], [0, 0, 1]]) @ Hr


def build(stitch: dict, size=(3490, 1400)) -> Lens:
    """EQUIPMENT (+ profile overrides) + a per-video stitch profile -> Lens.
    Keep in sync with lens_tool.html, which runs the same model in the browser."""
    W, H = size
    st = {**STITCH_DEFAULTS, **stitch}
    eq = {**EQUIPMENT, **stitch.get("equipment", {})}
    cam = {**STITCH_DEFAULTS["camera"], **st.get("camera", {})}
    seam = st["seam"]
    halves = {"left": (0, seam), "right": (seam, W)}
    cy = eq["cy"]
    base = {s: np.array(st["baseline"][s], float) - [x0, 0] for s, (x0, _) in halves.items()}
    hs = {}
    for s, (x0, x1) in halves.items():
        w, f = x1 - x0, eq["f"][s]
        xs = float(w) if s == "left" else 0.0
        K = np.array([[f, 0, xs], [0, f, cy], [0, 0, 1]], float)
        D = np.array([eq["k1"], eq["k2"], 0, 0, eq["k3"]], float)
        ends = undistort([[xs, 0.0], [xs, H - 1.0]], K, D, K)
        z = max(cy / (cy - ends[0, 1]), (H - cy) / (ends[1, 1] - cy)) * eq["zoom"]
        P = K.copy()
        P[0, 0] = P[1, 1] = f * z
        sign = 1 if s == "left" else -1
        h = Half(x0, w, H, K, D, P, _rotation(f * z, xs, cy, sign * cam["yaw"], cam["pitch"], sign * cam["roll"]))
        slope = np.polyfit(*h.points(base[s]).T, 1)[0] * st["shear"]
        h.Hm = np.array([[1, 0, 0], [-slope, 1, slope * xs], [0, 0, 1]]) @ h.Hm
        hs[s] = h

    def seam_y(s):
        h = hs[s]
        m, c = np.polyfit(*h.points(base[s]).T, 1)
        return m * (h.w if s == "left" else 0.0) + c

    if st["align"]:  # stretch each half vertically about the seam bottom so the baselines meet
        yt = (seam_y("left") + seam_y("right")) / 2
        for s, h in hs.items():
            sc = (H - yt) / (H - seam_y(s))
            h.Hm = np.array([[1, 0, 0], [0, sc, H * (1 - sc)], [0, 0, 1]]) @ h.Hm
    if st["collinear"]:  # shear each half about the seam so both segments lie on one line
        mids = {s: h.points(base[s]).mean(0) + [h.x0, 0] for s, h in hs.items()}
        mc = (mids["right"][1] - mids["left"][1]) / (mids["right"][0] - mids["left"][0])
        for s, h in hs.items():
            m = np.polyfit(*h.points(base[s]).T, 1)[0]
            xs = float(h.w) if s == "left" else 0.0
            h.Hm = np.array([[1, 0, 0], [mc - m, 1, -(mc - m) * xs], [0, 0, 1]]) @ h.Hm
    pins = st.get("corner_pin") or {}
    for s, h in hs.items():
        pin = {**STITCH_DEFAULTS["corner_pin"][s], **pins.get(s, {})}
        if any(any(v) for v in pin.values()):
            src = np.float32([[0, 0], [h.w, 0], [h.w, H], [0, H]])
            dst = src + np.float32([pin["top_left"], pin["top_right"], pin["bottom_right"], pin["bottom_left"]])
            h.Hm = cv2.getPerspectiveTransform(src, dst).astype(float) @ h.Hm
    par = {**STITCH_DEFAULTS["parallelogram"], **(st.get("parallelogram") or {})}
    lines = st.get("lines")
    for s, h in hs.items():
        k = par.get(s, 0.0)
        if not k or not lines:
            continue
        x0, xs = h.x0, (float(h.w) if s == "left" else 0.0)
        loc = lambda pts: h.points(np.asarray(pts, float) - [x0, 0])
        far, near = _tls_line(loc(st["baseline"][s])), _tls_line(loc(lines["near_wall"][s]))
        side, zone = _tls_line(loc(lines["side_wall"][s])), _tls_line(loc(lines["zone_line"][s]))
        horizon = np.cross(np.cross(far, near), np.cross(side, zone))  # line through both vanishing points
        Hp = np.array([[1, 0, 0], [0, 1, 0], [k * horizon[0] / horizon[2], k * horizon[1] / horizon[2], 1]])
        # pin the seam column (two points) and the far baseline's outer end back in place
        b = loc(st["baseline"][s])
        outer = b[np.argmin(np.abs(b[:, 0] - (h.w - xs)))]
        anchors = np.array([[xs, cy], [xs, 0.8 * H], outer])
        moved = _apply(Hp, anchors)
        h.Hm = _affine_from(moved, anchors) @ Hp @ h.Hm
    quads = {**STITCH_DEFAULTS["quad"], **(st.get("quad") or {})}
    for s, h in hs.items():
        if lines and quads.get(s):
            src = np.float32(quad_corners(st, h, s))
            dst = np.float32(np.asarray(quads[s], float) - [h.x0, 0])
            h.Hm = cv2.getPerspectiveTransform(src, dst).astype(float) @ h.Hm
    sw = st.get("seam_warp")
    if sw and not sw.get("enabled", True):
        sw = None
    return Lens(seam, size, hs, sw)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("seam", help="detect the stitch seam column in a video")
    sm.add_argument("video", type=Path)
    b = sub.add_parser("build", help="EQUIPMENT + stitch profile -> data/lens.json")
    b.add_argument("stitch", type=Path, help="data/stitch/<video>.json")
    b.add_argument("--video", type=Path, help="if given, check the profile's seam against this video")
    b.add_argument("--out", type=Path, default=Path("data/lens.json"))
    v = sub.add_parser("video", help="write a corrected video")
    v.add_argument("video", type=Path)
    v.add_argument("--lens", type=Path, default=Path("data/lens.json"))
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--crf", type=int, default=23)
    f = sub.add_parser("frame", help="write a corrected frame")
    f.add_argument("video", type=Path)
    f.add_argument("t", type=float)
    f.add_argument("--lens", type=Path, default=Path("data/lens.json"))
    f.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    if a.cmd == "seam":
        print(find_seam(a.video))
    elif a.cmd == "build":
        stitch = json.loads(a.stitch.read_text())
        if a.video:
            found = find_seam(a.video)
            if abs(found - stitch["seam"]) > 2:
                raise SystemExit(f"seam in video is x={found}, profile says {stitch['seam']}: this video was "
                                 "stitched differently; make a new profile (re-pick baseline + pillar)")
        lens = build(stitch)
        lens.save(a.out)
        base = lens.points(np.array(stitch["baseline"]["left"] + stitch["baseline"]["right"], float))
        m, c = np.polyfit(*base.T, 1)
        print(f"wrote {a.out}; far baseline after correction: slope {m:+.4f}, "
              f"rms off one line {np.sqrt(np.mean((base[:, 1] - (m * base[:, 0] + c)) ** 2)):.2f}px")
    elif a.cmd == "video":
        import subprocess
        lens = Lens.load(a.lens)
        cap = cv2.VideoCapture(str(a.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{lens.W}x{lens.H}",
             "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)
        k = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            enc.stdin.write(lens.frame(img).tobytes())
            k += 1
            if k % 600 == 0:
                print(f"{k}/{n} frames", flush=True)
        enc.stdin.close()
        enc.wait()
        print(f"wrote {a.out} ({k} frames)")
    else:
        cap = cv2.VideoCapture(str(a.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000)
        ok, img = cap.read()
        cv2.imwrite(str(a.out), Lens.load(a.lens).frame(img))


if __name__ == "__main__":
    main()
