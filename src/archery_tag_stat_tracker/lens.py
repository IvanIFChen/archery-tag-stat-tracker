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
    "f": {"left": 1822.0, "right": 1668.0},  # px, at the 3490x1400 stitched scale
    "cy": 190.0,  # lens-center height (on the far baseline row)
}


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
        u = cv2.undistortPoints(np.asarray(p, float).reshape(-1, 1, 2), self.K, self.D, P=self.P).reshape(-1, 2)
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


def build(stitch: dict, size=(3490, 1400)) -> Lens:
    """EQUIPMENT + a per-video stitch profile -> Lens."""
    W, H = size
    seam = stitch["seam"]
    halves = {"left": (0, seam), "right": (seam, W)}
    k1, cy = EQUIPMENT["k1"], EQUIPMENT["cy"]
    base = {s: np.array(stitch["baseline"][s], float) - [x0, 0] for s, (x0, _) in halves.items()}
    hs = {}
    for s, (x0, x1) in halves.items():
        w, f = x1 - x0, EQUIPMENT["f"][s]
        xs = float(w) if s == "left" else 0.0
        K = np.array([[f, 0, xs], [0, f, cy], [0, 0, 1]], float)
        D = np.array([k1, 0, 0, 0], float)
        ends = cv2.undistortPoints(np.array([[[xs, 0.0]], [[xs, H - 1.0]]]), K, D, P=K).reshape(2, 2)
        z = max(cy / (cy - ends[0, 1]), (H - cy) / (ends[1, 1] - cy))
        P = K.copy()
        P[0, 0] = P[1, 1] = f * z
        h = Half(x0, w, H, K, D, P, np.eye(3))
        slope = np.polyfit(*h.points(base[s]).T, 1)[0] * stitch["shear"]
        h.Hm = np.array([[1, 0, 0], [-slope, 1, slope * xs], [0, 0, 1]])
        hs[s] = h

    def seam_y(s):
        h = hs[s]
        m, c = np.polyfit(*h.points(base[s]).T, 1)
        return m * (h.w if s == "left" else 0.0) + c

    # stretch each half vertically about the seam bottom so the baselines meet
    yt = (seam_y("left") + seam_y("right")) / 2
    for s, h in hs.items():
        sc = (H - yt) / (H - seam_y(s))
        h.Hm = np.array([[1, 0, 0], [0, sc, H * (1 - sc)], [0, 0, 1]]) @ h.Hm
    # then shear each half about the seam so both segments lie on one line
    mids = {s: h.points(base[s]).mean(0) + [h.x0, 0] for s, h in hs.items()}
    mc = (mids["right"][1] - mids["left"][1]) / (mids["right"][0] - mids["left"][0])
    for s, h in hs.items():
        m = np.polyfit(*h.points(base[s]).T, 1)[0]
        xs = float(h.w) if s == "left" else 0.0
        h.Hm = np.array([[1, 0, 0], [mc - m, 1, -(mc - m) * xs], [0, 0, 1]]) @ h.Hm
    return Lens(seam, size, hs, stitch.get("seam_warp"))


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
