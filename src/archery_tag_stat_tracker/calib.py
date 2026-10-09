"""Per-camera fisheye calibration from straight lines in the scene.

The stitched frame is two GoPros cut at x = SEAM. For each half we take a few
features that are straight in the real world (field lines, wall bases, posts),
snap rough seed points onto them in a people-free background frame, and fit an
OpenCV fisheye model (f, cx, cy, k1, k2) that makes them straight again.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

SEAM = 1822
W, H = 3490, 1400

# Rough seeds in full-frame pixels: (kind, points). "ridge" = thin bright line
# (white paint), "edge" = boundary between two regions. Refined automatically.
SEEDS = {
    "left": [
        ("ridge", [(1615, 250), (1590, 300), (1490, 500), (1440, 600), (1390, 700), (1345, 800), (1295, 900),
                   (1250, 1000), (1200, 1100), (1150, 1200), (1100, 1300), (1040, 1390)]),  # neutral-zone line
        ("ridge", [(1460, 205), (1550, 195), (1630, 187)]),  # far baseline
        ("edge", [(310, 700), (450, 630), (600, 560), (750, 490), (900, 420), (1050, 350)]),  # side-wall base
        ("edge", [(1800, 30), (1800, 100), (1800, 170), (1800, 240)]),  # pillar edge
    ],
    "right": [
        ("ridge", [(1930, 250), (2000, 380), (2092, 500), (2207, 700), (2312, 900), (2437, 1100), (2602, 1300), (2662, 1385)]),
        ("ridge", [(1862, 180), (2022, 205), (2222, 232), (2280, 239)]),
        ("edge", [(2722, 420), (2872, 500), (3022, 575), (3172, 655)]),
        ("edge", [(2422, 220), (2422, 260), (2422, 300), (2422, 335)]),
    ],
}


def refine(gray: np.ndarray, kind: str, seeds, step=12, search=25):
    """Densely sample the seed polyline and snap each sample to the strongest
    ridge (brightness peak) or edge (gradient peak) along the local normal."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
    seeds = np.array(seeds, float)
    pts = []
    for a, b in zip(seeds[:-1], seeds[1:]):
        t = b - a
        n_s = max(2, int(np.linalg.norm(t) / step))
        tn = t / np.linalg.norm(t)
        nrm = np.array([-tn[1], tn[0]])
        for s in np.linspace(0, 1, n_s, endpoint=False):
            c = a + s * t
            offs = np.arange(-search, search + 1)
            xy = c + offs[:, None] * nrm
            vals = cv2.remap(g, xy[:, 0:1].astype(np.float32), xy[:, 1:2].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
            prof = vals if kind == "ridge" else np.abs(np.gradient(vals))
            k = int(np.argmax(prof))
            if 0 < k < len(prof) - 1:  # parabolic sub-pixel peak
                y0, y1, y2 = prof[k - 1:k + 2]
                den = y0 - 2 * y1 + y2
                k = k + (0.5 * (y0 - y2) / den if den != 0 else 0)
            pts.append(c + (k - search) * nrm)
    return np.array(pts)


def undistort(pts, f, cx, cy, k1, k2):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    D = np.array([k1, k2, 0, 0], float)
    return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)


def rays(pts, params):
    u = undistort(pts, *params)
    r = np.column_stack([u, np.ones(len(u))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def straightness(lines, params):
    """A straight 3D line seen from the camera spans a plane through the camera
    center, so its viewing rays lie on a great circle. Residual = sine of each
    ray's angle off the best-fit plane. Scale-free, so stretching points toward
    the edge of the lens can't fake straightness."""
    res = []
    for p in lines:
        r = rays(p, params)
        n = np.linalg.svd(r, full_matrices=False)[2][-1]
        # angular extent of the line; divide by it so a long focal length (which
        # squeezes all rays together) can't shrink the residual
        span = np.arccos(np.clip(r @ r.T, -1, 1)).max()
        res.append((r @ n) / max(span, 1e-3))
    return np.concatenate(res)


def fit(lines, w, h):
    # GoPro-like priors: ~100-130 deg horizontal FOV per camera for an equidistant
    # fisheye -> f ~ 600-1600 px at this scale; principal point within the middle
    # of the half; mild polynomial terms.
    x0 = [1000.0, w / 2, h / 2, 0.0, 0.0]
    lb = [500.0, w * 0.2, h * 0.2, -0.5, -0.5]
    ub = [2000.0, w * 0.8, h * 0.8, 0.5, 0.5]
    r = least_squares(lambda x: straightness(lines, x), x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01)
    return r.x, float(np.sqrt(np.mean(straightness(lines, x0) ** 2))), float(np.sqrt(np.mean(r.fun ** 2)))


def fit_shared(lines_by_side, sizes):
    """Both GoPros share f, k1, k2; each principal point is fixed at the center
    of its half. 3 parameters for all lines -> much better conditioned."""
    def resid(x):
        f, k1, k2 = x
        return np.concatenate([straightness(lines_by_side[s], (f, w / 2, h / 2, k1, k2))
                               for s, (w, h) in sizes.items()])
    x0 = [1000.0, 0.0, 0.0]
    r = least_squares(resid, x0, bounds=([500, -0.5, -0.5], [2000, 0.5, 0.5]), loss="soft_l1", f_scale=0.01)
    return r.x, float(np.sqrt(np.mean(resid(x0) ** 2))), float(np.sqrt(np.mean(r.fun ** 2)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("background", type=Path, help="people-free frame (median of many frames)")
    p.add_argument("--out", type=Path, default=Path("data/calib.json"))
    p.add_argument("--debug", type=Path, default=Path("out/calib"))
    a = p.parse_args()
    bg = cv2.imread(str(a.background))
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    a.debug.mkdir(parents=True, exist_ok=True)
    result = {"seam": SEAM}
    halves = {"left": (0, SEAM), "right": (SEAM, W)}
    lines = {side: [refine(gray, kind, seeds) - [x0, 0] for kind, seeds in SEEDS[side]]
             for side, (x0, _) in halves.items()}
    sizes = {side: (x1 - x0, H) for side, (x0, x1) in halves.items()}
    (f, k1, k2), before, after = fit_shared(lines, sizes)
    print(f"shared: f={f:.0f} k=({k1:+.3f},{k2:+.3f})  straightness rms {before:.4f} -> {after:.4f}")
    for side, (x0, x1) in halves.items():
        result[side] = {"x0": x0, "width": x1 - x0, "f": f, "cx": (x1 - x0) / 2, "cy": H / 2,
                        "k1": k1, "k2": k2, "rms_before": before, "rms_after": after}
        dbg = bg[:, x0:x1].copy()
        for ln in lines[side]:
            for q in ln:
                cv2.circle(dbg, tuple(int(v) for v in q), 3, (0, 0, 255), -1)
        cv2.imwrite(str(a.debug / f"seeds_{side}.jpg"), dbg)
    a.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
