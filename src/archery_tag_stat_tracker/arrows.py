"""Detect arrows in flight and the moment they cross into the neutral zone.

Each frame (full 60 fps) is double-differenced against frames k-2 and k+2, which
keeps only fast movers. Small blobs are linked frame-to-frame with a
constant-velocity model into "flights". A flight that is fast, straight and
enters the neutral zone (extended upward, since arrows fly at head height) is
a shot. Viewed as a long exposure, a flight is the dotted streak an arrow
leaves across the zone.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Band of the 3490x1400 frame searched for arrows (x0, x1): the neutral zone
# plus margin on both sides so flights are picked up before they reach it.
BAND = (900, 2750)
# Neutral zone edges in source pixels: (x at top y=178, x at bottom y=1388).
ZONE_LEFT = ((1626, 178), (1011, 1388))
ZONE_RIGHT = ((1864, 178), (2647, 1388))

DIFF_THRESH = 28
MIN_AREA, MAX_AREA = 5, 1500
MAX_MISS = 1  # frames a flight may go unseen
MIN_POINTS = 5
MIN_SPEED = 15.0  # px/frame, for seeding; far arrows move only ~20-55 px/frame
# Speed is judged in player-heights per frame so far (small) and near (big)
# arrows share one threshold. Player height (px) vs image y, from track.py boxes.
HEIGHT_AT_Y = ((100, 85), (400, 200), (700, 400), (1000, 500))
MIN_NSPEED = 0.25
EXTRAPOLATE = 12  # frames; flights split by the far pillar still count if their line crosses the zone
MAX_STEP = 260  # px/frame
MAX_RESID = 8.0  # px, line-fit residual (scaled up for fast, blurred arrows)


def edge_x(edge, y):
    (xa, ya), (xb, yb) = edge
    return xa + (xb - xa) * (y - ya) / (yb - ya)


def player_height(y):
    ys, hs = zip(*HEIGHT_AT_Y)
    return float(np.interp(y, ys, hs))


def in_zone(x, y):
    # Above the far end the zone continues straight up (arrows over the turf).
    yy = max(y, ZONE_LEFT[0][1])
    return edge_x(ZONE_LEFT, yy) <= x <= edge_x(ZONE_RIGHT, yy)


@dataclass
class Flight:
    pts: list = field(default_factory=list)  # (frame, x, y, area)

    def summary(self, fps):
        p = np.array([q[:3] for q in self.pts], float)
        f, x, y = p.T
        span = max(1.0, f[-1] - f[0])
        speed = float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / span)
        # straightness: residual of a total-least-squares line fit
        c = p[:, 1:] - p[:, 1:].mean(0)
        resid = float(np.linalg.svd(c, compute_uv=False)[-1] / np.sqrt(len(p))) if len(p) > 2 else 0.0
        vx, vy = (x[-1] - x[0]) / span, (y[-1] - y[0]) / span
        # sample the flight's line extended both ways; does it pass through the zone?
        k = np.arange(-EXTRAPOLATE, f[-1] - f[0] + EXTRAPOLATE + 1)
        crosses = any(in_zone(x[0] + vx * t, y[0] + vy * t) for t in k)
        return {
            "t0": float(f[0] / fps), "t1": float(f[-1] / fps), "n": len(p),
            "speed": speed, "nspeed": speed / player_height(float(y.mean())), "resid": resid,
            "vx": float(vx), "vy": float(vy), "crosses_zone": bool(crosses),
            "pts": [[int(a), round(b, 1), round(c_, 1)] for a, b, c_ in p.tolist()],
        }


def blobs(prev2, cur, next2):
    d = np.minimum(cv2.absdiff(cur, prev2), cv2.absdiff(cur, next2))
    m = (d > DIFF_THRESH).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = (stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA) & (stats[1:, cv2.CC_STAT_AREA] <= MAX_AREA)
    return [(float(cx) + BAND[0], float(cy), int(a)) for (cx, cy), a in
            zip(cent[1:][keep], stats[1:, cv2.CC_STAT_AREA][keep])]


def person_boxes(tracks: Path | None) -> dict:
    """frame -> (n, 4) boxes padded 5%, from track.py output (every `stride` frames)."""
    if tracks is None:
        return {}
    d = np.load(tracks)
    out = {}
    for f in np.unique(d["frame"]):
        b = d["box"][d["frame"] == f].copy()
        pw, ph = (b[:, 2] - b[:, 0]) * 0.05, (b[:, 3] - b[:, 1]) * 0.05
        b[:, 0] -= pw; b[:, 2] += pw; b[:, 1] -= ph; b[:, 3] += ph
        out[int(f)] = b
    return out


def collect(video: Path, start: float, duration: float, boxes: dict):
    """Per-frame fast-mover blobs, (n, 3) arrays of x, y, area; blobs on players dropped."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int(round(start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0 - 2))
    buf, frames = [], []
    for _ in range(int(duration * fps) + 4):
        ok, img = cap.read()
        if not ok:
            break
        buf.append(cv2.cvtColor(img[:, BAND[0]:BAND[1]], cv2.COLOR_BGR2GRAY))
        if len(buf) < 5:
            continue
        f = f0 + len(frames)
        b = np.array(blobs(buf[0], buf[2], buf[4]), float).reshape(-1, 3)
        buf.pop(0)
        pb = boxes.get(f, boxes.get(f - 1))
        if pb is not None and len(b):
            on = ((b[:, None, 0] >= pb[:, 0]) & (b[:, None, 0] <= pb[:, 2])
                  & (b[:, None, 1] >= pb[:, 1]) & (b[:, None, 1] <= pb[:, 3])).any(1)
            b = b[~on]
        frames.append(b)
    return fps, f0, frames


def link(frames: list, f0: int) -> list:
    """Seed flights from evenly spaced, collinear, fast blob triplets in consecutive
    frames (clutter almost never forms one), then extend both ways."""
    used = [np.zeros(len(a), bool) for a in frames]

    def extend(pts, step):
        (ka, a), (kb, b) = pts[-2], pts[-1]
        v = (frames[kb][b, :2] - frames[ka][a, :2]) / (kb - ka)
        k = kb
        while True:
            for gap in range(1, MAX_MISS + 2):
                n = k + step * gap
                if not 0 <= n < len(frames) or not len(frames[n]):
                    continue
                pred = frames[k][pts[-1][1], :2] + v * gap
                d = np.hypot(*(frames[n][:, :2] - pred).T)
                d[used[n]] = np.inf
                j = int(np.argmin(d))
                if d[j] <= 15 + 0.2 * np.hypot(*v) * gap:
                    v = (frames[n][j, :2] - frames[k][pts[-1][1], :2]) / gap
                    pts.append((n, j))
                    used[n][j] = True
                    k = n
                    break
            else:
                return pts

    flights = []
    for k in range(len(frames) - 2):
        A, B, C = frames[k], frames[k + 1], frames[k + 2]
        if not (len(A) and len(B) and len(C)):
            continue
        D = B[None, :, :2] - A[:, None, :2]
        dist = np.hypot(D[..., 0], D[..., 1])
        for a, b in zip(*np.nonzero((dist >= MIN_SPEED) & (dist <= MAX_STEP))):
            if used[k][a] or used[k + 1][b]:
                continue
            dc = np.hypot(*(C[:, :2] - (B[b, :2] + D[a, b])).T)
            dc[used[k + 2]] = np.inf
            c = int(np.argmin(dc))
            if dc[c] > 15 + 0.2 * dist[a, b]:
                continue
            for kk, i in ((k, a), (k + 1, b), (k + 2, c)):
                used[kk][i] = True
            fwd = extend([(k, a), (k + 1, b), (k + 2, c)], +1)
            back = extend([(k + 1, b), (k, a)], -1)[2:]
            pts = back[::-1] + fwd
            flights.append(Flight(pts=[(f0 + kk, *frames[kk][i]) for kk, i in pts]))
    return flights


def track(video: Path, start: float, duration: float, tracks: Path | None = None):
    fps, f0, frames = collect(video, start, duration, person_boxes(tracks))
    return fps, link(frames, f0)


def is_shot(s, min_points=MIN_POINTS, min_nspeed=MIN_NSPEED):
    return (s["n"] >= min_points and s["nspeed"] >= min_nspeed
            and s["resid"] <= max(MAX_RESID, 0.15 * s["speed"]) and s["crosses_zone"])


def dedupe(shots, max_gap_s=0.15):
    """One arrow can leave several flights (streak split by blur, the pillar or
    the stitch seam). Merge flights going the same way that follow each other
    closely in time; keep the longest."""
    out = []
    for s in sorted(shots, key=lambda s: s["t0"]):
        prev = out[-1] if out else None
        if prev and s["t0"] - prev["t1"] <= max_gap_s and (s["vx"] > 0) == (prev["vx"] > 0):
            if s["n"] > prev["n"]:
                out[-1] = {**s, "t0": prev["t0"]}
            else:
                prev["t1"] = max(prev["t1"], s["t1"])
            continue
        out.append(dict(s))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--tracks", type=Path, help="track.py output; blobs on players are ignored")
    p.add_argument("--out", type=Path, default=Path("out/arrows.json"))
    a = p.parse_args()
    fps, flights = track(a.video, a.start, a.duration, a.tracks)
    summ = [fl.summary(fps) for fl in flights if len(fl.pts) >= MIN_POINTS]
    shots = [s for s in summ if is_shot(s)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"fps": fps, "flights": summ}))
    print(f"{len(flights)} flights, {len(summ)} with >= {MIN_POINTS} points, {len(shots)} arrow shots")


if __name__ == "__main__":
    main()
