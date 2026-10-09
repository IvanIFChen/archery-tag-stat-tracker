"""Attribute arrow flights to shooters by tracing each flight back into the
shooter's half and picking the tracked player whose upper body lies closest to
the flight's line, shortly before the arrow was first seen."""

import argparse
import json
from pathlib import Path

import numpy as np

from .arrows import ZONE_LEFT, ZONE_RIGHT, dedupe, edge_x, is_shot

L_SH, R_SH = 5, 6
LOOKBACK_S = 0.6  # how long before the first arrow blob the release may be
MAX_PERP = 0.8  # max distance from line to shooter, in player heights
MAX_AHEAD = 0.5  # shooter may sit at most this many heights *past* the flight start


def zone_mid(y):
    y = max(y, ZONE_LEFT[0][1])
    return (edge_x(ZONE_LEFT, y) + edge_x(ZONE_RIGHT, y)) / 2


def upper_body(kp, box):
    sh = kp[[L_SH, R_SH]]
    sh = sh[sh[:, 2] > 0.3]
    if len(sh):
        return sh[:, :2].mean(0)
    return np.array([(box[0] + box[2]) / 2, box[1] + 0.25 * (box[3] - box[1])])


def fit_line(pts):
    p = np.array([q[1:3] for q in pts], float)
    c = p.mean(0)
    d = np.linalg.svd(p - c)[2][0]
    if np.dot(p[-1] - p[0], d) < 0:  # point along travel
        d = -d
    return p[0], d


def attribute(d: dict, flights: list, **shot_kw) -> list:
    fps = float(d["fps"])
    frames = d["frame"]
    out = []
    for s in dedupe([s for s in flights if is_shot(s, **shot_kw)]):
        p0, u = fit_line(s["pts"])
        f0 = s["pts"][0][0]
        shooter_left = s["vx"] > 0
        cand = np.flatnonzero((frames >= f0 - LOOKBACK_S * fps) & (frames <= f0 + 2))
        best = None
        for i in cand:
            box = d["box"][i]
            h = box[3] - box[1]
            q = upper_body(d["kp"][i], box)
            if (q[0] < zone_mid(q[1])) != shooter_left:
                continue
            v = q - p0
            along = float(np.dot(v, u))  # negative = behind the flight start
            perp = abs(float(v[0] * u[1] - v[1] * u[0]))
            if along > MAX_AHEAD * h or perp > MAX_PERP * h:
                continue
            score = perp / h + 0.3 * abs(f0 - frames[i]) / fps
            if best is None or score < best[0]:
                best = (score, i, along)
        if best is None:
            continue
        _, i, along = best
        # release ~ when the arrow left the shooter: back off by travel time
        lag = max(0.0, -along) / max(1.0, s["speed"])
        out.append({"frame": int(round(f0 - lag)), "tid": int(d["tid"][i]),
                    "box": d["box"][i].tolist(), "flight_t0": s["t0"], "score": round(best[0], 3)})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tracks", type=Path)
    p.add_argument("arrows", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/attributed.json"))
    a = p.parse_args()
    d = dict(np.load(a.tracks))
    shots = attribute(d, json.loads(a.arrows.read_text())["flights"])
    a.out.write_text(json.dumps(shots, indent=1))
    print(f"{len(shots)} attributed shots -> {a.out}")


if __name__ == "__main__":
    main()
