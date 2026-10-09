"""Compare lens corrections by how straight real arrow flights come out.

An arrow flies in a (nearly) straight line, or a gentle gravity arc, so a good
correction should make every detected flight straight, near side included.
Detected flights (source pixels) are mapped through each candidate lens and
scored by how far their points sit off a best-fit line and off a best-fit
parabola. Shooter attribution is re-scored with each lens too.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import lens as L
from .arrows import dedupe, is_shot
from .attribute import attribute
from .evaluate import score, score_arrows


def bend(p):
    """(line RMS px, arc RMS px, length px) for one flight's points."""
    p = np.asarray(p, float)
    m = p.mean(0)
    d = np.linalg.svd(p - m, full_matrices=False)[2][0]
    t, e = (p - m) @ d, (p - m) @ np.array([-d[1], d[0]])
    line = float(np.sqrt(np.mean(e ** 2)))
    arc = float(np.sqrt(np.mean((e - np.polyval(np.polyfit(t, e, 2), t)) ** 2))) if len(p) >= 4 else line
    return line, arc, float(np.ptp(t))


def variants(stitch: dict) -> dict:
    """name -> stitch-profile overrides (None = no correction at all)."""
    def par_quads():  # complete each half's corners into a parallelogram: c3 = c2 + c4 - c1
        lens = L.build(stitch)
        q = {}
        for s, h in lens.halves.items():
            c = np.array(L.quad_corners(stitch, h, s)) + [h.x0, 0]
            c[2] = c[1] + c[3] - c[0]
            q[s] = c.tolist()
        return q
    return {
        "none (raw stitched frame)": None,
        "T4 (current)": {},
        "T4, k1=+0.1": {"equipment": {"k1": 0.1}},
        "T4, k1=+0.3": {"equipment": {"k1": 0.3}},
        "T4, k2=-0.08": {"equipment": {"k2": -0.08}},
        "T4, k2=-0.15": {"equipment": {"k2": -0.15}},
        "T4 + halves as parallelograms": {"quad": par_quads()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stitch", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("arrows", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/straightness.json"))
    a = ap.parse_args()
    stitch = json.loads(a.stitch.read_text())
    d = dict(np.load(a.tracks))
    flights = json.loads(a.arrows.read_text())["flights"]
    labels = json.loads(a.labels.read_text())
    t_max = max(l["t"] for l in labels) + 1.0

    # real arrows = flights matched to a labeled shot (lenient detector setting)
    r = score_arrows(flights, labels, t_max, min_points=4, min_nspeed=0.25)
    shots = dedupe([s for s in flights if is_shot(s, min_points=4, min_nspeed=0.25) and s["t0"] <= t_max])
    real = [shots[si] for _, si, _ in r["matched"]]
    cands = [s for s in flights if s["n"] >= 6 and s["nspeed"] >= 0.25 and s["t0"] <= t_max]

    rows = []
    for name, over in variants(stitch).items():
        lens = None if over is None else L.build({**stitch, **over,
                                                  "equipment": {**stitch.get("equipment", {}), **(over or {}).get("equipment", {})}})
        fix = (lambda p: np.asarray(p, float)) if lens is None else lens.points

        def stats(fl):
            out = {"far": [], "near": []}
            for s in fl:
                src = np.array([q[1:3] for q in s["pts"]], float)
                line, arc, length = bend(fix(src))
                out["near" if src[:, 1].mean() > 600 else "far"].append((line, arc, length))
            return {k: {"n": len(v), "line_px": float(np.median([x[0] for x in v])) if v else None,
                        "arc_px": float(np.median([x[1] for x in v])) if v else None} for k, v in out.items()}

        att = {}
        for mp in (5, 6):
            sh = attribute(d, flights, lens, min_points=mp, min_nspeed=0.25)
            sc = score(d, [(s["frame"], s["tid"]) for s in sh], labels, t_max)
            att[mp] = (sc["recall"], sc["precision"])
        rows.append({"variant": name, "real": stats(real), "candidates": stats(cands), "attribution": att})

    a.out.write_text(json.dumps(rows, indent=1))
    print(f"real arrows: {len(real)} matched flights; candidate flights: {len(cands)}\n")
    print(f"{'variant':32s} | real arrows: median bend off line / arc (px), far | near"
          f" | all flights line far|near | who+when recall/precision  >=5 pts | >=6 pts")
    for row in rows:
        rf, rn = row["real"]["far"], row["real"]["near"]
        cf, cn = row["candidates"]["far"], row["candidates"]["near"]
        f = lambda v: "  -  " if v is None else f"{v:5.2f}"
        (r5, p5), (r6, p6) = row["attribution"][5], row["attribution"][6]
        print(f"{row['variant']:32s} | {f(rf['line_px'])}/{f(rf['arc_px'])} (n{rf['n']:2d}) | "
              f"{f(rn['line_px'])}/{f(rn['arc_px'])} (n{rn['n']:2d}) | {f(cf['line_px'])}|{f(cn['line_px'])}"
              f" | {r5:4.0%}/{p5:4.0%} | {r6:4.0%}/{p6:4.0%}")


if __name__ == "__main__":
    main()
