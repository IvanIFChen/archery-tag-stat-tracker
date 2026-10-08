"""Grid-search shot-rule parameters against hand labels and print the best by F1."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .evaluate import score
from .shots import Params, detect

GRID = {
    "y_max": [0.15, 0.2, 0.25, 0.3, 0.35],
    "min_aim_s": [0.1, 0.2, 0.3, 0.45, 0.6],
    "gap_s": [0.05, 0.1, 0.2],
    "cooldown_s": [0.6, 1.0, 1.5],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    d = dict(np.load(a.tracks))
    labels = json.loads(a.labels.read_text())
    t_max = max(l["t"] for l in labels) + 1.0

    results = []
    for vals in itertools.product(*GRID.values()):
        p = Params(**dict(zip(GRID, vals)))
        r = score(d, detect(d, p)[1], labels, t_max)
        f1 = 2 * r["tp"] / max(1, r["labels"] + r["detected"])
        results.append((f1, r["recall"], r["precision"], r["detected"], p))
    results.sort(key=lambda x: -x[0])
    for f1, rec, prec, n, p in results[: a.top]:
        print(f"F1 {f1:.2f}  recall {rec:.0%}  precision {prec:.0%}  detected {n:3d}  {p}")


if __name__ == "__main__":
    main()
