"""Score detected shots against hand labels (time + click position)."""

import argparse
import json
from pathlib import Path

import numpy as np

from .shots import detect

MAX_DT = 0.6  # seconds between labeled and detected release
BOX_PAD = 0.3  # grow detection box by this fraction when testing the click


def track_box(d: dict, tid: int, frame: int) -> np.ndarray:
    idx = np.flatnonzero(d["tid"] == tid)
    return d["box"][idx[np.argmin(np.abs(d["frame"][idx] - frame))]]


def score(d: dict, shots: list, labels: list, t_max: float) -> dict:
    fps = float(d["fps"])
    dets = [(f / fps, tid, track_box(d, tid, f)) for f, tid, *_ in shots if f / fps <= t_max]
    pairs = []
    for li, l in enumerate(labels):
        for di, (t, _, b) in enumerate(dets):
            dt = abs(t - l["t"])
            pw, ph = (b[2] - b[0]) * BOX_PAD, (b[3] - b[1]) * BOX_PAD
            inside = b[0] - pw <= l["x"] <= b[2] + pw and b[1] - ph <= l["y"] <= b[3] + ph
            if dt <= MAX_DT and inside:
                pairs.append((dt, li, di))
    used_l, used_d, matched = set(), set(), []
    for dt, li, di in sorted(pairs):
        if li not in used_l and di not in used_d:
            used_l.add(li), used_d.add(di), matched.append((li, di, dt))
    tp = len(matched)
    return {
        "labels": len(labels), "detected": len(dets), "tp": tp,
        "recall": tp / max(1, len(labels)), "precision": tp / max(1, len(dets)),
        "missed": [labels[i] for i in range(len(labels)) if i not in used_l],
        "false": [(round(dets[i][0], 2), dets[i][1]) for i in range(len(dets)) if i not in used_d],
        "matched": matched,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tracks", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    d = dict(np.load(a.tracks))
    labels = json.loads(a.labels.read_text())
    t_max = max(l["t"] for l in labels) + 1.0
    _, shots = detect(d)
    r = score(d, shots, labels, t_max)
    print(f"labels {r['labels']}  detected {r['detected']}  matched {r['tp']}  "
          f"recall {r['recall']:.0%}  precision {r['precision']:.0%}")
    if a.verbose:
        print("missed:", [(l["t"], l["x"], l["y"]) for l in r["missed"]])
        print("false:", r["false"])


if __name__ == "__main__":
    main()
