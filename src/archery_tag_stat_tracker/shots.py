"""Detect shots from tracked pose keypoints and render an overlay video.

A shot is an "aim" (both wrists raised to head/shoulder height and held there)
lasting long enough, followed by the aim ending: at release the draw hand
snaps away and the bow arm drops.

Wrist height alone is used because the camera sees near players from the
front or back, where the extended bow arm points at the lens and lands next to
the draw hand in 2D; only side-on (far) players show the textbook pose.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HEAD = [0, 1, 2, 3, 4]
L_WRIST, R_WRIST = 9, 10


@dataclass(frozen=True)
class Params:
    kp_conf: float = 0.3
    y_max: float = 0.20  # wrists at most this * box height below head center
    min_aim_s: float = 0.30  # aim must last this long to count
    gap_s: float = 0.20  # tolerate brief dropouts inside an aim
    cooldown_s: float = 1.0  # min time between shots for one track


def aim_score(kp: np.ndarray, box: np.ndarray, p: Params) -> float:
    """Return 1.0 if both wrists are raised to aiming height, else 0.0 (nan if unknown)."""
    h = box[3] - box[1]
    head = kp[HEAD]
    head = head[head[:, 2] > p.kp_conf]
    if len(head) == 0 or h <= 0 or kp[L_WRIST, 2] < p.kp_conf or kp[R_WRIST, 2] < p.kp_conf:
        return np.nan
    hy = head[:, 1].mean()
    return float(max(kp[L_WRIST, 1], kp[R_WRIST, 1]) - hy < p.y_max * h)


def detect(d: dict, p: Params = Params()) -> tuple[dict, list]:
    fps = float(d["fps"])
    by_tid = defaultdict(list)
    for i, t in enumerate(d["tid"]):
        by_tid[int(t)].append(i)

    state = {}  # detection index -> aim score
    shots = []  # (frame, tid, x, y)
    for tid, idx in by_tid.items():
        idx = sorted(idx, key=lambda i: d["frame"][i])
        run_start, last_on, last_shot = None, None, -1e9
        for i in idx:
            f = int(d["frame"][i])
            s = aim_score(d["kp"][i], d["box"][i], p)
            state[i] = s
            if s == 1.0:
                if run_start is None or (f - last_on) / fps > p.gap_s:
                    run_start = f
                last_on = f
            elif run_start is not None and (f - last_on) / fps > p.gap_s:
                if (last_on - run_start) / fps >= p.min_aim_s and (last_on - last_shot) / fps > p.cooldown_s:
                    b = d["box"][i]
                    shots.append((last_on, tid, float(b[0]), float(b[1])))
                    last_shot = last_on
                run_start = None
    return state, sorted(shots)


def render(video: Path, d: dict, state: dict, shots: list, out: Path, scale: float) -> None:
    fps, stride = float(d["fps"]), int(d["stride"])
    frames = d["frame"]
    by_frame = defaultdict(list)
    for i, f in enumerate(frames):
        by_frame[int(f)].append(i)
    shot_frames = defaultdict(list)
    for f, tid, *_ in shots:
        for k in range(0, int(0.5 * fps), stride):
            shot_frames[f + k].append(tid)
    counts = defaultdict(int)
    shot_at = {(f, tid) for f, tid, *_ in shots}

    cap = cv2.VideoCapture(str(video))
    f0, f1 = int(frames.min()), int(frames.max())
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))
    for f in range(f0, f1 + 1):
        if (f - f0) % stride:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (W, H))
        for i in by_frame.get(f, []):
            tid = int(d["tid"][i])
            if (f, tid) in shot_at:
                counts[tid] += 1
            x0, y0, x1, y1 = (d["box"][i] * scale).astype(int)
            s = state.get(i, np.nan)
            col = (0, 0, 255) if tid in shot_frames.get(f, []) else (0, 200, 255) if s == 1.0 else (255, 160, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            for k in (*HEAD[:1], 5, 6, 7, 8, 9, 10):
                x, y, c = d["kp"][i][k]
                if c > KP_CONF:
                    cv2.circle(img, (int(x * scale), int(y * scale)), 3, col, -1)
            label = f"#{tid} shots:{counts[tid]}"
            if tid in shot_frames.get(f, []):
                label += " SHOT"
            cv2.putText(img, label, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(img, f"{f / fps:7.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(img)
    vw.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tracks", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/overlay.mp4"))
    p.add_argument("--scale", type=float, default=0.5)
    a = p.parse_args()

    d = dict(np.load(a.tracks))
    state, shots = detect(d)
    fps = float(d["fps"])
    per = defaultdict(int)
    for f, tid, *_ in shots:
        per[tid] += 1
        print(f"{f / fps:7.2f}s  track {tid}")
    print("per track:", dict(sorted(per.items())), "total:", len(shots))
    if a.video:
        render(a.video, d, state, shots, a.out, a.scale)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
