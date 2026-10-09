"""Render "long exposure" images of arrow flights for docs and debugging.

For a short window, every pixel shows the largest change from the background
seen during the window (an arrow becomes a streak). A second panel colors each
changed pixel by the frame it first changed in (blue = early, red = late), so
the direction of travel is visible. Detected flights are drawn on top.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .arrows import ZONE_LEFT, ZONE_RIGHT, dedupe, is_shot


def read(cap, fps, t0, t1, step=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
    out = []
    for k in range(int((t1 - t0) * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            out.append(f)
    return out


def render(video: Path, t: float, crop, window=(-0.1, 0.7), flights=(), label=None, title=""):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    bg = np.median(np.stack(read(cap, fps, t - 1.6, t - 1.0, 6)), axis=0).astype(np.int16)
    frames = read(cap, fps, t + window[0], t + window[1])
    comp = np.zeros(bg.shape[:2], np.int16)
    first = np.full(bg.shape[:2], -1, np.int16)
    for k, f in enumerate(frames):
        d = np.abs(f.astype(np.int16) - bg).max(axis=2)
        first[(d > 40) & (first < 0)] = k
        comp = np.maximum(comp, d)

    x0, y0, x1, y1 = crop
    still = frames[min(len(frames) - 1, int(-window[0] * fps))].copy()
    exp = cv2.applyColorMap(np.clip(comp * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    hsv = np.zeros((*first.shape, 3), np.uint8)
    m = first >= 0
    hsv[..., 0][m] = (120 - first[m] * 120 // max(1, len(frames))).astype(np.uint8)
    hsv[..., 1][m] = 255
    hsv[..., 2][m] = 255
    timed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    for img in (still, exp, timed):
        for edge in (ZONE_LEFT, ZONE_RIGHT):
            cv2.line(img, *[tuple(map(int, p)) for p in edge], (255, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.circle(still, (label["x"], label["y"]), 40, (0, 0, 255), 4, cv2.LINE_AA)
    for fl in flights:
        pts = np.array([p[1:] for p in fl["pts"]], np.int32)
        for img in (exp, timed):
            cv2.polylines(img, [pts], False, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.arrowedLine(img, tuple(pts[-2]), tuple(pts[-1]), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.5)

    names = ("frame at release", "long exposure", "first change: blue=early, red=late")
    panels = []
    for img, name in zip((still, exp, timed), names):
        c = img[y0:y1, x0:x1].copy()
        cv2.rectangle(c, (0, 0), (c.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(c, f"{title}  {name}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(c)
    return np.hstack(panels) if (x1 - x0) < (y1 - y0) * 1.2 else np.vstack(panels)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("t", type=float, help="seconds (labeled release)")
    p.add_argument("--crop", default="900,0,2750,1000", help="x0,y0,x1,y1 in source pixels")
    p.add_argument("--arrows", type=Path, help="arrows.py output; draws flights in the window")
    p.add_argument("--labels", type=Path, help="circles the labeled shooter nearest t")
    p.add_argument("--title", default="")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    crop = tuple(map(int, a.crop.split(",")))
    window = (-0.1, 0.7)
    flights = []
    if a.arrows:
        fl = json.loads(a.arrows.read_text())["flights"]
        flights = [s for s in dedupe([s for s in fl if is_shot(s)])
                   if a.t + window[0] <= s["t0"] <= a.t + window[1]]
    label = None
    if a.labels:
        label = min(json.loads(a.labels.read_text()), key=lambda l: abs(l["t"] - a.t))
    img = render(a.video, a.t, crop, window, flights, label, a.title)
    img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out} ({len(flights)} flights)")


if __name__ == "__main__":
    main()
