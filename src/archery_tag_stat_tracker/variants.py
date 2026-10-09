"""Lens-correction variants vs arrow-flight quality.

For each variant: render the corrected video, carry the player tracks, labels
and neutral-zone geometry into its coordinates, run the arrow detector with
identical settings, then score the flights it finds:

* detection: real (labeled) shots found, time+direction matched
* flight quality of the real arrows: points per flight, length, px off a
  straight line, px off a gravity arc (parabola)
* attribution: right shooter at the right time

    uv run python -m archery_tag_stat_tracker.variants data/stitch/va-S1pJyK5U.json \\
        out/tracks_v2.npz data/labels_0-60s.json --video data/sample_2min.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import arrows
from . import lens as L
from .attribute import attribute
from .evaluate import score, score_arrows
from .straightness import bend

VARIANTS = {
    "raw": None,
    "raw_resampled": "resample",  # control: half-pixel shift, i.e. same resampling blur, no geometry change
    "T4": {},
    "T4_k2-0.08": {"equipment": {"k2": -0.08}},
    "T4_k2-0.15": {"equipment": {"k2": -0.15}},
    "T4_k1+0.3": {"equipment": {"k1": 0.3}},
}


class _Resample:
    """Identity geometry, but frames go through a bilinear half-pixel remap like a real correction."""
    def __init__(self, W=3490, H=1400):
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.maps = (gx + 0.5, gy + 0.5)

    def frame(self, img):
        return cv2.remap(img, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def points(self, p):
        return np.asarray(p, float).reshape(-1, 2) - 0.5


def lens_for(stitch, over):
    if over is None:
        return None
    if over == "resample":
        return _Resample()
    eq = {**stitch.get("equipment", {}), **over.get("equipment", {})}
    return L.build({**stitch, **over, "equipment": eq})


def render(video: Path, lens, out: Path, seconds: float):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
         "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE)
    for _ in range(int(seconds * fps)):
        ok, img = cap.read()
        if not ok:
            break
        enc.stdin.write(lens.frame(img).tobytes())
    enc.stdin.close()
    enc.wait()


def map_tracks(d: dict, fix) -> dict:
    out = dict(d)
    b = d["box"]
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], 1).reshape(-1, 2)
    c = fix(corners).reshape(-1, 4, 2)
    out["box"] = np.concatenate([c.min(1), c.max(1)], 1)
    kp = d["kp"].copy()
    kp[..., :2] = fix(kp[..., :2].reshape(-1, 2)).reshape(kp.shape[0], kp.shape[1], 2)
    out["kp"] = kp
    return out


def geometry(stitch, d_mapped, fix, size):
    W, H = size
    top = float(np.mean(fix(np.array(stitch["baseline"]["left"] + stitch["baseline"]["right"], float))[:, 1]))
    edges = []
    for s in ("left", "right"):
        p = fix(np.array(stitch["lines"]["zone_line"][s], float))
        a, b = np.polyfit(p[:, 1], p[:, 0], 1)  # x = a y + b
        edges.append([[a * top + b, top], [a * (H - 12) + b, H - 12]])
    xb = [edges[0][1][0], edges[1][1][0]]
    band = [max(0, int(min(xb) - 110)), min(W, int(max(xb) + 110))]
    cy = (d_mapped["box"][:, 1] + d_mapped["box"][:, 3]) / 2
    hh = d_mapped["box"][:, 3] - d_mapped["box"][:, 1]
    hy = []
    for lo in range(0, H, 150):
        m = (cy >= lo) & (cy < lo + 150)
        if m.sum() >= 30:
            hy.append([lo + 75, float(np.median(hh[m]))])
    return {"band": band, "zone_left": edges[0], "zone_right": edges[1], "height_at_y": hy}


def evaluate(flights, d, labels, t_max):
    out = {"per_label": {}}
    for mp in (4, 5, 6):
        r = score_arrows(flights, labels, t_max, min_points=mp, min_nspeed=0.25)
        real = [r["shots"][si] for _, si, _ in r["matched"]]
        if mp == 4:  # per labeled shot, for paired comparisons across variants
            for li, si, _ in r["matched"]:
                s = r["shots"][si]
                out["per_label"][li] = list(bend(np.array([p[1:3] for p in s["pts"]], float))) + [s["n"]]
        q = [bend(np.array([p[1:3] for p in s["pts"]], float)) + (s["n"],) for s in real]
        sh = attribute(d, flights, None, min_points=mp, min_nspeed=0.25)
        a = score(d, [(s["frame"], s["tid"]) for s in sh], labels, t_max)
        out[mp] = {
            "detected": r["detected"], "real": r["tp"], "recall": r["recall"], "precision": r["precision"],
            "median_points": float(np.median([x[3] for x in q])) if q else 0,
            "median_length_px": float(np.median([x[2] for x in q])) if q else 0,
            "median_line_px": float(np.median([x[0] for x in q])) if q else 0,
            "median_arc_px": float(np.median([x[1] for x in q])) if q else 0,
            "who_recall": a["recall"], "who_precision": a["precision"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stitch", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--video", type=Path, default=Path("data/sample_2min.mp4"))
    ap.add_argument("--seconds", type=float, default=61.0)
    ap.add_argument("--work", type=Path, default=Path("out/variants"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    stitch = json.loads(a.stitch.read_text())
    d0 = dict(np.load(a.tracks))
    labels0 = json.loads(a.labels.read_text())
    t_max = max(l["t"] for l in labels0) + 1.0
    results = {}
    for name, over in VARIANTS.items():
        if a.only and name not in a.only:
            continue
        work = a.work / name
        work.mkdir(parents=True, exist_ok=True)
        lens = lens_for(stitch, over)
        fix = (lambda p: np.asarray(p, float).reshape(-1, 2)) if lens is None else lens.points
        video = a.video if lens is None or over == "resample" else work / "video.mp4"
        if lens is not None and over != "resample" and not video.exists():
            print(f"[{name}] rendering", flush=True)
            render(a.video, lens, video, a.seconds)
        d = map_tracks(d0, fix)
        np.savez_compressed(work / "tracks.npz", **d)
        geo = geometry(stitch, d, fix, (3490, 1400))
        (work / "geometry.json").write_text(json.dumps(geo))
        labels = [{**l, "x": float(q[0]), "y": float(q[1])}
                  for l, q in zip(labels0, fix(np.array([[l["x"], l["y"]] for l in labels0], float)))]
        (work / "labels.json").write_text(json.dumps(labels))
        arrows.configure(geo)
        print(f"[{name}] detecting arrows", flush=True)
        # correct frames on the fly from the original video (no re-encode), so every
        # variant sees the same pixels except for the lens remap itself
        fps, fl = arrows.track(a.video, 0.0, 60.0, work / "tracks.npz", None if lens is None else lens.frame)
        summ = [f.summary(fps) for f in fl if len(f.pts) >= arrows.MIN_POINTS - 1]
        (work / "arrows.json").write_text(json.dumps({"fps": fps, "flights": summ}))
        results[name] = evaluate(summ, d, labels, t_max)
        r = results[name][5]
        print(f"[{name}] real shots {r['real']}/{len(labels)} (precision {r['precision']:.0%}), "
              f"median {r['median_points']:.0f} pts / {r['median_length_px']:.0f}px, "
              f"line {r['median_line_px']:.2f}px arc {r['median_arc_px']:.2f}px, "
              f"who {r['who_recall']:.0%}/{r['who_precision']:.0%}", flush=True)
    prev = json.loads((a.work / "results.json").read_text()) if (a.work / "results.json").exists() else {}
    allr = {**prev, **results}
    (a.work / "results.json").write_text(json.dumps(allr, indent=1, default=str))
    # paired: labeled shots whose arrow was found in every variant
    names = [n for n in VARIANTS if n in allr]
    common = set.intersection(*[set(map(str, allr[n]["per_label"])) for n in names])
    print(f"\npaired over {len(common)} labeled shots found in all {len(names)} variants:")
    for n in names:
        pl = allr[n]["per_label"]
        v = np.array([pl[k] if k in pl else pl[int(k)] for k in sorted(common, key=int)], float) if common else np.zeros((0, 4))
        if len(v):
            print(f"  {n:13s} line {np.median(v[:, 0]):5.2f}px  arc {np.median(v[:, 1]):5.2f}px  "
                  f"length {np.median(v[:, 2]):5.0f}px  points {np.median(v[:, 3]):4.1f}  | "
                  f"line/length {np.median(v[:, 0] / v[:, 2]) * 1000:5.2f}‰  arc/length {np.median(v[:, 1] / v[:, 2]) * 1000:5.2f}‰")


if __name__ == "__main__":
    main()
