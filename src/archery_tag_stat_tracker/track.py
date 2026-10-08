"""Detect and track players with pose keypoints; save per-frame results to .npz."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# Field region in the 3490x1400 source frame (x0, y0, x1, y1). Cropping before
# inference gives players more pixels at the same model input size.
ROI = (300, 0, 3300, 1000)
# Far end of the field, run again in overlapping tiles upscaled ~2x: far
# players are ~70-100px tall and are often missed mid-draw at 1x.
FAR_TILES = [(300, 0, 1900, 500), (1700, 0, 3300, 500)]
NMS_IOU = 0.5


def infer(model, img, region, imgsz, conf):
    x0, y0, x1, y1 = region
    r = model.predict(img[y0:y1, x0:x1], imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)[0]
    off = np.array([x0, y0], dtype=np.float32)
    box = r.boxes.xyxy.cpu().numpy() + np.tile(off, 2)
    kp = r.keypoints.data.cpu().numpy().copy()
    kp[..., :2] += off
    return box, r.boxes.conf.cpu().numpy(), kp


def detect(model, img, imgsz, conf):
    parts = [infer(model, img, reg, imgsz, conf) for reg in [ROI, *FAR_TILES]]
    box = np.concatenate([p[0] for p in parts])
    score = np.concatenate([p[1] for p in parts])
    kp = np.concatenate([p[2] for p in parts])
    keep = nms(torch.from_numpy(box), torch.from_numpy(score), NMS_IOU).numpy()
    return box[keep], score[keep], kp[keep]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--duration", type=float, default=60.0, help="seconds")
    p.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    p.add_argument("--model", default="models/yolo11s-pose.pt")
    p.add_argument("--imgsz", type=int, default=3008)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=Path("out/tracks.npz"))
    a = p.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    first = int(round(a.start * fps))
    n_frames = int(a.duration * fps)

    model = YOLO(a.model)
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    cfg.new_track_thresh = 0.2
    cfg.track_buffer = 60  # keep lost tracks ~2s (players duck behind bunkers)
    tracker = BYTETracker(cfg)
    rows = {"frame": [], "tid": [], "box": [], "conf": [], "kp": []}

    t0 = time.time()
    for i in range(n_frames):
        if i % a.stride:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        box, score, kp = detect(model, img, a.imgsz, a.conf)
        data = np.concatenate([box, score[:, None], np.zeros((len(box), 1))], axis=1)
        tracks = tracker.update(Boxes(data, img.shape[:2]).cpu().numpy(), img)
        if len(tracks):
            idx = tracks[:, -1].astype(int)
            rows["frame"] += [first + i] * len(tracks)
            rows["tid"] += tracks[:, 4].astype(int).tolist()
            rows["box"].append(box[idx])
            rows["conf"].append(score[idx])
            rows["kp"].append(kp[idx])
        if i % (a.stride * 300) == 0:
            print(f"{i / fps:6.1f}s  {time.time() - t0:6.1f}s elapsed", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, fps=fps, stride=a.stride,
        frame=np.array(rows["frame"]), tid=np.array(rows["tid"]),
        box=np.concatenate(rows["box"]), conf=np.concatenate(rows["conf"]),
        kp=np.concatenate(rows["kp"]),
    )
    print(f"saved {len(rows['frame'])} detections to {a.out}")


if __name__ == "__main__":
    main()
