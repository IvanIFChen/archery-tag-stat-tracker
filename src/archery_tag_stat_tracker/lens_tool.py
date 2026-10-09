"""Interactive lens-correction tuner.

Serves lens_tool.html, which re-runs the lens.py model live in WebGL while you
move sliders. Save writes the values into the video's stitch profile and
rebuilds data/lens.json, so the pipeline uses exactly what you saw.

    uv run python -m archery_tag_stat_tracker.lens_tool data/stitch/va-S1pJyK5U.json \\
        --video data/sample_2min.mp4      # -> http://127.0.0.1:8766/
"""

import argparse
import http.server
import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from . import lens as L

PAGE = Path(__file__).with_name("lens_tool.html")

COLORS = {"zone_line": "#33ff66", "near_wall": "#00e5ff", "side_wall": "#ffa500"}
NAMES = {"zone_line": "zone lines", "near_wall": "near walls (base of the wall the cameras are on)",
         "side_wall": "side-wall bases"}


def background(video: Path, n: int = 60) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for t in np.linspace(0.01, 0.99, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * total))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def prepare_images(video: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    names = []
    bg = out / "background.jpg"
    if not bg.exists():
        cv2.imwrite(str(bg), background(video), [cv2.IMWRITE_JPEG_QUALITY, 92])
    names.append("background.jpg")
    cap = cv2.VideoCapture(str(video))
    for t in (5, 30, 60, 100):
        p = out / f"frame_{t:03d}s.jpg"
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names.append(p.name)
    return names


class Handler(http.server.SimpleHTTPRequestHandler):
    stitch_path: Path
    lens_out: Path
    images: list

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/config":
            stitch = json.loads(self.stitch_path.read_text())
            eq = {**L.EQUIPMENT, **stitch.get("equipment", {})}
            st = {**L.STITCH_DEFAULTS, **stitch}
            st["camera"] = {**L.STITCH_DEFAULTS["camera"], **stitch.get("camera", {})}
            sw = {"enabled": True, **(stitch.get("seam_warp") or {})}
            pins = {h: {**L.STITCH_DEFAULTS["corner_pin"][h], **(stitch.get("corner_pin") or {}).get(h, {})}
                    for h in ("left", "right")}
            par = {**L.STITCH_DEFAULTS["parallelogram"], **(stitch.get("parallelogram") or {})}
            quad = {**L.STITCH_DEFAULTS["quad"], **(stitch.get("quad") or {})}
            lines = stitch.get("lines", {})
            features = {NAMES[k]: {"key": k, "color": COLORS[k], "lines": [v["left"], v["right"]]}
                        for k, v in lines.items()}
            self._json({"size": [3490, 1400], "equipment": eq, "seam": st["seam"], "shear": st["shear"],
                        "align": st["align"], "collinear": st["collinear"], "camera": st["camera"],
                        "seam_warp": sw, "corner_pin": pins, "parallelogram": par, "quad": quad, "lines": lines,
                        "baseline": st["baseline"], "features": features,
                        "images": self.images, "defaults": {"equipment": L.EQUIPMENT, **L.STITCH_DEFAULTS}})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/save":
            return self.send_error(404)
        new = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        stitch = json.loads(self.stitch_path.read_text())
        for k in ("equipment", "seam", "shear", "align", "collinear", "camera", "seam_warp", "corner_pin", "parallelogram", "quad"):
            if k in new:
                stitch[k] = new[k]
        self.stitch_path.write_text(json.dumps(stitch, indent=1))
        lens = L.build(stitch)
        lens.save(self.lens_out)
        b = lens.points(np.array(stitch["baseline"]["left"] + stitch["baseline"]["right"], float))
        m, c = np.polyfit(*b.T, 1)
        self._json({"ok": True, "saved": str(self.stitch_path), "lens": str(self.lens_out),
                    "baseline_rms": float(np.sqrt(np.mean((b[:, 1] - (m * b[:, 0] + c)) ** 2)))})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stitch", type=Path, help="data/stitch/<video>.json")
    p.add_argument("--video", type=Path, default=Path("data/sample_2min.mp4"))
    p.add_argument("--lens-out", type=Path, default=Path("data/lens.json"))
    p.add_argument("--work", type=Path, default=Path("out/lens_tool"))
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    images = prepare_images(a.video, a.work)
    Handler.stitch_path, Handler.lens_out, Handler.images = a.stitch, a.lens_out, images
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=str(a.work)))
    print(f"http://127.0.0.1:{a.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
