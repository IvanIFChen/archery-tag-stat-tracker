"""Serve the shot-labeling page; labels are saved to <dir>/labels.json."""

import argparse
import http.server
from functools import partial
from pathlib import Path

PAGE = Path(__file__).with_name("label.html")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/labels" and not (Path(self.directory) / "labels.json").exists():
            self.send_error(404)
        elif self.path == "/clip.mp4":
            self.send_video(Path(self.directory) / "clip.mp4")
        else:
            if self.path == "/labels":
                self.path = "/labels.json"
            super().do_GET()

    def send_video(self, path: Path):
        # Browsers need Range (206) responses to seek within a video.
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else size - int(b)
            end = int(b) if a and b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0 and (chunk := f.read(min(left, 1 << 20))):
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_POST(self):
        if self.path != "/labels":
            return self.send_error(404)
        data = self.rfile.read(int(self.headers["Content-Length"]))
        (Path(self.directory) / "labels.json").write_bytes(data)
        self.send_response(204)
        self.end_headers()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path, nargs="?", default=Path("out/label"), help="folder with clip.mp4")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=str(a.dir)))
    print(f"http://127.0.0.1:{a.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
