"""Simple HTTP server for SHARP WebUI.

Run with:
    python server.py [--checkpoint PATH] [--port 8080]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import socketserver
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply
from sharp.utils.io import convert_focallength, extract_exif

LOGGER = logging.getLogger(__name__)

VIEWER_VERSION = "1.26.3"
VIEWER_CDN_BASE = (
    f"https://cdn.jsdelivr.net/npm/@playcanvas/supersplat-viewer@{VIEWER_VERSION}/public"
)
VIEWER_DIR = Path("public/viewer")
OUTPUT_DIR = Path("output")

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css",
    ".js": "application/javascript",
    ".ply": "application/octet-stream",
    ".json": "application/json",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Global model state — set once at startup.
_model: object = None
_device: str = "cpu"
_inference_lock = threading.Lock()
_session_files: list[str] = []


def _ensure_viewer() -> None:
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["index.html", "index.js", "index.css"]:
        local_path = VIEWER_DIR / filename
        if not local_path.exists():
            url = f"{VIEWER_CDN_BASE}/{filename}"
            LOGGER.info("Downloading viewer asset: %s ...", filename)
            with urllib.request.urlopen(url) as resp:
                local_path.write_bytes(resp.read())
            LOGGER.info("Saved %s (%.1f KB)", filename, local_path.stat().st_size / 1024)


def _load_model(checkpoint_path: Path | None) -> None:
    global _model, _device

    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    LOGGER.info("Using device: %s", _device)

    if checkpoint_path:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)
    else:
        LOGGER.info("Downloading default checkpoint from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(_device)
    _model = predictor
    LOGGER.info("Model loaded and ready.")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/":
            self._serve_file(Path("public/index.html"))
        elif path == "/api/files":
            self._json_response(_session_files)
        elif path.startswith("/output/"):
            filename = Path(self.path.split("?")[0][len("/output/"):]).name
            self._serve_file(OUTPUT_DIR / filename)
        elif path.startswith("/viewer"):
            sub = path[len("/viewer"):] or "/index.html"
            if not sub.startswith("/"):
                sub = "/" + sub
            self._serve_file(VIEWER_DIR / sub.lstrip("/"))
        elif path.startswith("/public/"):
            self._serve_file(Path(path.lstrip("/")))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/predict":
            self._handle_predict()
        else:
            self.send_error(404)

    def _handle_predict(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Decode base64 image (strip optional data-URI prefix).
        raw_b64 = data.get("image", "")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception:
            self.send_error(400, "Invalid base64 image data")
            return

        try:
            img_pil = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            self.send_error(400, f"Cannot decode image: {exc}")
            return

        # Auto-rotate from EXIF orientation.
        exif_data: dict = {}
        try:
            exif_data = extract_exif(img_pil)
            orientation = exif_data.get("Orientation", 1)
            if orientation == 3:
                img_pil = img_pil.transpose(Image.ROTATE_180)
            elif orientation == 6:
                img_pil = img_pil.transpose(Image.ROTATE_270)
            elif orientation == 8:
                img_pil = img_pil.transpose(Image.ROTATE_90)
        except Exception:
            pass

        img_np = np.array(img_pil)
        if img_np.ndim < 3:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_np = img_np[:, :, :3]
        h, w = img_np.shape[:2]

        # Resolve focal length (explicit override → EXIF → default 30 mm).
        focal_mm_override = data.get("focal_length_mm")
        if focal_mm_override:
            f_px = convert_focallength(w, h, float(focal_mm_override))
        else:
            f_35mm = (
                exif_data.get("FocalLengthIn35mmFilm")
                or exif_data.get("FocalLenIn35mmFilm")
                or exif_data.get("FocalLength")
                or 30.0
            )
            if f_35mm < 1:
                f_35mm = 30.0
            elif f_35mm < 10:
                f_35mm *= 8.4
            f_px = convert_focallength(w, h, float(f_35mm))

        # Build output filename from the client-provided original name + timestamp.
        orig_stem = Path(data.get("filename", "predict")).stem
        # Strip any characters that could cause filesystem issues.
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in orig_stem)
        timestamp = int(time.time() * 1000)
        out_name = f"{safe_stem}_{timestamp}.ply"
        out_path = OUTPUT_DIR / out_name

        try:
            with _inference_lock:
                gaussians = predict_image(_model, img_np, f_px, torch.device(_device))
            save_ply(gaussians, f_px, (h, w), out_path)
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            self.send_error(500, str(exc))
            return

        _session_files.append(out_name)
        LOGGER.info("Saved %s", out_path)
        self._json_response({"filename": out_name, "f_px": round(f_px, 2)})

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        content_type = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, obj: object) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:  # suppress default stdout noise
        LOGGER.debug("%s - " + fmt, self.address_string(), *args)


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="SHARP WebUI server")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--port", type=int, default=8188, help="Port to listen on (default: 8188)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    Path("public").mkdir(exist_ok=True)

    LOGGER.info("Ensuring supersplat viewer assets are present...")
    _ensure_viewer()

    LOGGER.info("Loading SHARP model (this may take a moment on first run)...")
    _load_model(args.checkpoint)

    LOGGER.info("Listening on http://localhost:%d", args.port)
    with _ThreadingServer(("127.0.0.1", args.port), _Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
