"""
Interactive Live Online Handwriting Recognition Web UI.

Uses the trained model (./model/best_model.pt) to recognize text from live
handwritten strokes drawn on an interactive HTML5 canvas.

Usage:
    python app.py
    python app.py --port 8080 --host 0.0.0.0
"""

import argparse
import json
import os
import random
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

from src import config
from src.evaluate import load_model, predict
from src.data_parser import parse_stroke_xml

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# Global model instance
MODEL = None
MODEL_INFO = None

# Preloaded sample stroke files for the "Load Sample" button
SAMPLE_FILES = []


def find_sample_xml_files():
    """Find a selection of stroke XML files from datasets to use as presets."""
    sample_files = []
    strokes_dir = config.STROKES_DIR
    if os.path.isdir(strokes_dir):
        for root, _, files in os.walk(strokes_dir):
            for f in files:
                if f.endswith(".xml"):
                    sample_files.append(os.path.join(root, f))
                    if len(sample_files) >= 100:
                        break
            if len(sample_files) >= 100:
                break
    return sample_files


class HandwritingRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving frontend static files and prediction API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.lstrip("/")

        # Default to index.html
        if not path or path == "index.html":
            file_path = os.path.join(FRONTEND_DIR, "index.html")
            content_type = "text/html; charset=utf-8"
        elif path == "style.css":
            file_path = os.path.join(FRONTEND_DIR, "style.css")
            content_type = "text/css; charset=utf-8"
        elif path == "app.js":
            file_path = os.path.join(FRONTEND_DIR, "app.js")
            content_type = "application/javascript; charset=utf-8"
        elif path == "api/sample":
            # API endpoint returning random sample strokes
            sample_strokes = []
            if SAMPLE_FILES:
                xml_path = random.choice(SAMPLE_FILES)
                try:
                    sample_strokes = parse_stroke_xml(xml_path)
                except Exception as e:
                    print(f"Failed to parse sample {xml_path}: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"strokes": sample_strokes}).encode("utf-8"))
            return
        else:
            # Check if file exists in frontend/
            file_path = os.path.join(FRONTEND_DIR, path)
            if os.path.isfile(file_path):
                if file_path.endswith(".css"):
                    content_type = "text/css"
                elif file_path.endswith(".js"):
                    content_type = "application/javascript"
                elif file_path.endswith(".png"):
                    content_type = "image/png"
                else:
                    content_type = "text/plain"
            else:
                self.send_response(404)
                self.end_headers()
                return

        # Serve static file
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/predict":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len)

            try:
                data = json.loads(post_body.decode("utf-8"))
                strokes = data.get("strokes", [])

                t0 = time.time()
                predicted_text = ""
                if strokes and MODEL is not None:
                    predicted_text = predict(MODEL, strokes)
                latency_ms = round((time.time() - t0) * 1000, 1)

                response = {
                    "text": predicted_text,
                    "latency_ms": latency_ms,
                    "num_strokes": len(strokes),
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        # Support CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Clean server logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def start_server(host="127.0.0.1", port=7860, checkpoint_path=None):
    """Start the Handwriting Recognition HTTP Server."""
    global MODEL, MODEL_INFO, SAMPLE_FILES

    print("=" * 70)
    print("Online Handwriting to Text — Live Recognition Web UI")
    print("=" * 70)

    # 1. Load Model
    ckpt_path = checkpoint_path or os.path.join(config.PROJECT_ROOT, "model", "best_model.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    print(f"\n[1/2] Loading model from: {ckpt_path}")
    MODEL, MODEL_INFO = load_model(ckpt_path)
    print(f"  ✓ Model ready! (Epoch {MODEL_INFO.get('epoch', '?')}, Val CER: {MODEL_INFO.get('val_cer', '?'):.4f})")

    # 2. Find sample stroke files
    print("\n[2/2] Discovering sample stroke files...")
    SAMPLE_FILES = find_sample_xml_files()
    print(f"  ✓ Found {len(SAMPLE_FILES)} sample stroke files")

    # 3. Start HTTP Server
    server_address = (host, port)
    httpd = HTTPServer(server_address, HandwritingRequestHandler)

    url = f"http://{'localhost' if host in ['0.0.0.0', '127.0.0.1'] else host}:{port}"
    print("\n" + "=" * 70)
    print(f"🚀 Web UI running at: {url}")
    print(f"📁 Serving frontend files from: {FRONTEND_DIR}")
    print("=" * 70 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Handwriting Recognition Web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port number (default: 7860)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)

