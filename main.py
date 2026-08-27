import base64
import io
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import json

# ── Change this to your model path ───────────────────────────────────────────
MODEL_PATH = "model_final.pth"
LABELS_PATH = "labels.json"
# ─────────────────────────────────────────────────────────────────────────────

with open(LABELS_PATH) as f:
    _labels_config = json.load(f)

LABELS = _labels_config["classes"]
NUM_CLASSES = _labels_config["num_classes"]
MEAN, STD = 0.1307, 0.3081


def apply_orientation_fix(img: Image.Image) -> Image.Image:
    """
    Orientation correction is now handled at TRAINING-DATA-PREP time (see
    emnist_symbols_dataset.py: emnist_orientation_fix), not here. The
    training pipeline transforms EMNIST's raw storage orientation to match
    the canvas's natural upright drawing orientation, so the live image
    reaching this function should already be in the orientation the model
    expects — hence identity.

    If you swap in a differently-trained model and see systematic
    misclassifications, re-run sweep_test.py before assuming this needs
    to change.
    """
    return img

# ── Debug capture settings ────────────────────────────────────────────────
# While True, every drawn image is saved to debug_captures/ BEFORE any
# rotate/flip is applied. Use this to collect real samples for sweep_test.py,
# then set back to False once you've found the correct orientation.
CAPTURE_DEBUG_IMAGES = False
CAPTURE_DIR = "debug_captures"
# ─────────────────────────────────────────────────────────────────────────────


class LetterNet(nn.Module):
    """Improved deep architecture for 62-class case-sensitive recognition.

    Upgraded from the original 3-block model to handle the increased complexity
    of distinguishing uppercase, lowercase, and digits (62 classes vs 47).

    Key improvements:
    - 4 conv blocks instead of 3 (deeper feature extraction)
    - Doubled channel sizes: 64→128→256 (more representational capacity)
    - Deeper classifier with BatchNorm and higher dropout
    - Better suited for confusing pairs like O/o, I/l, 0/O, etc.
    """
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),

            # Block 3: 7x7 -> 3x3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),  # 7x7 -> 3x3, MPS-compatible
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 9, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load model once at startup
model = LetterNet(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

if CAPTURE_DEBUG_IMAGES:
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    print(f"⚠️  Debug capture is ON -- saving every drawn image to {CAPTURE_DIR}/")


app = FastAPI(title="Letter Recognizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG of the 28x28 rasterized stroke


class PredictResponse(BaseModel):
    character: str
    confidence: float
    top5: list[dict]


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    try:
        img_bytes = base64.b64decode(req.image.split(",")[-1])
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = img.resize((28, 28), Image.LANCZOS)

        # ── DEBUG CAPTURE ────────────────────────────────
        # Saves the image exactly as it looks BEFORE any rotate/flip,
        # so you can build a real labeled test set for sweep_test.py.
        # Rename each saved file to "<TRUE_LETTER>_anything.png" after
        # drawing, matching the order you drew letters in.
        if CAPTURE_DEBUG_IMAGES:
            img.save(f"{CAPTURE_DIR}/{int(time.time()*1000)}.png")
        # ────────────────────────────────────────────────

        # ── EMNIST orientation fix ──────────────────────
        # SINGLE SOURCE OF TRUTH for orientation correction. The frontend
        # no longer applies its own rotate/flip (see tester.html) — this is
        # the only place it happens now.
        #
        # STATUS: reset to identity (no-op) as of Phase 1 rebuild, because
        # the previous rotate(180) was tuned against images that had ALREADY
        # been rotated -90 + flipped by the frontend — i.e. it was tuned
        # against a bug, on only 12 samples. Not trustworthy.
        #
        # Run sweep_test.py against a properly labeled test set (see
        # debug_captures/) to determine the correct transform, then replace
        # `apply_orientation_fix` below with the winning one.
        img = apply_orientation_fix(img)
        # ────────────────────────────────────────────────

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    arr = np.array(img).astype(np.float32) / 255.0

    # ── Invert if needed (white bg → black bg) ─────────
    # Canvas sends white stroke on black bg already
    # but just in case:
    if arr.mean() < 0.5:
        pass   # already dark background, correct
    else:
        arr = 1.0 - arr   # invert if white background
    # ────────────────────────────────────────────────────

    arr = (arr - MEAN) / STD
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().tolist()

    ranked = sorted(
        [{"character": LABELS[i], "probability": round(p * 100, 1)}
         for i, p in enumerate(probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return PredictResponse(
        character=ranked[0]["character"],
        confidence=ranked[0]["probability"],
        top5=ranked[:5],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "classes": NUM_CLASSES}


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
