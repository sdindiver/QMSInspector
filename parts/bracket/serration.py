"""Offline serration present/missing classifier for Bracket parts.

Recall (perceptual-hash) only recognises near-duplicates of learned images, so a
genuinely new bracket photo lands in Needs Review with no serration answer. This
module adds a small trained MobileNetV2 classifier that generalises: given any
bracket image it predicts whether the big-hole rim serration is present or
missing. It is a second opinion layered on top of recall, never a replacement.

The model is trained offline by ``train_serration.py --final`` and saved to
``inspection_state/models/serration_bracket.pt``. If torch/torchvision or the
weights file are unavailable this module degrades gracefully (``predict`` returns
None) so the rest of the pipeline is never broken.

The classifier is trained on big-hole crops, so inference crops to the big hole
first (via ``holes.big_hole_crop``) and classifies only that region.
"""
from __future__ import annotations
import os
import threading

from inspector import settings

MODEL_PATH = os.path.join(os.path.dirname(settings.MODEL_PATH), "serration_bracket.pt")

# Above this probability of "missing" we assert a Serration Missing defect.
MISSING_THRESHOLD = 0.60

_LOCK = threading.Lock()
_STATE = {"loaded": False, "model": None, "meta": None, "tf": None}


def available():
    """True if the trained serration model file exists on disk."""
    return os.path.exists(MODEL_PATH)


def _load():
    if _STATE["loaded"]:
        return _STATE["model"] is not None
    with _LOCK:
        if _STATE["loaded"]:
            return _STATE["model"] is not None
        _STATE["loaded"] = True
        if not available():
            return False
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except Exception:
            return False
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            m = models.mobilenet_v2(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.3),
                                         nn.Linear(m.last_channel, len(ckpt["classes"])))
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            tf = transforms.Compose([
                transforms.Resize((ckpt.get("input", 224),) * 2),
                transforms.ToTensor(),
                transforms.Normalize(ckpt["mean"], ckpt["std"]),
            ])
            _STATE.update(model=m, meta=ckpt, tf=tf)
            return True
        except Exception:
            _STATE["model"] = None
            return False


def predict(path):
    """Return serration prediction for a bracket image or None if unavailable.

    The classifier is trained on big-hole crops, so inference crops to the big
    hole first (via holes.big_hole_crop) and classifies only that region. If the
    crop is unavailable it falls back to the whole image.

    {"status": "missing"|"present", "prob_missing": float, "confidence": int}
    """
    if not _load():
        return None
    try:
        import cv2
        import torch
        from PIL import Image
        from . import holes as HOLES
    except Exception:
        return None
    try:
        meta = _STATE["meta"]
        classes = meta["classes"]  # index 0 == "missing", 1 == "present"
        bgr = cv2.imread(path)
        crop = HOLES.big_hole_crop(bgr) if bgr is not None else None
        if crop is not None:
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(path).convert("RGB")
        x = _STATE["tf"](img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(_STATE["model"](x), dim=1)[0]
        prob_missing = float(probs[classes.index("missing")])
        status = "missing" if prob_missing >= MISSING_THRESHOLD else "present"
        conf = prob_missing if status == "missing" else (1.0 - prob_missing)
        return {"status": status, "prob_missing": round(prob_missing, 3),
                "confidence": int(round(conf * 100))}
    except Exception:
        return None
