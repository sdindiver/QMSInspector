"""Offline multi-class defect classifier for Bearing Cup parts.

A trained MobileNetV2 (see ``train_defect.py``) that predicts the bearing-cup
defect type: Corrosion, Dent, Edge Cut, Missing Punch, Out-of-Round, or OK. It
is trained with full 0-360 degree rotation + flip augmentation, so it recognises
the defect at ANY angle -- unlike perceptual-hash recall which only matches
stored orientations. It is a second opinion layered on top of recall, never a
replacement, and degrades gracefully (``predict`` returns None) if
torch/torchvision or the weights are unavailable.

The classifier looks at the whole segmented part (a defect can appear anywhere on
the cup), with the background flattened so it focuses on the metal.
"""
from __future__ import annotations
import os
import threading

from inspector import settings

MODEL_PATH = os.path.join(os.path.dirname(settings.MODEL_PATH), "bearing_cup_defect.pt")

# The non-OK class must beat this probability before we assert a defect.
DEFECT_THRESHOLD = 0.55
# Number of augmented views averaged at inference (test-time augmentation).
TTA_VIEWS = 6
# Training used ImageNet normalisation and a 224px input (see train_defect.py).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT = 224

# Map internal training labels -> human-readable defect names for the verdict.
LABEL_DISPLAY = {
    "Corrosion": "Corrosion",
    "Dent": "Dent",
    "Edge_Cut": "Edge Cut",
    "Missing_Punch": "Missing Punch",
    "Out-of-Round": "Out-of-Round",
    "OK": "OK",
}

_LOCK = threading.Lock()
_STATE = {"loaded": False, "model": None, "meta": None, "tf": None, "aug": None}


def available():
    """True if the trained bearing-cup model file exists on disk."""
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
            mean = ckpt.get("mean", IMAGENET_MEAN)
            std = ckpt.get("std", IMAGENET_STD)
            inp = ckpt.get("input", INPUT)
            base = transforms.Compose([
                transforms.Resize((inp, inp)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            # Rotation/flip augmentation for test-time averaging -> angle robust.
            aug = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(180, fill=128),
                transforms.RandomResizedCrop(inp, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            _STATE.update(model=m, meta=ckpt, tf=base, aug=aug)
            return True
        except Exception:
            _STATE["model"] = None
            return False


def _whole_image(path):
    """Load the image exactly as training did: whole frame, just resized.

    IMPORTANT: ``train_defect.py`` fed the model the WHOLE resized image (no crop,
    no background flatten). Inference must match that preprocessing or the model
    sees an out-of-distribution input and predicts unreliably. So we deliberately
    do NOT crop/segment here.
    """
    import cv2
    from PIL import Image
    from inspector import image_features as F

    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = F._resize(bgr)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def predict(path):
    """Return the bearing-cup defect prediction or None if unavailable.

    {"status": "present"|"absent", "defect": str, "prob": float,
     "confidence": int, "probs": {class: prob}}
    - status "present" means a non-OK defect class won above DEFECT_THRESHOLD.
    - "defect" is the human-readable defect name (empty when absent).
    """
    if not _load():
        return None
    try:
        import torch
        import numpy as np
    except Exception:
        return None
    try:
        img = _whole_image(path)
        if img is None:
            return None
        classes = _STATE["meta"]["classes"]
        # Average softmax over the base view + several augmented views (TTA).
        acc = None
        with torch.no_grad():
            views = [_STATE["tf"](img).unsqueeze(0)]
            for _ in range(TTA_VIEWS):
                views.append(_STATE["aug"](img).unsqueeze(0))
            for v in views:
                p = torch.softmax(_STATE["model"](v), 1)[0].numpy()
                acc = p if acc is None else acc + p
        probs = acc / len(views)
        probs_by_class = {c: float(round(probs[i], 3)) for i, c in enumerate(classes)}

        # Best NON-OK class decides whether a defect is present.
        best_defect, best_prob = None, -1.0
        for i, c in enumerate(classes):
            if c == "OK":
                continue
            if probs[i] > best_prob:
                best_defect, best_prob = c, float(probs[i])

        if best_defect is not None and best_prob >= DEFECT_THRESHOLD:
            return {"status": "present",
                    "defect": LABEL_DISPLAY.get(best_defect, best_defect.replace("_", " ")),
                    "prob": round(best_prob, 3),
                    "confidence": int(round(best_prob * 100)),
                    "probs": probs_by_class}
        ok_prob = float(probs[classes.index("OK")]) if "OK" in classes else (1.0 - best_prob)
        return {"status": "absent", "defect": "",
                "prob": round(best_prob if best_prob >= 0 else 0.0, 3),
                "confidence": int(round(ok_prob * 100)),
                "probs": probs_by_class}
    except Exception:
        return None
