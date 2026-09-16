"""Offline dark-spots present/absent classifier for Bracket parts.

A trained MobileNetV2 (see ``train_dark_spots.py``) that predicts whether a
bracket surface has a "Dark Spots" defect. It is trained with full 0-360 degree
rotation + flip augmentation, so it recognises the defect at ANY angle -- unlike
perceptual-hash recall which only matches stored orientations. It is a second
opinion layered on top of recall, never a replacement, and degrades gracefully
(``predict`` returns None) if torch/torchvision or the weights are unavailable.

The classifier looks at the whole segmented part (dark spots can appear anywhere
on the surface), with the background flattened so it focuses on the metal.
"""
from __future__ import annotations
import os
import threading

from inspector import settings

MODEL_PATH = os.path.join(os.path.dirname(settings.MODEL_PATH), "dark_spots_bracket.pt")

# Above this probability of "dark_spots" we assert a Dark Spots defect.
SPOT_THRESHOLD = 0.55
# Number of augmented views averaged at inference (test-time augmentation).
TTA_VIEWS = 6

_LOCK = threading.Lock()
_STATE = {"loaded": False, "model": None, "meta": None, "tf": None, "aug": None}


def available():
    """True if the trained dark-spots model file exists on disk."""
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
            inp = ckpt.get("input", 224)
            base = transforms.Compose([
                transforms.Resize((inp, inp)),
                transforms.ToTensor(),
                transforms.Normalize(ckpt["mean"], ckpt["std"]),
            ])
            # Rotation/flip augmentation for test-time averaging -> angle robust.
            aug = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(180, fill=128),
                transforms.RandomResizedCrop(inp, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(ckpt["mean"], ckpt["std"]),
            ])
            _STATE.update(model=m, meta=ckpt, tf=base, aug=aug)
            return True
        except Exception:
            _STATE["model"] = None
            return False


def _part_crop(path):
    import cv2
    import numpy as np
    from PIL import Image
    from inspector import image_features as F

    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = F._resize(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask, cnt = F.segment_part(gray)
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        pad = int(0.04 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
        flat = np.where(m3, bgr, 128).astype(np.uint8)
        crop = flat[y0:y1, x0:x1]
    else:
        crop = bgr
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def predict(path):
    """Return dark-spots prediction for a bracket image or None if unavailable.

    {"status": "present"|"absent", "prob_spot": float, "confidence": int}
    """
    if not _load():
        return None
    try:
        import torch
        import numpy as np
    except Exception:
        return None
    try:
        img = _part_crop(path)
        if img is None:
            return None
        classes = _STATE["meta"]["classes"]  # index 1 == "dark_spots"
        si = classes.index("dark_spots")
        probs = []
        with torch.no_grad():
            probs.append(float(torch.softmax(_STATE["model"](
                _STATE["tf"](img).unsqueeze(0)), 1)[0, si]))
            for _ in range(TTA_VIEWS):
                probs.append(float(torch.softmax(_STATE["model"](
                    _STATE["aug"](img).unsqueeze(0)), 1)[0, si]))
        prob = float(np.mean(probs))
        status = "present" if prob >= SPOT_THRESHOLD else "absent"
        conf = prob if status == "present" else (1.0 - prob)
        return {"status": status, "prob_spot": round(prob, 3),
                "confidence": int(round(conf * 100))}
    except Exception:
        return None
