"""Offline any-angle Line-Mark present/absent detector for Bracket parts.

Line marks are thin scratches. A single model can't separate them from the
brushed-metal finish reliably when only 5 defect examples exist, so this uses a
TWO-MODEL ensemble whose members fail on *different* parts and vote with OR:

1. Ridge-CNN (``train_line_mark.py`` -> ``line_mark_bracket.pt``): a MobileNetV2
   trained on 3-channel ridge-enhanced crops [grey, ridge map, Hough mask] with
   full 0-360 rotation augmentation, so it recognises the pattern at ANY angle.
2. Geometric longest-line: segment the part, flatten the background, suppress the
   rim and holes, run black-hat/top-hat ridge morphology + probabilistic Hough,
   and measure the longest coherent line normalised by part length. Rotation
   invariant by construction.

A part is flagged when EITHER detector fires. This matches the winning
leave-one-out result (CNN catches 163734/163750/164510, geometry catches
163810/163750/164510 -> union 4/5) while keeping false alarms low and defensible.

Both members degrade gracefully: if torch/weights are missing the CNN is skipped
and the geometric detector still votes. ``predict`` returns None only if the
image can't be read/segmented.
"""
from __future__ import annotations
import os
import threading

from inspector import settings

MODEL_PATH = os.path.join(os.path.dirname(settings.MODEL_PATH), "line_mark_bracket.pt")

# CNN probability above which the ridge-CNN votes "line mark".
CNN_THRESHOLD = 0.5
# Longest normalised line length above which the geometric detector votes.
# Native-orientation leave-one-out: line marks 0.155-0.207, top OK part 0.149.
GEOM_THRESHOLD = 0.152
# Number of augmented views averaged for the CNN at inference.
TTA_VIEWS = 6

_LOCK = threading.Lock()
_STATE = {"loaded": False, "model": None, "meta": None}


def available():
    """True if either ensemble member can run (geometry always can)."""
    return True


def _load_cnn():
    if _STATE["loaded"]:
        return _STATE["model"] is not None
    with _LOCK:
        if _STATE["loaded"]:
            return _STATE["model"] is not None
        _STATE["loaded"] = True
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except Exception:
            return False
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            m = models.mobilenet_v2(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.3),
                                         nn.Linear(m.last_channel, len(ckpt["classes"])))
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            _STATE.update(model=m, meta=ckpt)
            return True
        except Exception:
            _STATE["model"] = None
            return False


def _ridge_prep(path):
    """Return (grey_flattened, valid_mask, part_axis, contour) or None."""
    import cv2
    import numpy as np
    from inspector import image_features as F

    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = F._resize(bgr)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask, cnt = F.segment_part(g)
    if cnt is None or (mask > 0).sum() < 500:
        return None
    part = mask > 0
    axis = F._part_axis(cnt)
    med = int(np.median(g[part]))
    g2 = g.copy()
    g2[~part] = med
    er = max(20, int(0.045 * axis))
    inner = cv2.erode(mask, np.ones((er, er), np.uint8)) > 0
    holes = ((g < med - 60) & part).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    holes = cv2.dilate(holes, np.ones((max(5, int(0.015 * axis)),) * 2, np.uint8))
    valid = inner & (holes == 0)
    return g2, valid, axis, cnt


def _ridge_map(g2, valid):
    import cv2
    import numpy as np
    bh = cv2.morphologyEx(g2, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    th = cv2.morphologyEx(g2, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    r = np.maximum(bh, th).astype(np.float32)
    r[~valid] = 0
    return r


def _geom_score(prep):
    """Longest coherent ridge line, normalised by part length."""
    import cv2
    import numpy as np
    g2, valid, axis, _ = prep
    r = _ridge_map(g2, valid)
    rr = r[valid]
    thr = np.percentile(rr, 97) if rr.size else 255
    edges = (r > thr).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=40, maxLineGap=8)
    best = 0.0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            best = max(best, d)
    return best / axis if axis else 0.0


def _cnn_prob(path):
    """Ridge-CNN probability of line mark, TTA-averaged, or None if unavailable."""
    if not _load_cnn():
        return None
    try:
        import torch
        import numpy as np
        from torchvision import transforms
    except Exception:
        return None
    try:
        from parts.bracket.train_line_mark import ridge_stack, INPUT
    except Exception:
        return None
    try:
        stack = ridge_stack(path)  # HxWx3 uint8
        if stack is None:
            return None
        meta = _STATE["meta"]
        classes = meta["classes"]
        li = classes.index("line_mark")
        norm = transforms.Normalize(meta["mean"], meta["std"])
        import cv2

        def to_tensor(img):
            t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            return norm(t)

        probs = []
        with torch.no_grad():
            probs.append(float(torch.softmax(_STATE["model"](
                to_tensor(stack).unsqueeze(0)), 1)[0, li]))
            for _ in range(TTA_VIEWS):
                img = stack.copy()
                k = np.random.randint(0, 4)
                img = np.rot90(img, k).copy()
                ang = np.random.uniform(-180, 180)
                M = cv2.getRotationMatrix2D((INPUT / 2, INPUT / 2), ang,
                                            np.random.uniform(0.85, 1.15))
                img = cv2.warpAffine(img, M, (INPUT, INPUT),
                                     borderMode=cv2.BORDER_REFLECT)
                if np.random.rand() < 0.5:
                    img = img[:, ::-1].copy()
                if np.random.rand() < 0.5:
                    img = img[::-1, :].copy()
                probs.append(float(torch.softmax(_STATE["model"](
                    to_tensor(img).unsqueeze(0)), 1)[0, li]))
        return float(np.mean(probs))
    except Exception:
        return None


def predict(path):
    """Return line-mark prediction for a bracket image, or None if unreadable.

    {"status": "present"|"absent", "prob": float, "confidence": int,
     "cnn_prob": float|None, "geom_score": float, "votes": [str,...]}

    A line mark is asserted when EITHER ensemble member fires (OR vote).
    """
    prep = _ridge_prep(path)
    if prep is None:
        return None
    try:
        geom = _geom_score(prep)
    except Exception:
        geom = 0.0
    cnn = _cnn_prob(path)

    votes = []
    if cnn is not None and cnn >= CNN_THRESHOLD:
        votes.append("cnn")
    if geom >= GEOM_THRESHOLD:
        votes.append("geom")

    present = len(votes) > 0
    # Combined confidence: how strongly the firing member(s) exceed threshold.
    cnn_margin = (cnn - CNN_THRESHOLD) if cnn is not None else -1.0
    geom_margin = geom - GEOM_THRESHOLD
    if present:
        prob = 0.5 + 0.5 * max(cnn_margin / (1 - CNN_THRESHOLD) if cnn is not None else -1,
                               geom_margin / max(GEOM_THRESHOLD, 1e-6))
        prob = min(max(prob, 0.5), 0.99)
    else:
        prob = 0.5 - 0.5 * min(abs(cnn_margin) / max(CNN_THRESHOLD, 1e-6)
                               if cnn is not None else 1.0,
                               abs(geom_margin) / max(GEOM_THRESHOLD, 1e-6))
        prob = min(max(prob, 0.01), 0.5)
    status = "present" if present else "absent"
    conf = prob if present else (1.0 - prob)
    return {"status": status, "prob": round(prob, 3),
            "confidence": int(round(conf * 100)),
            "cnn_prob": round(cnn, 3) if cnn is not None else None,
            "geom_score": round(geom, 3), "votes": votes}
