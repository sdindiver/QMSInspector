"""Locate the big (serration) hole on a Bracket and crop to it.

The serration teeth live only on the rim of the single large splined hole; the two
other holes are plain. To make the serration classifier look ONLY at the big hole
(and ignore the small holes / part colour), we localise that hole and crop a padded
square around it before classifying.

Primary localiser is a small trained YOLO detector (inspection_state/models/
big_hole_yolo.pt) which handles glare/reflection robustly. If YOLO or its weights
are unavailable we fall back to a classical Hough search, and finally to the whole
image, so the pipeline never breaks.
"""
from __future__ import annotations
import os
import threading

import numpy as np
import cv2

from inspector import image_features as F
from inspector import settings

YOLO_PATH = os.path.join(os.path.dirname(settings.MODEL_PATH), "big_hole_yolo.pt")

_LOCK = threading.Lock()
_YOLO = {"loaded": False, "model": None}


def _yolo():
    if _YOLO["loaded"]:
        return _YOLO["model"]
    with _LOCK:
        if _YOLO["loaded"]:
            return _YOLO["model"]
        _YOLO["loaded"] = True
        if os.path.exists(YOLO_PATH):
            try:
                from ultralytics import YOLO
                _YOLO["model"] = YOLO(YOLO_PATH)
            except Exception:
                _YOLO["model"] = None
        return _YOLO["model"]


def _find_yolo(bgr):
    """Return (cx, cy, r) from the YOLO detector, or None."""
    m = _yolo()
    if m is None:
        return None
    try:
        r = m.predict(bgr, imgsz=512, conf=0.20, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rad = max(x2 - x1, y2 - y1) / 2.0
        return cx, cy, rad
    except Exception:
        return None


def _find_hough(bgr):
    """Classical fallback: largest circular hole inside the part."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, contour = F.segment_part(gray)
    if contour is None:
        return None
    pa = F._part_axis(contour)
    sil = np.zeros((h, w), np.uint8)
    cv2.drawContours(sil, [contour], -1, 255, -1)
    inner = cv2.erode(sil, np.ones((max(3, int(pa * 0.03)),) * 2, np.uint8))
    g = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(pa * 0.2),
                               param1=110, param2=28,
                               minRadius=int(pa * 0.05), maxRadius=int(pa * 0.18))
    if circles is None:
        return None
    cand = [(int(r), int(cx), int(cy)) for cx, cy, r in np.round(circles[0]).astype(int)
            if 0 <= cy < h and 0 <= cx < w and inner[cy, cx] > 0]
    if not cand:
        return None
    cand.sort(key=lambda t: -t[0])
    r, cx, cy = cand[0]
    return float(cx), float(cy), float(r)


def find_big_hole(bgr):
    """Return (cx, cy, r) of the big serration hole (YOLO first, then Hough)."""
    bgr = F._resize(bgr)
    return _find_yolo(bgr) or _find_hough(bgr)


def big_hole_crop(bgr, pad=1.7):
    """Crop a padded square around the big hole. Returns BGR array or None."""
    bgr = F._resize(bgr)
    res = find_big_hole(bgr)
    if res is None:
        return None
    h, w = bgr.shape[:2]
    cx, cy, r = res
    s = int(r * pad)
    x0, y0 = int(max(0, cx - s)), int(max(0, cy - s))
    x1, y1 = int(min(w, cx + s)), int(min(h, cy + s))
    crop = bgr[y0:y1, x0:x1]
    return crop if crop.size else None
