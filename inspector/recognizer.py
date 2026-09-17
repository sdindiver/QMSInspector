"""Offline inspection engine: run the packed model with ZERO LLM tokens.
 
Loads models/best.pt (all learnings in one file) and inspects images locally. Any
image that matches a learned sample by perceptual hash gets the confirmed verdict
plus the exact annotated masks pulled from the checkpoint. Genuinely-new images
come back Needs Review and (optionally) get copied into a "needs_review" folder so a
human can review the system later.
"""
from __future__ import annotations
import os
import json
import shutil

import numpy as np
import torch
import cv2

from . import image_features as F
from . import renderer as A
from . import recall_orient as RO
from parts.bracket import serration as SR
from parts.bracket import dark_spots as DSP
from parts.bracket import line_mark as LMK
from parts.bearing_cup import defect as BCD
from . import settings

DEFAULT_MODEL = settings.MODEL_PATH
DEFAULT_OUT = settings.OUT_DIR


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    E = ckpt["exemplars"]
    ckpt["_X"] = E["X"].numpy().astype(np.float64)
    ckpt["_mean"] = ckpt["norm"]["mean"].numpy().astype(np.float64)
    ckpt["_std"] = np.where(ckpt["norm"]["std"].numpy() < 1e-6, 1e-6,
                            ckpt["norm"]["std"].numpy()).astype(np.float64)
    return ckpt


def _priority(cat, severity_rules):
    c = (cat or "").lower()
    for score, kws in severity_rules:
        if any(k in c for k in kws):
            return score
    return 2


def _collect_needs_review(path, needs_review_dir):
    """Copy a Needs Review image into the collection folder for later retraining."""
    if not needs_review_dir:
        return None
    os.makedirs(needs_review_dir, exist_ok=True)
    dst = os.path.join(needs_review_dir, os.path.basename(path))
    try:
        shutil.copy2(path, dst)
    except shutil.SameFileError:
        pass
    return dst


def _points_to_line(points):
    """Fit a centerline through an annotated line-mark polygon."""
    if not points or len(points) < 2:
        return None
    pts = np.array(points, dtype=np.float64)
    center = pts.mean(axis=0)
    pts0 = pts - center
    try:
        _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0]
    norm = float(np.hypot(direction[0], direction[1]))
    if norm <= 1e-9:
        return None
    direction = direction / norm
    proj = pts0 @ direction
    p1 = center + direction * proj.min()
    p2 = center + direction * proj.max()
    return (round(float(p1[0]), 4), round(float(p1[1]), 4),
            round(float(p2[0]), 4), round(float(p2[1]), 4))


def _draw_line_trace(img, line, color):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = line
    p1 = (int(round(x1 * w)), int(round(y1 * h)))
    p2 = (int(round(x2 * w)), int(round(y2 * h)))
    cv2.line(img, p1, p2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.line(img, p1, p2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)


def inspect(path, m, out_dir, needs_review_dir=None):
    feat = F.extract(path)
    vec, signals = feat["vector"], feat["signals"]
    keys = m["feature_keys"]
    x = np.array([float(vec.get(k, 0.0)) for k in keys], dtype=np.float64)

    E = m["exemplars"]
    phashes, names, labels = E["phash"], E["names"], E["labels"]
    parts = E.get("parts", ["default"] * len(names))
    th = m["thresholds"]

    # zero-token perceptual-hash recall -- orientation-invariant so a flipped or
    # 90/180/270-rotated copy of a known part still recalls (box is mapped to the
    # matched orientation). Arbitrary angles (e.g. 45 deg) still won't recall.
    input_gray = cv2.cvtColor(F._resize(cv2.imread(path)), cv2.COLOR_BGR2GRAY)
    best_h, best_i, inv_pt, orient = RO.best_recall(input_gray, phashes)
    recalled = best_i >= 0 and best_h <= th.get("phash_recall_max", 6)

    img = cv2.imread(path)
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    collected = None
    pending_ok = False        # defer OK banner until serration second-opinion decides
    pending_review = False    # defer Needs-Review banner likewise
    had_boxes = False         # whether defect locator boxes were already drawn
    recalled_line_mark_lines = []

    # Line marks are a Bracket-only defect and the ensemble costs ~0.6s per image,
    # so it is computed LAZILY -- only once we know the part is a bracket -- to avoid
    # wasting that time on every non-bracket image (e.g. bearing cups). Memoised so
    # it runs at most once per inspect().
    _lm_cache = {}

    def _line_mark_pred():
        if "v" not in _lm_cache:
            _lm_cache["v"] = LMK.predict(path)
        return _lm_cache["v"]

    if recalled:
        ref = names[best_i]
        geo = m["geometry"].get(ref, {"defects": [], "result": labels[best_i]})
        part = geo.get("part") or parts[best_i]
        dets = geo.get("defects", [])
        if orient != "id":
            dets = RO.transform_defects(dets, inv_pt)
        if "bracket" in (part or "").lower():
            recalled_line_mark_lines = [
                ln for ln in (_points_to_line(d.get("points"))
                              for d in dets
                              if d.get("category", "").strip().lower() == "line mark")
                if ln is not None
            ]
        # When the model detects the line mark, drop the stored line-mark polygon
        # from generic polygon rendering. Exact recalled samples still reuse the
        # reviewed trace itself as a line overlay because it is the most precise mark.
        if "bracket" in (part or "").lower():
            lm0 = _line_mark_pred()
            if lm0 and lm0.get("status") == "present":
                dets = [d for d in dets
                        if d.get("category", "").strip().lower() != "line mark"]
        result = geo.get("result") or ("DEFECT" if dets else "OK")
        conf = 96 if best_h == 0 else 88
        if dets:
            A.annotate(img, dets, needs_review=(result == "NEEDS_REVIEW"))
            had_boxes = True
        elif result == "OK":
            pending_ok = True
        sdets = sorted(dets, key=lambda d: _priority(d.get("category", ""), m["severity_rules"]), reverse=True)
        verdict = {"result": result, "part": part, "part_confident": True,
                   "defects": [] if result == "OK" else [{
            "type": d["category"], "confidence": conf,
            "location": d.get("location", "see box"), "reason": d.get("reason", ""),
            "severity_priority": _priority(d["category"], m["severity_rules"]),
            "primary": (i == 0)} for i, d in enumerate(sdets)]}
        status = f"RESOLVED (part={part}, recall '{ref}' hamming={best_h}, orient={orient}, conf={conf}) -> 0 LLM tokens"
    else:
        # nearest-prototype hint (kNN), still local
        Xn = (m["_X"] - m["_mean"]) / m["_std"]
        xn = (x - m["_mean"]) / m["_std"]
        dists = np.sqrt(((Xn - xn) ** 2).sum(axis=1)) if len(Xn) else np.array([])
        order = np.argsort(dists) if len(dists) else []
        part = parts[order[0]] if len(order) else "default"   # nearest-neighbour part guess
        hint = ", ".join(f"{labels[i]}({dists[i]:.1f})" for i in order[:3]) if len(dists) else ""
        verdict = {"result": "NEEDS_REVIEW", "part": part, "part_confident": False,
                   "defects": [], "hint": hint}
        pending_review = True
        collected = _collect_needs_review(path, needs_review_dir)
        status = f"NEEDS_REVIEW (likely part={part}; no recall; nearest: {hint})"
        if collected:
            status += f"; copied to {collected}"

    # Serration second opinion (trained classifier) — generalises beyond recall
    # so even a never-seen bracket photo gets a serration present/missing answer.
    if "bracket" in (verdict.get("part") or "").lower():
        sr = SR.predict(path)
        if sr:
            verdict["serration"] = sr
            already = any("serration" in (d.get("type", "").lower())
                          for d in verdict.get("defects", []))
            if sr["status"] == "missing" and not already:
                verdict.setdefault("defects", []).append({
                    "type": "Serration Missing", "confidence": sr["confidence"],
                    "location": "big mounting hole rim",
                    "reason": "trained classifier: rim serration not detected",
                    "severity_priority": _priority("Serration Missing", m["severity_rules"]),
                    "primary": not verdict.get("defects"),
                    "source": "ml_classifier"})
                if verdict["result"] in ("OK", "NEEDS_REVIEW"):
                    verdict["result"] = "DEFECT"
                    verdict["part_confident"] = True
                    # Serration flipped a non-defect verdict: suppress the OK/Review
                    # banner and mark the frame as a defect so verdicts don't conflict.
                    pending_ok = False
                    pending_review = False
                    if not had_boxes:
                        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 8)
                sy = 80 if had_boxes else 40
                A.draw_label(img, 15, sy, f"Serration Missing (ML {sr['confidence']}%)", (0, 0, 255))
                status += f" | serration=MISSING ({sr['confidence']}%)"
            else:
                status += f" | serration={sr['status'].upper()} ({sr['confidence']}%)"

    # Dark-spots second opinion (trained classifier) — recognises the defect at
    # ANY angle (trained with full rotation/flip augmentation), unlike recall.
    if "bracket" in (verdict.get("part") or "").lower():
        ds = DSP.predict(path)
        if ds:
            verdict["dark_spots"] = ds
            already = any("dark spot" in (d.get("type", "").lower())
                          for d in verdict.get("defects", []))
            if ds["status"] == "present" and not already:
                verdict.setdefault("defects", []).append({
                    "type": "Dark Spots", "confidence": ds["confidence"],
                    "location": "part surface",
                    "reason": "trained classifier: dark spot pattern detected",
                    "severity_priority": _priority("Dark Spots", m["severity_rules"]),
                    "primary": not verdict.get("defects"),
                    "source": "ml_classifier"})
                if verdict["result"] in ("OK", "NEEDS_REVIEW"):
                    verdict["result"] = "DEFECT"
                    verdict["part_confident"] = True
                    pending_ok = False
                    pending_review = False
                    if not had_boxes:
                        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 8)
                sy = 120 if had_boxes else 40
                A.draw_label(img, 15, sy, f"Dark Spots (ML {ds['confidence']}%)", (190, 90, 90))
                status += f" | dark_spots=PRESENT ({ds['confidence']}%)"
            else:
                status += f" | dark_spots={ds['status'].upper()} ({ds['confidence']}%)"

    # Bearing-Cup defect second opinion (trained multi-class classifier). Trained
    # with full 0-360 rotation/flip augmentation so it recognises the defect at
    # ANY angle, unlike perceptual-hash recall. Layered on top of recall: it can
    # add a defect a never-seen orientation missed, but won't silence a recall.
    if "bearing" in (verdict.get("part") or "").lower():
        bc = BCD.predict(path)
        if bc:
            verdict["bearing_cup_defect"] = bc
            if bc["status"] == "present":
                dtype = bc["defect"]
                already = any(dtype.lower() in (d.get("type", "").lower())
                              for d in verdict.get("defects", []))
                if not already:
                    verdict.setdefault("defects", []).append({
                        "type": dtype, "confidence": bc["confidence"],
                        "location": "part surface",
                        "reason": f"trained classifier: {dtype} pattern detected",
                        "severity_priority": _priority(dtype, m["severity_rules"]),
                        "primary": not verdict.get("defects"),
                        "source": "ml_classifier"})
                    if verdict["result"] in ("OK", "NEEDS_REVIEW"):
                        verdict["result"] = "DEFECT"
                        verdict["part_confident"] = True
                        pending_ok = False
                        pending_review = False
                        if not had_boxes:
                            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 8)
                    sy = 200 if had_boxes else 40
                    A.draw_label(img, 15, sy, f"{dtype} (ML {bc['confidence']}%)", (0, 0, 255))
                status += f" | bearing_cup={dtype.upper()} ({bc['confidence']}%)"
            else:
                status += f" | bearing_cup=OK ({bc['confidence']}%)"

    # Line-mark ensemble (ridge-CNN OR geometric longest-line). Computed LAZILY and
    # only for Bracket parts (see _line_mark_pred) so the MODEL -- not the stored
    # review polygon -- marks line marks without slowing down non-bracket images.
    # Members fail on different parts, so OR-voting reaches the any-angle bar no
    # single model hit. The geometric member localises the line for drawing.
    lm = _line_mark_pred() if "bracket" in (verdict.get("part") or "").lower() else None
    if "bracket" in (verdict.get("part") or "").lower() and lm:
        verdict["line_mark"] = lm
        already = any("line mark" in (d.get("type", "").lower())
                      for d in verdict.get("defects", []))
        if lm["status"] == "present" and not already:
            verdict.setdefault("defects", []).append({
                "type": "Line Mark", "confidence": lm["confidence"],
                "location": "part surface",
                "reason": "ensemble detector: line/scratch pattern detected "
                          f"(votes: {', '.join(lm['votes'])})",
                "severity_priority": _priority("Line Mark", m["severity_rules"]),
                "primary": not verdict.get("defects"),
                "source": "ml_ensemble"})
            if verdict["result"] in ("OK", "NEEDS_REVIEW"):
                verdict["result"] = "DEFECT"
                verdict["part_confident"] = True
                pending_ok = False
                pending_review = False
            # For recalled samples, the reviewed trace is the exact geometry.
            # Keep it ahead of the live model so known images stay precise.
            if recalled_line_mark_lines:
                col = A.color_for("Line Mark")
                for line in recalled_line_mark_lines:
                    _draw_line_trace(img, line, col)
                had_boxes = True
            elif lm.get("line"):
                _draw_line_trace(img, lm["line"], A.color_for("Line Mark"))
                had_boxes = True
            elif lm.get("lines"):
                col = A.color_for("Line Mark")
                for line in lm["lines"]:
                    _draw_line_trace(img, line, col)
                had_boxes = True
            elif lm.get("polygons"):
                col = A.color_for("Line Mark")
                ih, iw = img.shape[:2]
                for pg in lm["polygons"]:
                    pts = np.array([[int(px * iw), int(py * ih)] for px, py in pg], np.int32)
                    cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
                had_boxes = True
            elif lm.get("polygon"):
                A.annotate(img, [{"category": "Line Mark", "points": lm["polygon"],
                                  "reason": "trained ensemble"}])
                had_boxes = True
            elif lm.get("box"):
                A.annotate(img, [{"category": "Line Mark", "bbox": lm["box"],
                                  "reason": "trained ensemble"}])
                had_boxes = True
            elif not had_boxes:
                cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 8)
            sy = 160 if had_boxes else 40
            A.draw_label(img, 15, sy, f"Line Mark (ML {lm['confidence']}%)", (0, 0, 255))
            status += f" | line_mark=PRESENT ({lm['confidence']}%)"
        else:
            status += f" | line_mark={lm['status'].upper()} ({lm['confidence']}%)"

    # Draw the deferred verdict banner now that serration has had its say.
    if pending_ok:
        A.draw_ok_banner(img)
    elif pending_review:
        A.draw_label(img, 15, 45, "Needs Review", (0, 140, 255))
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 140, 255), 6)


    out_img = os.path.join(out_dir, base + "_annotated.jpg")
    cv2.imwrite(out_img, img)
    with open(os.path.join(out_dir, base + ".json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    return verdict, status, out_img
