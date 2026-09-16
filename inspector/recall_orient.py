"""Orientation-invariant perceptual-hash recall.

The dHash recall in :mod:`inspector.recognizer` only matches a photo in the exact
orientation it was stored in, so a flipped or 90-degree-rotated copy of a known
part fails to recall and falls through to NEEDS_REVIEW.

This module compares an input against the stored exemplar hashes across the eight
dihedral orientations (identity, 90/180/270 rotations, horizontal/vertical flips
and the two diagonal transposes). It also returns the inverse point map for the
matched orientation so the stored defect geometry (polygons / boxes) is drawn in
the correct place on the rotated or flipped input.

Note: this handles the 8 right-angle/mirror orientations only. Arbitrary angles
(e.g. 45 degrees) change the dHash and are not recalled -- that needs a trained,
rotation-invariant detector, not memory recall.
"""
import numpy as np

from . import image_features as F


# Each entry: (name, image_op, inv_point)
#   image_op  : maps a grayscale array into the oriented frame that is hashed and
#               compared against the stored (canonical) exemplar hashes.
#   inv_point : maps a normalized (x, y) from the exemplar/oriented frame back to
#               the input-image frame, so stored polygons land correctly on the
#               rotated/flipped input. (Verified empirically against image_op.)
ORIENTATIONS = [
    ("id",        lambda a: a,                lambda x, y: (x, y)),
    ("rot90ccw",  lambda a: np.rot90(a, 1),   lambda x, y: (1.0 - y, x)),
    ("rot180",    lambda a: np.rot90(a, 2),   lambda x, y: (1.0 - x, 1.0 - y)),
    ("rot90cw",   lambda a: np.rot90(a, 3),   lambda x, y: (y, 1.0 - x)),
    ("flipH",     lambda a: a[:, ::-1],       lambda x, y: (1.0 - x, y)),
    ("flipV",     lambda a: a[::-1, :],       lambda x, y: (x, 1.0 - y)),
    ("transpose", lambda a: a.T,              lambda x, y: (y, x)),
    ("antidiag",  lambda a: a[::-1, ::-1].T,  lambda x, y: (1.0 - y, 1.0 - x)),
]


def oriented_hashes(gray):
    """Return [(name, phash_hex, inv_point), ...] for the 8 orientations of gray."""
    return [(name, F._phash(op(gray)), inv) for name, op, inv in ORIENTATIONS]


def best_recall(input_gray, exemplar_hashes):
    """Best orientation-invariant match against stored exemplar hashes.

    Returns (best_hamming, best_index, inv_point, orient_name). best_index is -1
    when there are no usable exemplar hashes.
    """
    oh = oriented_hashes(input_gray)
    best_h, best_i, best_inv, best_name = 999, -1, None, "id"
    for i, eh in enumerate(exemplar_hashes):
        if not eh:
            continue
        for name, ph, inv in oh:
            d = F.hamming(ph, eh)
            if d < best_h:
                best_h, best_i, best_inv, best_name = d, i, inv, name
    return best_h, best_i, best_inv, best_name


def transform_defects(dets, inv_point):
    """Copy dets with normalized geometry mapped through inv_point.

    Polygon ``points`` are mapped point-by-point. An axis-aligned ``bbox`` is
    converted to a 4-point polygon first, because a rotated box is no longer
    axis-aligned (its ``seg`` refinement is dropped for the same reason).
    """
    out = []
    for d in dets:
        d2 = dict(d)
        if d.get("points"):
            d2["points"] = [list(inv_point(x, y)) for x, y in d["points"]]
        elif d.get("bbox"):
            bx, by, bw, bh = d["bbox"]
            corners = [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)]
            d2.pop("bbox", None)
            d2.pop("seg", None)
            d2["points"] = [list(inv_point(x, y)) for x, y in corners]
        out.append(d2)
    return out
