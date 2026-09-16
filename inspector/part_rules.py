"""Per-part rule files: the ONE place a developer edits to add/change a part.

Each ``knowledge/rules/<part>.json`` fully describes a single part in one file:

    {
      "part": "Bracket",
      "default_part": false,
      "description": "what the part is / how it's photographed",
      "defects": {
        "Dark Burn": {
          "severity": 4,                # 4=critical .. 1=minor, 0=OK
          "color": [0, 0, 180],         # BGR for the annotation box
          "signature": "what it looks like / how to tell it apart"
        },
        ...
      },
      "rulings": ["confirmed inspection rules for this part", ...],
      "confusions": [["A", "B"], ...],          # optional look-alikes
      "segmentation_gotchas": ["...", ...]      # optional (masking tips)
    }

To add a new part: drop a new ``<part>.json`` in the rules folder, then run
``python qms.py build``. No Python edits, nothing split across taxonomy.json +
lessons.json anymore.

This module merges every rule file back into the two structures the rest of the
app already consumes, so no other module needs per-part special-casing:

    build_taxonomy() -> {default_part, parts, default_severity, defects{...}}
    build_lessons()  -> {parts{...}, confusions, segmentation_gotchas, ...}
"""
from __future__ import annotations
import glob
import json
import os

from . import settings

DEFAULT_SEVERITY = 2
FALLBACK_COLOR = [0, 0, 255]

# Shared "pass" class, auto-registered if a part file doesn't define its own.
_OK = {
    "severity": 0,
    "color": [0, 180, 0],
    "signature": "No defect present - the part passes inspection.",
}


def _rule_files():
    return sorted(glob.glob(os.path.join(settings.RULES_DIR, "*.json")))


def available():
    """True when at least one rules/<part>.json exists."""
    return bool(_rule_files())


def load_rules():
    """Return {part_name: rule_dict} for every rules/<part>.json (sorted)."""
    out = {}
    for path in _rule_files():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        name = data.get("part") or os.path.splitext(os.path.basename(path))[0]
        out[name] = data
    return out


def _signature(spec):
    return spec.get("signature") or spec.get("description") or ""


def build_taxonomy():
    """Assemble the taxonomy structure (names/severity/colors) from rules."""
    rules = load_rules()
    parts = list(rules.keys())
    default_part = next(
        (p for p, r in rules.items() if r.get("default_part")),
        parts[0] if parts else "default",
    )
    defects = {}
    for rule in rules.values():
        for name, spec in rule.get("defects", {}).items():
            entry = {
                "severity": int(spec.get("severity", DEFAULT_SEVERITY)),
                "color": spec.get("color", list(FALLBACK_COLOR)),
                "description": _signature(spec),
            }
            defects.setdefault(name, entry)
    defects.setdefault("OK", {
        "severity": _OK["severity"],
        "color": list(_OK["color"]),
        "description": _OK["signature"],
    })
    return {
        "default_part": default_part,
        "parts": parts,
        "default_severity": DEFAULT_SEVERITY,
        "defects": defects,
    }


def build_lessons():
    """Assemble the lessons structure (signatures/rulings/confusions) from rules."""
    rules = load_rules()
    parts = {}
    confusions = []
    gotchas = []
    mistakes = []
    by_image = {}
    for name, rule in rules.items():
        signatures = {dn: _signature(spec)
                      for dn, spec in rule.get("defects", {}).items()}
        parts[name] = {
            "description": rule.get("description", ""),
            "signatures": signatures,
            "confirmed_rulings": rule.get("rulings", rule.get("confirmed_rulings", [])),
        }
        confusions.extend(rule.get("confusions", []))
        gotchas.extend(rule.get("segmentation_gotchas", []))
        mistakes.extend(rule.get("recurring_mistakes", []))
        by_image.update(rule.get("confirmed_defects_by_image", {}))
    return {
        "note": "Assembled from knowledge/rules/<part>.json - edit those files, not this.",
        "parts": parts,
        "confusions": confusions,
        "segmentation_gotchas": gotchas,
        "recurring_mistakes": mistakes,
        "confirmed_defects_by_image": by_image,
    }
