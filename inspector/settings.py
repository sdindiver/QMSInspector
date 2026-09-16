"""Central configuration for QMSInspector.
 
Every path and tunable resolves from an environment variable first, then falls
back to an in-repo default, so the project is portable across machines. There is
NO cloud/LLM configuration here - inspection is 100% offline and token-free.
 
Environment variables (all optional):
  QMS_STATE_DIR       inspection state root        (default: <root>/inspection_state)
  QMS_DATA_DIR        images to inspect            (default: <state>/data/images)
  QMS_OUT_DIR         annotated outputs            (default: <root>/inspect_out)
  QMS_NEEDS_REVIEW_DIR collected Needs Review images (default: <root>/needs_review)
  QMS_MODEL_PATH      packed checkpoint            (default: <state>/models/best.pt)
  QMS_DB_PATH         inspection memory database   (default: <state>/data/inspection_memory.db)
  QMS_AES_DIR         external image folder        (default: legacy AES2 path)
  QMS_PHASH_RECALL_MAX near-duplicate threshold     (default: 6)
  QMS_LOG_LEVEL       logging level                (default: INFO)
"""
from __future__ import annotations
import os
import logging

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)


def _p(env, *default_parts):
   v = os.environ.get(env)
   return v if v else os.path.join(ROOT, *default_parts)


STATE_DIR = os.environ.get("QMS_STATE_DIR", os.path.join(ROOT, "inspection_state"))
LEGACY_STATE_DIR = ROOT
if not os.path.exists(os.path.join(STATE_DIR, "data")) and not os.path.exists(os.path.join(STATE_DIR, "reviews")):
   STATE_DIR = LEGACY_STATE_DIR


# --- inputs / outputs ---
DATA_DIR = _p("QMS_DATA_DIR", STATE_DIR, "data", "images")
OUT_DIR = _p("QMS_OUT_DIR", "inspect_out")
NEEDS_REVIEW_DIR = _p("QMS_NEEDS_REVIEW_DIR", "needs_review")

# --- durable learnings ---
MODEL_PATH = _p("QMS_MODEL_PATH", STATE_DIR, "models", "best.pt")
DB_PATH = _p("QMS_DB_PATH", STATE_DIR, "data", "inspection_memory.db")
LEGACY_DB_PATH = _p("QMS_DB_PATH", "data", "inspection_memory.db")
if not os.path.exists(DB_PATH) and os.path.exists(LEGACY_DB_PATH):
   DB_PATH = LEGACY_DB_PATH

# --- knowledge base / taxonomy ---
LEGACY_KNOWLEDGE_DIR = os.path.join(ROOT, "knowledge")
KNOWLEDGE_DIR = os.environ.get("QMS_KNOWLEDGE_DIR", os.path.join(STATE_DIR, "knowledge"))
if not os.path.exists(KNOWLEDGE_DIR) and os.path.exists(LEGACY_KNOWLEDGE_DIR):
   KNOWLEDGE_DIR = LEGACY_KNOWLEDGE_DIR
TAXONOMY_PATH = os.environ.get("QMS_TAXONOMY_PATH", os.path.join(KNOWLEDGE_DIR, "taxonomy.json"))
LESSONS_PATH = os.path.join(KNOWLEDGE_DIR, "lessons.json")
CORRECTIONS_PATH = os.path.join(KNOWLEDGE_DIR, "corrections.json")
CACHE_PATH = os.path.join(KNOWLEDGE_DIR, "defect_kb.json")

# --- per-part rule files (the ONE file a developer edits per part) ---
# Each rules/<part>.json fully describes a part: its defects (name, severity,
# color, signature), confirmed rulings, confusions and gotchas. Add a
# new part by dropping a new <part>.json here, then run: python qms.py build.
RULES_DIR = os.environ.get("QMS_RULES_DIR", os.path.join(KNOWLEDGE_DIR, "rules"))

# --- per-part human markings (one <part>.json per part) ---
REVIEWS_DIR = os.path.join(STATE_DIR, "reviews")

# Optional external image folder for parts whose photos live outside the repo
# (e.g. the original AES2 Bearing Cup captures). Overridable; not hardcoded.
LEGACY_AES_DIR = os.environ.get(
    "QMS_AES_DIR",
    r"C:\workspace-ai\AES2-20260830T112553Z-1-001\AES2")

# --- recall / decision ---
PHASH_RECALL_MAX = int(os.environ.get("QMS_PHASH_RECALL_MAX", "6"))


_LOG_CONFIGURED = False


def get_logger(name="qms"):
    """Return a configured logger (idempotent)."""
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        level = os.environ.get("QMS_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def ensure_dirs():
    for d in (DATA_DIR, OUT_DIR, NEEDS_REVIEW_DIR,
              os.path.dirname(MODEL_PATH), os.path.dirname(DB_PATH),
              KNOWLEDGE_DIR, RULES_DIR, REVIEWS_DIR):
        os.makedirs(d, exist_ok=True)
