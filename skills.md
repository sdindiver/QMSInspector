# QMSInspector - project context and expectations

This file is for future work sessions. Read this before making UI or labeling changes.

## What this project is

- Offline quality inspection workflow for manufactured parts
- No cloud, no LLM, no API calls during inspection
- Current parts in use: `Bearing Cup` and `Bracket`
- Training source of truth: `reviews\*.json`
- Learned memory: `data\inspection_memory.db`
- Packed deployable model: `models\best.pt`

## Core pipeline

1. `python qms.py train`
2. `python qms.py build`
3. `python qms.py serve --port 8000`

Important behavior:

- `train` is idempotent and skips already learned exemplar names
- If a review label changes for an already learned image, updating the JSON alone is NOT enough
- The trainer skips any image whose name already exists in the DB, so a changed label in
  `reviews\*.json` will be ignored unless the stale exemplar row is deleted first
- Restart the server after rebuilding because the Flask app caches the model in memory

## SKILL: Relabel a sample or add a new defect category

Use this exact procedure whenever a defect label is wrong, or a new defect type is needed.
Following all 5 steps keeps rules -> reviews -> DB -> cache -> model in sync (this is the
#1 source of "it still shows the old label" bugs).

1. (New category only) Add it to the part rule file
   `knowledge\rules\<part>.json` under `defects`:
   - `severity` (0=OK .. 5=critical; higher renders as PRIMARY), `color` [B,G,R],
     `signature` (one-line description)

2. Edit the human label + geometry in `reviews\<part>.json` for each affected image
   (`category`, tight `bbox` [x,y,w,h] normalized, `location`, `reason`).
   For small rim/edge features, use a tight box and DO NOT set `seg` (edge-seg locks onto the rim).

3. Delete the stale exemplar row(s) so the trainer re-learns the new label:
   ```python
   import sqlite3; con=sqlite3.connect(r"inspection_state\data\inspection_memory.db")
   con.execute("DELETE FROM exemplars WHERE name IN ('IMG...', 'IMG...')"); con.commit()
   ```

4. Retrain + rebuild:
   ```powershell
   python qms.py train   # re-adds the images with the new label
   python qms.py build   # repacks models\best.pt + regenerates knowledge\defect_kb.json
   ```

5. Verify: `python qms.py inspect "<image>" --no-collect` shows the new `type`/severity,
   and (if serving) restart the server so the cached model reloads.

Always back up `reviews\<part>.json` before editing (a `.bak` copy is enough).

## Startup

### Windows

```powershell
.\start.bat
```

Optional dependency install:

```powershell
.\start.bat --install
```

### Bash

```bash
./start.sh
./start.sh --install
```

Startup scripts should:

- clear `inspection_jobs\*`
- not force dependency install by default
- start the server on port `8000` unless overridden

## Current UI expectations

The UI should feel professional and clean while keeping workflow simple.

Preferred flow:

1. Create Project
2. Open Project
3. Add Parts
4. Click Part
5. View images again

The user also wants:

- delete part
- delete project

Naming constraint:

- Do not use the word `demo` in UI text, endpoint naming, file names, or output folder names

## Current UX decisions

- Use **Needs Review** consistently in the UI and docs
- Top summary cards should be clickable
- Bottom chips and top summary cards must stay synced to the same filter state
- `Images` summary card should reset to `All`
- Users must be able to reopen images later
- Avoid fake inspection percentages; real upload percent is fine, real processed-count progress is preferred
- Lightbox/popup should close on overlay click and on Escape
- Zoom/pan behavior should feel standard and intuitive for vertical drag direction
- Prefer custom lightbox controls if third-party zoom library worsens UX
- Keep both UI themes available with a user switch, with light theme as a preferred option

## Annotation expectations

- Labels must remain readable on bright metallic surfaces
- Locator boxes must be highly visible even for low-contrast defect colors
- Display outline visibility is more important than strictly using a muted taxonomy color for the box

Implemented direction:

- auto-scaled label text
- dark label background
- white text with dark stroke
- strong multi-stroke locator outline
- brighter focus color for low-saturation / low-contrast classes

## Domain labeling decisions

### Bearing Cup defect categories (current source of truth)

Defined in `knowledge\rules\bearing_cup.json`. Severity drives PRIMARY ranking:

- `Missing Punch` (severity 5, critical): central bore/hole absent - the piercing/punching
  step was missed, leaving a solid blind face where a through-hole is required. Highest priority.
- `Dent` (severity 4): metal pushed in / deformed but intact, no missing material.
- `Edge Cut` (severity 4): sheared/cut notch on the outer flange rim where edge material
  is missing - a straight/angular break in the smooth round flange silhouette.
- `Out-of-Round` (severity 4): bore not a clean circle (pinched / ovalised).
- `Corrosion` (severity 2): brown/orange rust/oxidation stain.
- `OK` (severity 0): no defect.

Labeling rule of thumb:
- Missing/sheared material at the rim -> `Edge Cut`
- Metal pushed in but intact -> `Dent`
- Central hole not punched -> `Missing Punch`
- Rust/stain -> `Corrosion`

Confirmed relabels (kept aligned across rules/reviews/DB/cache/model):
- `IMG20260824162142` -> `Edge Cut` (was Dent, originally mislabeled Corrosion)
- `IMG20260824162258` -> `Edge Cut` (was Dent)
- `IMG20260824162340` -> `Edge Cut` (was Dent)
- `IMG20260824162425` -> `Edge Cut` (was Dent)
- `IMG20260824162037` -> `Missing Punch` (was mislabeled Corrosion; the box marks the
  unpunched bottom cup, not the whole image)

### Legacy manual-mark category removal

Current user/domain preference:

- The old manual-mark category is not a real defect category for this project
- Remove that legacy category from the part rule file (`knowledge/rules/<part>.json`), training data, and model artifacts
- Do not show or train manual-mark / reject-paint categories in the UI

## Files most relevant for future edits

- `inspector\inspector_page.py` - UI HTML/JS
- `inspector\web_api.py` - Flask endpoints and UI routes
- `inspector\renderer.py` - annotation labels and boxes
- `inspector\recognizer.py` - offline inspection behavior
- `inspector\trainer.py` - training review data
- `inspector\knowledge_base.py` - SQLite learned exemplars
- `reviews\bearing_cup.json`
- `knowledge\rules\bearing_cup.json` - one rule file per part (defects, severity, rulings, confusions)

## Working rules for future sessions

- Favor clear, professional UX over noisy styling or unnecessary complexity
- Keep terminology clear for non-technical users
- If changing labels, keep `reviews`, `inspection_memory.db`, and `best.pt` aligned
- After model or renderer changes, restart the server and regenerate run outputs
- When the UI still shows old results, first suspect:
  1. old running server
  2. old `inspection_jobs` output
  3. stale learned exemplar rows in `data\inspection_memory.db`
