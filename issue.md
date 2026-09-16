# Issue Log — "IMG20260824163907 shows the *small* hole in the UI"

A debugging walkthrough of a single reported anomaly, written so the reasoning is
reproducible. The interesting part of this bug is that the **verdict was always correct**
— the problem was purely in *which hole the overlay box highlighted*, and the true cause
was in a code path I had not touched.

---

## TL;DR — Root cause (and why my earlier fix didn't touch it)

The "small hole" box seen in the UI was **not** coming from the YOLO detector I
retrained. It came from a **separate, hand-labeled polygon** stored in
`inspection_state/reviews/bracket.json`.

There are two independent paths for a recalled image:

1. **Recall path** (`A.annotate`) draws the stored review polygon → this is the magenta
   **"Serration Missing [P4]"** circle+box you see.
2. **YOLO second-opinion** → only adds the text verdict.

My previous fix corrected the YOLO training label (path 2), so the **verdict** was right —
but the **visible box** (path 1) still pointed at the top/small hole because its stored
polygon was centered at **(0.505, 0.262 = top)**.

**What I fixed**

- Rewrote 163907's stored polygon to circle the **bottom big hole** (center 0.43, 0.81),
  verified visually before writing.
- Ran `qms.py train` + `qms.py build` to repack `best.pt` (the geometry is baked into the
  checkpoint — `train` alone doesn't refresh it).
- Restarted the server (killed stale PID 37788) and confirmed through the live
  `/ui/inspect` endpoint: box now on the bottom big hole, verdict still
  **DEFECT / Serration Missing**.
- Checked the other 4 serration entries — only 163907 was wrong;
  164631 / 164648 / 164704 already matched.

Reload the UI and re-upload — 163907 will now highlight the correct bottom hole.

> Note: the "UI slow" question was set aside on request. First upload is slower because
> the YOLO/classifier models load **lazily** on the first bracket; subsequent uploads
> reuse the in-memory singletons.

---

## 1. The report

While clicking through the 22 brackets in the UI lightbox, image **5 / 22**
(`IMG20260824163907.jpg`) showed the magenta **"Serration Missing [P4]"** box drawn over
a **small hole near the top** of the bracket, not the large serration hole at the bottom.

The user had already, in a previous session, made me "fix" this — so the first honest
reaction was: *why is it back?* That question is itself a clue: if a fix didn't hold, the
fix probably addressed the wrong mechanism.

---

## 2. First, rule out the obvious: stale server / stale model

A recurring gotcha in this project is that `start.bat` starts a **new** server without
killing an old one on the same port, so an old process can serve stale code/weights.

Checks I ran:

```powershell
# who owns port 8000, and when did it start?
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Get-Process -Id $_ | Select-Object Id, StartTime }

# when were the models last written?
Get-ChildItem inspection_state\models\big_hole_yolo.pt,
              inspection_state\models\serration_bracket.pt |
  Select-Object Name, LastWriteTime
```

Result: the server (PID 37788) had started **after** the last retrain, and the model
files were older than the server start. **So it was not a stale-process issue.** The
server was genuinely producing this box from current data. That ruled out the easy
explanation and forced me to look at the actual render path.

**Lesson:** eliminate the environment/caching explanation with timestamps *before*
theorising about logic. One `StartTime` vs `LastWriteTime` comparison saved an hour of
chasing the wrong thing.

---

## 3. Read the pixels of the annotation, not my assumptions

I had *assumed* the earlier YOLO fix governed this box. But the box in the screenshot had
three tell-tale features:

- a **magenta polygon** tracing the hole rim,
- a **rectangle**, and
- a **`[P4]` priority tag**.

My serration second-opinion code only does `A.draw_label(...)` — **plain text, no
polygon, no box, no priority tag.** So whatever drew this box, it was **not** my YOLO
serration branch. That single observation redirected the entire investigation.

**Lesson:** the *style* of an artifact tells you which code produced it. Match the visual
to the drawing call before assuming which subsystem is responsible.

---

## 4. Trace the render path in `recognizer.inspect()`

Reading `inspector/recognizer.py` revealed two independent paths for a recalled image:

```python
# recall path (perceptual-hash hit)
if dets:
    A.annotate(img, dets, needs_review=...)   # <-- draws stored polygon + box + [P?]
...
# serration second opinion (my YOLO branch)
if "bracket" in part:
    sr = SR.predict(path)                     # <-- only adds a TEXT label
    A.draw_label(img, 15, 80, "Serration Missing (ML ..%)", ...)
```

So the visible polygon comes from **`A.annotate(img, dets)`**, and `dets` comes from:

```python
geo = m["geometry"].get(ref, ...)   # stored, hand-labelled geometry for this exemplar
dets = geo.get("defects", [])
```

**The box is driven by stored review geometry, completely separate from the YOLO
detector.** My earlier fix corrected the YOLO training label (so the *verdict* was right)
but never touched the polygon that actually gets drawn. That is exactly why the earlier
"fix" didn't change what the user saw.

**Lesson:** "the output is correct" and "the reason/overlay is correct" are two different
questions. A correct verdict produced via a wrong-looking overlay is still a bug worth
fixing, and the two can live in entirely different code paths.

---

## 5. Find the source of truth for the polygon

The geometry originates in the human-readable review files. I dumped the stored polygon
for 163907:

```python
import json
d = json.load(open(r"inspection_state\reviews\bracket.json"))
# 163907 "Serration Missing" points -> centre ~ (0.505, 0.262)  => TOP hole
```

Then I checked **every** bracket entry's polygon centre in one pass, to see whether this
was isolated or systemic:

| image  | stored polygon centre | YOLO big-hole centre | verdict           |
|--------|-----------------------|----------------------|-------------------|
| 163907 | (0.505, **0.262**)    | (0.43, **0.81**)     | ❌ wrong (top)     |
| 164631 | (0.215, 0.470)        | (0.21, 0.47)         | ✅ match           |
| 164648 | (0.196, 0.632)        | (0.21, 0.64)         | ✅ match           |
| 164704 | (0.539, 0.232)        | (0.53, 0.22)         | ✅ match (top part)|
| 164017 | *no stored polygon*   | (0.43, 0.84)         | ✅ text-label only |

Only **163907** had a stored polygon on the wrong end. (164017 has no stored polygon, so
it only ever gets the text label — no wrong box to fix there.)

**Lesson:** when you find one bad record, immediately audit the whole set with a single
script. It converts "is this a one-off or a pattern?" from a guess into a fact, and it
stops you from fixing #1 while #7 silently has the same defect.

---

## 6. Compute the corrected polygon *from the detector*, then eyeball it

Rather than hand-guess coordinates, I generated the replacement polygon from the
(already-correct) YOLO big-hole detection, then **drew it and looked at it** before
writing anything:

```python
cx, cy, r = H.find_big_hole(raw)          # bottom hole: (0.43, 0.81)
# build a 20-point ellipse at (cx,cy) with ~1.1x radius padding
# draw it on the image -> _dbg_163907_newpoly.jpg -> view
```

The verification crop showed the polygon sitting cleanly on the bottom big hole. Only
then did I overwrite the `points` array in `reviews/bracket.json`.

**Lesson:** never write coordinates you haven't visually confirmed. On glary metal,
"the numbers look right" is not the same as "the circle is on the hole." One throwaway
debug image is cheaper than shipping a second wrong box.

---

## 7. The non-obvious build step

Editing `reviews/bracket.json` alone changed nothing, because the geometry the server
uses is **baked into the packed checkpoint** `best.pt`, not read live from the JSON.

- `python qms.py train` — learns exemplars into the DB, but reported
  `+0 added, 40 already present` and did **not** refresh existing geometry.
- `python qms.py build` — **repacks `best.pt`** from the reviews, embedding the corrected
  polygon. *This* was the step that mattered.

I confirmed the fix propagated by reloading the model in-process:

```python
m = REC.load_model(settings.MODEL_PATH)
# geometry centre for 163907 now (0.429, 0.806)  -> bottom hole ✅
```

**Lesson:** know where the *served* copy of your data lives. Editing the human-readable
source is useless if a build step freezes a snapshot into an artifact the runtime loads.
Trace data all the way to the thing the server actually opens.

---

## 8. Restart, then verify through the *real* endpoint

Because the model is loaded once (lazy singleton) and cached in the running server, the
edit could not take effect until a restart:

```powershell
Stop-Process -Id 37788 -Force      # kill the old server holding the old model
python qms.py serve                # fresh process loads the repacked best.pt
```

Final proof was **not** an internal function call but the actual HTTP path the UI uses:

```powershell
curl.exe -s -F "files=@...IMG20260824163907.jpg" http://localhost:8000/ui/inspect
# -> download the returned annotated_url and view it
```

The UI-served annotated image showed **"Serration Missing [P4]" on the bottom big hole**.
Verdict unchanged: `DEFECT / Serration Missing`.

**Lesson:** verify through the same interface the user hit (HTTP endpoint + served image),
not just the convenient internal function. Bugs love the gap between "works in a REPL" and
"works through the server."

---

## 9. The fix (summary)

1. Rewrote 163907's `Serration Missing` polygon in `inspection_state/reviews/bracket.json`
   to a 20-point ellipse centred on the bottom big hole (~0.43, 0.81), padding ~1.1×.
2. `python qms.py build` to repack `inspection_state/models/best.pt` with the new
   geometry.
3. Killed the stale server, started a fresh one.
4. Verified through `/ui/inspect` + the served annotated image.
5. Cleaned up all `_dbg_*.jpg`, committed and pushed (`b691fad`).

No behavioural change (the verdict was always correct); the **overlay** now points at the
correct hole and the stored source-of-truth matches reality.

---

## 10. Why the earlier fix "didn't hold"

It *did* hold — for the thing it changed. The earlier session fixed the **YOLO training
annotation** (which feeds the classifier's crop and the text verdict). But the **visible
box** is drawn by the **recall path from stored review geometry**, a different subsystem.
The two were never connected. The symptom looked identical, so it appeared the fix
regressed, when in fact the fix had addressed a different mechanism than the one producing
the box.

**Meta-lesson:** if a bug "comes back" after you fixed it, strongly suspect there are
**two mechanisms that produce the same symptom**, and you fixed only one. Don't re-apply
the same fix — find the second path.

---

## Appendix — subsystems that can each place a "serration" mark

| subsystem                    | file                             | draws                         | governs |
|------------------------------|----------------------------------|-------------------------------|---------|
| Recall geometry (pHash hit)  | `recognizer.inspect` → `renderer.annotate` | polygon + box + `[P?]` | **the visible box** |
| YOLO big-hole + classifier   | `parts/bracket/serration.py` (`SR.predict`) | text label only        | the verdict text |
| YOLO training labels         | `parts/bracket/train_big_hole_yolo.py` (`BOXES`) | nothing at runtime | crop location for training |
| Stored review source-of-truth| `inspection_state/reviews/bracket.json` | nothing directly     | what `qms.py build` bakes into `best.pt` |
| Served checkpoint            | `inspection_state/models/best.pt`| —                             | what the running server loads |

Remember: **JSON is the source, `best.pt` is the snapshot, the running server is the
cache.** A change is only real once it has travelled through all three.
