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

## My complete thought process (the raw version)

This is the honest, first-person account of what actually went through my head — including
the wrong turns, the moments of doubt, and the assumptions I had to catch myself making.
The polished "steps + lessons" version is further below; this section is the messy
reality of how I got there.

**"Wait — didn't I already fix this?"**
The very first thing I felt was confusion, because the user had made me fix 163907 in a
previous session. My instinct was almost to be defensive — "I retrained YOLO, I verified
it, the crop was on the bottom hole, I saw it with my own eyes." But that reaction is a
trap. If the user is looking at the same wrong thing after a fix, then either (a) my fix
didn't deploy, or (b) my fix addressed something other than what they're looking at.
Both are *my* problem, not the user's. So I forced myself to drop the "but I already
fixed it" feeling and treat it as a fresh bug. The fact that it "came back" became my
biggest clue rather than an annoyance.

**First hypothesis: "It's stale — the server is running old code/weights."**
This is the most common failure mode in this project, and I've been bitten by it before
(two stale `qms.py serve` processes serving old code on port 8000). So my honest first
bet was: the fix is fine, the server just didn't pick it up. This felt likely and would
have been an easy win. I checked the server's `StartTime` against the model files'
`LastWriteTime`... and it killed my hypothesis: the server had started *after* the
retrain, and it was pointed at the current model files. Mild disappointment — the easy
explanation was wrong — but also useful, because now I *knew* the server was faithfully
rendering current data. The bug was real and in the logic, not the environment.

**The moment that reframed everything: "That box isn't even mine."**
I was still mentally anchored on "my YOLO serration detector drew this." Then I actually
*looked* at the screenshot properly instead of glancing at it: a magenta polygon tracing
the rim, a rectangle, and a `[P4]` priority tag. And something clicked — my serration
branch only calls `A.draw_label(...)`. It writes **text**. It does not draw a polygon, it
does not draw a rectangle, and it certainly doesn't compute a `[P4]` severity priority.
So whatever was drawing this box, *it was not the code I had been obsessing over.* That
was the turning point. I'd spent the whole previous session inside the YOLO subsystem, and
the actual culprit was somewhere I'd never looked. Slightly humbling: I had "fixed" a bug
in a subsystem that doesn't even produce the symptom.

**Following the evidence into `recognizer.inspect()`.**
Now I needed to find *what* draws polygons with priority tags. I grepped for `cv2.circle`,
`cv2.rectangle`, `Serration`, and traced the render calls. Reading `inspect()` top to
bottom, the structure became obvious: there are **two** completely separate things that
can put a "serration" mark on the image. The recall path — when a perceptual-hash match
is found — calls `A.annotate(img, dets)` where `dets` comes from
`m["geometry"][ref]["defects"]`. *That* draws the polygon+box+`[P4]`. My YOLO branch runs
afterwards and only appends a text label. So the polygon is driven by **stored geometry**,
not by any live detection. My earlier fix had corrected the YOLO training label, which
governs the crop and the text verdict — and it worked, for that. But the visible box was
never connected to it. This is why the "fix didn't hold": I had fixed a real thing, just
not the thing the user was pointing at. Two mechanisms, one symptom.

**"Where does that stored geometry actually come from?"**
`m["geometry"]` is loaded from the packed checkpoint, but the human-authored source is
`inspection_state/reviews/bracket.json`. I dumped 163907's polygon and there it was:
centre ≈ (0.505, **0.262**) — the top hole. Confirmed. But I didn't want to fix just this
one and declare victory, because if the labels were entered by hand, others could be wrong
too. So I did a paranoia pass: I computed the centre of **every** stored polygon and
compared it to where YOLO independently localises the big hole. That table was reassuring
— only 163907 disagreed. 164631/164648/164704 matched, and 164704 legitimately has its big
hole at the top (so a top-centre polygon there is correct, not a bug). 164017 had no stored
polygon at all, so it only ever gets a text label. Isolated to one record. Good.

**Deciding the correct coordinates — and not trusting my own numbers.**
I could have hand-typed a box over the bottom hole, but on this glary metal I don't trust
"looks about right." Instead I generated the replacement polygon *from the detector that I
already trusted*: take YOLO's big-hole centre/radius, build a 20-point ellipse, pad it
~1.1× so it sits just outside the rim. Then — and this is the part I refuse to skip — I
**drew it onto the image and looked at it** before writing anything to the JSON. It sat
cleanly on the bottom hole. Only then did I overwrite the `points` array. A discarded
debug image is cheap; a second wrong box shipped to the user is not.

**The step I almost missed: the JSON isn't what the server reads.**
I edited `reviews/bracket.json` and my instinct was to just restart and test. But I paused:
does the server read this JSON live, or a packed copy? It reads the packed checkpoint
`best.pt`. So I ran `qms.py train` — and it said `+0 added, 40 already present` and did
**not** refresh the geometry. That would have been a nasty trap: JSON edited, everything
"looks done," but the served artifact unchanged. The step that actually mattered was
`qms.py build`, which repacks `best.pt` from the reviews. I verified by reloading the model
in-process and checking the geometry centre had moved to (0.429, 0.806). Mental model I
locked in: **JSON is the source, `best.pt` is the snapshot, the running server is the
cache — a change is only real once it's travelled through all three.**

**Refusing to trust an internal call as "proof."**
The in-process check was encouraging, but the user experiences this through the web UI, not
a Python REPL. So the final verification was deliberately through the *same* path they use:
killed the stale server (PID 37788), started a fresh one, and `curl`-posted 163907 to
`/ui/inspect`, then downloaded the returned `annotated_url` and viewed it. The magenta box
was on the bottom big hole; verdict still `DEFECT / Serration Missing`. Only at that point
did I consider it actually fixed, clean up the debug images, and commit.

**What I'd do differently next time.**
The whole episode would have been shorter if, the *first* time the user reported "small
hole," I had matched the visual style of the box to the exact drawing call before assuming
it was my YOLO branch. I jumped to the subsystem I'd just built instead of following the
pixels to their source. The lesson I'm taking: when a symptom appears, identify the exact
line of code that renders it *before* forming a theory about which model is responsible.

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

---
---

# Issue #2 — "Two verdicts on one image" (backside 164017 shows OK *and* Serration Missing)

Reported right after Issue #1 was fixed. On `IMG20260824164017.jpg` (the **backside** of a
bracket, 6 / 22), the annotated image showed **two contradictory verdicts at once**: a green
**"OK — no defect"** banner with a green border, *and* a red **"Serration Missing (ML 100%)"**
label stacked on top of it. The user confirmed the serration call was the correct one — the
question was why an "OK" verdict was also being drawn on the same frame.

---

## TL;DR — Root cause and fix

**Root cause:** a **render-ordering** bug in `recognizer.inspect()`. For a recalled image
with no stored defect, the recall path drew the green OK banner+border **immediately**.
*Then* the YOLO serration second-opinion ran, correctly detected "missing", flipped the
verdict to `DEFECT`, and drew the red label — but the green OK visuals were already painted
and stayed. Two verdicts, one image.

**Fix:** defer the OK / Needs-Review banner until *after* the serration second-opinion has
decided the final verdict. If serration flips a non-defect verdict to DEFECT, suppress the
banner and draw a red defect border instead, so the frame shows a single consistent verdict.

- Verified: 164017 → one clean DEFECT (red border + serration label, no green banner);
  164314 (present) → still green OK; 163907 (recall defect) → still DEFECT. No regression.
- Committed `bfe5464`, pushed.

---

## My complete thought process (the raw version)

**"This is the third time the same bracket family bites me — but this symptom is different."**
My first reaction was to notice this was *not* the same bug as Issue #1. Issue #1 was about
*which hole* the box highlighted (a localization/geometry problem). This was about *two whole
verdicts* on one image — a green OK and a red DEFECT. Different symptom, so I resisted the urge
to assume it was the same cause. The user even told me the serration verdict was correct, which
was a gift: it meant I didn't need to question the ML at all. The bug was purely **visual/logic
consistency** — the picture was disagreeing with itself.

**"Which verdict is the 'wrong' one to draw — or is it an ordering problem?"**
The green banner said OK; the red label said DEFECT. The *final* verdict in the JSON was
DEFECT (correct). So the green OK banner was a **stale artifact** — something drew it before the
verdict was finalized. That immediately smelled like an ordering bug: draw-then-decide instead
of decide-then-draw. I didn't need to reproduce anything elaborate; I needed to read the draw
sequence in `inspect()`.

**Reading the sequence — and there it was, plain as day.**
The recall branch does, in order:
1. `elif result == "OK": A.draw_ok_banner(img)` — paints the green border + "OK - no defect".
2. Later, the serration block: if missing, append the defect, flip `result` OK→DEFECT, and
   `A.draw_label(..., "Serration Missing (ML ..%)", red)`.

So step 1 commits a visual decision *before* step 2 is even allowed to change the verdict.
`draw_ok_banner` (in `renderer.py`) draws a green border (`cv2.rectangle`, thickness 8) and the
"OK - no defect" text. Nothing ever erases them when the verdict flips. The red serration label
is simply drawn on top. Hence: green OK border + green banner + red serration label,
simultaneously. The logic (`verdict["result"]`) was right; only the **pixels** were
self-contradictory.

**Same shape of bug as Issue #1, one layer over.**
I noticed the family resemblance: Issue #1 was "the verdict was right but the overlay pointed at
the wrong hole"; Issue #2 is "the verdict is right but the overlay shows an extra stale verdict."
Both are cases where **the decision and its visualization drift apart.** That reframing told me
the fix should be structural: *never draw a verdict banner until the verdict is final.*

**Designing the fix — defer, don't patch-over.**
The tempting quick hack was: in the serration flip, just repaint a red border over the green one.
That "works" but leaves the latent ordering bug for the next verdict-changing feature to trip on.
The cleaner fix is to **defer** the banner: in the recall/kNN branches, instead of drawing the OK
or Needs-Review banner immediately, set a flag (`pending_ok` / `pending_review`). After the
serration block — the last thing that can change the verdict — draw the banner that matches the
*final* result. If serration flipped OK/Review → DEFECT, clear the flags and draw a red defect
border (and, since an ML-only defect has no locator box, that border is what signals "defect").

**Guarding the edge cases so I don't create Issue #3.**
Before writing, I walked the branches to make sure I wasn't trading one inconsistency for
another:
- *Recall = OK, serration = present* → `pending_ok` stays true → green banner drawn at the end.
  Unchanged behaviour. ✓
- *Recall = OK, serration = missing* → flip to DEFECT, `pending_ok` cleared, red border + label.
  The specific bug, now fixed. ✓
- *Recall had real defect boxes (had_boxes)* → those are drawn by `A.annotate` regardless; if
  serration is *also* missing I keep its label lower (y=80) so it doesn't overwrite the primary
  defect's label at the top. No border conflict because the verdict was already DEFECT. ✓
- *No recall → Needs-Review* → now deferred via `pending_review`; if serration flips it, red
  border replaces the orange "Needs Review" so we don't show "review" and "defect" together. ✓
I also added a `had_boxes` flag purely to decide the serration label's vertical position, so the
top-of-image slot isn't double-used.

**Verifying with pixels, through the real path — again.**
Numbers ("result == DEFECT") are necessary but not sufficient for a *rendering* bug — the whole
point is what the image looks like. So I rendered three representative cases and **viewed the
actual annotated JPEGs**: 164017 (the flip) now shows only the red border + serration label;
164314 (present) still shows the green OK banner with the visibly serrated bottom hole; 163907
(recall defect) still shows its DEFECT box. Then I restarted the server (the model/handlers are
loaded once) and left it verified through the same `/ui/inspect` flow the user uses. Only then
did I clean up the debug images and commit.

**What ties Issues #1 and #2 together.**
Both are "the model was right, the drawing lied." In #1 the drawing came from a stale stored
polygon; in #2 from a stale draw-order. The durable lesson: **treat the annotated image as an
output that must be derived from the *final* verdict, never assembled incrementally while the
verdict is still being decided.** Any feature that can change a verdict must run *before* the
verdict is visualized.

---

## The fix in code (what changed in `recognizer.inspect()`)

Before (simplified):

```python
if recalled:
    ...
    elif result == "OK":
        A.draw_ok_banner(img)          # drawn too early
else:
    ...
    A.draw_label(img, ..., "Needs Review", orange)   # drawn too early
    cv2.rectangle(img, ..., orange, 6)

# serration second opinion runs AFTER the banners are already painted
if serration missing:
    verdict["result"] = "DEFECT"
    A.draw_label(img, ..., "Serration Missing", red)   # stacks on top of green/orange
```

After:

```python
pending_ok = False
pending_review = False
had_boxes = False

if recalled:
    ...
    if dets:
        A.annotate(...); had_boxes = True
    elif result == "OK":
        pending_ok = True              # DEFER
else:
    ...
    pending_review = True              # DEFER

if serration missing and not already:
    verdict["defects"].append(...)
    if verdict["result"] in ("OK", "NEEDS_REVIEW"):
        verdict["result"] = "DEFECT"
        pending_ok = pending_review = False
        if not had_boxes:
            cv2.rectangle(img, (0,0), (w-1,h-1), (0,0,255), 8)   # red defect border
    A.draw_label(img, 15, 80 if had_boxes else 40, "Serration Missing (ML ..%)", red)

# draw the deferred banner ONLY now, matching the FINAL verdict
if pending_ok:
    A.draw_ok_banner(img)
elif pending_review:
    A.draw_label(img, 15, 45, "Needs Review", orange)
    cv2.rectangle(img, ..., orange, 6)
```

**One-line summary of the change:** move from *draw-then-decide* to *decide-then-draw* for the
verdict banner.

---

## Updated mental model (both issues)

The annotated image is produced by several subsystems that can each draw onto the frame. Two
independent failure modes have now shown up:

| issue | subsystem at fault | nature of bug | fix |
|-------|--------------------|---------------|-----|
| #1 — wrong hole highlighted | stored review polygon (recall path) | **wrong data** baked into `best.pt` | correct polygon → `qms.py build` → restart |
| #2 — two verdicts at once   | banner draw order in `inspect()`     | **wrong order** (draw before verdict final) | defer banner until after serration |

Unifying principle: **the drawing must always be a faithful function of the final verdict.**
Keep the *decision* and the *rendering* in that order, and keep the *served artifact* in sync
with its *source*.

