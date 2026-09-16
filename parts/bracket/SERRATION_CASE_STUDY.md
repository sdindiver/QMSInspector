# Case Study: Building Serration Detection — The Full Thought Process

> A narrated engineering log of how we went from "classical thresholding can't do
> this" to a **100% cross-validated** serration detector. Written to expose the
> *reasoning* — every option considered, every failure, and why each decision was
> made — so it can be studied as an approach-to-problem-solving reference.

---

## 0. The problem in one paragraph

A metal **bracket** has one large "big hole" whose rim is **splined** — it has fine
teeth/grooves called **serration**. On some parts the serration is missing (the rim
is smooth). "Serration Missing" is a **defect** we must catch. The bracket also has
two *other* holes (a tiny middle one and a medium one at the other end) whose rims
are plain — they are *not* serrated and must be ignored. The ask: **upload any
bracket photo and get a reliable present/missing answer**, even for parts the system
has never seen before, fully offline (no cloud, no LLM).

Sounds trivial ("just look at the rim and count teeth"). It was not. Here is why,
and how we got there.

---

## 1. Why the existing system couldn't already do it

The base engine (`inspector/`) recognises defects by **perceptual-hash recall**: it
stores hashes of human-reviewed images and matches a new image against them. If the
new image is a near-duplicate (hamming distance ≤ 6) it returns the stored verdict;
otherwise it goes to **Needs Review**.

- **Strength:** zero false positives on known parts, zero tokens, instant.
- **Fatal limitation for our goal:** recall only recognises *near-duplicates*. A
  genuinely new bracket photo — different lighting, angle, rotation — does **not**
  match, so it lands in Needs Review with **no serration answer at all**.

So the first real insight was a **framing** one:

> This is not a recall problem. It is a **generalization** problem. We need something
> that has *learned what serration looks like*, not something that memorises images.

That reframing is what ruled out "just add more reviewed images" as a solution and
forced us toward either (a) a hand-engineered feature that measures "teethiness", or
(b) a trained model. We tried (a) first because it needs no training data and is
explainable.

---

## 2. Attempt 1 — Classical computer vision (thresholding / signal features)

**The idea:** serration is a *periodic* pattern around a circle. Classical CV should
be able to score it. We prototyped several detectors:

| # | Method | Intuition | Result |
|---|--------|-----------|--------|
| 1 | **FFT teeth-score** on the unwrapped rim | teeth = a strong frequency peak when you "unroll" the circular rim into a line | **Failed to separate** present vs missing |
| 2 | **Boundary roughness** of the hole contour | a toothed rim has a wigglier contour perimeter than a smooth one | **Failed** — overlap |
| 3 | **Bright-blob / background-color hole finding** | find the hole first, then measure its rim | **Unreliable hole finding** |
| 4 | **Hough circles** to locate the big hole | classic circle detector | grabbed the **wrong** things |
| 5 | **Contour/topology + erosion** to isolate inner holes | use the part silhouette to constrain search | glare-dependent, inconsistent |

### The damning numbers

The FFT "teeth score" gave a **MISSING** part (`163907`) a score of **14.8**, while a
**PRESENT** part (`165152`) scored **14.9**. The distributions of present vs missing
*overlapped completely*. There was no threshold you could draw that separated them.
Global features (colorfulness, saturation mean, `hole_rough`) overlapped too.

### Root causes of failure (this is the important part)

1. **Reflective metal defeats periodicity.** The parts are shiny chromate. **Glare**
   creates bright periodic-looking streaks that *mimic* teeth, and glare also *fills
   in* real teeth so a serrated rim can look smooth. The very signal we were keying on
   (periodicity/brightness variation) is manufactured by lighting, not by the teeth.
2. **Hole localization was itself an unsolved problem.** You can't measure the rim
   until you know *which* circle is the big hole. Hough found **8 circles on a 3-hole
   part** — it locked onto the "VA" logo stamp, glare pools, and background. A detector
   that grabs the wrong hole gives a meaningless rim score no matter how good the rim
   metric is.
3. **Orientation varies.** The bracket is photographed at arbitrary rotations, so the
   big hole appears top/bottom/left/right. Any hard-coded spatial assumption breaks.

> **Lesson:** classical thresholding fails not because the metric is "bad" but because
> the **nuisance variation (glare, orientation, wrong-hole selection) is larger than
> the signal**. When your confounders have more variance than your target, no fixed
> threshold survives. This is the canonical "time to learn features instead of
> hand-crafting them" signal.

We did **not** give up on classical CV on a hunch — we **proved** it fails with an
empirical, side-by-side score comparison. That evidence is what justified the pivot.

---

## 3. The pivot decision — why a trained model, and which one

Constraints shaped the choice:
- **Offline / low-resource:** CPU-only machine, ~22 labeled images. No GPU, tiny data.
- **Must generalize** across lighting/rotation.

Options considered:
- **Train a classifier from scratch** → hopeless with 22 images (would overfit
  instantly).
- **Anomaly detection (one-class)** → we only had `sklearn`/`anomalib`? No — not
  installed; and "smooth vs toothed" is really a 2-class problem, not anomaly.
- **Transfer learning with a pretrained backbone** → ✅ the right tool for tiny data.
  A network pretrained on ImageNet already knows edges/textures; we only fine-tune a
  little. We chose **MobileNetV2** (small, CPU-friendly) and confirmed its pretrained
  weights actually download on this gated-internet machine.

Design to fight the tiny dataset:
- **Freeze the backbone**, unfreeze only the last 2 feature blocks (limit capacity →
  limit overfitting).
- **Heavy augmentation**: 180° rotation, RandomResizedCrop, H/V flips, ColorJitter.
  Rotation/flip augmentation directly teaches orientation-invariance — the thing that
  killed classical CV.
- **Class-weighted loss** (16 present vs 6 missing is imbalanced).
- **Stratified 3-fold cross-validation** (with only 6 missing, k=3 keeps ≥2 missing
  per fold) — because with 22 images a single train/test split is statistical noise.

---

## 4. Attempt 2 — Whole-image classifier: a "success" that was secretly wrong

Result: **90.9% cross-val**, and critically it **caught all 6 missing defects** (6/6
recall). For QC, catching every defect matters more than the odd false alarm. This
looked shippable.

But there was a nagging doubt:

> The parts that are *present* tend to be rainbow-chromate (colorful); the *missing*
> ones tend to be plainer silver. Did the model learn **serration**, or did it learn
> **color**? If it learned color, it will fail the moment a colorful part is missing
> serration, or a plain part has it.

This is the single most important habit in the whole story: **distrust a good score
until you know *why* it's good.** A model can be right for the wrong reason, and a
model that's right for the wrong reason is a landmine in production.

---

## 5. The skeptic's test — occlusion sensitivity (proving the shortcut)

To find out *where* the model looks, we ran an **occlusion sensitivity test**: slide a
gray patch across the image, and wherever hiding a region **changes the prediction the
most**, that region is what the model relies on. Plot it as a heatmap.

### A bug that almost lied to us

The first heatmap was nonsense — predictions flipped wildly (0.79 → 0.03) for trivial
reasons. Root cause: I pre-resized the huge source images with `cv2.resize`
(no antialiasing) before occluding. **Downsampling shiny fine teeth without
antialiasing destroys or *fabricates* the very texture we care about** (aliasing). The
fix: resize with **PIL antialiased resize**, exactly matching what the real
`predict()` path does.

> **Lesson:** your diagnostic must use the **identical preprocessing** as production,
> or you're measuring the diagnostic's artifacts, not the model. A measurement tool
> that isn't faithful to the system will confidently tell you lies.

### The verdict

The corrected heatmaps showed the model's high-influence regions were the **part body
/ center and the "VA" stamp — NOT the big-hole rim.** Confirmed: the whole-image model
was using a **color/surface shortcut**, not the serration. The 90.9% was real on this
dataset but **built on a feature that won't hold** (evidence archived to
`serration_occlusion.jpg`).

This is why we didn't ship the easy win. It would have failed silently later.

---

## 6. Attempt 3 — Force the model to look only at the big hole (crop-then-classify)

If the problem is "the model cheats by looking at the body," the fix is to **remove
the body from its view.** Crop to the big hole and classify only that crop. Then it
*physically cannot* use color/stamp shortcuts — the only thing in frame is the rim.

But we were now back to the original hard sub-problem: **reliably finding the big
hole** (the thing classical CV failed at). Two sub-attempts:

### 6a. Classical cropping — tested, and rejected with data

We cropped using a classical detector and re-ran CV. Result: still 90.9% but a
**worse error profile** — it now *missed a real defect* (5/6 recall). Proof that a
flaky localizer poisons the classifier: garbage crop in, garbage label out. Rejected,
**with evidence**, not vibes.

### 6b. A trained detector for the hole — YOLO

The insight: **localization is itself a learning problem**, and it deserves a learned
solution just like classification did. We trained a tiny **YOLOv8n** object detector
to find the big hole:

1. **Hand-annotated** the big (serration/splined) hole box on all 22 images. Doing
   this by eye taught us the domain: the serration hole is the **larger toothed hole
   at one END** of the elongated bracket; position varies with rotation; on
   missing-serration parts we boxed the structural spline hole in the same location.
2. Trained 120 epochs @ imgsz 512 → **mAP50 0.81**. On visual check it landed on a
   real end-hole in **22/22** images and the correct toothed hole on every verifiable
   present part.
3. Wrote `holes.py`: **YOLO-first**, with a Hough fallback and finally a whole-image
   fallback, so the pipeline **never breaks** even if a dependency/weight is missing.

Then re-trained the classifier on **YOLO crops**. Now it sees only the rim.

---

## 7. The plot twist — the model was right and *we* were wrong

Crop-model CV came back **90.9%** again, but with a *different* error profile:
- `164116`: labeled MISSING → model predicted PRESENT
- `165152`: labeled PRESENT → model predicted MISSING

Because the crops made the serration **crisp and unambiguous**, we could finally *look*
at exactly what the classifier sees. Zooming into `164116`'s crop: **it clearly has
teeth.** The label was wrong — I had mislabeled it earlier because in the full frame
the big hole was tiny and easy to misjudge. The model wasn't failing; it was
**correcting my label.**

We fixed the single label (`164116`: missing → present) and re-ran:

```
CROSS-VAL ACCURACY: 100.0%  (TP=17 TN=5 FP=0 FN=0)
missing-recall (caught defects): 5/5
```

> **Lesson:** when a *sound* model disagrees with a label, the label is a suspect too.
> The cropping didn't just improve accuracy — it made the data **auditable**, which is
> what let us find the ground-truth error. Improving the *representation the human can
> inspect* is often worth more than tuning the model.

### Why 100% here is believable but not smug

- It's cross-validated (each prediction is on a held-out fold), not train-accuracy.
- The model is **constrained to the rim**, so we know it's using the right feature —
  we didn't just get a number, we removed the shortcut that made the old number
  untrustworthy.
- **Caveat we keep stating honestly:** 22 images is tiny. 100% CV is strong evidence
  the *approach* is right, not a promise of 100% in the field. The design mitigates
  the small-data risk (crop focus + augmentation + frozen backbone), and the honest
  next step is to keep collecting labeled parts.

We also **validated generalization** directly: on flipped/rotated *unseen*
orientations the verdict stayed correct (missing→missing, present→present), confirming
it reads the relocated hole, not a memorised pixel layout.

---

## 8. Integration & verification

- `serration.py` `predict()` now **crops via `holes.big_hole_crop` before
  classifying**, so inference matches cropped training (with graceful fallback).
- `recognizer.py` raises a **"Serration Missing"** defect for Bracket parts when the
  classifier is confident.
- Verified **end-to-end through the real web upload endpoint** (`/ui/inspect`), not
  just a unit call: missing→DEFECT, present→OK, novel-flip→Needs Review with no false
  alarm.
- Caught a real-world gotcha: a **stale server process** on the port served old code
  and produced a wrong result — a reminder that "it works" must be tested against the
  *actually running* process.

---

## 9. The meta-lessons (the transferable thought process)

1. **Reframe before you solve.** "Detect serration" was really "generalize, not
   recall." The frame determines the whole solution space.
2. **Kill approaches with evidence, not intuition.** We had numbers proving classical
   CV overlapped, proving classical cropping hurt recall, and proving the shortcut.
   Each pivot was earned.
3. **Identify the dominant nuisance variable.** Here it was glare + orientation +
   wrong-hole selection. When nuisance variance > signal variance, hand-tuned
   thresholds are doomed; learned, augmentation-invariant features win.
4. **Distrust a good score until you know why it's good.** Right-for-the-wrong-reason
   is the most dangerous state in ML. Occlusion/attribution tests are cheap insurance.
5. **Make your diagnostics faithful.** Same preprocessing as production, or you debug
   ghosts (the antialiasing bug).
6. **Decompose recursively.** Classification needed localization; localization was its
   own learning problem. Solve the sub-problem with the same rigor.
7. **Constrain the model's input to force the right feature.** Cropping to the rim was
   a *representation* fix that beat any amount of model tuning — and it made the data
   auditable.
8. **When a sound model fights a label, audit the label.** Models can be more
   consistent than tired human annotators.
9. **Fail safe.** Every learned component degrades gracefully (YOLO→Hough→whole-image;
   missing weights → skip serration) so the base pipeline never breaks.
10. **Verify the real system.** Test the running server, watch for stale processes,
    check the actual upload path — not just an ideal in-process call.
11. **Stay honest about limits.** 22 images, small-data fragility, "keep collecting."
    Confidence in the *approach*, humility about the *sample size*.

---

## Appendix — key numbers, files, commands

**Journey of the accuracy number**

| Stage | Accuracy | Defect recall | Trustworthy? |
|-------|----------|---------------|--------------|
| Classical CV (FFT/roughness) | no separation | — | n/a (failed) |
| Whole-image CNN | 90.9% | 6/6 | ❌ color shortcut (occlusion-proven) |
| Classical crop + CNN | 90.9% | 5/6 (worse) | ❌ flaky localizer |
| YOLO crop + CNN (mislabel present) | 90.9% | 5/6 | ⚠️ label error hidden inside |
| **YOLO crop + CNN (label fixed)** | **100%** | **5/5** | ✅ rim-only, cross-validated |

**Files that resulted**
- `parts/bracket/holes.py` — YOLO-first big-hole localizer + `big_hole_crop()`, with
  Hough and whole-image fallbacks.
- `parts/bracket/serration.py` — MobileNetV2 present/missing classifier; crops before
  classifying.
- `parts/bracket/train_serration.py` — classifier trainer + 3-fold CV; `--final`
  saves the model.
- `parts/bracket/train_big_hole_yolo.py` — YOLO trainer; embeds the hand-annotated
  boxes as the source of truth.
- Models: `inspection_state/models/serration_bracket.pt`,
  `inspection_state/models/big_hole_yolo.pt`.

**Reproduce**
```powershell
# cross-validation report
python parts/bracket/train_serration.py

# retrain the big-hole detector -> models/big_hole_yolo.pt
python parts/bracket/train_big_hole_yolo.py --train

# retrain + save the serration classifier -> models/serration_bracket.pt
python parts/bracket/train_serration.py --final
```

**Two gotchas worth remembering**
- *Antialiasing:* always downsample these shiny-teeth images with PIL antialiased
  resize; `cv2.resize` without antialiasing fabricates/destroys teeth and flips
  predictions.
- *Stale server:* `start.bat` starts a new server but won't kill an old one on the
  same port; kill the previous `python qms.py serve` before trusting a UI result.
