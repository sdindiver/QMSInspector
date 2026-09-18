# Skill: Augmentation & Training for a new defect

> **Purpose.** Capture the user's expectations for how we augment data and train a
> per-defect classifier, so the same recipe can be reused for **any** new defect
> (line mark, dark spot, serration, corrosion, ...). This is the standard we agreed
> on while building the Bracket **Line Mark** detector.
>
> Reference implementation:
> - Augmenter: `parts/bracket/augment_line_mark.py`
> - Trainer:   `parts/bracket/train_line_mark.py`  (`--final --aug`)
>
> Visual overview: [`augmentation_pipeline.md`](./augmentation_pipeline.md) (mermaid diagram).

---

## 1. The expectation in one line

**Materialise a large, heavily-augmented image set to disk (300+ images), covering
every orientation AND real-world lighting, then train on it — and always augment
EVERY positive (defect) image, not just one.**

We previously used a lighter bearing-cup augmenter (`parts/bearing_cup/rotation_augment.py`).
This skill deliberately **fixes its weaknesses**. Any new-defect augmenter MUST cover
the checklist in §3.

---

## 2. Two-stage pipeline (always)

```
STAGE A  augment  ->  writes 300+ JPGs + manifest.json to
                      inspection_state/augmented/<part>_<defect>/
                      (inspectable BEFORE training — show the user a montage)

STAGE B  train    ->  reads that folder, trains, saves the .pt,
                      prints per-epoch loss + a sanity check on the real positives
```

Materialising to disk (not just on-the-fly) matters because the **user wants to eyeball
the augmented images** and confirm the defect is still visible and correctly placed
before we spend time training.

### 2.1 Standard location for augmented sets (mandatory)

**Every materialised augmented image set lives under `inspection_state/augmented/`** —
one sub-folder per part/defect. Never scatter them under `data/images/` or elsewhere;
`data/images/` is reserved for real inspection **input** images.

```
inspection_state/augmented/
    bracket_line_mark/      <- bracket line-mark set (+ manifest.json)
    bearing_cup_aug/        <- bearing-cup rotation/flip set
    <part>_<defect>/        <- every new defect follows the same pattern
```

The trainers point at this folder directly:
- `parts/bracket/train_line_mark.py`  -> `AUG_DIR   = inspection_state/augmented/bracket_line_mark`
- `parts/bearing_cup/train_defect.py` -> `AUG_IMG_DIR = inspection_state/augmented/bearing_cup_aug`

When you (re)generate a set with `rotation_augment.py`, pass the matching
`--output-dir inspection_state/augmented/<part>_<...>` so the trainer finds it.

---

## 3. Augmentation checklist (what "heavy" means)

Every augmenter must apply **all** of these. The bracket line-mark values are given as
the working defaults.

**Geometric**
- [x] **Continuous rotation** over `0–360°` (NOT just 45/135/225/315). The defect
      appears at *every* angle.
- [x] The **defect rotates WITH the part** — because we rotate the whole photo, the
      scratch/spot rotates too. (The user explicitly asked for this: "defect can be
      rotated also, right?" — yes.)
- [x] **Horizontal + vertical flips** (each ~50%).
- [x] **Scale jitter** (`0.88–1.12`) and small **translation jitter** (`±3%`) so the
      part is not always centred.

**Photometric (lighting robustness — the bearing-cup version lacked all of this)**
- [x] **Brightness** (additive `±28`) and **contrast** (multiplicative `0.75–1.30`
      about the mean).
- [x] **Gamma** (`0.7–1.4`).
- [x] **Smooth lighting gradient** across the frame (uneven illumination / glare),
      random direction, strength `0.10–0.45`.
- [x] **Mild sensor noise** (Gaussian, sigma `2–8`), ~60% of the time.

**Correctness / anti-artifacts**
- [x] Rotation fills exposed corners with the **sampled background colour**
      (`BORDER_CONSTANT`), **never `BORDER_REFLECT`** — reflecting the bright part edge
      creates a **phantom mirrored line** in the corner, which is fatal for a *line*
      detector and misleading for any defect.
- [x] No bright part silhouette left as a fake long line in the ridge/line channels.

**Class balance**
- [x] Give **positive (defect) images MORE variants** than negatives to counter the
      typical skew (bracket line mark: 5 positives vs 17 negatives → `40/positive`,
      `12/negative` ⇒ 426 images, `line_mark=205`, `other=221`).
- [x] Keep one **clean original** of each source image in the set too (`*_orig.jpg`).
- [x] **Augment ALL positive images.** (User caught this: never augment just one
      source. Verify with a per-source count and a montage that shows one sample from
      *each* positive.)

---

## 4. Trainer expectations

- **Model**: small transfer-learned CNN (MobileNetV2) fine-tuning only the last few
  feature blocks + a fresh 2-class head. Input `224`.
- **Signal amplification first**: for thin/low-contrast defects, feed a 3-channel
  **ridge stack** `[grey, ridge map, Hough line mask]` rather than the raw crop
  (see `ridge_stack()`), background flattened to the part median, holes/rim masked.
  Adapt the channels to the defect physics for other defects.
- **On-the-fly augmentation on top of the offline set**: per-epoch random
  rotation/flip **plus photometric jitter**, using **`BORDER_REPLICATE`** (again, never
  `REFLECT`).
- **Class weighting** in the loss to handle any residual imbalance.
- **Visible progress**: print **per-epoch loss** (run with `python -u ...`), and after
  saving, print a **sanity check** — the model's probability on each real defect image
  with `OK`/`MISS`.
- **Honest evaluation**: keep the **leave-one-out CV** mode (default, no `--final`) for
  measuring recall/false-positives on *held-out* parts. `--final` (or `--final --aug`)
  is only for producing the shipping weights.

---

## 5. Runbook for a new defect

1. **Confirm labels first.** Render the annotations (green centre-lines for line-type
   defects, boxes/polys otherwise) and get the user's explicit OK before training.
   Always **delete + re-render + verify** rendered pixels match the JSON (stale-file
   trap).
2. **Copy the augmenter** to `parts/<part>/augment_<defect>.py`; point `SRC_DIR`,
   `REVIEWS`, `OUT_DIR` at the new defect; set the `positive_tags()` category string.
3. **Generate** and **show a montage** (one sample per positive source). Get the user's
   OK on augmentation quality.
   ```powershell
   python parts\<part>\augment_<defect>.py --pos 40 --neg 12
   ```
4. **Train** (streams progress; run in the IDE terminal so the user can watch):
   ```powershell
   python -u parts\<part>\train_<defect>.py --final --aug
   ```
5. **Verify**: model file is fresh, sanity check shows the real positives as `OK`.
   If recall is weak, iterate (more variants, stronger ridge channels, or add a
   geometric ensemble member — see `docs/methodology.md`).
6. **Integrate** into the runtime detector and keep the leave-one-out numbers on record.

---

## 6. Provenance / defaults table

| Knob                    | Bracket line-mark default |
|-------------------------|---------------------------|
| variants per positive   | 40 (+1 original)          |
| variants per negative   | 12 (+1 original)          |
| rotation                | continuous 0–360°         |
| scale / translation     | 0.88–1.12 / ±3%           |
| flips                   | H 50%, V 50%              |
| brightness / contrast   | ±28 / 0.75–1.30           |
| gamma                   | 0.7–1.4                   |
| lighting gradient       | strength 0.10–0.45, 80%   |
| noise                   | sigma 2–8, 60%            |
| border fill             | background colour (CONSTANT) — never REFLECT |
| total set (bracket)     | 426 imgs (205 pos / 221 neg) |
| trainer epochs / reps   | 14 / 3 (on the 426 set)   |

---

## 7. Retraining when new photos arrive (future maintenance)

**The single biggest lever for accuracy is MORE REAL defect photos — not lower loss,
not more augmentation.** Augmentation multiplies the variety of the examples we have;
it cannot invent a scratch pattern we have never photographed. With only a handful of
real positives any CNN will overfit, which is why we ship the **CNN-OR-geometry
ensemble** (§4, `parts/bracket/line_mark.py`).

When you collect new line-mark (or other defect) photos:

1. **Drop the new images** into the defect image folder and **label** them in the
   reviews JSON (`inspection_state/reviews/<part>.json`) — same green-centreline /
   box / polygon convention, get the user's OK on the render first.
2. **Re-run the augmenter** so the new positives get their variants too:
   ```powershell
   python parts\<part>\augment_<defect>.py --pos 40 --neg 12
   ```
3. **Re-train the shipping model** on the refreshed augmented set:
   ```powershell
   python -u parts\<part>\train_<defect>.py --final --aug
   ```
   This overwrites `inspection_state/models/<defect>_<part>.pt`, which the runtime
   detector loads automatically — **no code change needed to integrate**; the new
   weights are live as soon as the file is saved.
4. **Re-measure honestly** before trusting it:
   - Leave-one-out CV (raw):   `python parts\<part>\train_<defect>.py`
   - Leak-free augmented CV:    `python parts\<part>\cv_aug_line_mark.py`  (bracket example)
   - End-to-end ensemble check on all parts (recall / false-positives).
5. **Ship** when held-out recall and false-positive rate are acceptable. The ensemble
   only gets stronger as the real-example count grows.

### How to read the numbers (don't be fooled by loss)
- **Training loss** going low (e.g. 0.03) is NOT the goal. On a subtle defect with few
  examples it means **memorisation / overfitting** — proven here: the 0.03-loss CNN
  scored **1/5** held-out. The regularised augmented model (loss ~0.20) scored
  **2-3/5** held-out. *Higher loss, better generalisation.*
- **In-sample / sanity check** (probability on the training images) will look great
  (~0.999). It only proves the wiring is correct, NOT that the model generalises.
- **Held-out CV recall + false-positives** is the number that predicts real-world
  behaviour. Judge by this.
- The deployed ensemble's **in-sample** headline for bracket line mark is
  **recall 5/5, FP 0/17**; the **held-out** view is more modest — the truth for brand
  new photos sits between, and the geometry member is what keeps it robust.

### Integration is automatic
The runtime wires the model in by **file path**, not by code:
`parts/<part>/line_mark.py` loads `inspection_state/models/<defect>_<part>.pt`, and
`inspector/recognizer.py` calls `predict()` for the matching part. Re-training and
saving the `.pt` is all that's required to deploy — there is nothing else to "hook up".
