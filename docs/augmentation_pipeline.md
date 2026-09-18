# Augmentation Pipeline (visual overview)

> Companion diagram for [`augmentation_training_skill.md`](./augmentation_training_skill.md).
> That doc is the authoritative recipe; this page is the picture. Use it to onboard
> quickly to **how a new per-defect classifier is produced** — from real photos to a
> deployed `.pt` that the runtime loads automatically.

The pipeline is deliberately **two-stage** (augment to disk, then train) so a human can
**eyeball the augmented images before spending time training** — the defect must still be
visible and correctly placed after every rotation and lighting change.

---

## End-to-end flow

```mermaid
flowchart TD
    subgraph SRC["Inputs"]
        A["Real defect photos<br/>data/images / source dir"]
        B["Reviews JSON<br/>inspection_state/reviews/&lt;part&gt;.json<br/>(labels: line mark / spot / serration ...)"]
    end

    A --> AUG
    B -->|positive_tags| AUG

    subgraph STAGEA["STAGE A - Augment (offline, to disk)"]
        AUG["augment_&lt;defect&gt;.py<br/>parts/&lt;part&gt;/"]
        AUG --> GEO["Geometric<br/>continuous 0-360 rotation<br/>H/V flips - scale 0.88-1.12 - translate +/-3%<br/>defect rotates WITH the part"]
        AUG --> PHOTO["Photometric<br/>brightness/contrast/gamma<br/>lighting gradient - sensor noise"]
        AUG --> SAFE["Anti-artifact<br/>BORDER_CONSTANT = background colour<br/>never REFLECT (phantom lines)"]
        GEO --> BAL["Class balance<br/>positives get MORE variants<br/>keep 1 clean *_orig per source"]
        PHOTO --> BAL
        SAFE --> BAL
    end

    BAL --> OUT["inspection_state/augmented/&lt;part&gt;_&lt;defect&gt;/<br/>300+ JPGs + manifest.json"]

    OUT --> EYE{"Human review<br/>montage OK?"}
    EYE -->|no - fix labels/knobs| AUG
    EYE -->|yes| TRAIN

    subgraph STAGEB["STAGE B - Train"]
        TRAIN["train_&lt;defect&gt;.py --final --aug<br/>parts/&lt;part&gt;/"]
        TRAIN --> SIG["Signal amplification<br/>ridge stack [grey, ridge, Hough]<br/>on-the-fly jitter (BORDER_REPLICATE)"]
        SIG --> FIT["MobileNetV2 transfer fine-tune<br/>class-weighted loss - per-epoch loss"]
        FIT --> SANITY["Sanity check<br/>prob on each real positive = OK/MISS"]
    end

    SANITY --> PT["inspection_state/models/&lt;defect&gt;_&lt;part&gt;.pt"]
    PT --> RUNTIME["Runtime loads by file path<br/>parts/&lt;part&gt;/&lt;defect&gt;.py -> recognizer.py<br/>(no code change to deploy)"]

    RUNTIME --> CV{"Held-out CV recall<br/>+ false-positives OK?"}
    CV -->|no - more real photos / variants| B
    CV -->|yes| SHIP["Ship"]
```

---

## Why two stages (and to disk)

```mermaid
flowchart LR
    OTF["On-the-fly only<br/>(augment inside training loop)"] -->|cannot inspect| RISK["Blind training<br/>bad labels found too late"]
    DISK["Materialise to disk first"] -->|montage before training| TRUST["Confirm defect still visible<br/>correct placement, no phantom lines"]
```

- **Materialising to disk** lets us show the user a montage (one sample per positive
  source) and get an explicit OK *before* training.
- Augmented sets are **regenerable**, so they are **not committed** to git
  (`inspection_state/augmented/` is gitignored). Re-run the augmenter to rebuild them.

---

## Reference implementations

| Stage | Bracket line-mark | Bearing-cup |
|-------|-------------------|-------------|
| Augment | `parts/bracket/augment_line_mark.py` | `parts/bearing_cup/rotation_augment.py` |
| Train | `parts/bracket/train_line_mark.py --final --aug` | `parts/bearing_cup/train_defect.py --final --aug` |
| Output set | `inspection_state/augmented/bracket_line_mark/` | `inspection_state/augmented/bearing_cup_aug/` |
| Model | `inspection_state/models/line_mark_bracket.pt` | `inspection_state/models/<defect>_bearing_cup.pt` |

For the full checklist (exact augmentation knobs, trainer expectations, and the runbook
for adding a brand-new defect) see **[`augmentation_training_skill.md`](./augmentation_training_skill.md)**.
