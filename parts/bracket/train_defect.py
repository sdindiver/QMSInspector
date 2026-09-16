"""Generic any-angle defect classifier trainer for Bracket parts.

Same recipe as ``train_dark_spots.py`` / ``train_serration.py`` (MobileNetV2
transfer learning + heavy rotation/flip augmentation so the defect *pattern* is
learned at ANY angle) but parameterized by ``--category`` so one script trains a
binary "present/absent" classifier for any defect that has enough labelled
examples in inspection_state/reviews/bracket.json.

Usage (run from anywhere; paths resolve to the repo root):
    python parts/bracket/train_defect.py --category "Line Mark"           # LOO CV
    python parts/bracket/train_defect.py --category "Line Mark" --final   # save model

The saved model file is ``<category-slug>_bracket.pt`` in the models dir, e.g.
"Line Mark" -> line_mark_bracket.pt. The generic predictor ``defect_clf.py``
loads it by the same slug.
"""
import os, sys, glob, json, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inspector import image_features as F

D = r"C:\workspace-ai\AES2-20260830T112553Z-1-001\ring\bracket\defect"
REVIEWS = os.path.join(REPO_ROOT, "inspection_state", "reviews", "bracket.json")
MODELS_DIR = os.path.join(REPO_ROOT, "inspection_state", "models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def slug(category):
    return category.strip().lower().replace("/", " ").replace("-", " ").replace("  ", " ").replace(" ", "_")


def model_path(category):
    return os.path.join(MODELS_DIR, f"{slug(category)}_bracket.pt")


def part_crop(path):
    """Whole part square-cropped with background flattened to mid-grey."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = F._resize(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask, cnt = F.segment_part(gray)
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        pad = int(0.04 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
        flat = np.where(m3, bgr, 128).astype(np.uint8)
        crop = flat[y0:y1, x0:x1]
    else:
        crop = bgr
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


_CACHE = {}
def load_crop(path):
    if path not in _CACHE:
        _CACHE[path] = part_crop(path)
    return _CACHE[path]


def positive_tags(category):
    """Set of 6-digit tags labelled with ``category`` in the reviews file."""
    rev = json.load(open(REVIEWS))
    out = set()
    tgt = category.strip().lower()
    for name, entry in rev.items():
        for d in entry.get("defects", []):
            if d.get("category", "").strip().lower() == tgt:
                out.add(name[11:17])
    return out


train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(180, fill=128),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DS(Dataset):
    def __init__(self, files, labels, tf, reps=1):
        self.files = files * reps
        self.labels = labels * reps
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(load_crop(self.files[i])), self.labels[i]


def make_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    for p in m.features[-2:].parameters():
        p.requires_grad = True
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, 2))
    return m


def _class_weights(labels):
    pos = sum(labels)
    neg = len(labels) - pos
    return torch.tensor([len(labels) / (2 * neg + 1e-9),
                         len(labels) / (2 * pos + 1e-9)], dtype=torch.float32)


def train_model(tr_files, tr_labels, epochs=16):
    torch.manual_seed(0)
    m = make_model()
    crit = nn.CrossEntropyLoss(weight=_class_weights(tr_labels))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    dl = DataLoader(DS(tr_files, tr_labels, train_tf, reps=8), batch_size=8, shuffle=True)
    m.train()
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward()
            opt.step()
    return m


def prob_present(m, path, tta=5):
    m.eval()
    ps = []
    with torch.no_grad():
        for _ in range(tta):
            ps.append(float(torch.softmax(m(train_tf(load_crop(path)).unsqueeze(0)), 1)[0, 1]))
        ps.append(float(torch.softmax(m(eval_tf(load_crop(path)).unsqueeze(0)), 1)[0, 1]))
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(D, "*.jpg")))
    pos = positive_tags(args.category)
    labels = [1 if os.path.basename(f)[11:17] in pos else 0 for f in files]
    print(f"category={args.category!r}  dataset={len(files)}  positive={sum(labels)}  other={len(labels)-sum(labels)}")
    if sum(labels) < 3:
        print("WARNING: fewer than 3 positive examples -- a trained classifier will be unreliable.")

    if args.final:
        m = train_model(files, labels, epochs=args.epochs)
        out = model_path(args.category)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"state_dict": m.state_dict(),
                    "classes": ["absent", "present"],
                    "category": args.category,
                    "arch": "mobilenet_v2", "input": 224,
                    "mean": IMAGENET_MEAN, "std": IMAGENET_STD}, out)
        print("saved final model ->", out)
        return

    idx_pos = [i for i, l in enumerate(labels) if l == 1]
    idx_neg = [i for i, l in enumerate(labels) if l == 0]
    rng = np.random.default_rng(0)
    rng.shuffle(idx_neg)
    neg_folds = np.array_split(idx_neg, len(idx_pos))

    rows = []
    for fi, pi in enumerate(idx_pos):
        va = [pi] + list(neg_folds[fi])
        tr = [i for i in range(len(files)) if i not in va]
        m = train_model([files[i] for i in tr], [labels[i] for i in tr], epochs=args.epochs)
        for i in va:
            p = prob_present(m, files[i])
            rows.append((os.path.basename(files[i])[11:17], labels[i], p))
            print(f"  fold{fi} {rows[-1][0]} true={labels[i]} prob={p:.3f}")

    thr = 0.5
    tp = sum(1 for _, t, p in rows if t == 1 and p >= thr)
    fn = sum(1 for _, t, p in rows if t == 1 and p < thr)
    fp = sum(1 for _, t, p in rows if t == 0 and p >= thr)
    tn = sum(1 for _, t, p in rows if t == 0 and p < thr)
    print(f"\nLEAVE-ONE-OUT @thr={thr}: TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  recall = {tp}/{tp+fn}   false positives = {fp}/{fp+tn}")


if __name__ == "__main__":
    main()
