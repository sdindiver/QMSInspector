"""Train the Bracket dark-spots present/absent classifier.

Same recipe as ``train_serration.py`` (MobileNetV2 transfer learning with heavy
rotation/flip augmentation) but for the "Dark Spots" defect. Because every image
is rotated through the full 0-360 range and flipped during training, the model
learns the dark-spot *pattern* rather than a fixed orientation, so it recognises
the defect at ANY angle -- no perceptual-hash recall needed.

The classifier looks at the whole segmented part (dark spots can appear anywhere
on the surface, unlike serration which is tied to the big-hole rim), with the
background flattened so the model focuses on the metal surface.

Usage (run from anywhere; paths resolve to the repo root):
    python parts/bracket/train_dark_spots.py           # leave-one-out CV report
    python parts/bracket/train_dark_spots.py --final   # train on ALL data -> dark_spots_bracket.pt

Labels come from inspection_state/reviews/bracket.json (category == "Dark Spots").
Add more labelled bracket photos there and retrain to improve reliability.
"""
import os, sys, glob, json, numpy as np, torch, torch.nn as nn
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
MODEL_OUT = os.path.join(REPO_ROOT, "inspection_state", "models", "dark_spots_bracket.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def part_crop(path):
    """Return the part square-cropped with background flattened to mid-grey.

    Segmenting + masking the background stops the classifier from keying on
    surroundings; a tight square crop around the part keeps the surface large.
    """
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


def dark_spot_names():
    """Set of 6-digit tags labelled 'Dark Spots' in the reviews file."""
    rev = json.load(open(REVIEWS))
    out = set()
    for name, entry in rev.items():
        for d in entry.get("defects", []):
            if d.get("category", "").strip().lower() == "dark spots":
                out.add(name[11:17])
    return out


POS = dark_spot_names()
def label_for(path):
    return 1 if os.path.basename(path)[11:17] in POS else 0


FILES = sorted(glob.glob(os.path.join(D, "*.jpg")))
LABELS = [label_for(f) for f in FILES]

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


def prob_spot(m, path, tta=5):
    """Average dark-spot probability over several rotated/flipped views (TTA)."""
    m.eval()
    ps = []
    with torch.no_grad():
        for _ in range(tta):
            x = train_tf(load_crop(path)).unsqueeze(0)
            ps.append(float(torch.softmax(m(x), 1)[0, 1]))
        x = eval_tf(load_crop(path)).unsqueeze(0)
        ps.append(float(torch.softmax(m(x), 1)[0, 1]))
    return float(np.mean(ps))


if __name__ == "__main__":
    print("dataset:", len(FILES), "dark_spots=", sum(LABELS), "other=", len(LABELS) - sum(LABELS))
    if "--final" in sys.argv:
        m = train_model(FILES, LABELS, epochs=16)
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        torch.save({"state_dict": m.state_dict(),
                    "classes": ["absent", "dark_spots"],
                    "arch": "mobilenet_v2", "input": 224,
                    "mean": IMAGENET_MEAN, "std": IMAGENET_STD}, MODEL_OUT)
        print("saved final model ->", MODEL_OUT)
        sys.exit(0)

    # Leave-one-out CV: honestly measure recall on unseen spots + false positives.
    idx_pos = [i for i, l in enumerate(LABELS) if l == 1]
    idx_neg = [i for i, l in enumerate(LABELS) if l == 0]
    rng = np.random.default_rng(0)
    rng.shuffle(idx_neg)
    neg_folds = np.array_split(idx_neg, len(idx_pos))

    rows = []
    for fi, pi in enumerate(idx_pos):
        va = [pi] + list(neg_folds[fi])
        tr = [i for i in range(len(FILES)) if i not in va]
        m = train_model([FILES[i] for i in tr], [LABELS[i] for i in tr], epochs=16)
        for i in va:
            p = prob_spot(m, FILES[i])
            rows.append((os.path.basename(FILES[i])[11:17], LABELS[i], p))
            print(f"  fold{fi} {rows[-1][0]} true={LABELS[i]} prob_spot={p:.3f}")

    thr = 0.5
    tp = sum(1 for _, t, p in rows if t == 1 and p >= thr)
    fn = sum(1 for _, t, p in rows if t == 1 and p < thr)
    fp = sum(1 for _, t, p in rows if t == 0 and p >= thr)
    tn = sum(1 for _, t, p in rows if t == 0 and p < thr)
    print(f"\nLEAVE-ONE-OUT @thr={thr}: TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  spot recall = {tp}/{tp+fn}   false positives = {fp}/{fp+tn}")
