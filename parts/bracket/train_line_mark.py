"""Train the Bracket Line-Mark present/absent classifier on RIDGE-ENHANCED crops.

Line marks are thin scratches that are almost invisible to a normal CNN after the
whole part is downscaled to 224px (proven: raw-crop CNN scored 2/5 recall). This
trainer amplifies the line signal BEFORE the network sees it: each crop is
converted to a ridge map (max of black-hat and top-hat morphology = dark or bright
thin lines) stacked with the raw grey and a Hough line mask. With full 0-360
rotation + flip augmentation the model learns the line-mark pattern at ANY angle.

Usage:
    python parts/bracket/train_line_mark.py           # leave-one-out CV
    python parts/bracket/train_line_mark.py --final    # save line_mark_bracket.pt
"""
import os, sys, glob, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inspector import image_features as F

D = r"C:\workspace-ai\AES2-20260830T112553Z-1-001\ring\bracket\defect"
REVIEWS = os.path.join(REPO_ROOT, "inspection_state", "reviews", "bracket.json")
MODEL_OUT = os.path.join(REPO_ROOT, "inspection_state", "models", "line_mark_bracket.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT = 224


def ridge_stack(path):
    """Return a 3-channel uint8 image: [grey, ridge map, Hough line mask].

    Background is flattened to the part median so the silhouette edge doesn't
    create a fake long line. Holes are masked out of the ridge/line channels.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = F._resize(bgr)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask, cnt = F.segment_part(g)
    part = mask > 0
    if cnt is None or part.sum() < 500:
        base = cv2.resize(g, (INPUT, INPUT))
        return cv2.merge([base, base, base])
    axis = F._part_axis(cnt)
    med = int(np.median(g[part]))
    g2 = g.copy(); g2[~part] = med
    er = max(20, int(0.045 * axis))
    inner = cv2.erode(mask, np.ones((er, er), np.uint8)) > 0
    holes = ((g < med - 60) & part).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    holes = cv2.dilate(holes, np.ones((max(5, int(0.015 * axis)),) * 2, np.uint8))
    valid = inner & (holes == 0)

    bh = cv2.morphologyEx(g2, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    th = cv2.morphologyEx(g2, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ridge = np.maximum(bh, th).astype(np.float32)
    ridge[~valid] = 0
    rr = ridge[valid]
    thr = np.percentile(rr, 97) if rr.size else 255
    rnorm = np.clip(ridge / (thr + 1e-6) * 180, 0, 255).astype(np.uint8)

    edges = (ridge > thr).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=8)
    lmask = np.zeros_like(g)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(lmask, (x1, y1), (x2, y2), 255, 5)

    # crop to part bbox (square) so the part fills the frame
    x, y, w, h = cv2.boundingRect(cnt)
    pad = int(0.04 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1b, y1b = min(g.shape[1], x + w + pad), min(g.shape[0], y + h + pad)
    gc = g2[y0:y1b, x0:x1b]
    rc = rnorm[y0:y1b, x0:x1b]
    lc = lmask[y0:y1b, x0:x1b]
    ch = [cv2.resize(c, (INPUT, INPUT)) for c in (gc, rc, lc)]
    return cv2.merge(ch)


_CACHE = {}
def load_stack(path):
    if path not in _CACHE:
        _CACHE[path] = ridge_stack(path)
    return _CACHE[path]


def positive_tags():
    rev = json.load(open(REVIEWS))
    out = set()
    for name, entry in rev.items():
        for d in entry.get("defects", []):
            if d.get("category", "").strip().lower() == "line mark":
                out.add(name[11:17])
    return out


POS = positive_tags()
FILES = sorted(glob.glob(os.path.join(D, "*.jpg")))
LABELS = [1 if os.path.basename(f)[11:17] in POS else 0 for f in FILES]

# Augmentation operates on the 3-channel ridge stack (numpy -> tensor).
NORM = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def aug_tensor(stack, train):
    img = stack.copy()
    if train:
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        ang = np.random.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((INPUT / 2, INPUT / 2), ang, np.random.uniform(0.85, 1.15))
        img = cv2.warpAffine(img, M, (INPUT, INPUT), borderMode=cv2.BORDER_REFLECT)
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        if np.random.rand() < 0.5:
            img = img[::-1, :].copy()
    t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return NORM(t)


class DS(Dataset):
    def __init__(self, files, labels, train, reps=1):
        self.files = files * reps
        self.labels = labels * reps
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return aug_tensor(load_stack(self.files[i]), self.train), self.labels[i]


def make_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    for p in m.features[-3:].parameters():
        p.requires_grad = True
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, 2))
    return m


def _cw(labels):
    pos = sum(labels); neg = len(labels) - pos
    return torch.tensor([len(labels) / (2 * neg + 1e-9),
                         len(labels) / (2 * pos + 1e-9)], dtype=torch.float32)


def train_model(tr_files, tr_labels, epochs=18):
    torch.manual_seed(0)
    m = make_model()
    crit = nn.CrossEntropyLoss(weight=_cw(tr_labels))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    dl = DataLoader(DS(tr_files, tr_labels, True, reps=10), batch_size=8, shuffle=True)
    m.train()
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
    return m


def prob_line(m, path, tta=6):
    m.eval(); ps = []
    with torch.no_grad():
        for _ in range(tta):
            ps.append(float(torch.softmax(m(aug_tensor(load_stack(path), True).unsqueeze(0)), 1)[0, 1]))
        ps.append(float(torch.softmax(m(aug_tensor(load_stack(path), False).unsqueeze(0)), 1)[0, 1]))
    return float(np.mean(ps))


if __name__ == "__main__":
    print("dataset:", len(FILES), "line_marks=", sum(LABELS), "other=", len(LABELS) - sum(LABELS))
    if "--final" in sys.argv:
        m = train_model(FILES, LABELS, epochs=18)
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "classes": ["absent", "line_mark"],
                    "arch": "mobilenet_v2_ridge3", "input": INPUT,
                    "mean": IMAGENET_MEAN, "std": IMAGENET_STD}, MODEL_OUT)
        print("saved ->", MODEL_OUT)
        sys.exit(0)

    idx_pos = [i for i, l in enumerate(LABELS) if l == 1]
    idx_neg = [i for i, l in enumerate(LABELS) if l == 0]
    rng = np.random.default_rng(0); rng.shuffle(idx_neg)
    neg_folds = np.array_split(idx_neg, len(idx_pos))
    rows = []
    for fi, pi in enumerate(idx_pos):
        va = [pi] + list(neg_folds[fi])
        tr = [i for i in range(len(FILES)) if i not in va]
        m = train_model([FILES[i] for i in tr], [LABELS[i] for i in tr], epochs=18)
        for i in va:
            p = prob_line(m, FILES[i])
            rows.append((os.path.basename(FILES[i])[11:17], LABELS[i], p))
            print(f"  fold{fi} {rows[-1][0]} true={LABELS[i]} prob={p:.3f}")
    for thr in (0.4, 0.5, 0.6):
        tp = sum(1 for _, t, p in rows if t == 1 and p >= thr)
        fn = sum(1 for _, t, p in rows if t == 1 and p < thr)
        fp = sum(1 for _, t, p in rows if t == 0 and p >= thr)
        tn = sum(1 for _, t, p in rows if t == 0 and p < thr)
        print(f"@thr={thr}: recall={tp}/{tp+fn}  FP={fp}/{fp+tn}")
