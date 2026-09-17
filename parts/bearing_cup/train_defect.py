"""Train the Bearing Cup defect classifier (MobileNetV2 transfer learning + rotation augmentation).

Used offline to build the model consumed by the main inspection pipeline.
Trained on bearing cup defects with full rotation/flip augmentation so it works
at ANY angle (0-360 degrees), not just 0/90/180/270.

Usage (run from anywhere; paths resolve to the repo root):
    python parts/bearing_cup/train_defect.py           # 3-fold cross-validation report
    python parts/bearing_cup/train_defect.py --final   # train on ALL data -> bearing_cup_defect.pt

Labels are set by visual inspection of the bearing cup defects (see categories below).
Retrain and add new labelled images here as more bearing cup photos are collected.
"""
import os, sys, glob, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2

# Make the repo root importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inspector import image_features as F

D = r"C:\workspace-ai\AES2-20260830T112553Z-1-001\AES2"
REVIEWS_FILE = os.path.join(REPO_ROOT, "inspection_state", "reviews", "bearing_cup.json")
AUG_REVIEWS_FILE = os.path.join(REPO_ROOT, "inspection_state", "reviews", "bearing_cup_augmented.json")
AUG_IMG_DIR = os.path.join(REPO_ROOT, "inspection_state", "data", "images", "bearing_cup_aug")
MODEL_OUT = os.path.join(REPO_ROOT, "inspection_state", "models", "bearing_cup_defect.pt")

# Load base review file to get defect categories and create labels
with open(REVIEWS_FILE) as f:
    base_reviews = json.load(f)

# Load augmented review file (rotations + flips for training robustness)
if os.path.exists(AUG_REVIEWS_FILE):
    with open(AUG_REVIEWS_FILE) as f:
        aug_reviews = json.load(f)
else:
    aug_reviews = base_reviews

# Build label mapping: "OK" vs defect type (use first defect category if multiple)
def get_label(entry):
    defects = entry.get("defects", [])
    if not defects:
        return "OK"
    # Use first defect category as the label
    return defects[0].get("category", "Unknown").replace(" ", "_")

# Collect all training files and labels
FILES = []
LABELS = []
for name, entry in sorted(aug_reviews.items()):
    src = os.path.join(AUG_IMG_DIR, name + ".jpg")
    if os.path.exists(src):
        FILES.append(src)
        LABELS.append(get_label(entry))
    else:
        # Fall back to original image directory if augmented doesn't exist
        src_orig = os.path.join(D, name + ".jpg")
        if os.path.exists(src_orig):
            FILES.append(src_orig)
            LABELS.append(get_label(entry))

# Get unique labels and create mapping
unique_labels = sorted(set(LABELS))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
label_indices = [label_to_idx[l] for l in LABELS]

print("dataset:", len(FILES), "files")
print("labels:", unique_labels)
for lbl in unique_labels:
    count = sum(1 for l in LABELS if l == lbl)
    print(f"  {lbl}: {count}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(180),           # Full 0-360 rotation
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
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
        bgr = cv2.imread(self.files[i])
        if bgr is None:
            # Return a blank image if file doesn't exist
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        bgr = F._resize(bgr)
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.tf(img), self.labels[i]

def make_model(num_classes):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze early layers, unfreeze last feature block for adaptation
    for p in m.features.parameters():
        p.requires_grad = False
    for p in m.features[-2:].parameters():
        p.requires_grad = True
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, num_classes))
    return m

def train_and_return(tr_files, tr_labels, num_classes, epochs=14):
    torch.manual_seed(0)
    m = make_model(num_classes)
    
    # Class weights to handle imbalance
    label_counts = {}
    for lbl in tr_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    cw = torch.tensor([1.0 * len(tr_labels) / (num_classes * (label_counts.get(i, 1) + 1e-9))
                       for i in range(num_classes)], dtype=torch.float32)
    
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    
    dl = DataLoader(DS(tr_files, tr_labels, train_tf, reps=8), batch_size=8, shuffle=True)
    m.train()
    
    for ep in range(epochs):
        tot_loss = 0.0
        for x, y in dl:
            opt.zero_grad()
            out = m(x)
            loss = crit(out, torch.tensor(y))
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"  epoch {ep + 1}/{epochs} loss={tot_loss / len(dl):.4f}")
    
    m.eval()
    return m

def eval_model(m, files, labels, num_classes):
    dl = DataLoader(DS(files, labels, eval_tf, reps=1), batch_size=8, shuffle=False)
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            out = m(x)
            preds = out.argmax(dim=1).numpy()
            correct += sum(preds == np.array(y))
            total += len(y)
    acc = 100.0 * correct / total if total else 0.0
    print(f"  accuracy: {correct}/{total} ({acc:.1f}%)")
    return acc

def save_model(m, path):
    ckpt = {
        "state_dict": m.state_dict(),
        "classes": unique_labels,
    }
    torch.save(ckpt, path)
    print(f"saved {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="Train on ALL data (no CV)")
    args = parser.parse_args()
    
    num_classes = len(unique_labels)
    
    if args.final:
        print(f"\nTraining on ALL {len(FILES)} images...")
        m = train_and_return(FILES, label_indices, num_classes, epochs=14)
        print(f"\nEvaluating on training set...")
        eval_model(m, FILES, label_indices, num_classes)
        save_model(m, MODEL_OUT)
    else:
        # 3-fold cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        folds = list(kf.split(FILES))
        
        accs = []
        for fold_i, (tr_idx, te_idx) in enumerate(folds):
            print(f"\n=== FOLD {fold_i + 1}/3 ===")
            tr_files = [FILES[i] for i in tr_idx]
            tr_labels = [label_indices[i] for i in tr_idx]
            te_files = [FILES[i] for i in te_idx]
            te_labels = [label_indices[i] for i in te_idx]
            
            m = train_and_return(tr_files, tr_labels, num_classes, epochs=14)
            print(f"Fold {fold_i + 1} validation:")
            acc = eval_model(m, te_files, te_labels, num_classes)
            accs.append(acc)
        
        print(f"\n=== CV SUMMARY ===")
        print(f"fold accuracies: {[f'{a:.1f}%' for a in accs]}")
        print(f"mean: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")
