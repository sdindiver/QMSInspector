"""Train the Bracket serration present/missing classifier (MobileNetV2 transfer
learning + augmentation). Used offline to (re)build the model consumed by
``inspector/serration.py``.

The classifier is trained on big-hole CROPS (produced by ``inspector/holes.py``
via the trained YOLO detector) so it looks ONLY at the large splined hole rim and
cannot use part colour / body shortcuts. Retrain the YOLO detector first with
``train_big_hole_yolo.py`` if you add new images.

Usage (run from anywhere; paths resolve to the repo root):
    python parts/bracket/train_serration.py           # 3-fold cross-validation report
    python parts/bracket/train_serration.py --final   # train on ALL data -> serration_bracket.pt

Labels are set by visual inspection of the big-hole rim (see MISSING set below).
Retrain and add new labelled images here as more bracket photos are collected.
"""
import os, sys, glob, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2

# Make the repo root importable so ``inspector`` resolves no matter the CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parts.bracket import holes as HOLES

_CROP_CACHE={}
def load_crop(path):
    """Load an image cropped to the big hole (falls back to full image)."""
    if path in _CROP_CACHE:
        return _CROP_CACHE[path]
    bgr = cv2.imread(path)
    crop = HOLES.big_hole_crop(bgr) if bgr is not None else None
    if crop is None:
        crop = bgr
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    _CROP_CACHE[path] = img
    return img

D = r"C:\workspace-ai\AES2-20260830T112553Z-1-001\ring\bracket\defect"
MODEL_OUT = os.path.join(REPO_ROOT, "inspection_state", "models", "serration_bracket.pt")

# Labels by visual inspection of the big-hole rim (1=serration present, 0=missing).
# Verified on YOLO big-hole crops where the teeth are crisp and unambiguous.
MISSING = {"163907","164017","164631","164648","164704"}
def label_for(name):
    t = os.path.basename(name)[11:17]
    return 0 if t in MISSING else 1

FILES = sorted(glob.glob(os.path.join(D,"*.jpg")))
LABELS = [label_for(f) for f in FILES]
print("dataset:", len(FILES), "present=",sum(LABELS),"missing=",len(LABELS)-sum(LABELS))

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4,0.4,0.4,0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
])

class DS(Dataset):
    def __init__(self, files, labels, tf, reps=1):
        self.files=files*reps; self.labels=labels*reps; self.tf=tf
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        img=load_crop(self.files[i])
        return self.tf(img), self.labels[i]

def make_model():
    m=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters(): p.requires_grad=False
    # unfreeze last feature block for a bit of adaptation
    for p in m.features[-2:].parameters(): p.requires_grad=True
    m.classifier=nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel,2))
    return m

def train_and_return(tr_files,tr_labels,epochs=14):
    torch.manual_seed(0)
    m=make_model()
    cw=torch.tensor([1.0*len(tr_labels)/(2*(len(tr_labels)-sum(tr_labels))+1e-9),
                     1.0*len(tr_labels)/(2*sum(tr_labels)+1e-9)],dtype=torch.float32)
    crit=nn.CrossEntropyLoss(weight=cw)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    dl=DataLoader(DS(tr_files,tr_labels,train_tf,reps=8),batch_size=8,shuffle=True)
    m.train()
    for ep in range(epochs):
        for x,y in dl:
            opt.zero_grad(); out=m(x); loss=crit(out,y); loss.backward(); opt.step()
    return m


def train_one(tr_files,tr_labels,va_files,va_labels,epochs=12):
    torch.manual_seed(0)
    m=make_model()
    cw=torch.tensor([1.0*len(tr_labels)/(2*(len(tr_labels)-sum(tr_labels))+1e-9),
                     1.0*len(tr_labels)/(2*sum(tr_labels)+1e-9)],dtype=torch.float32)
    crit=nn.CrossEntropyLoss(weight=cw)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    dl=DataLoader(DS(tr_files,tr_labels,train_tf,reps=8),batch_size=8,shuffle=True)
    m.train()
    for ep in range(epochs):
        for x,y in dl:
            opt.zero_grad(); out=m(x); loss=crit(out,y); loss.backward(); opt.step()
    # eval (average over a few augmentation-free passes)
    m.eval(); preds=[]
    with torch.no_grad():
        for f in va_files:
            x=eval_tf(load_crop(f)).unsqueeze(0)
            preds.append(int(m(x).argmax(1)))
    return preds

# Stratified k-fold (k=3 due to only 5 missing)
def folds(labels,k=3,seed=0):
    rng=np.random.default_rng(seed)
    idx0=[i for i,l in enumerate(labels) if l==0]; rng.shuffle(idx0)
    idx1=[i for i,l in enumerate(labels) if l==1]; rng.shuffle(idx1)
    F=[[] for _ in range(k)]
    for j,i in enumerate(idx0): F[j%k].append(i)
    for j,i in enumerate(idx1): F[j%k].append(i)
    return F

if __name__=="__main__":
    if "--final" in sys.argv:
        # Train final model on ALL data and save for production use.
        m=train_and_return(FILES,LABELS,epochs=14)
        os.makedirs(os.path.dirname(MODEL_OUT),exist_ok=True)
        torch.save({"state_dict":m.state_dict(),
                    "classes":["missing","present"],
                    "arch":"mobilenet_v2",
                    "input":224,
                    "mean":IMAGENET_MEAN,"std":IMAGENET_STD}, MODEL_OUT)
        print("saved final model ->", MODEL_OUT)
        sys.exit(0)
    F=folds(LABELS,k=3)
    all_true=[]; all_pred=[]
    for fi in range(3):
        va=F[fi]; tr=[i for i in range(len(FILES)) if i not in va]
        trf=[FILES[i] for i in tr]; trl=[LABELS[i] for i in tr]
        vaf=[FILES[i] for i in va]; val=[LABELS[i] for i in va]
        pr=train_one(trf,trl,vaf,val)
        for f,t,p in zip(vaf,val,pr):
            tag=os.path.basename(f)[11:17]
            all_true.append(t); all_pred.append(p)
            print(f"  fold{fi} {tag} true={t} pred={p} {'OK' if t==p else 'X'}")
    acc=np.mean([a==b for a,b in zip(all_true,all_pred)])
    tp=sum(1 for t,p in zip(all_true,all_pred) if t==1 and p==1)
    tn=sum(1 for t,p in zip(all_true,all_pred) if t==0 and p==0)
    fp=sum(1 for t,p in zip(all_true,all_pred) if t==0 and p==1)
    fn=sum(1 for t,p in zip(all_true,all_pred) if t==1 and p==0)
    print(f"\nCROSS-VAL ACCURACY: {acc*100:.1f}%  (TP={tp} TN={tn} FP={fp} FN={fn})")
    print("missing-recall (caught defects):", f"{tn}/{tn+fp}")
