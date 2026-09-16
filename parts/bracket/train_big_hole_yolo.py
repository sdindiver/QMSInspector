"""Train the big-hole YOLO detector used to crop bracket serration images.

The serration classifier (train_serration.py) must look ONLY at the bracket's
large splined/toothed "big hole", never the two small plain holes or the part
body. To make that reliable regardless of part orientation, a tiny yolov8n
detector is trained to localize the big hole; inspector/holes.py then crops to
it before classification.

Boxes below were hand-annotated on the 22 defect-set images (normalized
cx,cy,w,h). Run ``python parts/bracket/train_big_hole_yolo.py --train`` to
rebuild the dataset and train; the resulting weights are copied to
``inspection_state/models/big_hole_yolo.pt``.
"""
import os
import glob
import shutil
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = r"C:\workspace-ai\AES2-20260830T112553Z-1-001\ring\bracket\defect"

# Big (serration/spline) hole box per image, normalized [cx,cy,w,h].
# Present parts: box the visibly toothed hole. Missing parts: the structural
# spline hole at the same end. Orientation varies, so positions differ.
BOXES = {
    "163734": (0.47, 0.86, 0.22, 0.15), "163740": (0.44, 0.80, 0.20, 0.15),
    "163750": (0.39, 0.78, 0.20, 0.16), "163810": (0.43, 0.80, 0.19, 0.14),
    "163907": (0.50, 0.27, 0.15, 0.13), "164017": (0.50, 0.22, 0.16, 0.14),
    "164116": (0.60, 0.17, 0.17, 0.14), "164209": (0.31, 0.72, 0.17, 0.12),
    "164314": (0.42, 0.78, 0.16, 0.13), "164330": (0.46, 0.77, 0.16, 0.13),
    "164359": (0.48, 0.83, 0.16, 0.13), "164510": (0.45, 0.83, 0.16, 0.13),
    "164631": (0.21, 0.50, 0.14, 0.26), "164648": (0.19, 0.63, 0.14, 0.18),
    "164704": (0.51, 0.23, 0.16, 0.13), "164818": (0.80, 0.46, 0.14, 0.18),
    "164838": (0.45, 0.78, 0.16, 0.13), "164950": (0.48, 0.25, 0.16, 0.14),
    "165050": (0.70, 0.59, 0.14, 0.20), "165106": (0.47, 0.77, 0.16, 0.13),
    "165152": (0.50, 0.20, 0.16, 0.13), "165247": (0.47, 0.77, 0.16, 0.13),
}

ROOT = os.path.join(REPO_ROOT, "parts", "bracket", "_yolo")
IMG = os.path.join(ROOT, "images")
LAB = os.path.join(ROOT, "labels")
MODEL_OUT = os.path.join(REPO_ROOT, "inspection_state", "models", "big_hole_yolo.pt")


def keyfor(f):
    return os.path.basename(f)[11:17]


def build_dataset():
    for d in (IMG, LAB):
        os.makedirs(d, exist_ok=True)
    files = sorted(glob.glob(os.path.join(D, "*.jpg")))
    n = 0
    for f in files:
        k = keyfor(f)
        if k not in BOXES:
            continue
        shutil.copy2(f, os.path.join(IMG, k + ".jpg"))
        cx, cy, w, h = BOXES[k]
        with open(os.path.join(LAB, k + ".txt"), "w") as fh:
            fh.write(f"0 {cx} {cy} {w} {h}\n")
        n += 1
    data = {"path": ROOT, "train": "images", "val": "images",
            "names": {0: "big_hole"}}
    with open(os.path.join(ROOT, "data.yaml"), "w") as fh:
        yaml.safe_dump(data, fh)
    print("dataset images:", n)
    return os.path.join(ROOT, "data.yaml")


def train():
    from ultralytics import YOLO
    data_yaml = build_dataset()
    project = os.path.join(ROOT, "runs")
    m = YOLO("yolov8n.pt")
    m.train(data=data_yaml, epochs=120, imgsz=512, batch=8,
            degrees=180, fliplr=0.5, flipud=0.5, scale=0.5, translate=0.1,
            hsv_h=0.3, hsv_s=0.7, hsv_v=0.4, mosaic=0.0,
            project=project, name="train", exist_ok=True, verbose=False)
    best = os.path.join(project, "train", "weights", "best.pt")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    shutil.copy2(best, MODEL_OUT)
    print("training done -> ", MODEL_OUT)


if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    else:
        build_dataset()
        print("dataset built; run with --train to train the detector")
