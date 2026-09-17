"""Rotation + flip augmentation for Bearing Cup defect training.

Generates synthetic rotation/flip variants of defect images to train
orientation-robust models. This addresses the concrete failure mode where
the original model only detected defects at 0/90/180/270 angles but missed
45/135/225/315 degree rotations.

Usage:
    python -m parts.bearing_cup.rotation_augment \
        --input review.json \
        --image-dir /path/to/images \
        --output augmented.json \
        --output-dir /path/to/augmented_images
"""
from __future__ import annotations
import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path


def rotate_bbox(bbox, angle_deg, h, w):
    """Rotate a normalized bbox [cx, cy, half_w, half_h] by angle_deg around image center."""
    x, y, hw, hh = bbox
    px, py = x * w, y * h
    
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    cx, cy = w / 2, h / 2
    dx, dy = px - cx, py - cy
    
    px_rot = cx + dx * cos_a - dy * sin_a
    py_rot = cy + dx * sin_a + dy * cos_a
    
    x_rot = px_rot / w
    y_rot = py_rot / h
    
    return [x_rot, y_rot, hw, hh]


def flip_bbox_h(bbox, w):
    """Flip bbox horizontally."""
    x, y, hw, hh = bbox
    return [1.0 - x, y, hw, hh]


def flip_bbox_v(bbox, h):
    """Flip bbox vertically."""
    x, y, hw, hh = bbox
    return [x, 1.0 - y, hw, hh]


def rotate_polygon(points, angle_deg, h, w):
    """Rotate normalized polygon points around image center."""
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = w / 2, h / 2
    
    rotated = []
    for x, y in points:
        px, py = x * w, y * h
        dx, dy = px - cx, py - cy
        px_rot = cx + dx * cos_a - dy * sin_a
        py_rot = cy + dx * sin_a + dy * cos_a
        rotated.append([px_rot / w, py_rot / h])
    return rotated


def flip_polygon_h(points):
    """Flip polygon horizontally."""
    return [[1.0 - x, y] for x, y in points]


def flip_polygon_v(points):
    """Flip polygon vertically."""
    return [[x, 1.0 - y] for x, y in points]


def augment_review(review_data, image_dir, output_dir, angles=(45, 135, 225, 315)):
    """Generate rotation/flip augmented variants of all images.
    
    Returns (augmented_review, created_images) where augmented_review is the
    extended review dict and created_images lists all new image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    augmented = {}
    created = []
    
    for img_name, entry in sorted(review_data.items()):
        src = os.path.join(image_dir, img_name + '.jpg')
        if not os.path.exists(src):
            continue
        
        # Keep the original
        augmented[img_name] = entry
        
        img = cv2.imread(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Generate rotations
        for angle in angles:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
            
            aug_name = f"{img_name}_rot{angle}"
            aug_path = os.path.join(output_dir, aug_name + '.jpg')
            cv2.imwrite(aug_path, rotated_img)
            created.append(aug_path)
            
            # Create augmented review entry
            aug_entry = {"part": entry.get("part"), "result": entry.get("result"),
                        "defects": []}
            for defect in entry.get("defects", []):
                aug_defect = defect.copy()
                # Rotate bbox if present
                if "bbox" in defect:
                    aug_defect["bbox"] = rotate_bbox(defect["bbox"], angle, h, w)
                # Rotate points if present
                if "points" in defect:
                    aug_defect["points"] = rotate_polygon(defect["points"], angle, h, w)
                aug_entry["defects"].append(aug_defect)
            
            augmented[aug_name] = aug_entry
        
        # Generate flips (both h and v)
        for flip_type, flip_name in [("h", "fliph"), ("v", "flipv")]:
            if flip_type == "h":
                flipped_img = cv2.flip(img, 1)
                flip_fn_bbox = lambda b: flip_bbox_h(b, w)
                flip_fn_poly = lambda p: flip_polygon_h(p)
            else:
                flipped_img = cv2.flip(img, 0)
                flip_fn_bbox = lambda b: flip_bbox_v(b, h)
                flip_fn_poly = lambda p: flip_polygon_v(p)
            
            aug_name = f"{img_name}_{flip_name}"
            aug_path = os.path.join(output_dir, aug_name + '.jpg')
            cv2.imwrite(aug_path, flipped_img)
            created.append(aug_path)
            
            # Create augmented review entry
            aug_entry = {"part": entry.get("part"), "result": entry.get("result"),
                        "defects": []}
            for defect in entry.get("defects", []):
                aug_defect = defect.copy()
                if "bbox" in defect:
                    aug_defect["bbox"] = flip_fn_bbox(defect["bbox"])
                if "points" in defect:
                    aug_defect["points"] = flip_fn_poly(defect["points"])
                aug_entry["defects"].append(aug_defect)
            
            augmented[aug_name] = aug_entry
    
    return augmented, created


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input review.json file")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--output", required=True, help="Output augmented review.json file")
    parser.add_argument("--output-dir", required=True, help="Output directory for augmented images")
    parser.add_argument("--angles", default="45,135,225,315", help="Rotation angles (comma-separated)")
    
    args = parser.parse_args()
    
    with open(args.input) as f:
        review_data = json.load(f)
    
    angles = [int(a.strip()) for a in args.angles.split(",")]
    print(f"Loading {len(review_data)} images from {args.input}")
    print(f"Generating rotations: {angles}")
    
    augmented, created = augment_review(review_data, args.image_dir, args.output_dir, angles=angles)
    
    with open(args.output, "w") as f:
        json.dump(augmented, f, indent=2)
    
    print(f"✓ Created {len(augmented)} review entries (original + augmented)")
    print(f"✓ Generated {len(created)} augmented images")
    print(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
