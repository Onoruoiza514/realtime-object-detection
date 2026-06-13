import os
import sys
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import settings


def prepare():
    print(f"\n{'='*50}")
    print(f"  Preparing Classification Dataset from Detection Data")
    print(f"{'='*50}\n")

    src_dir = Path("data/asl_dataset")
    dst_dir = Path("data/asl_cls_dataset")

    if not src_dir.exists():
        print("[!] Detection dataset not found at data/asl_dataset/")
        return

    # Load class names
    with open(src_dir / "data.yaml") as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml["names"]

    splits = {"train": "train", "valid": "val", "test": "test"}

    for src_split, dst_split in splits.items():
        img_dir = src_dir / src_split / "images"
        label_dir = src_dir / src_split / "labels"

        if not img_dir.exists():
            continue

        print(f"\n[→] Processing {src_split} split...")

        for img_path in tqdm(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(label_path) as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Skip motion signs — they're handled by MediaPipe
                label = class_names[cls_id]
                if label in settings.MOTION_SIGNS:
                    continue

                # Convert normalized YOLO box to pixel coords
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # Add padding
                pad = settings.ROI_PADDING
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (settings.ROI_SIZE, settings.ROI_SIZE))

                # Save to destination
                out_dir = dst_dir / dst_split / label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{img_path.stem}_{i}.jpg"
                cv2.imwrite(str(out_path), crop)

    print(f"\n[✓] Classification dataset prepared at {dst_dir}")
    print("[✓] You can now run: python scripts/train_classifier.py\n")


if __name__ == "__main__":
    prepare()