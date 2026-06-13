import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from core.config import settings


def train():
    print(f"\n{'='*50}")
    print(f"  ASL Classifier Training (YOLOv8-cls)")
    print(f"{'='*50}\n")

    # -------------------------------------------------------
    # DATASET SETUP — classification format (different from detection!)
    # -------------------------------------------------------
    # YOLOv8 classification expects a folder structure like:
    #
    # data/asl_cls_dataset/
    #   ├── train/
    #   │   ├── A/  (folder of cropped images of letter A)
    #   │   ├── B/
    #   │   ├── ...
    #   │   └── Y/
    #   └── val/
    #       ├── A/
    #       ├── B/
    #       ├── ...
    #       └── Y/
    #
    # (J and Z excluded — handled via MediaPipe motion tracking)
    #
    # If you only have the detection dataset (with bounding boxes),
    # run scripts/prepare_classification_data.py first to crop
    # hand regions using the bounding box labels and sort them
    # into per-letter folders.
    # -------------------------------------------------------

    dataset_dir = Path("data/asl_cls_dataset")

    if not dataset_dir.exists():
        print("[!] Classification dataset not found at data/asl_cls_dataset/")
        print("\n  Run this first to convert your detection dataset:")
        print("  python scripts/prepare_classification_data.py\n")
        return

    print(f"[✓] Dataset found at {dataset_dir}")
    print(f"[✓] Device: {settings.DEVICE.upper()}")
    print(f"[✓] Starting training...\n")

    # Load base YOLOv8 classification model (nano)
    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=str(dataset_dir),
        epochs=20,
        imgsz=settings.ROI_SIZE,
        batch=8,
        device=settings.DEVICE,
        project="models",
        name="asl_classifier",
        exist_ok=True,
        patience=5,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n{'='*50}")
    print(f"[✓] Training complete!")
    print(f"[✓] Best weights saved to: models/asl_classifier/weights/best.pt")
    print(f"{'='*50}\n")

    # Validate
    print("[→] Running validation...\n")
    metrics = model.val()
    print(f"\n  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}\n")

    # Copy best weights to models/ root
    best_weights = Path("models/asl_classifier/weights/best.pt")
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, Path(settings.MODEL_PATH))
        print(f"[✓] Best weights copied to {settings.MODEL_PATH}")

    print("\n[✓] All done! You can now run the app with your trained classifier.\n")


if __name__ == "__main__":
    train()