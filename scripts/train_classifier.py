import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from core.config import settings


def train():
    print(f"\n{'='*50}")
    print(f"  ASL YOLOv8 Model Training")
    print(f"{'='*50}\n")

    # -------------------------------------------------------
    # DATASET SETUP
    # -------------------------------------------------------
    # We use the free Roboflow ASL Dataset (A-Z, 26 classes)
    # Steps to get it:
    # 1. Go to: https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters
    # 2. Click Export → YOLOv8 format → download zip
    # 3. Extract into: data/asl_dataset/
    # 4. Your data/ folder should look like:
    #    data/asl_dataset/
    #      ├── train/images/  & train/labels/
    #      ├── valid/images/  & valid/labels/
    #      ├── test/images/   & test/labels/
    #      └── data.yaml
    # -------------------------------------------------------

    dataset_yaml = Path("data/asl_dataset/data.yaml")

    if not dataset_yaml.exists():
        print("[!] Dataset not found at data/asl_dataset/data.yaml")
        print("\n  Please follow these steps:")
        print("  1. Visit: https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters")
        print("  2. Export in YOLOv8 format")
        print("  3. Extract into data/asl_dataset/")
        print("  4. Re-run this script\n")
        return

    print(f"[✓] Dataset found at {dataset_yaml}")
    print(f"[✓] Device: {settings.DEVICE.upper()}")
    print(f"[✓] Starting training...\n")

    # Load base YOLOv8 model
    model = YOLO("yolov8n.pt")  # nano — fast training, good accuracy

    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=20,
        imgsz=320,
        batch=4,
        device=settings.DEVICE,
        project="models",
        name="asl_yolov8",
        exist_ok=True,
        patience=10,           # early stopping
        save=True,
        save_period=10,        # save checkpoint every 10 epochs
        val=True,
        plots=True,            # save training curves
        verbose=True,

        # Augmentation — improves generalization
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.0,            # don't flip — hand signs are not symmetric
        flipud=0.0,
        mosaic=0.5,
        scale=0.2,
        translate=0.05,
        workers = 8,
        cache=True
    )

    print(f"\n{'='*50}")
    print(f"[✓] Training complete!")
    print(f"[✓] Best weights saved to: models/asl_yolov8/weights/best.pt")
    print(f"{'='*50}\n")

    # Validate on test set
    print("[→] Running validation on test set...\n")
    metrics = model.val()
    print(f"\n  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}\n")

    # Copy best weights to models/ root for easy access
    best_weights = Path("models/asl_yolov8/weights/best.pt")
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, Path(settings.MODEL_PATH))
        print(f"[✓] Best weights copied to {settings.MODEL_PATH}")

    print("\n[✓] All done! You can now run the app with your trained model.\n")


if __name__ == "__main__":
    train()