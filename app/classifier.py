import cv2
from ultralytics import YOLO
from core.config import settings


class ASLClassifier:
    """
    Loads a trained YOLOv8 classification model and predicts
    the ASL letter from a cropped hand image.
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.device = settings.DEVICE

    def load_model(self):
        """Load the classification model from MODEL_PATH."""
        try:
            self.model = YOLO(settings.MODEL_PATH)
            self.model.to(self.device)
            self.model_loaded = True
            print(f"[✓] Classifier loaded from {settings.MODEL_PATH} on {self.device}")
        except Exception as e:
            print(f"[!] Classifier model not found: {e}")
            self.model_loaded = False

    def predict(self, crop):
        """
        Classify a cropped hand image.

        Args:
            crop: BGR image (numpy array) of the cropped hand region

        Returns:
            label: predicted letter (str) or None
            confidence: float (0.0 - 1.0)
        """
        if not self.model_loaded or crop is None or crop.size == 0:
            return None, 0.0

        results = self.model(crop, device=self.device, verbose=False)

        if not results:
            return None, 0.0

        result = results[0]

        # Classification models return probs, not boxes
        if result.probs is None:
            return None, 0.0

        top1_idx = int(result.probs.top1)
        confidence = float(result.probs.top1conf)

        if confidence < settings.CONFIDENCE_THRESHOLD:
            return None, confidence

        # Map class index to label using model names or settings fallback
        names = result.names if hasattr(result, "names") else None
        if names and top1_idx in names:
            label = names[top1_idx]
        elif top1_idx < len(settings.ASL_CLASSES):
            label = settings.ASL_CLASSES[top1_idx]
        else:
            label = None

        # Exclude motion signs — handled separately by hand_tracker trajectory
        if label in settings.MOTION_SIGNS:
            return None, confidence

        return label, confidence