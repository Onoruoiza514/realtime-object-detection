import cv2
import numpy as np
from ultralytics import YOLO
from core.config import settings


class ASLClassifier:
    """
    Handles ASL letter classification.
    Uses cropped hand images only.
    """

    def __init__(self):

        self.device = settings.DEVICE

        self.model = YOLO(settings.MODEL_PATH)

        self.model.to(self.device)

        print(f"[✓] Classifier loaded on {self.device}")

    def preprocess(self, hand_crop):
        """
        Resize hand crop for classification.
        """

        if hand_crop is None or hand_crop.size == 0:
            return None

        resized = cv2.resize(hand_crop, (224, 224))

        return resized

    def predict(self, hand_crop):
        """
        Predict ASL letter from cropped hand image.
        Returns:
        - label
        - confidence
        """

        processed = self.preprocess(hand_crop)

        if processed is None:
            return None, 0.0

        results = self.model(
            processed,
            verbose=False,
            device=self.device
        )

        probs = results[0].probs

        if probs is None:
            return None, 0.0

        class_id = int(probs.top1)

        confidence = float(probs.top1conf)

        label = settings.ASL_CLASSES[class_id]

        if confidence < settings.CONFIDENCE_THRESHOLD:
            return None, confidence

        return label, confidence