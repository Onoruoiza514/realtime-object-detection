import cv2
from collections import deque
from app.hand_tracker import HandTracker
from app.classifier import ASLClassifier
from core.config import settings


class ASLPredictor:
    """
    Full per-frame prediction pipeline.

    - Uses HandTracker for hand detection, ROI extraction, and fingertip tracking
    - Uses ASLClassifier for static sign classification (A-I, K-Y)
    - Performs trajectory-based motion detection for J and Z
    - Returns a unified detection result dict (drop-in replacement for old detector)
    """

    def __init__(self):
        self.hand_tracker = HandTracker()
        self.classifier = ASLClassifier()
        self.model_loaded = False

        # Fingertip trajectory buffer for J/Z detection
        self.landmark_buffer = deque(maxlen=settings.MOTION_FRAME_BUFFER)

    def load_model(self):
        """Load the classification model."""
        self.classifier.load_model()
        self.model_loaded = self.classifier.model_loaded

    def process_frame(self, frame):
        """
        Run the full pipeline on one frame.

        Returns:
            frame: annotated frame (hand skeleton drawn)
            detection: dict with keys:
                - letter
                - confidence
                - bbox
                - is_motion_sign
                - hand_detected
        """
        detection = {
            "letter": None,
            "confidence": 0.0,
            "bbox": None,
            "is_motion_sign": False,
            "hand_detected": False
        }

        # Run hand tracking
        frame, landmarks, crop, bbox, fingertip = self.hand_tracker.process(frame)

        if landmarks is None:
            self.landmark_buffer.clear()
            return frame, detection

        detection["hand_detected"] = True
        detection["bbox"] = bbox

        # Track fingertip for motion detection
        if fingertip:
            self.landmark_buffer.append(fingertip)

        # Check for motion sign (J/Z) first
        motion_letter = self._detect_motion_sign()
        if motion_letter:
            detection["letter"] = motion_letter
            detection["is_motion_sign"] = True
            detection["confidence"] = 0.95
            return frame, detection

        # Otherwise classify static sign from crop
        label, confidence = self.classifier.predict(crop)
        if label:
            detection["letter"] = label
            detection["confidence"] = confidence
            detection["is_motion_sign"] = False

        return frame, detection

    def _detect_motion_sign(self):
        """
        Classify buffered fingertip trajectory as J or Z.

        J pattern: dominant downward curve
        Z pattern: horizontal zigzag with direction changes
        """
        if len(self.landmark_buffer) < settings.MOTION_FRAME_BUFFER:
            return None

        points = list(self.landmark_buffer)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        total_movement = sum(
            abs(points[i][0] - points[i-1][0]) + abs(points[i][1] - points[i-1][1])
            for i in range(1, len(points))
        )

        if total_movement < 40:
            return None

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        x_changes = sum(
            1 for i in range(1, len(xs) - 1)
            if (xs[i] - xs[i-1]) * (xs[i+1] - xs[i]) < 0
        )

        if x_range > y_range and x_changes >= 2:
            return "Z"

        downward = sum(1 for i in range(1, len(ys)) if ys[i] > ys[i-1])
        if downward > len(ys) * 0.6 and y_range > x_range:
            return "J"

        return None