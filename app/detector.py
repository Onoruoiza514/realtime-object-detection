import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from core.config import settings


class ASLDetector:
    """
    Handles all detection logic:
    - YOLOv8 for static signs (A–I, K–Y)
    - MediaPipe for motion signs (J, Z)
    """

    def __init__(self):
        self.device = settings.DEVICE
        self.confidence = settings.CONFIDENCE_THRESHOLD
        self.model = None
        self.model_loaded = False

        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        # Motion tracking buffer for J and Z
        self.landmark_buffer = deque(maxlen=settings.MOTION_FRAME_BUFFER)
        self.motion_sign_result = None

    def load_model(self):
        """Load YOLOv8 model from path."""
        try:
            self.model = YOLO(settings.MODEL_PATH)
            self.model.to(self.device)
            self.model_loaded = True
            print(f"[✓] YOLOv8 model loaded from {settings.MODEL_PATH} on {self.device}")
        except Exception as e:
            print(f"[!] Model not found — using YOLOv8n as placeholder: {e}")
            self.model = YOLO("yolov8n.pt")  # fallback for development
            self.model_loaded = True

    def detect_static_sign(self, frame):
        """
        Run YOLOv8 inference on frame.
        Returns detected letter, confidence, and bounding box.
        """
        if not self.model_loaded:
            return None, 0.0, None

        results = self.model(frame, conf=self.confidence,
                             device=self.device, verbose=False)

        best_label = None
        best_conf = 0.0
        best_bbox = None

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = settings.ASL_CLASSES[cls] if cls < len(settings.ASL_CLASSES) else None

                if label and label not in settings.MOTION_SIGNS and conf > best_conf:
                    best_conf = conf
                    best_label = label
                    best_bbox = box.xyxy[0].tolist()

        return best_label, best_conf, best_bbox

    def detect_motion_sign(self, frame):
        """
        Use MediaPipe to track index fingertip movement across frames.
        Classifies trajectory as J or Z based on movement pattern.
        Returns detected motion letter or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            self.landmark_buffer.clear()
            return None, None

        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand skeleton on frame
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
        )

        # Track index fingertip (landmark 8)
        h, w = frame.shape[:2]
        tip = hand_landmarks.landmark[settings.TRACKING_LANDMARK]
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        self.landmark_buffer.append((tip_x, tip_y))

        if len(self.landmark_buffer) < settings.MOTION_FRAME_BUFFER:
            return None, None

        sign = self._classify_motion()
        return sign, hand_landmarks

    def _classify_motion(self):
        """
        Classify buffered fingertip trajectory as J or Z.

        J pattern: curves downward then hooks left (U-shape going down)
        Z pattern: moves right, diagonals down-left, then right again (Z-shape)
        """
        points = list(self.landmark_buffer)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        total_movement = sum(
            abs(points[i][0] - points[i-1][0]) + abs(points[i][1] - points[i-1][1])
            for i in range(1, len(points))
        )

        # Not enough movement — not a motion sign
        if total_movement < 40:
            return None

        # Z detection: significant horizontal movement with direction changes
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        x_changes = sum(
            1 for i in range(1, len(xs) - 1)
            if (xs[i] - xs[i-1]) * (xs[i+1] - xs[i]) < 0
        )

        if x_range > y_range and x_changes >= 2:
            return "Z"

        # J detection: dominant downward curve
        downward = sum(1 for i in range(1, len(ys)) if ys[i] > ys[i-1])
        if downward > len(ys) * 0.6 and y_range > x_range:
            return "J"

        return None

    def process_frame(self, frame):
        """
        Full pipeline for one frame.
        Returns enriched frame + detection result dict.
        """
        detection = {
            "letter": None,
            "confidence": 0.0,
            "bbox": None,
            "is_motion_sign": False,
            "hand_detected": False
        }

        # Run MediaPipe on every frame for hand skeleton + J/Z
        motion_letter, hand_landmarks = self.detect_motion_sign(frame)
        if hand_landmarks:
            detection["hand_detected"] = True

        if motion_letter:
            detection["letter"] = motion_letter
            detection["is_motion_sign"] = True
            detection["confidence"] = 0.95
            return frame, detection

        # Run YOLOv8 for static signs
        label, conf, bbox = self.detect_static_sign(frame)
        if label:
            detection["letter"] = label
            detection["confidence"] = conf
            detection["bbox"] = bbox
            detection["is_motion_sign"] = False

        return frame, detection