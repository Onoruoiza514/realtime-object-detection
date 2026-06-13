import cv2
import numpy as np
import mediapipe as mp
from core.config import settings


class HandTracker:
    """
    Uses MediaPipe Hands to:
    - Detect and track a hand in the frame
    - Draw the hand skeleton overlay
    - Extract a padded bounding box and cropped hand image (ROI)
    - Provide fingertip landmark history for J/Z motion detection
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

    def process(self, frame):
        """
        Run MediaPipe on a frame.

        Returns:
            annotated_frame: frame with hand skeleton drawn
            landmarks: MediaPipe hand_landmarks object or None
            crop: cropped hand image (BGR) or None
            bbox: (x1, y1, x2, y2) in pixel coords or None
            fingertip: (x, y) pixel coords of index fingertip or None
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmarks = None
        crop = None
        bbox = None
        fingertip = None

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]

            # Draw skeleton overlay
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

            # Compute bounding box from all landmarks
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]

            pad = settings.ROI_PADDING
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)
            bbox = (x1, y1, x2, y2)

            # Extract crop
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    crop = cv2.resize(crop, (settings.ROI_SIZE, settings.ROI_SIZE))

            # Fingertip position (for J/Z motion tracking)
            tip = landmarks.landmark[settings.TRACKING_LANDMARK]
            fingertip = (int(tip.x * w), int(tip.y * h))

        return frame, landmarks, crop, bbox, fingertip