import cv2
import numpy as np
from core.config import settings


def draw_detection_box(frame, bbox, label, confidence, is_motion_sign=False):
    """Draw bounding box and label on frame."""
    x1, y1, x2, y2 = map(int, bbox)

    # Choose color based on sign type
    color = settings.UI_MOTION_BOX_COLOR if is_motion_sign else settings.UI_BOX_COLOR

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, settings.UI_BOX_THICKNESS)

    # Build label text
    label_text = f"{label} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = settings.UI_FONT_SCALE * 0.7
    thickness = 2

    # Draw label background
    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)

    # Draw label text
    cv2.putText(frame, label_text, (x1 + 4, y1 - 5),
                font, font_scale, settings.UI_TEXT_COLOR, thickness)

    return frame


def draw_sentence_overlay(frame, current_letter, current_word, sentence):
    """Draw the live text overlay panel at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark panel
    panel_height = 160
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, settings.UI_OVERLAY_ALPHA,
                    frame, 1 - settings.UI_OVERLAY_ALPHA, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Current detected letter
    cv2.putText(frame, "Letter:", (20, h - 120), font, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, current_letter or "-", (120, h - 120),
                font, 1.2, settings.UI_MOTION_BOX_COLOR
                if current_letter in settings.MOTION_SIGNS
                else settings.UI_BOX_COLOR, 2)

    # Current word being built
    cv2.putText(frame, "Word:", (20, h - 75), font, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, current_word or "-", (120, h - 75),
                font, 0.9, (255, 255, 255), 2)

    # Full sentence
    cv2.putText(frame, "Sentence:", (20, h - 30), font, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, sentence or "-", (140, h - 30),
                font, 0.8, (200, 230, 255), 2)

    return frame


def draw_fps(frame, fps):
    """Draw FPS counter on top-right corner."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame


def draw_title_bar(frame, title):
    """Draw app title bar at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, title, (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def letterbox_frame(frame, target_size=(640, 640)):
    """Resize frame while maintaining aspect ratio with padding."""
    h, w = frame.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    pad_top = (target_size[0] - new_h) // 2
    pad_left = (target_size[1] - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, pad_top, pad_left


def get_color_for_letter(letter):
    """Return a unique consistent color per letter for visual variety."""
    np.random.seed(ord(letter))
    color = tuple(int(c) for c in np.random.randint(100, 255, 3))
    return color