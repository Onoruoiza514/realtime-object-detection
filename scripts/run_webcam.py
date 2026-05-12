import cv2
import time
import sys
import os

# This to make sure app and core modules are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import ASLDetector
from app.text_builder import TextBuilder
from app.stream import VideoStream
from core.config import settings
from core.utils import (
    draw_detection_box,
    draw_sentence_overlay,
    draw_fps,
    draw_title_bar
)


def run():
    print(f"\n{'='*50}")
    print(f"  {settings.APP_TITLE} v{settings.APP_VERSION}")
    print(f"  Device: {settings.DEVICE.upper()}")
    print(f"{'='*50}")
    print("\n  Controls:")
    print("  [B] — Backspace (remove last letter)")
    print("  [C] — Clear sentence")
    print("  [Q] — Quit\n")

    # Initialize
    detector = ASLDetector()
    detector.load_model()
    text_builder = TextBuilder()

    # Open webcam
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[!] Could not open webcam. Check CAMERA_INDEX in .env")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read frame.")
            break

        # Mirror flip
        frame = cv2.flip(frame, 1)

        # Run detection
        frame, detection = detector.process_frame(frame)

        # Update text builder
        text_state = text_builder.update(detection["letter"])

        # Draw bounding box for static signs
        if detection["bbox"] and detection["letter"]:
            frame = draw_detection_box(
                frame,
                detection["bbox"],
                detection["letter"],
                detection["confidence"],
                is_motion_sign=detection["is_motion_sign"]
            )

        # Draw UI
        frame = draw_title_bar(frame, settings.APP_TITLE)
        frame = draw_sentence_overlay(
            frame,
            text_state["current_letter"],
            text_state["current_word"],
            text_state["sentence"]
        )

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        frame = draw_fps(frame, fps)

        # Show frame
        cv2.imshow(settings.APP_TITLE, frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[✓] Quitting...")
            break
        elif key == ord('b'):
            text_builder.backspace()
            print("[✓] Backspace")
        elif key == ord('c'):
            text_builder.clear()
            print("[✓] Cleared")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()