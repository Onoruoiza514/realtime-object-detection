import cv2
import time
from core.config import settings
from core.utils import (
    draw_detection_box,
    draw_sentence_overlay,
    draw_fps,
    draw_title_bar
)


class VideoStream:
    """
    Captures webcam frames, runs the ASL prediction pipeline,
    and yields annotated MJPEG frames for the browser stream.
    """

    def __init__(self, predictor, text_builder):
        self.predictor = predictor
        self.text_builder = text_builder
        self.cap = None
        self.is_running = False
        self.current_fps = 0.0

    def start(self):
        """Open webcam capture."""
        self.cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        self.is_running = True
        print(f"[✓] Webcam started on index {settings.CAMERA_INDEX}")

    def stop(self):
        """Release webcam."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("[✓] Webcam stopped.")

    def generate_frames(self):
        """
        Generator that yields annotated MJPEG frames.
        Used by FastAPI streaming response.
        """
        if not self.cap or not self.cap.isOpened():
            self.start()

        prev_time = time.time()

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("[!] Failed to read frame from webcam.")
                break

            # Mirror flip
            frame = cv2.flip(frame, 1)

            # Run full prediction pipeline
            frame, detection = self.predictor.process_frame(frame)

            # Update text builder
            text_state = self.text_builder.update(detection["letter"])

            # Draw bounding box for detected hand/sign
            if detection["bbox"] and detection["letter"]:
                frame = draw_detection_box(
                    frame,
                    detection["bbox"],
                    detection["letter"],
                    detection["confidence"],
                    is_motion_sign=detection["is_motion_sign"]
                )

            # Draw UI overlays
            frame = draw_title_bar(frame, settings.APP_TITLE)
            frame = draw_sentence_overlay(
                frame,
                text_state["current_letter"],
                text_state["current_word"],
                text_state["sentence"]
            )

            # Draw buffer progress bar
            frame = self._draw_buffer_progress(frame, text_state["buffer_progress"])

            # Calculate and draw FPS
            curr_time = time.time()
            self.current_fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            frame = draw_fps(frame, self.current_fps)

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )

    def _draw_buffer_progress(self, frame, progress):
        """
        Draw a small progress bar showing letter confirmation progress.
        """
        h, w = frame.shape[:2]
        bar_x, bar_y = 20, h - 175
        bar_w, bar_h = 250, 10

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        fill_w = int(bar_w * progress)
        if fill_w > 0:
            color = (0, 200, 100) if progress < 1.0 else (0, 255, 150)