import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/asl_yolov8.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"

    # Camera
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", 0))
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720

    # Text builder
    LETTER_BUFFER_FRAMES: int = 15        # frames a letter must be stable before confirming
    WORD_PAUSE_SECONDS: float = 2.0       # seconds of no detection = space between words
    MAX_SENTENCE_LENGTH: int = 100

    # ASL Classes — full A-Z
    ASL_CLASSES: list = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "J",  # motion — MediaPipe tracked
        "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X", "Y",
        "Z"   # motion — MediaPipe tracked
    ]

    # Motion signs handled by MediaPipe instead of YOLOv8
    MOTION_SIGNS: list = ["J", "Z"]

    # Number of frames to track for motion detection
    MOTION_FRAME_BUFFER: int = 20

    # Keypoint indices used for J and Z motion tracking (MediaPipe)
    # Index 8 = index fingertip, Index 4 = thumb tip
    TRACKING_LANDMARK: int = 8

    # UI
    APP_TITLE: str = "ASL Sign Language Recognition"
    APP_VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # Interface
    UI_FONT_SCALE: float = 1.0
    UI_BOX_THICKNESS: int = 2
    UI_TEXT_COLOR: tuple = (255, 255, 255)      # white
    UI_BOX_COLOR: tuple = (0, 200, 100)         # green for static signs
    UI_MOTION_BOX_COLOR: tuple = (0, 165, 255)  # orange for motion signs (J, Z)
    UI_OVERLAY_ALPHA: float = 0.4               # transparency of text overlay panel

settings = Settings()