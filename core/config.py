import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/asl_classifier.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
    DEVICE: str = os.getenv("DEVICE", "cpu")

    # Camera
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", 0))
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720

    # Hand ROI
    ROI_PADDING: int = 30          # pixels to pad around hand crop
    ROI_SIZE: int = 224            # classifier input size

    # Text builder
    LETTER_BUFFER_FRAMES: int = 15
    WORD_PAUSE_SECONDS: float = 2.0
    MAX_SENTENCE_LENGTH: int = 100

    # ASL Classes — full A-Z
    ASL_CLASSES: list = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X", "Y",
        "Z"
    ]

    # Motion signs — handled by MediaPipe trajectory tracking
    MOTION_SIGNS: list = ["J", "Z"]

    # Motion detection
    MOTION_FRAME_BUFFER: int = 20
    TRACKING_LANDMARK: int = 8     # index fingertip

    # UI
    APP_TITLE: str = "ASL Sign Language Recognition"
    APP_VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    UI_FONT_SCALE: float = 1.0
    UI_BOX_THICKNESS: int = 2
    UI_TEXT_COLOR: tuple = (255, 255, 255)
    UI_BOX_COLOR: tuple = (0, 200, 100)
    UI_MOTION_BOX_COLOR: tuple = (0, 165, 255)
    UI_OVERLAY_ALPHA: float = 0.4


settings = Settings()