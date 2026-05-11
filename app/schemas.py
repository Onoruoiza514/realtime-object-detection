from pydantic import BaseModel
from typing import Optional


class DetectionResult(BaseModel):
    """Single frame detection result."""
    letter: Optional[str] = None
    confidence: float = 0.0
    is_motion_sign: bool = False
    hand_detected: bool = False
    bbox: Optional[list] = None


class TextState(BaseModel):
    """Current state of the text builder."""
    current_letter: Optional[str] = None
    current_word: str = ""
    sentence: str = ""
    buffer_progress: float = 0.0


class FrameResponse(BaseModel):
    """Full response for a single processed frame."""
    detection: DetectionResult
    text_state: TextState
    fps: float = 0.0


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str


class ClearResponse(BaseModel):
    """Response after clearing the sentence."""
    success: bool
    message: str


class BackspaceResponse(BaseModel):
    """Response after backspace action."""
    success: bool
    text_state: TextState