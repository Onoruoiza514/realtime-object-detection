import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from app.detector import ASLDetector
from app.text_builder import TextBuilder
from app.stream import VideoStream
from app.schemas import (
    FrameResponse, DetectionResult, TextState,
    HealthResponse, ClearResponse, BackspaceResponse
)
from core.config import settings


# --- App lifespan: load model on startup ---
detector = ASLDetector()
text_builder = TextBuilder()
video_stream = VideoStream(detector, text_builder)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[✓] Starting {settings.APP_TITLE} v{settings.APP_VERSION}")
    detector.load_model()
    video_stream.start()
    yield
    video_stream.stop()
    print("[✓] App shutdown complete.")


# --- FastAPI app ---
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Real-time ASL hand sign recognition — full A-Z including motion signs J and Z.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the live demo UI."""
    return HTMLResponse(content=get_ui_html())


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms model is loaded and ready."""
    return HealthResponse(
        status="ok",
        model_loaded=detector.model_loaded,
        device=settings.DEVICE,
        version=settings.APP_VERSION
    )


@app.get("/stream")
async def stream():
    """Live MJPEG webcam stream with detection overlays."""
    return StreamingResponse(
        video_stream.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/detect", response_model=FrameResponse)
async def detect(file: UploadFile = File(...)):
    """
    Detect ASL sign from an uploaded image.
    Returns detected letter, confidence, and current text state.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return FrameResponse(
            detection=DetectionResult(),
            text_state=TextState(**text_builder.get_state()),
            fps=0.0
        )

    frame, detection = detector.process_frame(frame)
    text_state = text_builder.update(detection["letter"])

    return FrameResponse(
        detection=DetectionResult(**detection),
        text_state=TextState(**text_state),
        fps=video_stream.current_fps
    )


@app.post("/clear", response_model=ClearResponse)
async def clear():
    """Clear the current sentence and reset text builder."""
    text_builder.clear()
    return ClearResponse(success=True, message="Sentence cleared.")


@app.post("/backspace", response_model=BackspaceResponse)
async def backspace():
    """Remove the last confirmed letter."""
    text_builder.backspace()
    return BackspaceResponse(
        success=True,
        text_state=TextState(**text_builder.get_state())
    )


@app.get("/state", response_model=TextState)
async def get_state():
    """Get current text builder state."""
    return TextState(**text_builder.get_state())


# --- Embedded UI ---

def get_ui_html():
    """Beautiful embedded HTML interface served at /"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ASL Sign Language Recognition</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f0f0f;
      color: #fff;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    header {
      width: 100%;
      padding: 18px 40px;
      background: #1a1a1a;
      border-bottom: 1px solid #2a2a2a;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    header h1 { font-size: 20px; font-weight: 600; }
    header span { font-size: 13px; color: #888; }
    .badge {
      background: #00c864;
      color: #000;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 99px;
    }
    .main {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 24px;
      padding: 28px 40px;
      width: 100%;
      max-width: 1300px;
    }
    .video-card {
      background: #1a1a1a;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #2a2a2a;
    }
    .video-card img {
      width: 100%;
      display: block;
    }
    .sidebar { display: flex; flex-direction: column; gap: 16px; }
    .card {
      background: #1a1a1a;
      border-radius: 16px;
      padding: 20px;
      border: 1px solid #2a2a2a;
    }
    .card h3 {
      font-size: 12px;
      font-weight: 500;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 14px;
    }
    .letter-display {
      font-size: 80px;
      font-weight: 700;
      text-align: center;
      color: #00c864;
      line-height: 1;
      padding: 10px 0;
      min-height: 100px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .letter-display.motion { color: #ffa500; }
    .word-display {
      font-size: 28px;
      font-weight: 600;
      color: #fff;
      min-height: 44px;
      word-break: break-all;
    }
    .sentence-display {
      font-size: 16px;
      color: #ccc;
      line-height: 1.7;
      min-height: 60px;
      word-break: break-word;
    }
    .progress-bar {
      width: 100%;
      height: 6px;
      background: #2a2a2a;
      border-radius: 99px;
      overflow: hidden;
      margin-top: 10px;
    }
    .progress-fill {
      height: 100%;
      background: #00c864;
      border-radius: 99px;
      transition: width 0.1s;
      width: 0%;
    }
    .btn-row { display: flex; gap: 10px; margin-top: 4px; }
    .btn {
      flex: 1;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid #333;
      background: #222;
      color: #fff;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn:hover { background: #2a2a2a; }
    .btn.danger { border-color: #ff4444; color: #ff4444; }
    .btn.danger:hover { background: #2a1010; }
    .conf-bar {
      height: 4px;
      background: #2a2a2a;
      border-radius: 99px;
      margin-top: 8px;
      overflow: hidden;
    }
    .conf-fill {
      height: 100%;
      background: #378ADD;
      border-radius: 99px;
      width: 0%;
      transition: width 0.2s;
    }
    .conf-label {
      font-size: 12px;
      color: #888;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <header>
    <h1>✋ ASL Sign Language Recognition</h1>
    <span>Full A–Z · J & Z motion tracking</span>
    <span class="badge" id="status">LIVE</span>
  </header>

  <div class="main">
    <div class="video-card">
      <img src="/stream" alt="Live stream"/>
    </div>

    <div class="sidebar">
      <div class="card">
        <h3>Detected Letter</h3>
        <div class="letter-display" id="letterDisplay">—</div>
        <div class="conf-bar"><div class="conf-fill" id="confFill"></div></div>
        <div class="conf-label" id="confLabel">Confidence: —</div>
        <div class="progress-bar">
          <div class="progress-fill" id="progressFill"></div>
        </div>
      </div>

      <div class="card">
        <h3>Current Word</h3>
        <div class="word-display" id="wordDisplay">—</div>
      </div>

      <div class="card">
        <h3>Sentence</h3>
        <div class="sentence-display" id="sentenceDisplay">Start signing to build a sentence...</div>
      </div>

      <div class="card">
        <h3>Controls</h3>
        <div class="btn-row">
          <button class="btn" onclick="doBackspace()">⌫ Backspace</button>
          <button class="btn danger" onclick="doClear()">✕ Clear</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function fetchState() {
      try {
        const res = await fetch('/state');
        const data = await res.json();

        const letter = data.current_letter;
        const el = document.getElementById('letterDisplay');
        el.textContent = letter || '—';
        el.className = 'letter-display' + (letter === 'J' || letter === 'Z' ? ' motion' : '');

        document.getElementById('wordDisplay').textContent = data.current_word || '—';
        document.getElementById('sentenceDisplay').textContent =
          data.sentence || 'Start signing to build a sentence...';
        document.getElementById('progressFill').style.width =
          (data.buffer_progress * 100) + '%';
      } catch(e) {}
    }

    async function doBackspace() {
      await fetch('/backspace', { method: 'POST' });
    }

    async function doClear() {
      await fetch('/clear', { method: 'POST' });
    }

    setInterval(fetchState, 300);
  </script>
</body>
</html>
"""