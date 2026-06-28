from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Detection model ───────────────────────────────────────────────────────
    # .pt  → uses Ultralytics YOLO (for development / verification)
    # .tflite → uses TFLite interpreter (for on-device mobile deployment)
    MODEL_PATH: str = "yolo11n.pt"   # swap to custom_yolo_updated.tflite for mobile
    INPUT_SIZE: int = 640
    DETECTION_CONF: float = 0.35   # Low threshold; VLM filters FPs
    NMS_THRESHOLD: float = 0.45

    # Class names in your model.
    # yolo11n.pt (COCO): person=0, cell phone=67  → mapped automatically
    # Custom .pt/.tflite: list your actual training labels in order
    CLASS_NAMES: List[str] = field(default_factory=lambda: [
        "person", "phone"])
    TARGET_CLASSES: List[str] = field(default_factory=lambda: [
        "phone"])

    # ── Spatial / ROI ─────────────────────────────────────────────────────────
    ROI_EXPAND: float = 0.20       # Expand person bbox by 20% before crop
    PROXIMITY_RATIO: float = 0.80  # Object within 80% of person diagonal = associated

    # ── Cloud API ────────────────────────────────────────────────────────────
    CLOUD_HOST: str = "0.0.0.0"
    CLOUD_PORT: int = 8000
    CLOUD_API_URL: str = "http://localhost:8000"

    # ── VLM ──────────────────────────────────────────────────────────────────
    VLM_MODEL: str = "llava:7b"
    VLM_WORKERS: int = 3           # Thread pool size for concurrent VLM calls
    JPEG_QUALITY: int = 75         # Compress crop before sending (bandwidth)

    VLM_ALERT_THRESHOLD: float = 0.75   # Must be this confident to fire an alert
    VLM_REVIEW_THRESHOLD: float = 0.40  # Below this → clear FP, above → borderline

    # ── Alert dedup ───────────────────────────────────────────────────────────
    ALERT_COOLDOWN_SEC: float = 5.0   # Mobile: don't re-send same activity < 5s
    DEDUP_COOLDOWN_SEC: float = 5.0   # Cloud: suppress duplicate server-side

    # ── Mobile sim ───────────────────────────────────────────────────────────
    FRAME_SKIP: int = 3            # Process every 3rd frame (5-10 FPS effective)
    UPLOAD_QUEUE_SIZE: int = 5
    DRIVER_ID: str = "driver_001"

    # ── Storage ───────────────────────────────────────────────────────────────
    DB_PATH: str = "events.db"
    LOG_FILE: str = "logs/driver_safety.log"   # rotating log file
    EVIDENCE_DIR: str = "evidence"             # confirmed alerts
    REVIEW_DIR: str = "review"                 # borderline — human review needed
    FALSE_DETECTIONS_DIR: str = "false_detections"  # clear FPs — for model retraining


cfg = Config()
