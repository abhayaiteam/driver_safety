"""
Driver Safety Cloud Server — FastAPI (Phase 1 internal endpoint)

Every detection from the mobile app lands here first.
VLM verdict → three buckets:
  ✅ conf >= ALERT_THRESHOLD  → evidence/          → downstream alert fired
  ⚠️ conf >= REVIEW_THRESHOLD → review/            → logged, no alert
  ❌ conf <  REVIEW_THRESHOLD → false_detections/  → logged, no alert, use for retraining
"""

import asyncio
import base64
import json
import logging
import logging.handlers
import os
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from cloud.vlm import verify as vlm_verify

# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(cfg.LOG_FILE), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Rotating file — 10 MB per file, keep 7 days of history
    fh = logging.handlers.RotatingFileHandler(
        cfg.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("cloud.server")

log = _setup_logging()

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Driver Safety Cloud (internal)", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_executor = ThreadPoolExecutor(max_workers=cfg.VLM_WORKERS)
_dedup: dict[str, float] = {}   # driver_id:activity → last verified timestamp

# ── DB init ───────────────────────────────────────────────────────────────────

def _init_db() -> None:
    for d in (cfg.EVIDENCE_DIR, cfg.REVIEW_DIR, cfg.FALSE_DETECTIONS_DIR):
        os.makedirs(d, exist_ok=True)

    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          TEXT PRIMARY KEY,
            driver_id   TEXT NOT NULL,
            activity    TEXT NOT NULL,
            bucket      TEXT NOT NULL,   -- 'alert' | 'review' | 'false'
            vlm_conf    REAL NOT NULL,
            det_conf    REAL,
            reason      TEXT,
            image_path  TEXT,
            ts          REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

_init_db()

# ── Bucket classification ─────────────────────────────────────────────────────

def _classify(vlm_conf: float, verified: bool) -> str:
    """
    Map VLM output to one of three buckets.

    alert  → send downstream alert, save to evidence/
    review → borderline, save to review/, do NOT alert
    false  → clear FP, save to false_detections/, do NOT alert
    """
    if verified and vlm_conf >= cfg.VLM_ALERT_THRESHOLD:
        return "alert"
    if vlm_conf >= cfg.VLM_REVIEW_THRESHOLD:
        return "review"
    return "false"

# ── Persistence (background tasks) ───────────────────────────────────────────

def _save_crop(dest_dir: str, stem: str, image_b64: str) -> str:
    """Write JPEG to dest_dir/{stem}.jpg. Returns the path or '' on failure."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{stem}.jpg")
    try:
        with open(path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return path
    except (ValueError, OSError) as exc:
        log.error("Crop save failed path=%s err=%s", path, exc)
        return ""


def _persist(event_id: str, driver_id: str, activity: str, bucket: str,
             vlm_conf: float, det_conf: float, reason: str,
             image_b64: str) -> None:
    """Save image crop + DB row in the background."""
    stem = f"{event_id}_dc{det_conf:.2f}"

    dest = {
        "alert":  cfg.EVIDENCE_DIR,
        "review": os.path.join(cfg.REVIEW_DIR, activity),
        "false":  os.path.join(cfg.FALSE_DETECTIONS_DIR, activity),
    }[bucket]

    image_path = _save_crop(dest, stem, image_b64)

    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?)",
        (event_id, driver_id, activity, bucket,
         vlm_conf, det_conf, reason, image_path, time.time()),
    )
    conn.commit()
    conn.close()


def _send_alert(event_id: str, driver_id: str, activity: str,
                vlm_conf: float, reason: str) -> None:
    """
    Downstream alert stub — extend to webhook / MQTT / push notification.
    Called only for bucket='alert'.
    """
    log.info(
        "ALERT fired event=%s driver=%s activity=%s vlm_conf=%.2f reason=%r",
        event_id, driver_id, activity, vlm_conf, reason,
    )
    # e.g. requests.post(cfg.BACKEND_WEBHOOK_URL, json={...}, timeout=5)

# ── Pydantic models ───────────────────────────────────────────────────────────

class DetectionPayload(BaseModel):
    driver_id: str
    activity: str
    detection_confidence: float
    image_b64: str
    timestamp: Optional[float] = None


class VerifyResponse(BaseModel):
    event_id: str
    bucket: str          # alert | review | false
    verified: bool       # True only when bucket == 'alert'
    vlm_confidence: float
    reason: str
    latency_ms: float

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/v1/verify", response_model=VerifyResponse)
async def verify_detection(payload: DetectionPayload,
                           background: BackgroundTasks) -> VerifyResponse:
    dedup_key = f"{payload.driver_id}:{payload.activity}"
    now = time.time()

    if dedup_key in _dedup and (now - _dedup[dedup_key]) < cfg.DEDUP_COOLDOWN_SEC:
        log.debug("DEDUP suppressed driver=%s activity=%s", payload.driver_id, payload.activity)
        return VerifyResponse(
            event_id="dedup", bucket="false", verified=False,
            vlm_confidence=0.0, reason="duplicate suppressed", latency_ms=0.0,
        )

    event_id = uuid.uuid4().hex[:10]
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, vlm_verify,
        payload.activity, payload.image_b64, cfg.VLM_MODEL,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    bucket = _classify(result["confidence"], result["verified"])
    is_alert = bucket == "alert"

    # Structured log — one line per decision, machine-parseable
    log.info(
        "VERIFY event=%s driver=%s activity=%s det_conf=%.2f "
        "vlm_conf=%.2f bucket=%s latency_ms=%.0f reason=%r",
        event_id, payload.driver_id, payload.activity,
        payload.detection_confidence, result["confidence"],
        bucket, latency_ms, result["reason"],
    )

    if is_alert:
        _dedup[dedup_key] = now

    background.add_task(
        _persist, event_id, payload.driver_id, payload.activity,
        bucket, result["confidence"], payload.detection_confidence,
        result["reason"], payload.image_b64,
    )

    if is_alert:
        background.add_task(
            _send_alert, event_id, payload.driver_id,
            payload.activity, result["confidence"], result["reason"],
        )

    return VerifyResponse(
        event_id=event_id,
        bucket=bucket,
        verified=is_alert,
        vlm_confidence=result["confidence"],
        reason=result["reason"],
        latency_ms=round(latency_ms, 1),
    )


@app.get("/api/v1/events")
async def get_events(bucket: Optional[str] = None, limit: int = 100) -> list:
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.row_factory = sqlite3.Row
    where = "WHERE bucket=?" if bucket else ""
    params = [bucket, limit] if bucket else [limit]
    rows = conn.execute(
        f"SELECT * FROM events {where} ORDER BY ts DESC LIMIT ?", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/v1/stats")
async def get_stats() -> dict:
    conn = sqlite3.connect(cfg.DB_PATH)
    rows = conn.execute(
        "SELECT activity, bucket, COUNT(*) as n FROM events GROUP BY activity, bucket"
    ).fetchall()
    conn.close()
    stats: dict = {}
    for activity, bucket, n in rows:
        stats.setdefault(activity, {"alert": 0, "review": 0, "false": 0})
        stats[activity][bucket] = n
    return stats


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ts": time.time()}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cloud.server:app", host=cfg.CLOUD_HOST, port=cfg.CLOUD_PORT,
                reload=False, workers=1)
