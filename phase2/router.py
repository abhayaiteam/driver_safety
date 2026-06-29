"""
Phase 2 — Public API Router

Two ways to send an image:
  1. Multipart upload  POST /api/v2/verify/upload  (browser / Postman / frontend)
  2. JSON base64       POST /api/v2/verify          (mobile / internal services)

Auth: X-API-Key header (or ?api_key= query param).
      Set API_KEY env var in production (default: "dev-key-change-me").

Bucket logic (mirrors cloud/server.py — one source of truth via config):
  alert  → vlm_conf >= VLM_ALERT_THRESHOLD  and verified=True  → alert fired
  review → vlm_conf >= VLM_REVIEW_THRESHOLD (borderline)       → logged only
  false  → vlm_conf <  VLM_REVIEW_THRESHOLD (clear FP)         → logged only
"""

import asyncio
import base64
import logging
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests as _requests
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, UploadFile

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vlm import verify as _vlm_verify    # phase2/vlm.py
from config import cfg                   # phase2/config.py

from models import (
    ErrorResponse,
    HealthResponse,
    StatsResponse,
    VerifyJsonRequest,
    VerifyResponse,
)

log = logging.getLogger("phase2.router")

# ── Constants ─────────────────────────────────────────────────────────────────

API_VERSION    = "2.0.0"
VALID_ACTIVITIES = {"phone", "cigarette", "food", "drink"}
_API_KEY       = os.getenv("API_KEY", "dev-key-change-me")
_executor      = ThreadPoolExecutor(max_workers=cfg.VLM_WORKERS)

router = APIRouter()

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key:   Optional[str] = None,
) -> None:
    if (x_api_key or api_key) != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

# ── Bucket classification (mirrors cloud/server.py) ───────────────────────────

def _classify(verified: bool, vlm_conf: float) -> str:
    if verified and vlm_conf >= cfg.VLM_ALERT_THRESHOLD:
        return "alert"
    if vlm_conf >= cfg.VLM_REVIEW_THRESHOLD:
        return "review"
    return "false"

# ── Core VLM call ─────────────────────────────────────────────────────────────

async def _run_verify(activity: str, image_b64: str) -> tuple[dict, float]:
    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _vlm_verify, activity, image_b64, cfg.VLM_MODEL)
    return result, (time.perf_counter() - t0) * 1000

# ── Persistence (background) ──────────────────────────────────────────────────

def _persist(event_id: str, driver_id: str, activity: str, bucket: str,
             vlm_conf: float, det_conf: float, reason: str, image_b64: str) -> None:
    dest = {
        "alert":  cfg.EVIDENCE_DIR,
        "review": os.path.join(cfg.REVIEW_DIR, activity),
        "false":  os.path.join(cfg.FALSE_DETECTIONS_DIR, activity),
    }[bucket]
    os.makedirs(dest, exist_ok=True)

    stem = f"{event_id}_dc{det_conf:.2f}"
    jpg  = os.path.join(dest, f"{stem}.jpg")
    try:
        with open(jpg, "wb") as f:
            f.write(base64.b64decode(image_b64))
    except (ValueError, OSError) as exc:
        log.error("Crop save failed path=%s err=%s", jpg, exc)
        jpg = ""

    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """INSERT OR IGNORE INTO events
           (id, driver_id, activity, bucket, vlm_conf, det_conf, reason, image_path, ts)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (event_id, driver_id, activity, bucket, vlm_conf, det_conf, reason, jpg, time.time()),
    )
    conn.commit()
    conn.close()

    if bucket == "alert":
        _send_alert(event_id, driver_id, activity, vlm_conf, reason)


def _lookup_driver(device_tablet_id: str) -> dict:
    """
    GET https://proximity-driver-api.prod-app.in/api/drivers/by-device/{deviceTabletId}
    Returns driver info dict, or empty dict on any error.
    """
    url = f"{cfg.MOBILE_API_BASE_URL}{cfg.DRIVER_LOOKUP_PATH}/{device_tablet_id}"
    headers = {}
    if cfg.MOBILE_API_KEY:
        headers["Authorization"] = f"Bearer {cfg.MOBILE_API_KEY}"
    try:
        resp = _requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        info = resp.json()
        log.debug("DRIVER_LOOKUP device=%s → %s", device_tablet_id, info)
        return info if isinstance(info, dict) else {}
    except Exception as exc:
        log.warning("DRIVER_LOOKUP failed device=%s err=%s", device_tablet_id, exc)
        return {}


def _send_alert(event_id: str, driver_id: str, activity: str, vlm_conf: float, reason: str) -> None:
    """
    1. Look up driver details from proximity-driver-api using the deviceTabletId.
    2. POST enriched alert to the mobile team's alert webhook.
    """
    url = cfg.ALERT_WEBHOOK_URL
    if not url:
        log.debug("ALERT_WEBHOOK_URL not configured — skipping push")
        return

    # Enrich with driver info from their backend
    driver_info = _lookup_driver(driver_id)

    headers = {"Content-Type": "application/json"}
    if cfg.ALERT_WEBHOOK_TOKEN:
        headers["Authorization"] = f"Bearer {cfg.ALERT_WEBHOOK_TOKEN}"

    payload = {
        "event_id":        event_id,
        "device_id":       driver_id,          # matches their deviceTabletId field
        "driver":          driver_info,         # full driver object from their /by-device API
        "activity":        activity,
        "confidence":      round(vlm_conf, 2),
        "reason":          reason,
        "source":          "driver_safety_ai",
        "timestamp":       time.time(),
    }

    try:
        resp = _requests.post(url, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        log.info("ALERT_PUSH event=%s device=%s activity=%s status=%s",
                 event_id, driver_id, activity, resp.status_code)
    except Exception as exc:
        log.error("ALERT_PUSH failed event=%s device=%s err=%s", event_id, driver_id, exc)


# ── Shared response builder ───────────────────────────────────────────────────

def _build_response(event_id: str, activity: str, result: dict, latency_ms: float) -> VerifyResponse:
    bucket = _classify(result["verified"], result["confidence"])
    return VerifyResponse(
        event_id=event_id,
        verified=(bucket == "alert"),
        confidence=result["confidence"],
        activity=activity,
        reason=result["reason"],
        bucket=bucket,
        latency_ms=round(latency_ms, 1),
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/verify",
    response_model=VerifyResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Verify detection — JSON / base64",
    tags=["Verification"],
)
async def verify_json(
    body: VerifyJsonRequest,
    background: BackgroundTasks,
    _auth: None = Depends(require_api_key),
) -> VerifyResponse:
    if body.activity not in VALID_ACTIVITIES:
        raise HTTPException(status_code=422, detail=f"activity must be one of {sorted(VALID_ACTIVITIES)}")

    event_id          = uuid.uuid4().hex[:10]
    result, latency   = await _run_verify(body.activity, body.image_b64)
    bucket            = _classify(result["verified"], result["confidence"])

    log.info(
        "VERIFY event=%s driver=%s activity=%s det_conf=%.2f "
        "vlm_conf=%.2f bucket=%s latency_ms=%.0f reason=%r",
        event_id, body.driver_id, body.activity, body.det_confidence or 0.0,
        result["confidence"], bucket, latency, result["reason"],
    )

    background.add_task(
        _persist, event_id, body.driver_id or "unknown", body.activity,
        bucket, result["confidence"], body.det_confidence or 0.0,
        result["reason"], body.image_b64,
    )

    return _build_response(event_id, body.activity, result, latency)


@router.post(
    "/verify/upload",
    response_model=VerifyResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Verify detection — multipart image upload",
    tags=["Verification"],
)
async def verify_upload(
    background:     BackgroundTasks,
    file:           UploadFile = File(..., description="JPEG/PNG of the driver crop"),
    activity:       str   = Form(...,    description="phone | cigarette | food | drink"),
    driver_id:      str   = Form(default="unknown"),
    det_confidence: float = Form(default=0.0, ge=0.0, le=1.0),
    _auth: None = Depends(require_api_key),
) -> VerifyResponse:
    if activity not in VALID_ACTIVITIES:
        raise HTTPException(status_code=422, detail=f"activity must be one of {sorted(VALID_ACTIVITIES)}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    image_b64       = base64.b64encode(raw).decode()
    event_id        = uuid.uuid4().hex[:10]
    result, latency = await _run_verify(activity, image_b64)
    bucket          = _classify(result["verified"], result["confidence"])

    log.info(
        "VERIFY event=%s driver=%s activity=%s det_conf=%.2f "
        "vlm_conf=%.2f bucket=%s latency_ms=%.0f reason=%r",
        event_id, driver_id, activity, det_confidence,
        result["confidence"], bucket, latency, result["reason"],
    )

    background.add_task(
        _persist, event_id, driver_id, activity,
        bucket, result["confidence"], det_confidence,
        result["reason"], image_b64,
    )

    return _build_response(event_id, activity, result, latency)


@router.get("/events", summary="Recent alerts", tags=["Events"])
async def get_events(
    bucket:    Optional[str] = None,
    activity:  Optional[str] = None,
    driver_id: Optional[str] = None,
    limit:     int = 50,
    _auth: None = Depends(require_api_key),
) -> list:
    """
    Query verified alert history.
    Filter by `bucket` (alert/review/false), `activity`, or `driver_id`.
    """
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.row_factory = sqlite3.Row

    wheres, params = ["1=1"], []
    if bucket:
        wheres.append("bucket=?");    params.append(bucket)
    if activity:
        wheres.append("activity=?");  params.append(activity)
    if driver_id:
        wheres.append("driver_id=?"); params.append(driver_id)
    params.append(limit)

    rows = conn.execute(
        f"SELECT id, driver_id, activity, bucket, vlm_conf, det_conf, reason, ts "
        f"FROM events WHERE {' AND '.join(wheres)} ORDER BY ts DESC LIMIT ?",
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/stats", response_model=StatsResponse, summary="Activity counts", tags=["Events"])
async def get_stats(_auth: None = Depends(require_api_key)) -> StatsResponse:
    conn = sqlite3.connect(cfg.DB_PATH)
    rows = conn.execute(
        "SELECT activity, COUNT(*) FROM events WHERE bucket='alert' GROUP BY activity"
    ).fetchall()
    conn.close()
    counts = {r[0]: r[1] for r in rows}
    return StatsResponse(
        phone=counts.get("phone", 0),
        cigarette=counts.get("cigarette", 0),
        food=counts.get("food", 0),
        drink=counts.get("drink", 0),
        total=sum(counts.values()),
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness", tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION, vlm_model=cfg.VLM_MODEL)
