import asyncio
import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vlm import verify as _vlm_verify
from config import cfg

from models import ErrorResponse, HealthResponse, VerifyJsonRequest, VerifyResponse

log = logging.getLogger("phase2.router")

API_VERSION      = "2.0.0"

# Only object-detection (YOLO) activities plus eye-state drowsiness need a VLM
# double-check here — unauthorized-driver and hardware events are verified elsewhere
# and are not this service's concern, so they're passed straight through to the
# backend untouched.
# The app sends free-text labels (e.g. "phone being used", not just "phone"), so match by
# keyword rather than exact string.
_OBJECT_DETECTION_KEYWORDS: dict[str, str] = {
    "phone":      "phone",
    "cigarette":  "cigarette",
    "smoking":    "cigarette",
    "smoke":      "cigarette",
    "eating":     "food",
    "food":       "food",
    "drinking":   "drink",
    "drink":      "drink",
    "drowsy":     "drowsy",
    "drowsiness": "drowsy",
    "sleepy":     "drowsy",
}
_API_KEY         = os.getenv("API_KEY", "dev-key-change-me")
_executor        = ThreadPoolExecutor(max_workers=cfg.VLM_WORKERS)

router = APIRouter()


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key:   Optional[str] = None,
) -> None:
    if (x_api_key or api_key) != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


async def _run_verify(activity: str, image_b64: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _vlm_verify, activity, image_b64, cfg.VLM_MODEL)


def _resolve_object_detection_activity(activity: str) -> Optional[str]:
    """Map a free-text activity label to a VLM prompt key (phone/cigarette/food/drink/drowsy)
    if it needs a VLM double-check; None for anything else (distraction,
    unauthorized driver, hardware events, ...)."""
    lowered = activity.strip().lower()
    for keyword, canonical in _OBJECT_DETECTION_KEYWORDS.items():
        if keyword in lowered:
            return canonical
    return None


def _pass_through(activity: str, driver_id: str) -> VerifyResponse:
    """Non object-detection activities skip the VLM double-check and are accepted as-is."""
    log.info("VERIFY driver=%s activity=%s verified=True (pass-through, no VLM check)",
              driver_id, activity)
    return VerifyResponse(
        verified=True,
        confidence=1.0,
        activity=activity,
        reason="Not an object-detection activity — passed through without VLM check.",
    )


@router.post(
    "/verify/upload",
    response_model=VerifyResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Verify detection — multipart image upload",
    tags=["Verification"],
)
async def verify_upload(
    file:      UploadFile = File(..., description="JPEG/PNG of the driver crop"),
    activity:  str        = Form(..., description="Object-detection activity label, e.g. 'phone being used'"),
    driver_id: str        = Form(default="unknown"),
    _auth: None = Depends(require_api_key),
) -> VerifyResponse:
    canonical = _resolve_object_detection_activity(activity)
    if canonical is None:
        return _pass_through(activity, driver_id)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    result = await _run_verify(canonical, base64.b64encode(raw).decode())
    verified = result["verified"] and result["confidence"] >= cfg.VLM_ALERT_THRESHOLD

    log.info("VERIFY driver=%s activity=%s verified=%s conf=%.2f reason=%r",
             driver_id, activity, verified, result["confidence"], result["reason"])

    return VerifyResponse(
        verified=verified,
        confidence=result["confidence"],
        activity=activity,
        reason=result["reason"],
    )


@router.post(
    "/verify",
    response_model=VerifyResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Verify detection — JSON / base64",
    tags=["Verification"],
)
async def verify_json(
    body: VerifyJsonRequest,
    _auth: None = Depends(require_api_key),
) -> VerifyResponse:
    canonical = _resolve_object_detection_activity(body.activity)
    if canonical is None:
        return _pass_through(body.activity, body.driver_id)

    result = await _run_verify(canonical, body.image_b64)
    verified = result["verified"] and result["confidence"] >= cfg.VLM_ALERT_THRESHOLD

    log.info("VERIFY driver=%s activity=%s verified=%s conf=%.2f reason=%r",
             body.driver_id, body.activity, verified, result["confidence"], result["reason"])

    return VerifyResponse(
        verified=verified,
        confidence=result["confidence"],
        activity=body.activity,
        reason=result["reason"],
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness", tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION, vlm_model=cfg.VLM_MODEL)
