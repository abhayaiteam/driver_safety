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

_OBJECT_DETECTION_KEYWORDS: dict[str, str] = {
    "phone":      "phone",
    "mobile":     "phone",
    "call":       "phone",
    "texting":    "phone",
    "cigarette":  "cigarette",
    "smoking":    "cigarette",
    "smoke":      "cigarette",
    "vape":       "cigarette",
    "vaping":     "cigarette",
    # "eating":     "food",      # disabled — not needed right now
    # "food":       "food",      # disabled — not needed right now
    # "drinking":   "drink",     # disabled — not needed right now
    # "drink":      "drink",     # disabled — not needed right now
    "drowsy":     "drowsy",
    "drowsiness": "drowsy",
    "sleepy":     "drowsy",
    "seatbelt":       "seatbelt",
    "seat belt":      "seatbelt",
    "seat_belt":      "seatbelt",
    "fasten seatbelt": "seatbelt",
    "buckle":         "seatbelt",
}

# Food/drink detection is temporarily disabled (see commented entries above). Matched
# labels are suppressed (verified=False) here rather than falling through to
# _pass_through's default verified=True, so they don't show up as confirmed alerts
# while disabled. Re-enable by uncommenting the entries above and deleting this block.
_DISABLED_KEYWORDS = ("eating", "food", "drinking", "drink")


def _is_disabled_activity(activity: str) -> bool:
    lowered = activity.strip().lower()
    return any(keyword in lowered for keyword in _DISABLED_KEYWORDS)


_ACTIVITY_DISPLAY_NAMES: dict[str, str] = {
    # static labels for canonical activities (seatbelt is handled dynamically
    # in _display_activity because its label depends on the outcome)
}
_API_KEY         = os.getenv("API_KEY", "dev-key-change-me")
_executor        = ThreadPoolExecutor(max_workers=cfg.VLM_WORKERS)

router = APIRouter()


def _display_activity(canonical: str, original: str, verified: bool) -> str:
    """Human-readable activity label for the response.
    For seatbelt the label states the OUTCOME (verified=true means the
    violation was confirmed → 'Seatbelt Not Worn'; verified=false means the
    belt was seen or the violation could not be confirmed → 'Seatbelt Worn').
    All other activities keep their original / mapped label."""
    if canonical == "seatbelt":
        return "Seatbelt Not Worn" if verified else "Seatbelt Worn"
    return _ACTIVITY_DISPLAY_NAMES.get(canonical, original)


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
    """Map a free-text activity label to a VLM prompt key (phone/cigarette/food/drink/drowsy/seatbelt)
    if it needs a VLM double-check; None for anything else (distraction,
    unauthorized driver, hardware events, ...)."""
    lowered = activity.strip().lower()
    for keyword, canonical in _OBJECT_DETECTION_KEYWORDS.items():
        if keyword in lowered:
            return canonical
    return None


def _pass_through(activity: str, driver_id: str) -> VerifyResponse:
    """Activity with no VLM verifier (e.g. distraction from Flutter) — trust the
    upstream detection and pass through to the backend as an alert, no VLM check."""
    log.info("VERIFY driver=%s activity=%s verified=True (pass-through — no VLM check, "
             "trusting upstream detection)", driver_id, activity)
    return VerifyResponse(
        verified=True,
        confidence=1.0,
        activity=activity,
        reason="Passed through as detected (no VLM verification for this activity).",
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
    if _is_disabled_activity(activity):
        log.info("VERIFY driver=%s activity=%s verified=False (food/drink detection disabled)",
                  driver_id, activity)
        return VerifyResponse(
            verified=False,
            confidence=0.0,
            activity=activity,
            reason="Food/drink verification is temporarily disabled.",
        )

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
        activity=_display_activity(canonical, activity, verified),
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
    if _is_disabled_activity(body.activity):
        log.info("VERIFY driver=%s activity=%s verified=False (food/drink detection disabled)",
                  body.driver_id, body.activity)
        return VerifyResponse(
            verified=False,
            confidence=0.0,
            activity=body.activity,
            reason="Food/drink verification is temporarily disabled.",
        )

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
        activity=_display_activity(canonical, body.activity, verified),
        reason=result["reason"],
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness", tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION, vlm_model=cfg.VLM_MODEL)