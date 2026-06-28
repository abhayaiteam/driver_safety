"""
Request / Response schemas for the Phase 2 public API.
Frontend team uses these as the contract.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────────

class VerifyJsonRequest(BaseModel):
    """
    Used when the client sends an image as base64 (e.g. mobile app, internal service).
    """
    image_b64: str = Field(
        ...,
        description="Base64-encoded JPEG/PNG of the driver crop (expanded person ROI)",
        examples=["<base64 string>"],
    )
    activity: str = Field(
        ...,
        description="What the detector flagged. One of: phone, cigarette, food, drink",
        examples=["phone"],
    )
    driver_id: Optional[str] = Field(
        default="unknown",
        description="Identifier for the driver / vehicle / camera",
        examples=["driver_001"],
    )
    det_confidence: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="YOLO detection confidence (0–1). Sent for logging; not used in VLM.",
        examples=[0.72],
    )


# ── Responses ────────────────────────────────────────────────────────────────

class VerifyResponse(BaseModel):
    """Result returned to the frontend for every verification request."""
    event_id:   str   = Field(description="Unique ID for this event (use for dedup / tracking)")
    verified:   bool  = Field(description="True only when bucket='alert' — safe to fire an alert")
    confidence: float = Field(ge=0.0, le=1.0, description="VLM confidence score (0–1)")
    activity:   str   = Field(description="Activity that was checked")
    reason:     str   = Field(description="VLM's one-line explanation of the decision")
    bucket:     str   = Field(description="alert | review | false  — classification bucket")
    latency_ms: float = Field(description="End-to-end verification time in milliseconds")


class ErrorResponse(BaseModel):
    """Returned on 4xx / 5xx errors."""
    error: str
    detail: Optional[str] = None


class StatsResponse(BaseModel):
    """Per-activity event counts."""
    phone: int = 0
    cigarette: int = 0
    food: int = 0
    drink: int = 0
    total: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    vlm_model: str
