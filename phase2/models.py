from typing import Optional
from pydantic import BaseModel, Field


class VerifyJsonRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG of the driver crop")
    activity: str = Field(..., description="phone | cigarette | drowsy | distracted")
    driver_id: Optional[str] = Field(default="unknown")
    det_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class VerifyResponse(BaseModel):
    verified:   bool  = Field(description="True = confirmed detection, False = not detected")
    confidence: float = Field(ge=0.0, le=1.0, description="VLM confidence score (0–1)")
    activity:   str   = Field(description="Activity that was checked")
    reason:     str   = Field(description="One-line explanation from VLM")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    vlm_model: str
