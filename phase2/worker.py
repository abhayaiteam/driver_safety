"""
Incident Verification Worker
=============================
Polls the mobile team's incident API for pending detections,
verifies each one with LLaVA, then updates the incident with the result.

Only incidents where VLM confirms the detection (verified=True, conf >= threshold)
are marked as confirmed. All others are marked as false positives.

Flow:
    1. GET  {MOBILE_API_BASE_URL}{INCIDENT_LIST_PATH}?status=pending
    2. For each incident → extract image → run VLM
    3. PATCH {MOBILE_API_BASE_URL}{INCIDENT_UPDATE_PATH}/{id}
            { verified, confidence, reason, verified_by }
"""

import asyncio
import base64
import logging
import os
import time

import requests as _req

from config import cfg
from vlm import verify as _vlm_verify

log = logging.getLogger("phase2.worker")

POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "10"))
INCIDENT_STATUS_FIELD = os.getenv("INCIDENT_STATUS_FIELD", "status")
INCIDENT_PENDING_VALUE = os.getenv("INCIDENT_PENDING_VALUE", "pending")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if cfg.MOBILE_API_KEY:
        h["Authorization"] = f"Bearer {cfg.MOBILE_API_KEY}"
    return h


def _get_pending_incidents() -> list:
    """
    GET all incidents with status=pending from the mobile backend.
    Handles both a plain list response and a wrapped { data: [...] } shape.
    """
    url = f"{cfg.MOBILE_API_BASE_URL}{cfg.INCIDENT_LIST_PATH}"
    params = {INCIDENT_STATUS_FIELD: INCIDENT_PENDING_VALUE}
    resp = _req.get(url, params=params, headers=_auth_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    # unwrap common envelope shapes: { data: [] } / { incidents: [] } / { results: [] }
    for key in ("data", "incidents", "results", "items"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _extract_image_b64(incident: dict) -> str | None:
    """
    Pull the image out of the incident object.
    Tries base64 fields first, then a URL to download.
    """
    # Direct base64 field — common names
    for field in ("image_b64", "imageBase64", "image", "frame", "snapshot"):
        val = incident.get(field)
        if val and isinstance(val, str) and len(val) > 100:
            return val

    # Image URL — download and encode
    for field in ("image_url", "imageUrl", "frame_url", "snapshot_url"):
        val = incident.get(field)
        if val and isinstance(val, str) and val.startswith("http"):
            try:
                r = _req.get(val, timeout=10)
                r.raise_for_status()
                return base64.b64encode(r.content).decode()
            except Exception as exc:
                log.warning("Image download failed url=%s err=%s", val, exc)

    return None


def _incident_id(incident: dict) -> str | None:
    for field in ("id", "_id", "incidentId", "incident_id"):
        if incident.get(field):
            return str(incident[field])
    return None


def _update_incident(incident_id: str, verified: bool, confidence: float, reason: str) -> None:
    """
    PATCH the incident back with the VLM verdict.
    The mobile team's backend marks it confirmed or false-positive.
    """
    url = f"{cfg.MOBILE_API_BASE_URL}{cfg.INCIDENT_UPDATE_PATH}/{incident_id}"
    payload = {
        "verified":     verified,
        "confidence":   round(confidence, 2),
        "reason":       reason,
        "verified_by":  "driver_safety_ai",
        "verified_at":  time.time(),
        # Mobile team can use this to filter confirmed alerts in their dashboard
        "status":       "confirmed" if verified else "false_positive",
    }
    resp = _req.patch(url, json=payload, headers=_auth_headers(), timeout=5)
    resp.raise_for_status()


# ── One poll cycle ────────────────────────────────────────────────────────────

async def _process_pending() -> None:
    loop = asyncio.get_event_loop()

    incidents = await loop.run_in_executor(None, _get_pending_incidents)
    if not incidents:
        return

    log.info("Worker found %d pending incident(s)", len(incidents))

    for incident in incidents:
        iid      = _incident_id(incident)
        activity = (incident.get("activity") or incident.get("type") or "phone").lower()
        device   = incident.get("deviceTabletId") or incident.get("device_id") or "unknown"

        if not iid:
            log.warning("Worker: incident has no id, skipping: %s", incident)
            continue

        image_b64 = await loop.run_in_executor(None, _extract_image_b64, incident)
        if not image_b64:
            log.warning("Worker: no image in incident=%s device=%s, skipping", iid, device)
            continue

        # Run VLM (blocking Ollama call → executor keeps FastAPI async)
        result = await loop.run_in_executor(None, _vlm_verify, activity, image_b64, cfg.VLM_MODEL)

        verified = result["verified"] and result["confidence"] >= cfg.VLM_ALERT_THRESHOLD

        log.info(
            "Worker VERIFY incident=%s device=%s activity=%s verified=%s conf=%.2f reason=%r",
            iid, device, activity, verified, result["confidence"], result["reason"],
        )

        try:
            await loop.run_in_executor(
                None, _update_incident, iid, verified, result["confidence"], result["reason"]
            )
            log.info("Worker UPDATED incident=%s → status=%s", iid,
                     "confirmed" if verified else "false_positive")
        except Exception as exc:
            log.error("Worker UPDATE failed incident=%s err=%s", iid, exc)


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run_worker() -> None:
    """
    Called once at FastAPI startup — runs forever in the background.
    Polls every POLL_INTERVAL_SEC seconds (default 10).
    """
    if not cfg.INCIDENT_LIST_PATH:
        log.info("Worker disabled — INCIDENT_LIST_PATH not configured")
        return

    log.info("Worker started — polling %s%s every %ds",
             cfg.MOBILE_API_BASE_URL, cfg.INCIDENT_LIST_PATH, POLL_INTERVAL_SEC)

    while True:
        try:
            await _process_pending()
        except Exception as exc:
            log.error("Worker cycle error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL_SEC)
