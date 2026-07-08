"""
Incident Verification Worker
=============================
Polls GET /api/incidents from proximity-driver-api.prod-app.in,
downloads snapshotUrl for each unprocessed incident, verifies with LLaVA,
then PATCHes the result back.

  verified=True  → PATCH /api/incidents/{id}/resolve      (real incident)
  verified=False → PATCH /api/incidents/{id}/acknowledge  (false positive, reviewed)

Processed incident IDs are stored in the local SQLite DB to avoid reprocessing.
"""

import asyncio
import base64
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests as _req

from config import cfg
from vlm import verify as _vlm_verify

log = logging.getLogger("phase2.worker")

POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "10"))

# eventType values the mobile team might use → map to our VLM prompt keys
# eventType values the mobile team might use → map to our VLM prompt keys
_EVENT_TYPE_MAP: dict[str, str] = {
    "phone":         "phone",
    "phone_usage":   "phone",
    "mobile_phone":  "phone",
    "mobile":        "phone",
    "call":          "phone",
    "texting":       "phone",
    "smoking":       "cigarette",
    "cigarette":     "cigarette",
    "smoke":         "cigarette",
    "vape":          "cigarette",
    "vaping":        "cigarette",
    "food":          "food",
    "eating":        "food",
    "drink":         "drink",
    "drinking":      "drink",
    "drowsy":        "drowsy",
    "drowsiness":    "drowsy",
    "sleepy":        "drowsy",
    "seatbelt":      "seatbelt",
    "seat_belt":     "seatbelt",
    "no_seatbelt":   "seatbelt",
    "no_seat_belt":  "seatbelt",
    "buckle":        "seatbelt",
}


# ── Auth headers ──────────────────────────────────────────────────────────────

def _headers(json: bool = True) -> dict:
    h = {}
    if json:
        h["Content-Type"] = "application/json"
    if cfg.MOBILE_API_KEY:
        h["Authorization"] = f"Bearer {cfg.MOBILE_API_KEY}"
    return h


# ── SQLite: track processed incidents ─────────────────────────────────────────

def _ensure_processed_table() -> None:
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed_incidents "
        "(id TEXT PRIMARY KEY, ts REAL)"
    )
    conn.commit()
    conn.close()


def _already_processed(incident_id: str) -> bool:
    conn = sqlite3.connect(cfg.DB_PATH)
    row = conn.execute(
        "SELECT id FROM processed_incidents WHERE id=?", (incident_id,)
    ).fetchone()
    conn.close()
    return row is not None


def _mark_processed(incident_id: str) -> None:
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT OR IGNORE INTO processed_incidents (id, ts) VALUES (?, ?)",
        (incident_id, time.time()),
    )
    conn.commit()
    conn.close()


# ── API calls ─────────────────────────────────────────────────────────────────

def _get_incidents() -> list:
    """GET /api/incidents from the mobile backend."""
    url = f"{cfg.MOBILE_API_BASE_URL}/api/incidents"
    resp = _req.get(url, headers=_headers(json=False), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    for key in ("data", "incidents", "results", "items"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _download_snapshot(snapshot_url: str) -> str | None:
    """Download image from snapshotUrl and return as base64."""
    try:
        resp = _req.get(snapshot_url, timeout=10)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()
    except Exception as exc:
        log.warning("Snapshot download failed url=%s err=%s", snapshot_url, exc)
        return None


def _resolve_incident(incident_id: str) -> None:
    """PATCH /api/incidents/{id}/resolve — confirmed real incident."""
    url = f"{cfg.MOBILE_API_BASE_URL}/api/incidents/{incident_id}/resolve"
    resp = _req.patch(url, headers=_headers(), timeout=5)
    resp.raise_for_status()


def _acknowledge_incident(incident_id: str) -> None:
    """PATCH /api/incidents/{id}/acknowledge — reviewed, false positive."""
    url = f"{cfg.MOBILE_API_BASE_URL}/api/incidents/{incident_id}/acknowledge"
    resp = _req.patch(url, headers=_headers(), timeout=5)
    resp.raise_for_status()


# ── One poll cycle ────────────────────────────────────────────────────────────

async def _process_pending() -> None:
    loop = asyncio.get_event_loop()
    incidents = await loop.run_in_executor(None, _get_incidents)

    new_incidents = [
        inc for inc in incidents
        if inc.get("id") and not _already_processed(str(inc["id"]))
    ]

    if not new_incidents:
        return

    log.info("Worker: %d new incident(s) to verify", len(new_incidents))

    for incident in new_incidents:
        iid          = str(incident["id"])
        event_type   = (incident.get("eventType") or "").lower().replace(" ", "_").strip()
        activity     = _EVENT_TYPE_MAP.get(event_type)
        device       = incident.get("deviceTabletId") or "unknown"
        snapshot_url = incident.get("snapshotUrl")

        if not event_type:
            log.warning("Worker: incident=%s has no eventType, skipping (left pending)", iid)
            continue

        if activity is None:
            # Unmapped event type: do NOT assume phone. Fall back to the strict
            # generic VLM prompt using the raw label, and flag it in the log so
            # the map above can be extended.
            log.warning("Worker: incident=%s has unmapped eventType=%r — using generic "
                        "strict prompt; add it to _EVENT_TYPE_MAP", iid, event_type)
            activity = event_type

        if not snapshot_url:
            log.warning("Worker: incident=%s has no snapshotUrl, skipping", iid)
            _mark_processed(iid)
            continue

        # Download frame
        image_b64 = await loop.run_in_executor(None, _download_snapshot, snapshot_url)
        if not image_b64:
            log.warning("Worker: could not download image for incident=%s", iid)
            continue

        # VLM verification
        result = await loop.run_in_executor(None, _vlm_verify, activity, image_b64, cfg.VLM_MODEL)
        verified = result["verified"] and result["confidence"] >= cfg.VLM_ALERT_THRESHOLD

        log.info(
            "Worker VERIFY incident=%s device=%s activity=%s verified=%s "
            "conf=%.2f reason=%r",
            iid, device, activity, verified, result["confidence"], result["reason"],
        )

        # Update incident on mobile backend
        try:
            if verified:
                await loop.run_in_executor(None, _resolve_incident, iid)
                log.info("Worker RESOLVED incident=%s (real detection)", iid)
            else:
                await loop.run_in_executor(None, _acknowledge_incident, iid)
                log.info("Worker ACKNOWLEDGED incident=%s (false positive)", iid)

            _mark_processed(iid)

        except Exception as exc:
            log.error("Worker update failed incident=%s err=%s", iid, exc)


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run_worker() -> None:
    """Starts at FastAPI startup. Polls forever."""
    _ensure_processed_table()

    log.info("Worker started — polling %s/api/incidents every %ds",
             cfg.MOBILE_API_BASE_URL, POLL_INTERVAL_SEC)

    while True:
        try:
            await _process_pending()
        except Exception as exc:
            log.error("Worker cycle error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL_SEC)
