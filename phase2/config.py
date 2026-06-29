"""
Cloud-only config for phase2.
Every value reads from an environment variable first, falls back to a safe default.
Set these on EC2 via the systemd service file or a .env file.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── VLM ──────────────────────────────────────────────────────────────────
    VLM_MODEL:             str   = os.getenv("VLM_MODEL",              "llava:7b")
    VLM_WORKERS:           int   = int(os.getenv("VLM_WORKERS",        "3"))
    JPEG_QUALITY:          int   = int(os.getenv("JPEG_QUALITY",       "75"))
    VLM_ALERT_THRESHOLD:   float = float(os.getenv("VLM_ALERT_THRESHOLD",  "0.75"))
    VLM_REVIEW_THRESHOLD:  float = float(os.getenv("VLM_REVIEW_THRESHOLD", "0.40"))

    # ── Dedup ─────────────────────────────────────────────────────────────────
    ALERT_COOLDOWN_SEC:  float = float(os.getenv("ALERT_COOLDOWN_SEC",  "5.0"))
    DEDUP_COOLDOWN_SEC:  float = float(os.getenv("DEDUP_COOLDOWN_SEC",  "5.0"))

    # ── Alert webhook (mobile team's backend) ────────────────────────────────
    # Set ALERT_WEBHOOK_URL to the mobile backend endpoint that receives alerts.
    # e.g. https://their-backend.com/api/safety-alerts
    ALERT_WEBHOOK_URL:    str = os.getenv("ALERT_WEBHOOK_URL",   "")
    ALERT_WEBHOOK_TOKEN:  str = os.getenv("ALERT_WEBHOOK_TOKEN", "")  # Bearer token if they require auth

    # ── Storage ───────────────────────────────────────────────────────────────
    DB_PATH:               str = os.getenv("DB_PATH",              "events.db")
    LOG_FILE:              str = os.getenv("LOG_FILE",             "logs/driver_safety.log")
    EVIDENCE_DIR:          str = os.getenv("EVIDENCE_DIR",         "evidence")
    REVIEW_DIR:            str = os.getenv("REVIEW_DIR",           "review")
    FALSE_DETECTIONS_DIR:  str = os.getenv("FALSE_DETECTIONS_DIR", "false_detections")


cfg = Config()
