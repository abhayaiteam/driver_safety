"""
Driver Safety — Phase 2 Cloud API
===================================
Self-contained FastAPI service. No dependency on the parent repo directories.

Run from project root:
    uvicorn phase2.main:app --host 0.0.0.0 --port 8001

Run from inside the phase2/ folder (or via Docker):
    uvicorn main:app --host 0.0.0.0 --port 8001

Environment variables:
    API_KEY   — secret key clients must send as X-API-Key header (required in production)
    VLM_MODEL — Ollama model name (default: llava:7b)
    DB_PATH   — SQLite file path (default: events.db)
    LOG_FILE  — rotating log file path (default: logs/driver_safety.log)
"""

import asyncio
import logging
import logging.handlers
import os
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup: works whether invoked as `phase2.main` or bare `main` ─────────
sys.path.insert(0, str(Path(__file__).parent))

from config import cfg          # phase2/config.py
from router import router      # phase2/router.py
from worker import run_worker  # phase2/worker.py


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    os.makedirs(Path(cfg.LOG_FILE).parent, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        cfg.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7,
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


_setup_logging()
log = logging.getLogger("phase2.main")


# ── DB init ───────────────────────────────────────────────────────────────────

def _init_db() -> None:
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          TEXT PRIMARY KEY,
            driver_id   TEXT,
            activity    TEXT,
            bucket      TEXT,
            vlm_conf    REAL,
            det_conf    REAL,
            reason      TEXT,
            image_path  TEXT,
            ts          REAL
        )
    """)
    conn.commit()
    conn.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Driver Safety API",
    version="2.0.0",
    summary="Real-time driver distraction detection via VLM verification.",
    description="""
## Overview

The **Driver Safety API** sits between your mobile/edge detector and your
alert dashboard. The mobile app detects a potential distraction event with
YOLO, crops the driver ROI, and calls this API. The API uses a Vision
Language Model (LLaVA) to confirm whether the distraction is genuine before
sending it to the backend as a verified alert.

## Authentication

All endpoints (except `/api/v2/health`) require an API key:

```
X-API-Key: <your-key>
```

Or as a query parameter: `?api_key=<your-key>`

## Supported Activities

| Value | Meaning |
|---|---|
| `phone` | Driver using a mobile phone |
| `cigarette` | Driver smoking |
| `food` | Driver eating |
| `drink` | Driver drinking |
""",
    contact={
        "name": "Driver Safety Team",
        "email": "abhay.a@ngxptechnologies.com",
    },
    license_info={"name": "Private"},
    openapi_tags=[
        {"name": "Verification", "description": "Submit a detection crop for VLM verification"},
        {"name": "Events",       "description": "Query the history of verified alerts"},
        {"name": "System",       "description": "Health and status"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v2")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    for d in [cfg.EVIDENCE_DIR, cfg.REVIEW_DIR, cfg.FALSE_DETECTIONS_DIR]:
        os.makedirs(d, exist_ok=True)
    _init_db()
    key = os.getenv("API_KEY", "dev-key-change-me")
    log.info("=" * 60)
    log.info("  Driver Safety Phase 2 API  —  v2.0.0")
    log.info("  VLM model  : %s", cfg.VLM_MODEL)
    log.info("  API key    : %s%s", key[:4], "*" * (len(key) - 4))
    log.info("  DB         : %s", cfg.DB_PATH)
    log.info("  Docs       : http://0.0.0.0:8001/docs")
    if cfg.INCIDENT_LIST_PATH:
        log.info("  Worker     : polling %s%s every %ss",
                 cfg.MOBILE_API_BASE_URL, cfg.INCIDENT_LIST_PATH,
                 os.getenv("POLL_INTERVAL_SEC", "10"))
    log.info("=" * 60)

    # Start background incident verification worker
    asyncio.create_task(run_worker())


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
