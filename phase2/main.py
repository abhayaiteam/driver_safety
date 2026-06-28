"""
Driver Safety — Phase 2 Cloud API
===================================
Clean public-facing FastAPI service for the frontend team.

Run:
    conda activate v12
    cd ~/Downloads/driver_safety
    uvicorn phase2.main:app --host 0.0.0.0 --port 8001 --reload

Swagger docs:  http://localhost:8001/docs
ReDoc:         http://localhost:8001/redoc

Environment variables:
    API_KEY   — secret key clients must send as X-API-Key header
                (default: "dev-key-change-me", change in production)
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg   # noqa: E402  (path must be set first)
from phase2.router import router

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

Contact the backend team for a production key.

## Supported Activities

| Value | Meaning |
|---|---|
| `phone` | Driver using a mobile phone |
| `cigarette` | Driver smoking |
| `food` | Driver eating |
| `drink` | Driver drinking |

## Quick Start

```bash
# Upload an image file (easiest from frontend)
curl -X POST http://localhost:8001/api/v2/verify/upload \\
  -H "X-API-Key: dev-key-change-me" \\
  -F "file=@driver_crop.jpg" \\
  -F "activity=phone" \\
  -F "driver_id=cam_01"

# Or send base64 JSON (from mobile / internal service)
curl -X POST http://localhost:8001/api/v2/verify \\
  -H "X-API-Key: dev-key-change-me" \\
  -H "Content-Type: application/json" \\
  -d '{"image_b64": "<base64>", "activity": "phone", "driver_id": "cam_01"}'
```
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

# ── CORS — allow frontend to call from browser ────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v2")

# ── Startup log ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    import os
    os.makedirs(cfg.EVIDENCE_DIR, exist_ok=True)
    os.makedirs(cfg.FALSE_DETECTIONS_DIR, exist_ok=True)
    key = os.getenv("API_KEY", "dev-key-change-me")
    print(f"\n{'='*60}")
    print(f"  Driver Safety Phase 2 API  —  v2.0.0")
    print(f"  VLM model  : {cfg.VLM_MODEL}")
    print(f"  API key    : {key[:4]}{'*'*(len(key)-4)}")
    print(f"  Docs       : http://localhost:8001/docs")
    print(f"{'='*60}\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "phase2.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
