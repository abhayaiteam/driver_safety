import logging
import logging.handlers
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from router import router


def _setup_logging() -> None:
    os.makedirs(Path(cfg.LOG_FILE).parent, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(cfg.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


_setup_logging()
log = logging.getLogger("phase2.main")

app = FastAPI(
    title="Driver Safety API",
    version="2.0.0",
    summary="Real-time driver distraction detection via VLM verification.",
    description="""
## Authentication
All endpoints (except `/api/v2/health`) require:
```
X-API-Key: <your-key>
```

## Supported Activities
| Value | Meaning |
|---|---|
| `phone` | Driver using a mobile phone |
| `cigarette` | Driver smoking |
| `drowsy` | Driver showing signs of drowsiness |
| `distracted` | Driver looking away from road |
""",
    contact={"name": "Driver Safety Team", "email": "abhay.a@ngxptechnologies.com"},
    license_info={"name": "Private"},
    openapi_tags=[
        {"name": "Verification", "description": "Submit a detection crop for VLM verification"},
        {"name": "System",       "description": "Health and status"},
    ],
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router, prefix="/api/v2")


@app.on_event("startup")
async def _startup() -> None:
    key = os.getenv("API_KEY", "dev-key-change-me")
    log.info("=" * 60)
    log.info("  Driver Safety Phase 2 API  —  v2.0.0")
    log.info("  VLM model  : %s", cfg.VLM_MODEL)
    log.info("  API key    : %s%s", key[:4], "*" * (len(key) - 4))
    log.info("  Docs       : http://0.0.0.0:8001/docs")
    log.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
