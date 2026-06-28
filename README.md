# Driver Safety AI

Real-time driver distraction detection using a two-stage pipeline:  
**On-device YOLO detection → Cloud VLM verification → Alert**

Detects: 📱 Mobile phone usage &nbsp;|&nbsp; 🚬 Smoking &nbsp;|&nbsp; 🍽️ Eating &nbsp;|&nbsp; 🥤 Drinking

---

## Architecture

```
Mobile / Edge Device                    Cloud (this repo)
────────────────────                    ─────────────────
Video / RTSP stream
        │
   Frame extraction
   (5–10 FPS)
        │
   YOLO11n TFLite          ──── JPEG crop ────▶   Phase 1 API
   Object detection                              POST /api/v1/verify
        │                                               │
   Spatial association                          LLaVA VLM verification
   (phone → person)                                     │
        │                                       Three-bucket classification
   Expand person ROI                                    │
   20% padding                         ┌───────┬───────┴───────┐
        │                           alert   review          false
   JPEG compress                  evidence/  review/    false_detections/
   (~75KB crop)                   + alert    + log only   + log only
        │                         fired
   POST to cloud ◀──────────────────────
```

**Why two stages?**  
YOLO runs at low confidence (0.35) to catch all potential events. The VLM acts as a smart filter — it sees the actual image and decides if the detection is real. This eliminates false positives without missing genuine distractions.

---

## Project Structure

```
driver_safety/
│
├── config.py               # All tuneable parameters (thresholds, paths, model)
│
├── mobile_sim.py           # Simulates the mobile app
│                           # YOLO detection → spatial association → send to cloud
│
├── cloud/
│   ├── vlm.py              # LLaVA verification via Ollama (driver-context prompts)
│   └── server.py           # Phase 1 FastAPI — internal endpoint for mobile
│
└── phase2/
    ├── main.py             # Phase 2 FastAPI — public endpoint for frontend team
    ├── router.py           # Routes: /verify, /verify/upload, /events, /stats
    └── models.py           # Pydantic request / response schemas
```

---

## VLM Confidence Buckets

Every detection is classified into one of three buckets:

| Bucket | Condition | Action | Saved to |
|--------|-----------|--------|----------|
| `alert` | verified=True AND conf ≥ 0.75 | Alert fired to backend | `evidence/` |
| `review` | 0.40 ≤ conf < 0.75 | Logged only — borderline | `review/{activity}/` |
| `false` | conf < 0.40 | Logged only — clear false positive | `false_detections/{activity}/` |

`false_detections/` and `review/` are used to retrain and improve the YOLO model.

---

## Quickstart

### Requirements

| Component | Environment | Key packages |
|-----------|-------------|--------------|
| Mobile sim | `conda activate yolo` | ultralytics, tensorflow, opencv |
| Cloud API | `conda activate v12` | fastapi, uvicorn, ollama, pydantic |
| VLM | Ollama running locally | `ollama pull llava:7b` |

### 1 — Start Ollama

```bash
ollama serve
ollama pull llava:7b
```

### 2 — Start Cloud API (Phase 2 — frontend-facing)

```bash
conda activate v12
cd ~/Downloads/driver_safety
uvicorn phase2.main:app --host 0.0.0.0 --port 8001
```

Swagger docs: [http://localhost:8001/docs](http://localhost:8001/docs)

### 3 — Start Mobile Simulator

```bash
conda activate yolo
cd ~/Downloads/driver_safety
python mobile_sim.py 0              # webcam
python mobile_sim.py video.mp4      # video file
python mobile_sim.py rtsp://...     # RTSP stream
```

---

## API Reference (Phase 2)

All endpoints require `X-API-Key` header.  
Default dev key: `dev-key-change-me` — set `API_KEY` env var in production.

### POST `/api/v2/verify/upload` — multipart (frontend / Postman)

```bash
curl -X POST http://localhost:8001/api/v2/verify/upload \
  -H "X-API-Key: dev-key-change-me" \
  -F "file=@driver_crop.jpg" \
  -F "activity=phone" \
  -F "driver_id=cam_01" \
  -F "det_confidence=0.72"
```

### POST `/api/v2/verify` — JSON base64 (mobile / internal service)

```bash
curl -X POST http://localhost:8001/api/v2/verify \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "<base64-encoded-jpeg>",
    "activity": "phone",
    "driver_id": "cam_01",
    "det_confidence": 0.72
  }'
```

### Response

```json
{
  "event_id":   "a3f2b1c4d5",
  "verified":   true,
  "confidence": 0.88,
  "activity":   "phone",
  "reason":     "Driver holding phone to right ear with left hand off wheel",
  "bucket":     "alert",
  "latency_ms": 2340.0
}
```

`verified=true` only when `bucket="alert"`. Frontend should only fire an alert on `verified=true`.

### GET `/api/v2/events` — query alert history

```
?bucket=alert&activity=phone&driver_id=cam_01&limit=50
```

### GET `/api/v2/stats` — per-activity alert counts

```json
{"phone": 12, "cigarette": 3, "food": 1, "drink": 0, "total": 16}
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `yolo11n.pt` | Detection model — swap to `.tflite` for on-device |
| `DETECTION_CONF` | `0.35` | YOLO confidence threshold (low — VLM filters FPs) |
| `VLM_MODEL` | `llava:7b` | Ollama model for verification |
| `VLM_ALERT_THRESHOLD` | `0.75` | Minimum VLM confidence to fire an alert |
| `VLM_REVIEW_THRESHOLD` | `0.40` | Below this = clear false positive |
| `VLM_WORKERS` | `3` | Thread pool size for concurrent VLM calls |
| `ALERT_COOLDOWN_SEC` | `5.0` | Minimum seconds between same-activity alerts per driver |

---

## Logs

Structured log line for every decision:

```
2026-06-28 14:23:11 INFO  VERIFY event=a3f2b1 driver=cam_01 activity=phone
    det_conf=0.72 vlm_conf=0.88 bucket=alert latency_ms=2340 reason='Driver holding phone to ear'
```

Log files rotate at 10 MB, keeping 7 files → `logs/driver_safety.log`.

---

## Improving the Model

Crops saved to `false_detections/{activity}/` are YOLO false positives confirmed by the VLM.  
Each `.jpg` has a `.json` sidecar with detection confidence and VLM reasoning.  
Use these as **hard negatives** when retraining your YOLO model to reduce future false positives.

---

## Roadmap

- [ ] Swap `llava:7b` for `qwen2.5-vl` when available in Ollama for better accuracy
- [ ] Add RTSP multi-camera support in mobile sim  
- [ ] Webhook / push notification in `_send_alert()`
- [ ] Auto-labelling pipeline using `false_detections/` for model retraining
