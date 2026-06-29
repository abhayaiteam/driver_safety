FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8001

# Ollama runs on the host (or a sidecar container) — point to it via OLLAMA_HOST env var
# Default: Ollama on the same host machine
ENV OLLAMA_HOST=http://host.docker.internal:11434

ENV API_KEY=dev-key-change-me
ENV VLM_MODEL=llava:7b
ENV DB_PATH=/data/events.db
ENV LOG_FILE=/data/logs/driver_safety.log
ENV EVIDENCE_DIR=/data/evidence
ENV REVIEW_DIR=/data/review
ENV FALSE_DETECTIONS_DIR=/data/false_detections

# /data is a volume — mount it to persist events.db, logs, and crops
VOLUME ["/data"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
