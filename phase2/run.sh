#!/usr/bin/env bash
# Start the Driver Safety Phase 2 API
# Run from inside the phase2/ folder OR from the project root.
#
# Usage:
#   cd phase2 && bash run.sh
#
# Override any setting via env vars before calling:
#   API_KEY=my-secret VLM_MODEL=llava:7b bash run.sh

set -euo pipefail

: "${API_KEY:?ERROR: Set API_KEY before running. Example: export API_KEY=your-secret-key}"

export VLM_MODEL="${VLM_MODEL:-llava:7b}"
export DB_PATH="${DB_PATH:-events.db}"
export LOG_FILE="${LOG_FILE:-logs/driver_safety.log}"
export EVIDENCE_DIR="${EVIDENCE_DIR:-evidence}"
export REVIEW_DIR="${REVIEW_DIR:-review}"
export FALSE_DETECTIONS_DIR="${FALSE_DETECTIONS_DIR:-false_detections}"

mkdir -p logs evidence review false_detections

echo "=============================="
echo "  Driver Safety API  v2.0.0"
echo "  VLM  : $VLM_MODEL"
echo "  Port : 8001"
echo "  Docs : http://0.0.0.0:8001/docs"
echo "=============================="

# Detect whether we're inside phase2/ or the project root
if [[ -f "main.py" ]]; then
    # Running from inside phase2/
    uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
else
    # Running from project root
    uvicorn phase2.main:app --host 0.0.0.0 --port 8001 --workers 1
fi
