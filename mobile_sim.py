"""
Mobile Simulator — mimics what the Android/iOS app does on-device.

Pipeline per frame:
  1. Capture frame from webcam / video file / RTSP
  2. Run YOLO (.pt via Ultralytics, or .tflite via TFLite) → bboxes
  3. Spatial association → which object belongs to which person
  4. Expand person ROI by 20%, JPEG-compress the crop
  5. POST crop + metadata to Cloud API (non-blocking queue + uploader thread)
  6. Show live preview with overlays

Run with:
    conda activate yolo
    python mobile_sim.py                     # webcam
    python mobile_sim.py /path/to/video.mp4  # video file
    python mobile_sim.py rtsp://...          # RTSP stream
"""

import base64
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from config import cfg

# ── Model loader — auto-detects .pt vs .tflite ───────────────────────────────

_model_path = str(cfg.MODEL_PATH)
print(f"[SIM] Loading model: {_model_path}")

if _model_path.endswith(".pt"):
    from ultralytics import YOLO as _YOLO
    _yolo = _YOLO(_model_path)

    # Build a mapping from COCO / custom class ids → our cfg.CLASS_NAMES
    # For yolo11n.pt (COCO): person=0, cell phone=67
    # For a custom .pt trained on ["person","phone"]: person=0, phone=1
    _RAW_NAMES = _yolo.names           # {id: "class_name"}
    _NAME_TO_OURS = {}
    for cid, raw in _RAW_NAMES.items():
        raw_l = raw.lower().replace(" ", "_")
        for ours in cfg.CLASS_NAMES:
            # "cell_phone" matches "phone", "person" matches "person"
            if ours in raw_l or raw_l in ours:
                _NAME_TO_OURS[cid] = ours
                break

    # COCO fallback: explicitly map "cell phone" → "phone"
    for cid, raw in _RAW_NAMES.items():
        if "phone" in raw.lower() and cid not in _NAME_TO_OURS:
            _NAME_TO_OURS[cid] = "phone"

    _DETECT_CLASS_IDS = list(_NAME_TO_OURS.keys())
    print(f"[SIM] .pt model ready | mapped classes: {_NAME_TO_OURS}")

    def detect(frame: np.ndarray) -> list:
        results = _yolo.predict(
            frame,
            conf=cfg.DETECTION_CONF,
            classes=_DETECT_CLASS_IDS,
            imgsz=cfg.INPUT_SIZE,
            verbose=False,
        )[0]

        dets = []
        for box in results.boxes:
            cid  = int(box.cls[0])
            name = _NAME_TO_OURS.get(cid, _RAW_NAMES.get(cid, str(cid)))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append({
                "class_id":   cid,
                "name":       name,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
            })
        return dets

else:
    # ── TFLite path (kept for actual mobile deployment) ───────────────────
    import tensorflow as tf

    _interpreter = tf.lite.Interpreter(model_path=_model_path)
    _interpreter.allocate_tensors()
    _inp = _interpreter.get_input_details()[0]
    _out = _interpreter.get_output_details()[0]
    print(f"[SIM] .tflite ready  in={_inp['shape']}  out={_out['shape']}")

    def _preprocess_tflite(frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (cfg.INPUT_SIZE, cfg.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def _nms_tflite(raw_dets: list) -> list:
        if not raw_dets:
            return []
        final = []
        for cls_id in set(d["class_id"] for d in raw_dets):
            sub = [d for d in raw_dets if d["class_id"] == cls_id]
            xywh  = [[d["bbox"][0], d["bbox"][1],
                       d["bbox"][2]-d["bbox"][0], d["bbox"][3]-d["bbox"][1]]
                      for d in sub]
            confs = [d["confidence"] for d in sub]
            idxs  = cv2.dnn.NMSBoxes(xywh, confs, cfg.DETECTION_CONF, cfg.NMS_THRESHOLD)
            for i in idxs:
                final.append(sub[i])
        return final

    def detect(frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        _interpreter.set_tensor(_inp["index"], _preprocess_tflite(frame))
        _interpreter.invoke()
        raw = _interpreter.get_tensor(_out["index"])[0].T  # (8400, 9)

        bbox_norm    = raw[:, :4]
        class_scores = raw[:, 4:]
        best_cls     = class_scores.argmax(axis=1)
        best_conf    = class_scores.max(axis=1)
        mask         = best_conf >= cfg.DETECTION_CONF

        dets = []
        for box, cls_id, conf in zip(bbox_norm[mask], best_cls[mask], best_conf[mask]):
            cx, cy, bw, bh = box
            x1 = max(0, int((cx - bw/2) * w))
            y1 = max(0, int((cy - bh/2) * h))
            x2 = min(w, int((cx + bw/2) * w))
            y2 = min(h, int((cy + bh/2) * h))
            name = cfg.CLASS_NAMES[int(cls_id)] if int(cls_id) < len(cfg.CLASS_NAMES) else str(cls_id)
            dets.append({"class_id": int(cls_id), "name": name,
                         "confidence": float(conf), "bbox": [x1, y1, x2, y2]})
        return _nms_tflite(dets)

# ── Upload queue + uploader thread ───────────────────────────────────────────

_upload_q: queue.Queue = queue.Queue(maxsize=cfg.UPLOAD_QUEUE_SIZE)


def _uploader() -> None:
    """Runs in a daemon thread; sends crops to the cloud API."""
    while True:
        item = _upload_q.get()
        if item is None:
            break
        try:
            r = requests.post(
                f"{cfg.CLOUD_API_URL}/api/v1/verify",
                json=item,
                timeout=20,
            )
            res = r.json()
            verified = res.get("verified", False)
            conf = res.get("confidence", 0.0)
            latency = res.get("latency_ms", 0.0)
            symbol = "✅" if verified else "❌"
            print(
                f"[CLOUD] {symbol} {item['activity']:12s} | "
                f"verified={verified}  conf={conf:.2f}  latency={latency:.0f}ms",
                flush=True,
            )
        except Exception as exc:
            print(f"[UPLOAD ERR] {exc}", flush=True)
        _upload_q.task_done()


threading.Thread(target=_uploader, daemon=True, name="uploader").start()

# ── Spatial association ───────────────────────────────────────────────────────

def _center(bbox: list) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _diag(bbox: list) -> float:
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5


def associate(detections: list) -> list:
    """
    For each target object (phone/cigarette/food/drink), find the nearest person.

    Returns list of:
      {person_bbox, activity, obj_confidence}
    """
    persons = [d for d in detections if d["name"] == "person"]
    objects = [d for d in detections if d["name"] in cfg.TARGET_CLASSES]

    events = []
    for obj in objects:
        ocx, ocy = _center(obj["bbox"])
        best_person = None
        best_score  = -1.0

        for person in persons:
            px1, py1, px2, py2 = person["bbox"]

            # Score 1: object center inside person bbox (highest priority)
            inside = px1 <= ocx <= px2 and py1 <= ocy <= py2
            if inside:
                events.append({
                    "person_bbox":   person["bbox"],
                    "activity":      obj["name"],
                    "obj_confidence": obj["confidence"],
                })
                best_person = person
                break

            # Score 2: proximity — normalised distance from object center to person center
            pcx, pcy = _center(person["bbox"])
            dist = ((ocx - pcx) ** 2 + (ocy - pcy) ** 2) ** 0.5
            pd   = _diag(person["bbox"])
            norm_dist = dist / max(pd, 1.0)

            if norm_dist < cfg.PROXIMITY_RATIO:
                score = 1.0 - norm_dist
                if score > best_score:
                    best_score  = score
                    best_person = person

        if best_person and best_person not in [e.get("_person") for e in events]:
            events.append({
                "person_bbox":    best_person["bbox"],
                "activity":       obj["name"],
                "obj_confidence": obj["confidence"],
            })

    return events


# ── ROI crop ─────────────────────────────────────────────────────────────────

def crop_roi(frame: np.ndarray, bbox: list) -> np.ndarray | None:
    """Expand bbox by ROI_EXPAND and return the frame crop."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * cfg.ROI_EXPAND)
    py = int(bh * cfg.ROI_EXPAND)

    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)
    rx2 = min(w, x2 + px)
    ry2 = min(h, y2 + py)

    crop = frame[ry1:ry2, rx1:rx2]
    return crop if crop.size > 0 else None


def encode_jpeg(img: np.ndarray) -> str:
    """Return base64-encoded JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode()


# ── Drawing helpers ───────────────────────────────────────────────────────────

_COLORS = {
    "person":    (100, 220, 100),
    "phone":     (0,   0,   255),
    "cigarette": (0,   140, 255),
    "food":      (255, 60,  60),
    "drink":     (0,   200, 100),
}

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    disp = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = _COLORS.get(d["name"], (200, 200, 200))
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        label = f"{d['name']} {d['confidence']:.2f}"
        cv2.putText(disp, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return disp


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(source) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[SIM] Cannot open source: {source}")
        sys.exit(1)

    print(f"[SIM] Source opened: {source}")
    print(f"[SIM] Cloud API   : {cfg.CLOUD_API_URL}")
    print(f"[SIM] Press 'q' to quit\n")

    frame_idx    = 0
    last_sent: dict[str, float] = {}   # activity → last send epoch

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[SIM] Stream ended.")
            break

        frame_idx += 1

        # ── Skip frames for target FPS ────────────────────────────────────
        if frame_idx % cfg.FRAME_SKIP != 0:
            display = cv2.resize(frame, (960, 540))
            cv2.imshow("Driver Safety [Mobile Sim]", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Detect ────────────────────────────────────────────────────────
        detections = detect(frame)
        events     = associate(detections)

        # ── Draw ──────────────────────────────────────────────────────────
        display = draw_detections(frame, detections)

        now = time.time()
        for ev in events:
            activity = ev["activity"]
            last_t   = last_sent.get(activity, 0.0)

            if (now - last_t) < cfg.ALERT_COOLDOWN_SEC:
                continue

            crop = crop_roi(frame, ev["person_bbox"])
            if crop is None:
                continue

            if _upload_q.full():
                print("[SIM] Upload queue full — dropping frame", flush=True)
                continue

            image_b64 = encode_jpeg(crop)
            _upload_q.put({
                "driver_id":            cfg.DRIVER_ID,
                "activity":             activity,
                "detection_confidence": ev["obj_confidence"],
                "image_b64":            image_b64,
                "timestamp":            now,
            })
            last_sent[activity] = now

            # Visual feedback on the preview
            cv2.putText(
                display,
                f"SENT → {activity.upper()}",
                (10, 34 + list(cfg.TARGET_CLASSES).index(activity) * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
            )

        # ── FPS overlay ───────────────────────────────────────────────────
        cv2.putText(display, f"frame {frame_idx}", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Driver Safety [Mobile Sim]", cv2.resize(display, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    _upload_q.put(None)   # stop uploader thread


if __name__ == "__main__":
    raw_src = sys.argv[1] if len(sys.argv) > 1 else "0"
    try:
        src = int(raw_src)   # webcam index
    except ValueError:
        src = raw_src        # file path or RTSP URL
    run(src)
