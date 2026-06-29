"""
RTSP + YOLO TFLite + VLM Verification Test
============================================
Usage:
    python test_rtsp_vlm.py --rtsp 0                    # webcam
    python test_rtsp_vlm.py --rtsp rtsp://user:pass@IP/stream --show
    python test_rtsp_vlm.py --debug                     # inspect model output shape only
"""

import argparse
import base64
import time
import cv2
import numpy as np
import requests
import tensorflow as tf

# ── CONFIG ────────────────────────────────────────────────────────────────────

API_BASE   = "https://textile-immediately-muscles-calculation.trycloudflare.com"
API_KEY    = "033bd1b5ee860eb2b3ade6768f7cd2d8df503f2813e7937d458b11ed0ec90d71"
MODEL_PATH = "custom_yolo_dynamic_int8.tflite"

CLASS_NAMES = ["cigarette", "phone", "seatbelt", "eating", "drinking"]

# Map YOLO class name → VLM activity name
VLM_ACTIVITY = {
    "phone":     "phone",
    "cigarette": "cigarette",
    "eating":    "food",
    "drinking":  "drink",
    "seatbelt":  None,   # no VLM — YOLO confidence is enough
}

DETECT_THRESHOLD = 0.35
NMS_THRESHOLD    = 0.45
VLM_COOLDOWN_SEC = 3.0


# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model(path: str):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    outs = interp.get_output_details()

    print(f"\n{'='*50}")
    print(f"[MODEL] Input  : shape={inp['shape']}  dtype={inp['dtype'].__name__}")
    for i, o in enumerate(outs):
        print(f"[MODEL] Output[{i}]: shape={o['shape']}  dtype={o['dtype'].__name__}")
    print(f"{'='*50}\n")

    return interp, inp, outs


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_yolo(interp, inp_detail, out_details, frame):
    ih, iw = inp_detail["shape"][1], inp_detail["shape"][2]
    img = cv2.resize(frame, (iw, ih))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if inp_detail["dtype"] == np.uint8:
        tensor = img.astype(np.uint8)
    else:
        tensor = (img / 255.0).astype(np.float32)

    interp.set_tensor(inp_detail["index"], tensor[np.newaxis])
    interp.invoke()

    raw = interp.get_tensor(out_details[0]["index"])
    return raw


def parse_detections(raw, orig_h, orig_w, model_h, model_w, threshold, debug=False):
    """
    Handles both YOLOv5 and YOLOv8 TFLite output formats automatically.

    YOLOv8 TFLite: shape [1, 4+num_classes, num_anchors] → need to transpose
    YOLOv5 TFLite: shape [1, num_anchors, 4+1+num_classes] → has objectness score
    """
    num_classes = len(CLASS_NAMES)

    if debug:
        print(f"[DEBUG] raw shape: {raw.shape}  min={raw.min():.3f}  max={raw.max():.3f}")

    # ── Detect format and reshape ──────────────────────────────────────────────
    if raw.ndim == 3:
        # YOLOv8: [1, 9, 8400] → transpose to [8400, 9]
        if raw.shape[1] == (4 + num_classes):
            rows = raw[0].T           # [8400, 9]
            has_objectness = False
        # YOLOv5: [1, 25200, 10] → [25200, 10]
        elif raw.shape[2] == (4 + 1 + num_classes):
            rows = raw[0]             # [N, 10]
            has_objectness = True
        elif raw.shape[2] == (4 + num_classes):
            rows = raw[0]             # [N, 9]
            has_objectness = False
        else:
            # Unknown — try treating as [N, ?]
            rows = raw[0] if raw.shape[1] > raw.shape[2] else raw[0].T
            has_objectness = raw.shape[-1] > (4 + num_classes)
    else:
        rows = raw
        has_objectness = rows.shape[-1] > (4 + num_classes)

    if debug:
        print(f"[DEBUG] rows shape after reshape: {rows.shape}  has_objectness={has_objectness}")

    # ── Extract boxes, scores ─────────────────────────────────────────────────
    boxes_xywh, confs, class_ids = [], [], []

    for row in rows:
        if has_objectness:
            obj_conf  = float(row[4])
            scores    = row[5:5 + num_classes] * obj_conf
        else:
            scores    = row[4:4 + num_classes]

        class_id = int(np.argmax(scores))
        conf     = float(scores[class_id])

        if conf < threshold:
            continue

        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])

        # Normalise if values look like pixel coords
        if cx > 2.0:
            cx /= model_w; bw /= model_w
        if cy > 2.0:
            cy /= model_h; bh /= model_h

        # Convert to pixel space in original frame
        x1 = int((cx - bw / 2) * orig_w)
        y1 = int((cy - bh / 2) * orig_h)
        x2 = int((cx + bw / 2) * orig_w)
        y2 = int((cy + bh / 2) * orig_h)

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(x1 + 1, min(orig_w, x2))
        y2 = max(y1 + 1, min(orig_h, y2))

        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        confs.append(conf)
        class_ids.append(class_id)

    if not boxes_xywh:
        return []

    # ── NMS ───────────────────────────────────────────────────────────────────
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confs, threshold, NMS_THRESHOLD)
    if len(indices) == 0:
        return []

    results = []
    for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
        x, y, w, h = boxes_xywh[i]
        results.append((x, y, x + w, y + h, confs[i], class_ids[i]))

    return results


# ── VLM CALL ──────────────────────────────────────────────────────────────────

def call_vlm(crop_bgr, activity: str, driver_id: str = "test") -> dict:
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    files   = {"file": ("crop.jpg", buf.tobytes(), "image/jpeg")}
    data    = {"activity": activity, "driver_id": driver_id}
    headers = {"X-API-Key": API_KEY}

    try:
        t0   = time.perf_counter()
        resp = requests.post(
            f"{API_BASE}/api/v2/verify/upload",
            files=files, data=data, headers=headers, timeout=30,
        )
        latency = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        result = resp.json()
        result["latency_ms"] = round(latency, 1)
        return result
    except Exception as e:
        return {"verified": False, "confidence": 0.0, "reason": str(e), "latency_ms": 0}


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp",     default="0")
    parser.add_argument("--activity", default=None, help="Force activity override")
    parser.add_argument("--driver",   default="test")
    parser.add_argument("--show",     action="store_true")
    parser.add_argument("--debug",    action="store_true", help="Print raw output shape and exit")
    args = parser.parse_args()

    interp, inp_detail, out_details = load_model(MODEL_PATH)
    model_h, model_w = int(inp_detail["shape"][1]), int(inp_detail["shape"][2])

    # Debug mode: run one frame and print raw output to diagnose bbox issues
    if args.debug:
        dummy = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        raw   = run_yolo(interp, inp_detail, out_details, dummy)
        print(f"[DEBUG] raw output shape: {raw.shape}")
        print(f"[DEBUG] raw output dtype: {raw.dtype}")
        print(f"[DEBUG] raw sample values (first row): {raw.flat[:10]}")
        print("\nUse this info to verify coordinate parsing is correct.")
        return

    src = int(args.rtsp) if args.rtsp.isdigit() else args.rtsp
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {args.rtsp}")
        return

    print(f"[STREAM] Connected: {args.rtsp}")
    print(f"[READY] model_input={model_w}x{model_h}  threshold={DETECT_THRESHOLD}")
    print(f"        VLM API: {API_BASE}")
    print(f"        Press Q to quit\n")

    last_vlm_time: dict[int, float] = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        raw         = run_yolo(interp, inp_detail, out_details, frame)
        detections  = parse_detections(raw, orig_h, orig_w, model_h, model_w, DETECT_THRESHOLD,
                                       debug=(frame_count == 1))  # debug first frame only

        for (x1, y1, x2, y2, conf, class_id) in detections:
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
            activity   = args.activity or VLM_ACTIVITY.get(class_name, class_name)

            # If --activity is specified, skip all other classes
            if args.activity and VLM_ACTIVITY.get(class_name, class_name) != args.activity:
                continue
            now        = time.time()

            if args.show:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Seatbelt: no VLM needed
            if activity is None:
                worn = conf >= 0.6
                print(f"[SEATBELT] frame={frame_count} {'✅ WORN' if worn else '⚠️  NOT WORN'} conf={conf:.2f}")
                continue

            if now - last_vlm_time.get(class_id, 0) < VLM_COOLDOWN_SEC:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            last_vlm_time[class_id] = now
            print(f"\n[YOLO] frame={frame_count} class={class_name} yolo_conf={conf:.2f} box=({x1},{y1},{x2},{y2})")

            result     = call_vlm(crop, activity, args.driver)
            verified   = result.get("verified", False)
            vlm_conf   = result.get("confidence", 0.0)
            reason     = result.get("reason", "")
            latency_ms = result.get("latency_ms", 0)

            status = "✅ ALERT" if verified else "❌ FALSE POSITIVE"
            print(f"[VLM]  {status}")
            print(f"       verified={verified}  confidence={vlm_conf:.2f}  latency={latency_ms}ms")
            print(f"       reason: {reason}")

            if args.show:
                color = (0, 0, 255) if verified else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} VLM:{vlm_conf:.2f} {'✓' if verified else '✗'}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if args.show:
            cv2.imshow("Driver Safety", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
