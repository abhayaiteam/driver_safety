"""
RTSP + YOLO TFLite + VLM Verification Test
============================================
Usage:
    python test_rtsp_vlm.py --rtsp rtsp://user:pass@192.168.1.1/stream
    python test_rtsp_vlm.py --rtsp rtsp://... --activity phone
    python test_rtsp_vlm.py --rtsp 0   # use webcam instead of RTSP

Sends detected crops to the VLM API and prints verified true/false.
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

# YOLO class index → VLM activity name (matches your trained model)
CLASS_MAP = {
    0: "cigarette",
    1: "phone",
    2: "seatbelt",   # no VLM needed — YOLO result is used directly
    3: "food",
    4: "drink",
}

DETECT_THRESHOLD = 0.40   # YOLO confidence to trigger VLM
VLM_COOLDOWN_SEC = 3.0    # min seconds between VLM calls per class


# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model(path: str):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    outs = interp.get_output_details()
    print(f"\n[MODEL] Input  : {inp['shape']}  dtype={inp['dtype'].__name__}")
    for i, o in enumerate(outs):
        print(f"[MODEL] Output[{i}]: {o['shape']}  dtype={o['dtype'].__name__}")
    return interp, inp, outs


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_yolo(interp, inp_detail, out_details, frame):
    h, w = inp_detail["shape"][1], inp_detail["shape"][2]
    img  = cv2.resize(frame, (w, h))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if inp_detail["dtype"] == np.uint8:
        tensor = img.astype(np.uint8)
    else:
        tensor = (img / 255.0).astype(np.float32)

    interp.set_tensor(inp_detail["index"], tensor[np.newaxis])
    interp.invoke()

    # Most YOLO TFLite models output [1, num_boxes, 4+num_classes]
    raw = interp.get_tensor(out_details[0]["index"])[0]  # shape (N, 4+C)
    return raw, h, w


def parse_detections(raw, orig_h, orig_w, model_h, model_w, threshold):
    """
    Parse raw YOLO output into list of (x1,y1,x2,y2,conf,class_id).
    Supports both [cx,cy,w,h,...] and [x1,y1,x2,y2,...] formats.
    """
    detections = []
    num_classes = raw.shape[-1] - 4

    for row in raw:
        box   = row[:4]
        scores = row[4:4 + num_classes]
        class_id = int(np.argmax(scores))
        conf     = float(scores[class_id])

        if conf < threshold:
            continue

        # Normalise coords (assume cx,cy,w,h in [0..model_size])
        cx, cy, bw, bh = box
        if cx > 1.0:  # pixel coords → normalise
            cx /= model_w; cy /= model_h
            bw /= model_w; bh /= model_h

        x1 = int((cx - bw / 2) * orig_w)
        y1 = int((cy - bh / 2) * orig_h)
        x2 = int((cx + bw / 2) * orig_w)
        y2 = int((cy + bh / 2) * orig_h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, conf, class_id))

    return detections


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
    parser.add_argument("--rtsp",     default="0",    help="RTSP URL or 0 for webcam")
    parser.add_argument("--activity", default=None,   help="Force activity (phone/cigarette/drowsy/distracted)")
    parser.add_argument("--driver",   default="test", help="Driver ID")
    parser.add_argument("--show",     action="store_true", help="Show OpenCV window")
    args = parser.parse_args()

    src = int(args.rtsp) if args.rtsp.isdigit() else args.rtsp
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {args.rtsp}")
        return

    print(f"[STREAM] Connected: {args.rtsp}")
    interp, inp_detail, out_details = load_model(MODEL_PATH)
    model_h, model_w = inp_detail["shape"][1], inp_detail["shape"][2]

    last_vlm_time: dict[int, float] = {}
    frame_count = 0

    print(f"\n[READY] Watching stream — YOLO threshold={DETECT_THRESHOLD}")
    print(f"        VLM API: {API_BASE}")
    print(f"        Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.5)
            continue

        frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # Run YOLO every frame
        raw, mh, mw = run_yolo(interp, inp_detail, out_details, frame)
        detections   = parse_detections(raw, orig_h, orig_w, model_h, model_w, DETECT_THRESHOLD)

        for (x1, y1, x2, y2, conf, class_id) in detections:
            activity = args.activity or CLASS_MAP.get(class_id, "phone")
            now      = time.time()

            # Cooldown: don't spam VLM for same class
            if now - last_vlm_time.get(class_id, 0) < VLM_COOLDOWN_SEC:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            last_vlm_time[class_id] = now
            print(f"\n[YOLO] frame={frame_count} activity={activity} yolo_conf={conf:.2f} box=({x1},{y1},{x2},{y2})")

            # Seatbelt: YOLO result is enough, no VLM needed
            if activity == "seatbelt":
                verified = conf >= 0.6
                print(f"[SEATBELT] {'✅ WORN' if verified else '⚠️  NOT WORN'}  yolo_conf={conf:.2f}")
                if args.show:
                    color = (0, 255, 0) if verified else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"seatbelt {'worn' if verified else 'missing'} {conf:.2f}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                continue

            result = call_vlm(crop, activity, args.driver)
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
                label = f"{activity} {conf:.2f} | VLM:{vlm_conf:.2f} {'✓' if verified else '✗'}"
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
