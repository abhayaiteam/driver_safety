"""
VLM verification via Ollama — two-pass cross-checked version, with a fast-path
that skips pass-2 when pass-1 is highly confident (halves latency on clear cases).

Pipeline per detection:
  Pass 1 (strict prompt)  → must answer verified=true with an evidence-bearing reason
  Reason consistency gate → reason must actually mention the object; contradictions flip to false
  Fast path               → if pass-1 confidence ≥ 0.90 and passed the gate, accept immediately
  Pass 2 (skeptical)      → for borderline positives (< 0.90), independent re-check
  Final verdict           → verified only if both agree; confidence = min(both)

Seatbelt is asked as the POSITIVE question ("is the belt worn?") and inverted in
code so verified=true always means "violation confirmed".

Never raises — returns verified=False, confidence=0.0 on any error.
"""

import json
import logging
import re
import time  
import os
import requests


from ollama import ResponseError, chat

log = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "First write down exactly what you observe, then decide based on that.\n\n"
    "Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n"
    '{"reason": "one concise sentence describing exactly what you observe", '
    '"verified": true or false, '
    '"confidence": your honest certainty as a number between 0.0 and 1.0 — '
    "use 0.9+ only when the evidence is unmistakable, 0.5-0.7 when partially "
    "visible or ambiguous, below 0.5 when guessing}"
)

# ── Pass 1: strict detection prompts ─────────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "phone": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands, face, and ear area.\n\n"
        "Question: Is the driver holding or actively using a mobile phone right now — "
        "calling, texting, browsing, or taking a photo?\n"
        "Answer YES only if you can CLEARLY identify an actual mobile phone: a distinct "
        "rectangular device, usually with a visible screen, flat edges, or camera bump, "
        "held in the hand, held to the ear, or held up toward the face.\n"
        "Answer NO for all of these common false alarms: an empty hand near the ear or face, "
        "scratching or touching the head, adjusting hair or glasses, resting the chin or "
        "cheek on the hand, holding the steering wheel, a wallet, sunglasses case, card, "
        "cup, food item, or any dark or blurry shape that is not clearly identifiable as a "
        "phone. If you cannot clearly see the phone itself as a distinct device, answer NO — "
        "a hand posture alone is never enough.\n\n" + _JSON_INSTRUCTION
    ),
    "cigarette": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's mouth, lips, and hand area.\n\n"
        "Question: Is the driver actively smoking a cigarette, cigar, bidi, or vape right now? "
        "Answer YES only if you can clearly identify an actual cigarette/cigar/vape "
        "(a thin stick or device held at the lips or between fingers near the mouth) or "
        "visible smoke coming from the mouth or hand. "
        "Answer NO for an empty hand, a finger, pen, phone, food, or any object that is not "
        "clearly identifiable as a cigarette/cigar/vape — if you are not certain it is one, "
        "answer NO.\n\n" + _JSON_INSTRUCTION
    ),
    "food": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively eating food right now? "
        "Answer YES only if you can clearly identify an actual food item (snack, fruit, "
        "sandwich, wrapper with food, etc.) in the hand or at the mouth, or clear chewing "
        "with food visible. "
        "Answer NO for an empty hand near the mouth, yawning, talking, a phone, cup, "
        "cigarette, or any object not clearly identifiable as food — if you are not certain "
        "it is food, answer NO.\n\n" + _JSON_INSTRUCTION
    ),
    "drink": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively drinking from a cup, bottle, or can right now? "
        "Answer YES only if you can clearly identify an actual cup, bottle, can, or flask "
        "held in the hand or raised toward the mouth. "
        "Answer NO for an empty hand, a phone, food, or any object not clearly identifiable "
        "as a drink container — if you are not certain it is a drink container, answer "
        "NO.\n\n" + _JSON_INSTRUCTION
    ),
    "drowsy": (
        "This is a cropped image of a vehicle driver's face taken from a dashcam or cabin camera.\n\n"
        "Question: Is this driver showing signs of drowsiness or falling asleep right now — "
        "purely from their eyes and facial state, NOT because they are looking down at "
        "something? "
        "Answer YES only if the eyes/eyelids themselves show drowsiness: eyes closed or "
        "nearly closed, heavy drooping eyelids, or a slack, sleepy expression with the head "
        "lolling as if dozing off with no purposeful engagement. "
        "Answer NO if the driver is engaged in any other activity, even if the head is tilted "
        "down or the eyes look partly shut because of it — holding or using a phone (call, "
        "text, browse), smoking a cigarette/cigar/vape, eating food, or drinking from a cup, "
        "bottle, or can. Those are their own separate activities, not drowsiness, even if the "
        "posture looks similar to dozing off. "
        "Answer NO if the driver looks clearly alert and awake — eyes open and actively "
        "looking forward, or just a single quick blink. A brief glance down or sideways while "
        "still alert also counts as NO. "
        "If the eyes or head position are not visible at all in the frame, answer NO.\n\n"
        + _JSON_INSTRUCTION
    ),
    "seatbelt": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera, "
        "possibly angled or partially cropped.\n\n"
        "Follow these steps in order:\n"
        "Step 1: Locate the driver's visible torso — shoulder, chest, and lap, whatever is in frame.\n"
        "Step 2: Look for a diagonal strap of fabric webbing (black, grey, or beige, about "
        "5-8 cm wide) running from near one shoulder diagonally across the chest toward the "
        "opposite hip. It may be faint, low-contrast, similar in color to the clothing, or "
        "partially cut off by the crop — trace slowly across the whole visible torso.\n"
        "Step 3: Check for a metal or plastic buckle/latch plate where the strap crosses the body.\n\n"
        "Question: Is the driver WEARING a seatbelt?\n"
        "Answer YES (verified=true) if ANY part of a diagonal strap or buckle is visible — "
        "even faint, partial, at an unusual angle, or cut off by the crop.\n"
        "Answer NO (verified=false) only if the shoulder-to-chest region is clearly visible "
        "AND you are certain no strap crosses it anywhere.\n"
        "If almost none of the torso is visible (e.g. a tight face-only crop), answer YES "
        "and state that visibility was insufficient in the reason.\n\n" + _JSON_INSTRUCTION
    ),
}

# ── Pass 2: skeptical confirmation prompts ───────────────────────────────────

_CONFIRM_PROMPTS: dict[str, str] = {
    "phone": (
        "This is a cropped image of a vehicle driver. An automatic system flagged this "
        "driver for PHONE USE, but such systems very often raise FALSE alarms when the "
        "driver merely has a hand near the ear or face, scratches their head, adjusts hair "
        "or glasses, rests the chin on the hand, or holds a wallet, card, or other small "
        "object.\n\n"
        "Your job is to double-check skeptically. Answer YES only if you can point to an "
        "actual, clearly visible mobile phone — a distinct rectangular device — in the "
        "driver's hand or at their ear/face. If the 'phone' could plausibly be an empty "
        "hand, another object, or an unclear blur, answer NO.\n\n" + _JSON_INSTRUCTION
    ),
    "cigarette": (
        "This is a cropped image of a vehicle driver. An automatic system flagged this "
        "driver for SMOKING, but such systems often raise FALSE alarms for a finger near "
        "the lips, a pen, food, a toothpick, or a hand resting on the mouth.\n\n"
        "Your job is to double-check skeptically. Answer YES only if you can point to an "
        "actual, clearly visible cigarette, cigar, bidi, or vape device (or clear smoke). "
        "If it could plausibly be anything else, answer NO.\n\n" + _JSON_INSTRUCTION
    ),
    "food": (
        "This is a cropped image of a vehicle driver. An automatic system flagged this "
        "driver for EATING, but such systems often raise FALSE alarms for a hand near the "
        "mouth, yawning, talking, or holding a phone or cup.\n\n"
        "Your job is to double-check skeptically. Answer YES only if you can point to an "
        "actual, clearly visible food item. If it could plausibly be anything else, answer "
        "NO.\n\n" + _JSON_INSTRUCTION
    ),
    "drink": (
        "This is a cropped image of a vehicle driver. An automatic system flagged this "
        "driver for DRINKING, but such systems often raise FALSE alarms for a phone held "
        "up, a hand near the face, or other objects.\n\n"
        "Your job is to double-check skeptically. Answer YES only if you can point to an "
        "actual, clearly visible cup, bottle, can, or flask. If it could plausibly be "
        "anything else, answer NO.\n\n" + _JSON_INSTRUCTION
    ),
    "drowsy": (
        "This is a cropped image of a vehicle driver's face. An automatic system flagged "
        "this driver as DROWSY, but such systems often raise FALSE alarms when the driver "
        "is simply blinking, glancing down briefly, looking at a phone, smoking, eating, or "
        "drinking.\n\n"
        "Your job is to double-check skeptically. Answer YES only if the eyes and face "
        "genuinely show sleepiness — eyes closed or nearly closed with heavy eyelids, or a "
        "slack dozing expression — and the driver is NOT engaged in any other activity. "
        "Otherwise answer NO.\n\n" + _JSON_INSTRUCTION
    ),
    "seatbelt": (
        "This is a cropped image of a vehicle driver. An automatic system believes this "
        "driver has no seatbelt on, but such systems often miss belts that are faint, "
        "low-contrast, the same color as the clothing, partially hidden by an arm, or cut "
        "off by the crop edge.\n\n"
        "Follow these steps in order:\n"
        "Step 1: Locate the driver's visible shoulder, chest, and lap areas.\n"
        "Step 2: Trace slowly from each shoulder diagonally down across the chest, hunting "
        "for ANY continuous strap, webbing edge, or buckle — even a short, faint segment counts.\n\n"
        "Question: Is the driver WEARING a seatbelt?\n"
        "Answer YES (verified=true) if any trace of a strap or buckle is visible anywhere, "
        "or if too little of the torso is visible to judge.\n"
        "Answer NO (verified=false) only if the shoulder and chest are clearly visible and "
        "you are certain no strap crosses them anywhere. "
        "Image cropping is NOT evidence of a missing belt.\n\n" + _JSON_INSTRUCTION
    ),
}

_GENERIC_PROMPT = (
    "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera.\n\n"
    "Question: Is the driver clearly doing '{activity}' right now? "
    "Answer YES only if the evidence for this specific activity is clearly visible and "
    "unmistakable in the image. If the evidence is ambiguous, partially visible, or you are "
    "not certain, answer NO.\n\n" + _JSON_INSTRUCTION
)

_GENERIC_CONFIRM_PROMPT = (
    "This is a cropped image of a vehicle driver. An automatic system flagged this driver "
    "for '{activity}', but such systems often raise false alarms.\n\n"
    "Your job is to double-check skeptically. Answer YES only if clear, unmistakable visual "
    "evidence of '{activity}' is present. If it could plausibly be something else, answer "
    "NO.\n\n" + _JSON_INSTRUCTION
)



_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/") + "/api/generate"


def _ask(prompt: str, image_b64: str, model: str, activity: str, pass_name: str) -> dict:
    """One VLM round-trip via Ollama's /api/generate (the /api/chat path returns
    empty responses with qwen3-vl). Up to 2 attempts; conservative fallback."""
    raw = ""
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            resp = requests.post(_OLLAMA_URL, json={
                "model":   model,
                "prompt":  prompt,
                "images":  [image_b64],
                "stream":  False,
                "options": {"temperature": 0.0 if attempt == 0 else 0.2,
                            "num_predict": 256},
            }, timeout=60)
            raw = (resp.json().get("response") or "").strip()
        except Exception as e:
            log.warning("VLM request failed [%s/%s a%d]: %s", activity, pass_name, attempt, e)
            raw = ""
        dt = time.perf_counter() - t0
        log.info("VLM call [%s/%s a%d] took %.1fs, got %d chars: %s",
                 activity, pass_name, attempt, dt, len(raw), raw[:120])
        if raw:
            result = _parse(raw)
            if result:
                return _apply_inversion(activity, result)
        log.warning("VLM parse/empty fail [%s/%s a%d]: %r", activity, pass_name, attempt, raw[:150])

    return {"verified": False, "confidence": 0.0,
            "reason": f"unparseable_vlm_output: {raw[:100]}"}

# ── Inverted activities ──────────────────────────────────────────────────────

_INVERTED_ACTIVITIES = {"seatbelt"}


def _apply_inversion(activity: str, result: dict) -> dict:
    if activity in _INVERTED_ACTIVITIES:
        result["verified"] = not result["verified"]
    return result


# ── Reason-consistency gate ──────────────────────────────────────────────────

_EVIDENCE_WORDS: dict[str, tuple[str, ...]] = {
    "phone":     ("phone", "mobile", "smartphone", "cell", "device"),
    "cigarette": ("cigarette", "cigar", "bidi", "vape", "smok", "e-cig"),
    "food":      ("food", "eat", "snack", "sandwich", "fruit", "burger", "chew", "bite"),
    "drink":     ("drink", "cup", "bottle", "can", "flask", "sip", "beverage"),
}


def _reason_contradicts(activity: str, reason: str) -> bool:
    """True when a verified=true reason actually denies or fails to evidence the object."""
    words = _EVIDENCE_WORDS.get(activity)
    if not words:
        return False
    lower = reason.lower()
    mentioned = [w for w in words if w in lower]
    if not mentioned:
        return True
    for w in mentioned:
        for neg in ("no ", "not ", "without ", "isn't ", "is not ", "cannot see", "can't see",
                    "doesn't ", "does not ", "unable to see", "absence of "):
            idx = lower.find(neg)
            while idx != -1:
                w_idx = lower.find(w, idx)
                if w_idx != -1 and 0 <= w_idx - idx <= 40:
                    return True
                idx = lower.find(neg, idx + 1)
    return False


def _parse(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    matches = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
    candidate = matches[-1] if matches else None
    if candidate is None:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        candidate = m.group() if m else None
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
        verified = data.get("verified", False)
        if isinstance(verified, str):
            verified = verified.strip().lower() in ("true", "yes", "1")
        return {
            "verified":   bool(verified),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            "reason":     str(data.get("reason", "")).strip(),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

def _ask(prompt: str, image_b64: str, model: str, activity: str, pass_name: str) -> dict:
    """One VLM round-trip with up to 2 attempts. On repeated failure, returns a
    conservative result that never confirms a detection (verified=False)."""
    raw = ""
    for attempt in range(2):
        t0 = time.perf_counter()
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
            options={"temperature": 0.0 if attempt == 0 else 0.2,
                     "num_predict": 256},
        )
        dt = time.perf_counter() - t0
        raw = (response.get("message", {}).get("content") or "").strip()
        log.info("VLM call [%s/%s a%d] took %.1fs, got %d chars: %s",
                 activity, pass_name, attempt, dt, len(raw), raw[:120])
        if raw:
            result = _parse(raw)
            if result:
                return _apply_inversion(activity, result)
        log.warning("VLM parse/empty fail [%s/%s a%d]: %r", activity, pass_name, attempt, raw[:150])

    return {"verified": False, "confidence": 0.0,
            "reason": f"unparseable_vlm_output: {raw[:100]}"}


def verify(activity: str, image_b64: str, model: str = "llava:7b") -> dict:
    """
    Two-pass cross-checked verification with a high-confidence fast path.

    Pass 1: strict detection. NO or contradictory reason → rejected immediately.
    Fast path: pass-1 confidence ≥ 0.90 and reason gate passed → accept, skip pass 2.
    Pass 2: for borderline positives (< 0.90), skeptical re-check; both must agree.

    Never raises — returns verified=False, confidence=0.0 on any error.
    """
    prompt  = _PROMPTS.get(activity) or _GENERIC_PROMPT.format(activity=activity)
    confirm = _CONFIRM_PROMPTS.get(activity) or _GENERIC_CONFIRM_PROMPT.format(activity=activity)

    try:
        # ── Pass 1: strict detection ─────────────────────────────────────
        first = _ask(prompt, image_b64, model, activity, "pass1")

        if not first["verified"]:
            return first

        # Reason-consistency gate
        if _reason_contradicts(activity, first["reason"]):
            log.info("VLM verdict overturned by reason-consistency gate [%s]: %r",
                     activity, first["reason"])
            return {
                "verified":   False,
                "confidence": min(first["confidence"], 0.3),
                "reason":     f"rejected (reason contradicts verdict): {first['reason']}",
            }

        # ── Fast path: highly confident pass 1 → skip pass 2 (halves latency) ─
        if first["confidence"] >= 0.90:
            log.info("VLM fast-path accept [%s] conf=%.2f (skipped pass-2): %r",
                     activity, first["confidence"], first["reason"])
            return {
                "verified":   True,
                "confidence": first["confidence"],
                "reason":     first["reason"],
            }

        # ── Pass 2: skeptical confirmation (borderline positives only) ───
        second = _ask(confirm, image_b64, model, activity, "pass2-confirm")

        if not second["verified"] or _reason_contradicts(activity, second["reason"]):
            log.info("VLM detection rejected on confirmation pass [%s]: %r",
                     activity, second["reason"])
            return {
                "verified":   False,
                "confidence": min(first["confidence"], second["confidence"]),
                "reason":     f"failed skeptical re-check: {second['reason'] or first['reason']}",
            }

        return {
            "verified":   True,
            "confidence": min(first["confidence"], second["confidence"]),
            "reason":     second["reason"] or first["reason"],
        }

    except (ResponseError, ConnectionError, TimeoutError, ValueError) as exc:
        log.error("VLM call failed for '%s': %s", activity, exc)
        return {"verified": False, "confidence": 0.0, "reason": f"vlm_error: {exc}"}