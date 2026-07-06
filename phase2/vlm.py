"""
VLM verification via Ollama.
Standalone copy — no dependency on parent directories.
"""

import json
import logging
import re

from ollama import ResponseError, chat

log = logging.getLogger(__name__)

_PROMPTS: dict[str, str] = {
    "phone": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands, face, and ear area.\n\n"
        "Question: Is the driver holding or actively using a mobile phone right now, for ANY "
        "reason — calling, texting, browsing, taking a photo or selfie, or anything else? "
        "A phone visible in the driver's hand, held to the ear, held up toward the face, or "
        "mounted but being actively touched all count as YES. "
        "Answer NO only if no phone is visible at all — empty hand, steering wheel grip, or "
        "hand on gear shift.\n\n"
        "First write down exactly what you observe, then decide based on that.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": true, "confidence": 0.95}\n'
        'or\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": false, "confidence": 0.95}'
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
        "answer NO.\n\n"
        "First write down exactly what you observe, then decide based on that.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": true, "confidence": 0.95}\n'
        'or\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": false, "confidence": 0.95}'
    ),
    "food": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively eating food right now? "
        "Food item in hand, at mouth, or chewing motion counts as YES. "
        "Empty hands or no food visible counts as NO.\n\n"
        "First write down exactly what you observe, then decide based on that.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": true, "confidence": 0.95}\n'
        'or\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": false, "confidence": 0.95}'
    ),
    "drink": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively drinking from a cup, bottle, or can right now? "
        "Container raised toward mouth or drinking motion counts as YES. "
        "No container visible or hands on wheel counts as NO.\n\n"
        "First write down exactly what you observe, then decide based on that.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": true, "confidence": 0.95}\n'
        'or\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": false, "confidence": 0.95}'
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
        "First write down exactly what you observe, then decide based on that.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": true, "confidence": 0.95}\n'
        'or\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": false, "confidence": 0.95}'
    ),
    "seatbelt": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera, "
        "possibly from an angled, side, or partially obstructed view. "
        "A worn seatbelt looks like a straight or slightly diagonal strap of fabric webbing "
        "(usually black, grey, or beige, about 5-8 cm wide) running from near one shoulder "
        "diagonally across the chest down to the opposite hip, sometimes with a metal or "
        "plastic buckle/latch plate visible where it crosses the body. It often contrasts in "
        "color or texture with the driver's clothing. At an angle it can look thin, faint, "
        "low-contrast, or partially cut off by the crop — scan the driver's entire chest, "
        "shoulder, and lap area carefully before concluding no strap is present.\n\n"
        "Question: Is the driver NOT wearing a seatbelt right now? "
        "Answer YES (violation, not wearing) only if no strap, webbing, or buckle matching that "
        "description is visible anywhere across the chest, shoulder, or lap. "
        "If ANY part of a strap or buckle is visible — even faint, partial, or at an unusual "
        "angle — answer NO (belt is worn); it is better to miss a real violation than to "
        "falsely flag a driver who is already wearing their seatbelt.\n\n"
        "First write down exactly what you observe, then decide based on that.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text), in this field order:\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": true, "confidence": 0.95}\n'
        'or\n'
        '{"reason": "one concise sentence describing exactly what you observe", "verified": false, "confidence": 0.95}'
    ),
}


def _parse(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return {
            "verified":   bool(data.get("verified", False)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            "reason":     str(data.get("reason", "")).strip(),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def verify(activity: str, image_b64: str, model: str = "llava:7b") -> dict:
    """
    Ask the VLM a binary driver-context question about the cropped image.
    Never raises — returns verified=False, confidence=0.0 on any error.
    """
    prompt = _PROMPTS.get(activity) or (
        f"This is a cropped image of a vehicle driver. "
        f"Is the driver doing '{activity}' right now? First write down exactly what you observe, "
        f"then decide based on that.\n\n"
        'Answer ONLY with JSON, in this field order: '
        '{"reason": "one sentence describing what you observe", "verified": true_or_false, "confidence": 0.0_to_1.0}'
    )

    try:
        response = chat(
            model=model,
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  [image_b64],
            }],
            options={"temperature": 0.0, "num_predict": 100},
        )
        raw = response["message"]["content"].strip()
        log.debug("VLM raw response for '%s': %s", activity, raw[:200])

        result = _parse(raw)
        if result:
            return result

        # JSON parse failed — LLaVA returned plain text instead
        # Detect negations to determine if the activity was present or not
        lower = raw.lower()
        negations = ["not ", "no ", "doesn't", "isn't", "cannot", "can't", "do not", "without", "absent"]
        has_negation = any(n in lower for n in negations)
        verified = not has_negation
        confidence = 0.3 if has_negation else 0.7
        log.warning("VLM JSON parse failed for '%s', using heuristic verified=%s. raw=%s",
                    activity, verified, raw[:120])
        return {"verified": verified, "confidence": confidence, "reason": raw[:120]}

    except (ResponseError, ConnectionError, TimeoutError, ValueError) as exc:
        log.error("VLM call failed for '%s': %s", activity, exc)
        return {"verified": False, "confidence": 0.0, "reason": f"vlm_error: {exc}"}
