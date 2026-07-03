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
        "Question: Is the driver actively holding or using a mobile phone right now? "
        "A phone held to the ear, in hand while texting, or mounted but being actively touched counts as YES. "
        "An empty hand, steering wheel grip, or hand on gear shift counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true, "confidence": 0.95, "reason": "one concise sentence"}\n'
        'or\n'
        '{"verified": false, "confidence": 0.95, "reason": "one concise sentence"}'
    ),
    "cigarette": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's mouth, lips, and hand area.\n\n"
        "Question: Is the driver actively smoking a cigarette, cigar, or similar right now? "
        "Smoke visible, cigarette between fingers or lips counts as YES. "
        "No smoke or cigarette visible counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true, "confidence": 0.95, "reason": "one concise sentence"}\n'
        'or\n'
        '{"verified": false, "confidence": 0.95, "reason": "one concise sentence"}'
    ),
    "food": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively eating food right now? "
        "Food item in hand, at mouth, or chewing motion counts as YES. "
        "Empty hands or no food visible counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true, "confidence": 0.95, "reason": "one concise sentence"}\n'
        'or\n'
        '{"verified": false, "confidence": 0.95, "reason": "one concise sentence"}'
    ),
    "drink": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively drinking from a cup, bottle, or can right now? "
        "Container raised toward mouth or drinking motion counts as YES. "
        "No container visible or hands on wheel counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true, "confidence": 0.95, "reason": "one concise sentence"}\n'
        'or\n'
        '{"verified": false, "confidence": 0.95, "reason": "one concise sentence"}'
    ),
    "drowsy": (
        "This is a cropped image of a vehicle driver's face taken from a dashcam or cabin camera. "
        "Look carefully at the driver's eyes.\n\n"
        "Question: Are the driver's eyes closed or nearly closed right now, indicating drowsiness? "
        "Fully or mostly closed eyelids count as YES (drowsy). "
        "Open eyes — even if looking away, blinking, or squinting — count as NO (not drowsy).\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true, "confidence": 0.95, "reason": "one concise sentence"}\n'
        'or\n'
        '{"verified": false, "confidence": 0.95, "reason": "one concise sentence"}'
    ),
    "seatbelt": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera, "
        "possibly from an angled, side, or partially obstructed view. Scan the driver's entire "
        "chest, shoulder, and lap area for any sign of a seatbelt — a diagonal strap, webbing, "
        "or buckle. At an angle, the strap can look thin, faint, low-contrast against clothing, "
        "or partially cut off by the crop — look carefully before concluding it's absent.\n\n"
        "Question: Is the driver NOT wearing a seatbelt right now? "
        "Answer YES (violation, not wearing) only if no strap, webbing, or buckle is visible "
        "anywhere across the chest, shoulder, or lap. "
        "If ANY part of a strap or buckle is visible — even faint, partial, or at an unusual "
        "angle — answer NO (belt is worn); it is better to miss a real violation than to "
        "falsely flag a driver who is already wearing their seatbelt.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true, "confidence": 0.95, "reason": "one concise sentence"}\n'
        'or\n'
        '{"verified": false, "confidence": 0.95, "reason": "one concise sentence"}'
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
        f"Is the driver doing '{activity}' right now?\n\n"
        'Answer ONLY with JSON: {"verified": true_or_false, "confidence": 0.0_to_1.0, "reason": "one sentence"}'
    )

    try:
        response = chat(
            model=model,
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  [image_b64],
            }],
            options={"temperature": 0.0, "num_predict": 60},
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
