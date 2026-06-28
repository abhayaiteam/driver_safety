"""
VLM verification via Ollama.

Design for accuracy:
- Prompts are tightly scoped to the driver-in-vehicle context
- Two-sentence chain: first ask what IS visible, then ask binary question
  → forces the model to ground its answer in observation, not assumption
- temperature=0 for deterministic output
- num_predict capped at 100 for speed without sacrificing JSON completeness
"""

import json
import logging
import re
import sys

from ollama import ResponseError, chat

log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────
# Each prompt has two parts separated by \n\n:
#   1. Grounding: describe what you actually see (prevents hallucination)
#   2. Binary question: answer in strict JSON

_PROMPTS: dict[str, str] = {
    "phone": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands, face, and ear area.\n\n"
        "Question: Is the driver actively holding or using a mobile phone right now? "
        "A phone held to the ear, in hand while texting, or mounted but being actively touched counts as YES. "
        "An empty hand, steering wheel grip, or hand on gear shift counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true_or_false, "confidence": 0.0_to_1.0, "reason": "one concise sentence"}'
    ),
    "cigarette": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's mouth, lips, and hand area.\n\n"
        "Question: Is the driver actively smoking a cigarette, cigar, or similar right now? "
        "Smoke visible, cigarette between fingers or lips counts as YES. "
        "No smoke or cigarette visible counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true_or_false, "confidence": 0.0_to_1.0, "reason": "one concise sentence"}'
    ),
    "food": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively eating food right now? "
        "Food item in hand, at mouth, or chewing motion counts as YES. "
        "Empty hands or no food visible counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true_or_false, "confidence": 0.0_to_1.0, "reason": "one concise sentence"}'
    ),
    "drink": (
        "This is a cropped image of a vehicle driver taken from a dashcam or cabin camera. "
        "Look carefully at the driver's hands and mouth area.\n\n"
        "Question: Is the driver actively drinking from a cup, bottle, or can right now? "
        "Container raised toward mouth or drinking motion counts as YES. "
        "No container visible or hands on wheel counts as NO.\n\n"
        'Answer ONLY with this exact JSON (no markdown, no extra text):\n'
        '{"verified": true_or_false, "confidence": 0.0_to_1.0, "reason": "one concise sentence"}'
    ),
}


def _parse(raw: str) -> dict | None:
    """Extract and parse the JSON block from VLM output. Returns None on failure."""
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # Greedy match: first { to last } — captures complete JSON even with newlines
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
    Ask the VLM a binary, driver-context question about the cropped image.

    Returns:
        {
          "verified":   bool,
          "confidence": float (0.0–1.0),
          "reason":     str
        }

    Never raises — on any error returns verified=False, confidence=0.0.
    """
    prompt = _PROMPTS.get(activity)
    if not prompt:
        prompt = (
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
            options={
                "temperature": 0.0,
                "num_predict": 100,
            },
        )
        raw = response["message"]["content"].strip()
        log.debug("VLM raw response for '%s': %s", activity, raw[:200])

        result = _parse(raw)
        if result:
            return result

        # Fallback: plain-text heuristic
        lower = raw.lower()
        verified = "true" in lower and "false" not in lower
        log.warning("VLM JSON parse failed for '%s', using heuristic. raw=%s", activity, raw[:120])
        return {"verified": verified, "confidence": 0.5, "reason": raw[:120]}

    except (ResponseError, ConnectionError, TimeoutError, ValueError) as exc:
        log.error("VLM call failed for '%s': %s", activity, exc)
        return {"verified": False, "confidence": 0.0, "reason": f"vlm_error: {exc}"}
