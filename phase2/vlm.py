"""
VLM verification via Ollama /api/generate — single-pass, short prompts.
YOLO/mobile already detected the event; the VLM does a quick confirm.
Short prompts keep qwen3-vl from over-thinking (which caused empty responses).
Seatbelt is asked positively and inverted so verified=true means "violation".
Never raises.
"""

import json
import logging
import os
import re
import time

import requests

log = logging.getLogger(__name__)

_OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/")

_JSON_INSTRUCTION = (
    "Reply with ONLY this JSON, nothing else: "
    '{"verified": true or false, "confidence": 0.0 to 1.0, "reason": "brief reason"}'
)

# ── Short detection prompts (long prompts made qwen3-vl think itself empty) ───
_PROMPTS: dict[str, str] = {
    "phone": (
        "Look at this driver image. Is the driver holding or using a mobile phone "
        "(in hand or at ear)? " + _JSON_INSTRUCTION
    ),
    "cigarette": (
        "Look at this driver image. Is the driver smoking a cigarette, cigar, bidi, or "
        "vape (visible at lips or in hand)? " + _JSON_INSTRUCTION
    ),
    "food": (
        "Look at this driver image. Is the driver eating food (food item visible in hand "
        "or at mouth)? " + _JSON_INSTRUCTION
    ),
    "drink": (
        "Look at this driver image. Is the driver drinking from a cup, bottle, or can? "
        + _JSON_INSTRUCTION
    ),
    "drowsy": (
        "Look at this driver's face. Are the eyes closed or nearly closed showing "
        "drowsiness/sleepiness (not just a brief blink)? " + _JSON_INSTRUCTION
    ),
    "seatbelt": (
        "Look at this driver image. Is a seatbelt strap visible crossing diagonally across "
        "the chest/torso? Answer YES if any strap is visible, NO if clearly no strap. "
        + _JSON_INSTRUCTION
    ),
}

_GENERIC_PROMPT = (
    "Look at this driver image. Is the driver clearly doing '{activity}'? "
    + _JSON_INSTRUCTION
)

# ── Seatbelt inversion: prompt asks "is belt worn?", pipeline wants "violation?" ─
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
    raw = ""
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{_OLLAMA_BASE}/api/generate",
                json={
                    "model":   model,
                    "prompt":  prompt,
                    "images":  [image_b64],
                    "stream":  False,
                    "options": {"temperature": 0.0 if attempt == 0 else 0.3,
                                "num_predict": 400},
                },
                timeout=90,
            )
            resp.raise_for_status()
            raw = (resp.json().get("response") or "").strip()
        except Exception as e:
            log.warning("VLM request error [%s/%s a%d]: %s", activity, pass_name, attempt, e)
            raw = ""
        dt = time.perf_counter() - t0
        log.info("VLM call [%s/%s a%d] took %.1fs, %d chars: %s",
                 activity, pass_name, attempt, dt, len(raw), raw[:100])
        if raw:
            result = _parse(raw)
            if result:
                return _apply_inversion(activity, result)
    return {"verified": False, "confidence": 0.0,
            "reason": f"unparseable_vlm_output: {raw[:80]}"}


def verify(activity: str, image_b64: str, model: str = "qwen3-vl:8b") -> dict:
    """Single-pass: mobile/YOLO detected, the VLM confirms or rejects. One VLM call."""
    prompt = _PROMPTS.get(activity) or _GENERIC_PROMPT.format(activity=activity)
    try:
        result = _ask(prompt, image_b64, model, activity, "verify")
        if not result["verified"]:
            return result
        if _reason_contradicts(activity, result["reason"]):
            log.info("VLM verdict overturned by reason gate [%s]: %r", activity, result["reason"])
            return {
                "verified":   False,
                "confidence": min(result["confidence"], 0.3),
                "reason":     f"rejected (reason contradicts verdict): {result['reason']}",
            }
        return result
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as exc:
        log.error("VLM call failed for '%s': %s", activity, exc)
        return {"verified": False, "confidence": 0.0, "reason": f"vlm_error: {exc}"}