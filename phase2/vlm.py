"""
VLM verification via Ollama — two-pass cross-checked version.

Pipeline per detection:
  Pass 1 (strict prompt)  → must answer verified=true with an evidence-bearing reason
  Reason consistency gate → reason must actually mention the object; contradictions flip to false
  Pass 2 (skeptical confirmation prompt, only if pass 1 was positive)
                          → an independent re-check framed to hunt for false positives
  Final verdict           → verified only if BOTH passes agree; confidence = min(both)

Never raises — returns verified=False, confidence=0.0 on any error.
"""

import json
import logging
import re

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
# Every prompt follows the same rule: YES requires the object/state to be
# CLEARLY identifiable; anything ambiguous or uncertain is NO.

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
        "possibly from an angled, side, or partially obstructed view. "
        "A worn seatbelt looks like a straight or slightly diagonal strap of fabric webbing "
        "(usually black, grey, or beige, about 5-8 cm wide) running from near one shoulder "
        "diagonally across the chest down to the opposite hip, sometimes with a metal or "
        "plastic buckle/latch plate visible where it crosses the body. It often contrasts in "
        "color or texture with the driver's clothing. At an angle it can look thin, faint, "
        "low-contrast, or partially cut off by the crop — scan whatever part of the driver's "
        "chest, shoulder, and lap is visible carefully before concluding no strap is present.\n\n"
        "Question: Is the driver NOT wearing a seatbelt right now? "
        "Answer YES (violation, not wearing) only if a meaningful portion of the chest, "
        "shoulder, or lap area is clearly visible and you are certain no strap, webbing, or "
        "buckle matching that description appears anywhere in it. "
        "Answer NO (belt is worn) if ANY part of a strap or buckle is visible anywhere — even "
        "faint, partial, or at an unusual angle. "
        "Also answer NO if the chest/shoulder/lap area is too small, dark, or blurry to judge "
        "reliably (e.g. a tight face-only close-up) — say so explicitly in the reason when "
        "that's the case.\n\n" + _JSON_INSTRUCTION
    ),
}

# ── Pass 2: skeptical confirmation prompts (run only after a positive pass 1) ─

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
        "This is a cropped image of a vehicle driver. An automatic system flagged this "
        "driver as NOT WEARING A SEATBELT, but such systems often miss belts that are "
        "faint, low-contrast, the same color as the clothing, partially hidden by an arm or "
        "the crop edge, or seen at an unusual angle.\n\n"
        "Your job is to double-check skeptically. Scan the entire visible chest, shoulder, "
        "and lap area for ANY trace of a strap, webbing, or buckle. "
        "Answer YES (confirmed: no seatbelt) only if the chest/shoulder area is clearly "
        "visible and you are certain there is no strap anywhere. "
        "Answer NO if any part of a belt is visible, or if the area is too small, dark, or "
        "blurry to be certain.\n\n" + _JSON_INSTRUCTION
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

# ── Reason-consistency gate ──────────────────────────────────────────────────
# For object-presence classes, a verified=true answer must actually mention the
# object in its reason, and must not simultaneously deny it. Catches LLaVA's
# habit of returning verified=true with a reason like "no phone is visible".
# Not applied to seatbelt/drowsy (their positive reason legitimately describes
# an ABSENCE, e.g. "no strap visible").

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
        # verified=true but the reason never mentions the object at all
        return True
    # object mentioned, but in a negated form: "no phone", "not holding a phone", ...
    for w in mentioned:
        for neg in ("no ", "not ", "without ", "isn't ", "is not ", "cannot see", "can't see",
                    "doesn't ", "does not ", "unable to see", "absence of "):
            idx = lower.find(neg)
            while idx != -1:
                # negation within ~40 chars before the evidence word counts as denial
                w_idx = lower.find(w, idx)
                if w_idx != -1 and 0 <= w_idx - idx <= 40:
                    return True
                idx = lower.find(neg, idx + 1)
    return False


def _parse(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
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
    """One VLM round-trip. On JSON-parse failure, falls back to a conservative
    heuristic that can never confirm a detection on its own (verified=False)."""
    response = chat(
        model=model,
        messages=[{
            "role":    "user",
            "content": prompt,
            "images":  [image_b64],
        }],
        options={"temperature": 0.0, "num_predict": 160},
    )
    raw = response["message"]["content"].strip()
    log.debug("VLM raw [%s/%s]: %s", activity, pass_name, raw[:200])

    result = _parse(raw)
    if result:
        return result

    # JSON parse failed — plain-text fallback. Be conservative: an unparseable
    # answer is never allowed to confirm an alert by itself.
    log.warning("VLM JSON parse failed [%s/%s]; treating as NOT verified. raw=%s",
                activity, pass_name, raw[:120])
    return {"verified": False, "confidence": 0.0,
            "reason": f"unparseable_vlm_output: {raw[:100]}"}


def verify(activity: str, image_b64: str, model: str = "llava:7b") -> dict:
    """
    Two-pass cross-checked verification.

    Pass 1: strict detection prompt. If it answers NO (or its reason contradicts
    the positive verdict), the detection is rejected immediately.
    Pass 2: independent skeptical confirmation. The detection passes only if
    BOTH passes answer YES. Final confidence = min of the two passes.

    Never raises — returns verified=False, confidence=0.0 on any error.
    """
    prompt  = _PROMPTS.get(activity) or _GENERIC_PROMPT.format(activity=activity)
    confirm = _CONFIRM_PROMPTS.get(activity) or _GENERIC_CONFIRM_PROMPT.format(activity=activity)

    try:
        # ── Pass 1: strict detection ─────────────────────────────────────
        first = _ask(prompt, image_b64, model, activity, "pass1")

        if not first["verified"]:
            return first

        # Reason-consistency gate: verified=true must be backed by the reason
        if _reason_contradicts(activity, first["reason"]):
            log.info("VLM verdict overturned by reason-consistency gate [%s]: %r",
                     activity, first["reason"])
            return {
                "verified":   False,
                "confidence": min(first["confidence"], 0.3),
                "reason":     f"rejected (reason contradicts verdict): {first['reason']}",
            }

        # ── Pass 2: skeptical confirmation ───────────────────────────────
        second = _ask(confirm, image_b64, model, activity, "pass2-confirm")

        if not second["verified"] or _reason_contradicts(activity, second["reason"]):
            log.info("VLM detection rejected on confirmation pass [%s]: %r",
                     activity, second["reason"])
            return {
                "verified":   False,
                "confidence": min(first["confidence"], second["confidence"]),
                "reason":     f"failed skeptical re-check: {second['reason'] or first['reason']}",
            }

        # Both passes agree → confirmed
        return {
            "verified":   True,
            "confidence": min(first["confidence"], second["confidence"]),
            "reason":     second["reason"] or first["reason"],
        }

    except (ResponseError, ConnectionError, TimeoutError, ValueError) as exc:
        log.error("VLM call failed for '%s': %s", activity, exc)
        return {"verified": False, "confidence": 0.0, "reason": f"vlm_error: {exc}"}