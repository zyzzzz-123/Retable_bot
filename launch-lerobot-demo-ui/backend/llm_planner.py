"""
LLM Vision Planner — uses OpenRouter (Gemini) to plan which pipeline stages to run.

Reads the front camera frame, sends it to the LLM with a structured prompt,
and returns which objects still need to be handled by the robot.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

from config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PLANNABLE_OBJECTS,
    FRAME_DIR,
    PIPELINE_STAGES,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Prompt
# ═══════════════════════════════════════════════════════════════════════

PLANNING_PROMPT = """\
You are a visual planner for a table-clearing robot. Look at this photo of a table and determine the status of each object listed below.

Context: The robot arm is visible in the image. The **original/starting positions** of the lemon, tissue box, and cup are on the **left side of the robot arm**. When these objects have been tidied, they will have been moved away from the left side (e.g. into a box on the right, or off the table entirely).

For each object, determine one of THREE statuses:
- **"todo"** — The object IS visible at its original position and needs to be moved.
- **"done"** — The object has clearly been moved away from its original position (tidied).
- **"not_found"** — The object is NOT visible anywhere in the image (not present on the table).

Objects to evaluate:
1. **Lemon** — A yellow lemon. Original position: left side of the robot arm. If you can see a lemon on the left side → "todo". If it has been moved to a box or away from the left → "done". If no lemon is visible at all → "not_found".
2. **Tissue** — A tissue box. Original position: left side of the robot arm. If you can see a tissue box on the left side → "todo". If it has been moved away → "done". If no tissue box is visible → "not_found".
3. **Cup** — A water cup. Original position: left side of the robot arm. If you can see a cup on the left side → "todo". If it has been moved away → "done". If no cup is visible → "not_found".
4. **Cloth** — A cleaning cloth / rag. Look specifically at the **bottom-right area** of the image. If you can see a cloth/rag in the bottom-right corner → "todo". If there is NO cloth in the bottom-right corner → "not_found".

Respond with ONLY the following JSON, no other text:
{
  "Lemon": {"status": "todo" or "done" or "not_found", "reason": "brief reason"},
  "Tissue": {"status": "todo" or "done" or "not_found", "reason": "brief reason"},
  "Cup": {"status": "todo" or "done" or "not_found", "reason": "brief reason"},
  "Cloth": {"status": "todo" or "done" or "not_found", "reason": "brief reason"}
}
"""


# ═══════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ObjectStatus:
    status: str  # "done" | "todo" | "not_found"
    reason: str = ""


@dataclass
class PlanResult:
    objects: dict[str, ObjectStatus] = field(default_factory=dict)
    stages_to_run: list[str] = field(default_factory=list)
    raw_response: str = ""


class LLMPlannerError(Exception):
    """Raised when LLM planning fails (API error, parse error, etc.)."""
    pass


# ═══════════════════════════════════════════════════════════════════════
#  Core planner
# ═══════════════════════════════════════════════════════════════════════

async def plan_from_camera(frame_path: Optional[str] = None) -> PlanResult:
    """Read the front camera frame and ask the LLM which stages to run.

    Args:
        frame_path: Path to JPEG frame. Defaults to FRAME_DIR/front.jpg.

    Returns:
        PlanResult with per-object status and ordered list of stages to run.

    Raises:
        LLMPlannerError: If the API call fails or the response cannot be parsed.
    """
    if frame_path is None:
        frame_path = os.path.join(FRAME_DIR, "front.jpg")

    # ── Read and encode image ──
    if not os.path.exists(frame_path):
        raise LLMPlannerError(f"Camera frame not found: {frame_path}")

    try:
        with open(frame_path, "rb") as f:
            img_bytes = f.read()
        if len(img_bytes) < 100:
            raise LLMPlannerError(f"Camera frame too small ({len(img_bytes)} bytes), possibly corrupt")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except OSError as e:
        raise LLMPlannerError(f"Failed to read camera frame: {e}")

    # ── Call OpenRouter API ──
    logger.info(f"Calling LLM planner ({LLM_MODEL})...")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PLANNING_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_b64}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
            )
    except httpx.TimeoutException:
        raise LLMPlannerError("LLM API call timed out (20s)")
    except httpx.RequestError as e:
        raise LLMPlannerError(f"LLM API request failed: {e}")

    if resp.status_code != 200:
        raise LLMPlannerError(
            f"LLM API returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    # ── Parse response ──
    try:
        api_data = resp.json()
    except json.JSONDecodeError:
        raise LLMPlannerError(f"LLM API returned invalid JSON: {resp.text[:300]}")

    # Extract the assistant message content
    try:
        content = api_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMPlannerError(f"Unexpected API response structure: {json.dumps(api_data)[:300]}")

    logger.info(f"LLM raw response: {content}")

    # ── Parse the JSON from the LLM response ──
    plan_json = _extract_json(content)
    if plan_json is None:
        raise LLMPlannerError(f"Could not extract JSON from LLM response: {content[:300]}")

    # ── Build PlanResult ──
    result = PlanResult(raw_response=content)

    for obj_name in LLM_PLANNABLE_OBJECTS:
        if obj_name not in plan_json:
            raise LLMPlannerError(
                f"LLM response missing object '{obj_name}'. Got: {list(plan_json.keys())}"
            )
        obj_data = plan_json[obj_name]
        status = obj_data.get("status", "").lower().strip().replace(" ", "_")
        if status not in ("done", "todo", "not_found"):
            raise LLMPlannerError(
                f"Invalid status for '{obj_name}': '{status}'. Expected 'done', 'todo', or 'not_found'."
            )
        reason = obj_data.get("reason", "")
        result.objects[obj_name] = ObjectStatus(status=status, reason=reason)

    # Build stages_to_run: only "todo" objects, in original PIPELINE_STAGES order
    todo_names = {name for name, obj in result.objects.items() if obj.status == "todo"}
    result.stages_to_run = [
        s["name"] for s in PIPELINE_STAGES if s["name"] in todo_names
    ]

    logger.info(f"LLM plan: {_plan_summary(result)}")
    logger.info(f"Stages to run: {result.stages_to_run}")

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from LLM response text.

    Handles cases where the LLM wraps JSON in markdown code fences or
    adds extra text before/after.
    """
    import re

    # Try 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try 2: extract from markdown code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: find first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _plan_summary(result: PlanResult) -> str:
    """One-line summary for logging."""
    parts = []
    for name, obj in result.objects.items():
        icon = "✅" if obj.status == "done" else "⏳" if obj.status == "todo" else "❌"
        parts.append(f"{icon} {name}={obj.status}")
    return " | ".join(parts)


def plan_to_dict(result: PlanResult) -> dict:
    """Convert PlanResult to a JSON-serializable dict for WebSocket broadcast."""
    return {
        name: {"status": obj.status, "reason": obj.reason}
        for name, obj in result.objects.items()
    }
