"""Groq-based screening assistant for diabetic retinopathy grading review."""

import json
import logging
import os

import httpx

from drstage.api.schemas import AssistantMessage, AssistantResponse, VisitContext
from drstage.data.registry import CLASS_NAMES

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = (
    "You are the RetinaScan screening assistant. You support trained screening staff who are "
    "reviewing diabetic retinopathy (DR) grades produced by an AI model from fundus photographs.\n"
    "Facts: before answering anything about this visit, including letters, call get_visit_summary "
    "and get_eye_details for each photographed eye, and use lookup_guideline for clinical "
    "definitions and referral timings. Only state facts that the tools returned. The model outputs "
    "a grade, probabilities, image-quality checks and a heatmap region; it does not detect "
    "individual lesions, macular oedema or other eye disease, so never claim findings like these. "
    "You cannot see the images. Describe an eye without a photograph simply as not photographed.\n"
    "Grades: never change or overrule a grade. A clinician confirms every result, and if a "
    "clinician review exists its grade is final. Point out uncertainty, ungradable images and any "
    "disagreement with the dataset's reference grade.\n"
    "Style: plain clinical English with British spelling, under about 180 words unless asked for a "
    "letter. Short paragraphs or bullet lists; no tables.\n"
    "Letters: fill in the grades, probabilities, image quality and referral timing from the "
    "tools, and use placeholders such as [Patient name], [Date of birth] and [Clinician name] for "
    "anything you were not given. Never invent patient details.\n"
    "Scope: answer only about this screening visit or diabetic retinopathy; otherwise decline in "
    "one sentence. You provide decision support, not a diagnosis."
)

GUIDELINE_CONTENT = {
    "grade_0": "No DR: no abnormalities; re-screen in 12 months.",
    "grade_1": "Mild DR: microaneurysms only; re-screen in 6 to 12 months.",
    "grade_2": (
        "Moderate DR: more than microaneurysms but less than severe "
        "(e.g. dot/blot haemorrhages, hard exudates, cotton-wool spots); "
        "refer within 3 months."
    ),
    "grade_3": (
        "Severe DR: any of the 4-2-1 rule (more than 20 intraretinal "
        "haemorrhages in each of 4 quadrants, definite venous beading in "
        "2 or more quadrants, prominent IRMA in 1 or more quadrant) and no "
        "signs of proliferation; refer within 4 weeks."
    ),
    "grade_4": (
        "Proliferative DR: neovascularisation or vitreous/preretinal "
        "haemorrhage; urgent referral within 1 week."
    ),
    "image_quality": (
        "Ungradable photographs should be retaken before grading. Quality "
        "checks cover exposure, contrast, focus and field of view, with "
        "thresholds from the 1st/5th percentile of training images."
    ),
    "heatmap": (
        "Grad-CAM shows which regions most influenced the predicted grade; "
        "it is not a lesion segmentation and can highlight the optic disc "
        "or artefacts."
    ),
    "model": (
        "ConvNeXt-Base (ImageNet-22k pretrained) fine-tuned on the DDR "
        "dataset at 512 px; on 3,381 held-out DDR test images QWK 0.938, "
        "accuracy 90.4%, any-DR sensitivity 95.4% and specificity 96.3%; "
        "weakest on mild and severe grades; on an external dataset (APTOS "
        "2019) its ranking stayed excellent (AUC 0.99) but probabilities "
        "were overconfident, so a new site should calibrate thresholds "
        "locally."
    ),
}


def _is_uncertain(class_probs: list[float]) -> bool:
    """Same rule as the UI: the top two grades are close, or the top grade is weak."""
    first, second = sorted(class_probs, reverse=True)[:2]
    return first - second < 0.2 or first < 0.5


class AssistantError(Exception):
    """Exception raised when the assistant service fails."""

    pass


class Assistant:
    """Groq-based screening assistant with tool support."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = GROQ_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        max_rounds: int = 4,
    ):
        """Initialize the assistant with Groq API credentials."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.transport = transport
        self.max_rounds = max_rounds

    async def reply(
        self, messages: list[AssistantMessage], visit: VisitContext
    ) -> AssistantResponse:
        """Get assistant reply to a list of messages in the context of a visit."""
        chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        chat.extend([{"role": m.role, "content": m.content} for m in messages])

        tools_used: list[str] = []
        tools_list = self._build_tools()

        for round_num in range(self.max_rounds):
            if round_num == self.max_rounds - 1:
                tool_choice = "none"
            elif round_num == 0:
                tool_choice = "required"
            else:
                tool_choice = "auto"

            response = await self._call_groq(chat, tools_list, tool_choice)

            if response.get("tool_calls"):
                chat.append(
                    {
                        "role": "assistant",
                        "content": response.get("content") or "",
                        "tool_calls": response["tool_calls"],
                    }
                )

                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)

                    try:
                        tool_args = json.loads(tool_call["function"].get("arguments") or "{}")
                        if not isinstance(tool_args, dict):
                            raise ValueError("arguments must be a JSON object")
                        result = self._execute_tool(tool_name, tool_args, visit)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        result = {"error": f"Invalid tool arguments: {e}"}

                    chat.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )
            else:
                break

        content = (response.get("content") or "").strip()
        if not content:
            raise AssistantError("The assistant returned an empty answer. Try asking again.")
        return AssistantResponse(reply=content, tools_used=tools_used)

    async def _call_groq(
        self, chat: list[dict], tools: list[dict], tool_choice: str, retry: bool = True
    ) -> dict:
        """Call the Groq API with the given messages and tools."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": chat,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0.2,
            "max_tokens": 2000,
        }
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"

        try:
            async with httpx.AsyncClient(transport=self.transport) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )

                if resp.status_code == 400 and "tool_use_failed" in resp.text and retry:
                    # Groq rejects replies that skip a required tool or emit a malformed call
                    logger.info(f"Groq tool_use_failed with tool_choice={tool_choice}; retrying")
                    fallback = "auto" if tool_choice == "required" else tool_choice
                    return await self._call_groq(chat, tools, fallback, retry=False)
                if resp.status_code == 429:
                    raise AssistantError(
                        "The assistant service is rate limited. Try again in a moment."
                    )
                elif resp.status_code == 401:
                    raise AssistantError("The API key was rejected. Check your credentials.")
                elif resp.status_code >= 400:
                    logger.warning(f"Groq API error {resp.status_code}: {resp.text}")
                    raise AssistantError(
                        "The assistant service could not be reached. Try again in a moment."
                    )

                data = resp.json()
                message = data["choices"][0]["message"]

                result = {"content": message.get("content")}

                if message.get("tool_calls"):
                    result["tool_calls"] = message["tool_calls"]

                return result

        except httpx.HTTPError as e:
            logger.warning(f"Groq HTTP error: {e}")
            raise AssistantError(
                "The assistant service could not be reached. Try again in a moment."
            ) from e
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Groq response parsing error: {e}")
            raise AssistantError(
                "The assistant service returned an invalid response. Try again in a moment."
            ) from e

    def _execute_tool(self, name: str, args: dict, visit: VisitContext) -> dict:
        """Execute a tool with the given arguments."""
        if name == "get_visit_summary":
            return self._tool_get_visit_summary(visit)
        elif name == "get_eye_details":
            return self._tool_get_eye_details(args, visit)
        elif name == "lookup_guideline":
            return self._tool_lookup_guideline(args)
        else:
            return {"error": f"Unknown tool: {name}"}

    def _tool_get_visit_summary(self, visit: VisitContext) -> dict:
        """Get a summary of the visit."""
        eyes_summary = []
        for eye in visit.eyes:
            eye_summary = {
                "side": eye.side,
                "grade": eye.grade,
                "grade_name": eye.grade_name,
                "grade_probability": round(eye.class_probs[eye.grade], 3),
                "uncertain": _is_uncertain(eye.class_probs),
                "probability_any_dr": round(eye.prob_dr, 3),
                "referable": eye.referable,
                "gradable": eye.gradable,
                "quality_problems": [
                    f"{c.label}: {c.message}" for c in eye.checks if c.status != "ok"
                ],
                "heatmap_focus": f"{eye.attention.region} of the photograph",
            }
            if eye.reference_grade is not None:
                eye_summary["dataset_reference_grade"] = CLASS_NAMES[eye.reference_grade]
                eye_summary["model_agrees_with_reference"] = eye.reference_grade == eye.grade
            if eye.review:
                eye_summary["review"] = {
                    "decision": eye.review.decision,
                    "grade": eye.review.grade,
                    "reason": eye.review.reason,
                }
            eyes_summary.append(eye_summary)

        not_photographed = []
        if len(visit.eyes) < 2:
            sides = {eye.side for eye in visit.eyes}
            for side in ["right", "left"]:
                if side not in sides:
                    not_photographed.append(side)

        return {
            "outcome_title": visit.outcome_title,
            "outcome_detail": visit.outcome_detail,
            "eyes": eyes_summary,
            "not_photographed": not_photographed,
        }

    def _tool_get_eye_details(self, args: dict, visit: VisitContext) -> dict:
        """Get detailed information about one eye."""
        side = args.get("side")
        eye = None
        for e in visit.eyes:
            if e.side == side:
                eye = e
                break

        if eye is None:
            return {"error": f"No photograph for {side} eye"}

        class_prob_dict = {CLASS_NAMES[i]: round(eye.class_probs[i], 3) for i in range(5)}

        top_two_grades = sorted(enumerate(eye.class_probs), key=lambda x: x[1], reverse=True)[:2]
        margin = round(top_two_grades[0][1] - top_two_grades[1][1], 3)
        uncertain = _is_uncertain(eye.class_probs)

        result = {
            "side": eye.side,
            "grade": eye.grade,
            "grade_name": eye.grade_name,
            "class_probs": class_prob_dict,
            "top_grades": [
                {
                    "grade": top_two_grades[0][0],
                    "name": CLASS_NAMES[top_two_grades[0][0]],
                },
                {
                    "grade": top_two_grades[1][0],
                    "name": CLASS_NAMES[top_two_grades[1][0]],
                },
            ],
            "margin": margin,
            "uncertain": uncertain,
            "quality_checks": [
                {
                    "label": check.label,
                    "status": check.status,
                    "message": check.message,
                }
                for check in eye.checks
            ],
            "heatmap": {
                "focus": f"{eye.attention.region} of the photograph",
                "fraction_of_image_highlighted": eye.attention.spread,
            },
        }

        if eye.reference_grade is not None:
            result["dataset_reference_grade"] = CLASS_NAMES[eye.reference_grade]
            result["model_agrees_with_reference"] = eye.reference_grade == eye.grade

        if eye.review:
            result["review"] = {
                "decision": eye.review.decision,
                "grade": eye.review.grade,
                "reason": eye.review.reason,
                "note": eye.review.note,
            }

        return result

    def _tool_lookup_guideline(self, args: dict) -> dict:
        """Look up guideline information."""
        topic = args.get("topic")
        content = GUIDELINE_CONTENT.get(topic)
        if content is None:
            return {"error": f"Unknown topic: {topic}"}
        return {"topic": topic, "content": content}

    def _build_tools(self) -> list[dict]:
        """Build the OpenAI-format tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_visit_summary",
                    "description": (
                        "Get summary of the entire visit including outcomes and per-eye grades"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_eye_details",
                    "description": (
                        "Get detailed information about one eye including class "
                        "probabilities, quality checks and attention"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "side": {
                                "type": "string",
                                "enum": ["right", "left"],
                                "description": "Which eye to get details for",
                            }
                        },
                        "required": ["side"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_guideline",
                    "description": (
                        "Look up clinical guideline information on grades, image "
                        "quality, heatmap interpretation or model details"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "enum": [
                                    "grade_0",
                                    "grade_1",
                                    "grade_2",
                                    "grade_3",
                                    "grade_4",
                                    "image_quality",
                                    "heatmap",
                                    "model",
                                ],
                                "description": "Which topic to look up",
                            }
                        },
                        "required": ["topic"],
                    },
                },
            },
        ]


def load_assistant() -> Assistant | None:
    """Load and return an assistant instance, or None if not configured."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.info("Assistant disabled: GROQ_API_KEY not set")
        return None

    model = os.getenv("DRSTAGE_ASSISTANT_MODEL", DEFAULT_MODEL)
    logger.info(f"Assistant enabled with model {model}")
    return Assistant(api_key=api_key, model=model)
