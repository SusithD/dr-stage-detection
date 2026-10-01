"""Tests for the assistant module."""

import asyncio
import json

import httpx
import numpy as np
import pytest
from conftest import make_fundus
from fastapi.testclient import TestClient

from drstage.api.app import create_app
from drstage.api.assistant import Assistant, AssistantError, load_assistant
from drstage.api.imaging import summarise_attention
from drstage.api.predictors import DemoPredictor
from drstage.api.schemas import (
    AssistantMessage,
    AssistantRequest,
    Attention,
    CheckContext,
    EyeContext,
    VisitContext,
)


def test_summarise_attention_upper_left():
    """Attention in upper left corner maps to 'upper left'."""
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[10:20, 10:20] = 1.0
    region, spread = summarise_attention(cam)
    assert region == "upper left"
    assert 0 <= spread <= 1


def test_summarise_attention_centre():
    """Attention in centre maps to 'centre'."""
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[40:60, 40:60] = 1.0
    region, spread = summarise_attention(cam)
    assert region == "centre"
    assert 0 <= spread <= 1


def test_summarise_attention_lower_right():
    """Attention in lower right corner maps to 'lower right'."""
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[80:90, 80:90] = 1.0
    region, spread = summarise_attention(cam)
    assert region == "lower right"
    assert 0 <= spread <= 1


def test_summarise_attention_clipping():
    """CAM values outside [0,1] are clipped."""
    cam = np.array([[-0.5, 1.5], [2.0, 0.5]], dtype=np.float32)
    region, spread = summarise_attention(cam)
    assert isinstance(region, str)
    assert 0 <= spread <= 1


def test_summarise_attention_spread_decimal():
    """Spread is rounded to 3 decimals."""
    cam = np.random.random((100, 100)).astype(np.float32)
    region, spread = summarise_attention(cam)
    assert isinstance(spread, float)
    assert len(str(spread).split(".")[-1]) <= 3


RIGHT_EYE = EyeContext(
    side="right",
    grade=2,
    grade_name="Moderate",
    class_probs=[0.01, 0.02, 0.55, 0.40, 0.02],
    prob_dr=0.99,
    referable=True,
    gradable=True,
    checks=[CheckContext(label="Focus", status="ok", message="In focus.")],
    attention=Attention(region="upper left", spread=0.25),
    reference_grade=0,
)
VISIT = VisitContext(
    eyes=[RIGHT_EYE], outcome_title="Refer", outcome_detail="Moderate retinopathy."
)
QUESTION = [AssistantMessage(role="user", content="Explain this result")]


def completion(content=None, tool_calls=None) -> httpx.Response:
    """Build a fake Groq chat completion response."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"choices": [{"message": message}]})


def tool_call(name: str, arguments: str, call_id: str = "call_1") -> dict:
    """One OpenAI-format tool call."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeGroq:
    """Replays scripted responses and records every request body."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0)

    def assistant(self, **kwargs) -> Assistant:
        return Assistant("test-key", transport=httpx.MockTransport(self), **kwargs)


def run(assistant: Assistant, visit: VisitContext = VISIT):
    return asyncio.run(assistant.reply(QUESTION, visit))


def tool_results(request: dict) -> list[dict]:
    return [json.loads(m["content"]) for m in request["messages"] if m["role"] == "tool"]


def test_tool_loop_feeds_eye_details_back_to_the_model():
    """A tool call is executed locally and its result sent back before the final answer."""
    fake = FakeGroq(
        completion(tool_calls=[tool_call("get_eye_details", '{"side": "right"}')]),
        completion("Moderate, but close to severe."),
    )
    response = run(fake.assistant())

    assert response.reply == "Moderate, but close to severe."
    assert response.tools_used == ["get_eye_details"]
    details = tool_results(fake.requests[1])[0]
    assert details["grade_name"] == "Moderate"
    assert details["uncertain"] is True
    assert details["dataset_reference_grade"] == "No DR"
    assert details["model_agrees_with_reference"] is False
    assert fake.requests[0]["messages"][0]["role"] == "system"
    assert "Moderate" not in json.dumps(fake.requests[0]["messages"])


def test_tool_without_arguments_accepts_empty_string():
    """Models often send '' as the arguments of a parameterless tool."""
    fake = FakeGroq(
        completion(tool_calls=[tool_call("get_visit_summary", "")]),
        completion("Refer the patient."),
    )
    run(fake.assistant())
    summary = tool_results(fake.requests[1])[0]
    assert summary["outcome_title"] == "Refer"
    assert summary["not_photographed"] == ["left"]


def test_parallel_tool_calls_each_get_a_reply():
    fake = FakeGroq(
        completion(
            tool_calls=[
                tool_call("get_eye_details", '{"side": "left"}', "a"),
                tool_call("lookup_guideline", '{"topic": "grade_2"}', "b"),
            ]
        ),
        completion("Done."),
    )
    response = run(fake.assistant())
    tool_messages = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    left, guideline = tool_results(fake.requests[1])
    assert "error" in left
    assert "3 months" in guideline["content"]
    assert response.tools_used == ["get_eye_details", "lookup_guideline"]


@pytest.mark.parametrize(
    "name,arguments",
    [("unknown_tool", "{}"), ("get_eye_details", "{not json"), ("lookup_guideline", "[1]")],
)
def test_bad_tool_calls_become_error_results(name, arguments):
    fake = FakeGroq(completion(tool_calls=[tool_call(name, arguments)]), completion("Sorry."))
    assert run(fake.assistant()).reply == "Sorry."
    assert "error" in tool_results(fake.requests[1])[0]


def test_last_round_disables_tools():
    looping = completion(tool_calls=[tool_call("get_visit_summary", "{}")])
    fake = FakeGroq(looping, looping, completion("Summary."))
    response = run(fake.assistant(max_rounds=3))
    assert [r["tool_choice"] for r in fake.requests] == ["required", "auto", "none"]
    assert response.reply == "Summary."


def test_required_tool_refusal_falls_back_to_auto():
    """Groq returns 400 tool_use_failed when the model declines a forced tool call."""
    refusal = httpx.Response(400, json={"error": {"code": "tool_use_failed"}})
    fake = FakeGroq(refusal, completion("I can only help with this screening visit."))
    response = run(fake.assistant())
    assert [r["tool_choice"] for r in fake.requests] == ["required", "auto"]
    assert response.tools_used == []


def test_visit_summary_reports_grade_probability():
    fake = FakeGroq(completion(tool_calls=[tool_call("get_visit_summary", "{}")]), completion("ok"))
    run(fake.assistant())
    eye = tool_results(fake.requests[1])[0]["eyes"][0]
    assert eye["grade_probability"] == 0.55
    assert eye["uncertain"] is True
    assert eye["probability_any_dr"] == 0.99
    assert eye["model_agrees_with_reference"] is False
    assert eye["quality_problems"] == []


def test_empty_answer_is_an_error():
    with pytest.raises(AssistantError, match="empty"):
        run(FakeGroq(completion("  ")).assistant())


@pytest.mark.parametrize(
    "status_code,match", [(401, "API key"), (429, "rate limited"), (500, "could not be reached")]
)
def test_upstream_errors_are_readable(status_code, match):
    fake = FakeGroq(httpx.Response(status_code, json={"error": "x"}))
    with pytest.raises(AssistantError, match=match):
        run(fake.assistant())


def test_network_failure_is_readable():
    def fail(request):
        raise httpx.ConnectError("offline")

    assistant = Assistant("k", transport=httpx.MockTransport(fail))
    with pytest.raises(AssistantError, match="could not be reached"):
        run(assistant)


def test_gpt_oss_uses_low_reasoning_effort():
    fake = FakeGroq(completion("Hi."))
    run(fake.assistant(model="openai/gpt-oss-120b"))
    assert fake.requests[0]["reasoning_effort"] == "low"


def test_load_assistant_reads_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-xyz")
    monkeypatch.setenv("DRSTAGE_ASSISTANT_MODEL", "qwen/qwen3.8-27b")
    assistant = load_assistant()
    assert assistant.api_key == "test-key-xyz"
    assert assistant.model == "qwen/qwen3.8-27b"


def test_load_assistant_without_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", " ")
    assert load_assistant() is None


@pytest.fixture
def payload():
    return AssistantRequest(messages=QUESTION, visit=VISIT).model_dump()


def client_with(tmp_path, assistant=None) -> TestClient:
    app = create_app(
        predictor=DemoPredictor(), split_csv=tmp_path / "missing.csv", assistant=assistant
    )
    return TestClient(app)


def test_endpoint_returns_the_reply(tmp_path, payload):
    fake = FakeGroq(
        completion(tool_calls=[tool_call("get_visit_summary", "{}")]), completion("Refer.")
    )
    client = client_with(tmp_path, fake.assistant())
    assert client.get("/api/health").json()["assistant"] is True
    response = client.post("/api/assistant", json=payload)
    assert response.status_code == 200
    assert response.json() == {"reply": "Refer.", "tools_used": ["get_visit_summary"]}


def test_endpoint_maps_upstream_failure_to_502(tmp_path, payload):
    fake = FakeGroq(httpx.Response(429, json={}))
    response = client_with(tmp_path, fake.assistant()).post("/api/assistant", json=payload)
    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


def test_endpoint_without_assistant(tmp_path, payload):
    client = client_with(tmp_path)
    assert client.get("/api/health").json()["assistant"] is False
    response = client.post("/api/assistant", json=payload)
    assert response.status_code == 503


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(messages=[]),
        lambda p: p.update(messages=p["messages"] * 21),
        lambda p: p["messages"][0].update(role="system"),
        lambda p: p["visit"].update(eyes=p["visit"]["eyes"] * 3),
        lambda p: p["visit"]["eyes"][0].update(class_probs=[1.0]),
    ],
)
def test_endpoint_validates_requests(tmp_path, payload, mutate):
    mutate(payload)
    client = client_with(tmp_path, FakeGroq().assistant())
    assert client.post("/api/assistant", json=payload).status_code == 422


def test_predict_includes_attention(tmp_path):
    import io

    import cv2

    _, jpg = cv2.imencode(".jpg", cv2.cvtColor(make_fundus(640, 480, seed=42), cv2.COLOR_RGB2BGR))
    response = client_with(tmp_path).post(
        "/api/predict", files={"file": ("test.jpg", io.BytesIO(jpg.tobytes()), "image/jpeg")}
    )
    assert response.status_code == 200
    attention = response.json()["attention"]
    assert isinstance(attention["region"], str)
    assert 0 <= attention["spread"] <= 1
