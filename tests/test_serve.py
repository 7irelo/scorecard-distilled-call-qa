import asyncio
import logging
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from scorecard.serve.app import Settings, create_app
from scorecard.serve.cost_model import (
    ApiOption,
    CostInputs,
    Fixed,
    SelfHosted,
    always_on_break_even_calls,
    gpus_needed,
    load_cost_inputs,
    quality_adjusted_usd_per_1k,
    report_markdown,
    scheduled_monthly_usd,
    self_hosted_usd_per_1k,
    volume_table,
)
from scorecard.serve.inference import BackendUnavailable, FakeBackend, Generation, VLLMBackend
from scorecard.serve.loadgen import run_load
from tests.fixtures.rubric_calls import FIXTURES

KEY = {"x-api-key": "secret"}
SECRET_TEXT = "my very private ID number 8001015009087"


def body(text="Hello, my name is Thandi."):
    return {
        "call_id": "c1",
        "turns": [
            {"id": 1, "speaker": "AGENT", "text": text},
            {"id": 2, "speaker": "CUSTOMER", "text": "My bill is wrong."},
        ],
    }


def client(backend=None, **settings):
    app = create_app(Settings(api_key="secret", **settings), backend or FakeBackend())
    return TestClient(app)


def test_happy_path_returns_card_and_versions():
    reply = client(model_revision="abc", adapter_hash="def").post(
        "/v1/scorecards", json=body(), headers=KEY
    )
    assert reply.status_code == 200
    payload = reply.json()
    assert payload["model_info"] == {
        "model_revision": "abc",
        "adapter_hash": "def",
        "schema_version": "1.0",
    }
    assert payload["attempts"] == 1
    assert payload["scorecard"]["schema_version"] == "1.0"


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}])
def test_auth_required(headers):
    assert client().post("/v1/scorecards", json=body(), headers=headers).status_code == 401


def test_oversize_is_rejected_not_truncated():
    backend = FakeBackend()
    reply = client(backend, max_transcript_chars=20).post(
        "/v1/scorecards", json=body("x" * 100), headers=KEY
    )
    assert reply.status_code == 413
    assert backend.calls == 0


def test_huge_body_rejected_before_parsing():
    reply = client(max_transcript_chars=10).post(
        "/v1/scorecards", json=body("x" * 200_000), headers=KEY
    )
    assert reply.status_code == 413


def test_duplicate_turn_ids_rejected():
    request = body()
    request["turns"][1]["id"] = 1
    assert client().post("/v1/scorecards", json=request, headers=KEY).status_code == 422


def test_empty_turns_rejected():
    request = {"call_id": "x", "turns": []}
    assert client().post("/v1/scorecards", json=request, headers=KEY).status_code == 422


def test_invalid_output_retries_once_then_succeeds():
    backend = FakeBackend(invalid_first=1)
    reply = client(backend).post("/v1/scorecards", json=body(), headers=KEY)
    assert reply.status_code == 200
    assert reply.json()["attempts"] == 2


def test_invalid_output_twice_is_an_explicit_error():
    reply = client(FakeBackend(invalid_first=2)).post("/v1/scorecards", json=body(), headers=KEY)
    assert reply.status_code == 502


def test_backend_failure_returns_503():
    reply = client(FakeBackend(fail_first=1)).post("/v1/scorecards", json=body(), headers=KEY)
    assert reply.status_code == 503


class _Slow(FakeBackend):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    async def generate(self, transcript):
        await asyncio.sleep(self.delay)
        return await super().generate(transcript)


def test_timeout_returns_504():
    reply = client(_Slow(0.5), request_timeout_s=0.01).post(
        "/v1/scorecards", json=body(), headers=KEY
    )
    assert reply.status_code == 504


def test_overload_returns_429_with_retry_after():
    async def run():
        app = create_app(Settings(api_key="secret", max_concurrency=1, max_queue=1), _Slow(0.2))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            return await asyncio.gather(
                *(http.post("/v1/scorecards", json=body(), headers=KEY) for _ in range(6))
            )

    replies = asyncio.run(run())
    codes = sorted(r.status_code for r in replies)
    assert codes.count(200) == 2
    assert codes.count(429) == 4
    assert next(r for r in replies if r.status_code == 429).headers["retry-after"] == "5"


def test_transcript_text_stays_out_of_logs_and_metrics(caplog):
    caplog.set_level(logging.DEBUG, logger="scorecard.api")
    http = client(FakeBackend(invalid_first=1))
    assert http.post("/v1/scorecards", json=body(SECRET_TEXT), headers=KEY).status_code == 200
    assert "8001015009087" not in caplog.text
    assert "8001015009087" not in http.get("/metrics").text


def test_health_ready_and_metrics():
    http = client()
    assert http.get("/healthz").json() == {"status": "ok"}
    assert http.get("/readyz").status_code == 200
    http.post("/v1/scorecards", json=body(), headers=KEY)
    metrics = http.get("/metrics").text
    assert 'scorecard_requests_total{outcome="ok"} 1.0' in metrics
    assert "scorecard_request_seconds_bucket" in metrics


class _NotReady(FakeBackend):
    async def ready(self):
        return False


def test_readyz_reports_unready_backend():
    assert client(_NotReady()).get("/readyz").status_code == 503


def _vllm(handler):
    return VLLMBackend("http://v/v1", "student", transport=httpx.MockTransport(handler))


def test_vllm_backend_sends_schema_and_disables_thinking():
    seen = {}

    def handler(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            },
        )

    result = asyncio.run(_vllm(handler).generate(FIXTURES[0][0]))
    assert result == Generation("{}", 9, 4)
    assert seen["model"] == "student"
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_vllm_backend_errors_become_unavailable():
    with pytest.raises(BackendUnavailable):
        asyncio.run(_vllm(lambda _: httpx.Response(500)).generate(FIXTURES[0][0]))


def test_load_generator_counts_outcomes_against_real_app():
    async def run():
        app = create_app(Settings(api_key="secret"), FakeBackend())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            return await run_load(http, [body()], target_rps=200, duration_s=0.3, api_key="secret")

    result = asyncio.run(run())
    assert result.sent > 10
    assert result.completed_ok == result.sent
    assert result.error_rate == 0
    assert result.p95_s >= result.p50_s > 0


def _gpu(measured=True):
    return SelfHosted(name="L4", gpu_usd_per_hour=0.59, peak_calls_per_hour=1800, measured=measured)


def _api(i, o, measured=True):
    return ApiOption(
        name="api",
        input_per_mtok=i,
        output_per_mtok=o,
        batch_discount=0.5,
        input_tokens=6000,
        output_tokens=500,
        measured=measured,
    )


def test_api_costs_match_the_plan():
    assert _api(4, 20).usd_per_1k() == pytest.approx(34.0)
    assert _api(4, 20).usd_per_1k(batch=True) == pytest.approx(17.0)
    assert _api(0.1, 0.5).usd_per_1k(batch=True) == pytest.approx(0.425)


def test_self_hosted_cost_depends_on_utilisation():
    assert self_hosted_usd_per_1k(_gpu(), 1.0) == pytest.approx(0.328, abs=0.001)
    assert self_hosted_usd_per_1k(_gpu(), 0.1) == pytest.approx(3.28, abs=0.01)
    with pytest.raises(ValueError):
        self_hosted_usd_per_1k(_gpu(), 0)


def test_break_even_values_match_the_plan():
    fixed = Fixed()
    opus = always_on_break_even_calls(_gpu(), _api(4, 20), fixed, batch=True)
    haiku = always_on_break_even_calls(_gpu(), _api(0.1, 0.5), fixed, batch=True)
    assert opus == pytest.approx(25_334, rel=0.01)
    assert haiku == pytest.approx(1_013_000, rel=0.01)


def test_break_even_is_infinite_when_beyond_capacity():
    slow = SelfHosted(name="slow", gpu_usd_per_hour=0.59, peak_calls_per_hour=100)
    assert math.isinf(always_on_break_even_calls(slow, _api(0.1, 0.5), Fixed(), batch=True))


def test_fixed_costs_raise_break_even():
    fixed = Fixed(training_and_labelling_usd=1200, amortisation_months=12)
    assert always_on_break_even_calls(
        _gpu(), _api(4, 20), fixed, True
    ) > always_on_break_even_calls(_gpu(), _api(4, 20), Fixed(), True)


def test_scheduled_batch_beats_always_on_at_low_volume():
    assert scheduled_monthly_usd(_gpu(), 10_000, Fixed()) == pytest.approx(3.28, abs=0.01)


def test_capacity_adds_gpus():
    assert gpus_needed(_gpu(), 10_000) == 1
    assert gpus_needed(_gpu(), 2_000_000) == 2


def test_quality_adjustment():
    assert quality_adjusted_usd_per_1k(1.0, 0.5) == 2.0
    assert math.isinf(quality_adjusted_usd_per_1k(1.0, 0))


def test_report_warns_on_placeholders_only():
    placeholder = CostInputs(self_hosted=[_gpu(False)], apis=[_api(4, 20)])
    measured = CostInputs(self_hosted=[_gpu()], apis=[_api(4, 20)])
    assert "PLACEHOLDER" in report_markdown(placeholder)
    assert "PLACEHOLDER" not in report_markdown(measured)


def test_volume_table_converts_to_zar():
    inputs = CostInputs(
        self_hosted=[_gpu()],
        apis=[_api(4, 20)],
        fixed=Fixed(usd_to_zar=18.0),
        volumes_per_month=[1000],
    )
    rows = volume_table(inputs)
    assert len(rows) == 4
    assert rows[0]["monthly_zar"] == pytest.approx(rows[0]["monthly_usd"] * 18.0)


def test_example_costs_file_loads_and_is_marked_placeholder():
    inputs = load_cost_inputs(__import__("pathlib").Path("benchmarks/costs.example.yaml"))
    assert not inputs.fully_measured
    assert "PLACEHOLDER" in report_markdown(inputs)
