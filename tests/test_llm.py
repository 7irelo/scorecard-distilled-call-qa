import asyncio
import json

import httpx
import pytest

from scorecard.llm.backends import OpenAICompatibleBackend
from scorecard.llm.client import CostLedger, LLMClient, Price, ResponseCache
from scorecard.llm.config import LLMConfig, ModelConfig
from scorecard.llm.types import (
    LLMRequest,
    LLMResponse,
    Outcome,
    PolicyViolation,
    ProviderClass,
    Purpose,
    RetryableError,
)


class StubBackend:
    provider_class = ProviderClass.OPEN_WEIGHT

    def __init__(self, failures=0, outcome=Outcome.OK):
        self.calls, self.failures, self.outcome = 0, failures, outcome

    async def complete(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableError("busy")
        return LLMResponse("hello", self.outcome, input_tokens=1000, output_tokens=500)


def request(purpose=Purpose.TEACHER, salt=""):
    return LLMRequest("m", (("user", "hi"),), purpose, salt=salt)


def test_cache_prevents_second_call_and_ledger_charges_once(tmp_path):
    backend = StubBackend()
    ledger = CostLedger(tmp_path / "ledger.jsonl")
    cache = ResponseCache(tmp_path / "c.sqlite")
    client = LLMClient(backend, cache, ledger, Price(1.0, 2.0))

    async def run():
        return await client.complete(request()), await client.complete(request())

    first, second = asyncio.run(run())
    cache.close()
    assert backend.calls == 1
    assert not first.cached
    assert second.cached
    assert ledger.total_usd() == pytest.approx(0.002)
    assert len(ledger.path.read_text().splitlines()) == 2


def test_salt_changes_cache_key(tmp_path):
    backend = StubBackend()
    cache = ResponseCache(tmp_path / "c.sqlite")
    client = LLMClient(backend, cache)

    async def run():
        await client.complete(request(salt="a"))
        await client.complete(request(salt="b"))

    asyncio.run(run())
    cache.close()
    assert backend.calls == 2


def test_retries_then_succeeds():
    backend = StubBackend(failures=2)
    client = LLMClient(backend, base_delay=0)
    assert asyncio.run(client.complete(request())).outcome is Outcome.OK
    assert backend.calls == 3


def test_exhausted_retries_return_error_and_are_not_cached(tmp_path):
    backend = StubBackend(failures=99)
    cache = ResponseCache(tmp_path / "c.sqlite")
    client = LLMClient(backend, cache, max_attempts=2, base_delay=0)
    assert asyncio.run(client.complete(request())).outcome is Outcome.ERROR
    assert backend.calls == 2
    backend.failures = 0
    assert asyncio.run(client.complete(request())).outcome is Outcome.OK
    cache.close()


@pytest.mark.parametrize(
    "purpose", [Purpose.GENERATE, Purpose.VERIFY, Purpose.TEACHER, Purpose.JUDGE]
)
def test_closed_provider_is_blocked_outside_baselines(purpose):
    backend = StubBackend()
    backend.provider_class = ProviderClass.CLOSED
    with pytest.raises(PolicyViolation):
        asyncio.run(LLMClient(backend).complete(request(purpose)))
    assert backend.calls == 0


def test_closed_provider_allowed_for_baseline():
    backend = StubBackend()
    backend.provider_class = ProviderClass.CLOSED
    assert asyncio.run(LLMClient(backend).complete(request(Purpose.BASELINE))).text == "hello"


def _backend(handler):
    return OpenAICompatibleBackend("http://x/v1", "k", transport=httpx.MockTransport(handler))


def _reply(finish):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "{}"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


@pytest.mark.parametrize(
    ("finish", "outcome"),
    [("stop", Outcome.OK), ("length", Outcome.TRUNCATED), ("content_filter", Outcome.REFUSED)],
)
def test_openai_backend_maps_finish_reason(finish, outcome):
    response = asyncio.run(_backend(lambda _: _reply(finish)).complete(request()))
    assert response.outcome is outcome
    assert (response.input_tokens, response.output_tokens) == (7, 3)


def test_openai_backend_sends_schema():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return _reply("stop")

    req = LLMRequest("m", (("user", "hi"),), Purpose.TEACHER, json_schema={"type": "object"})
    asyncio.run(_backend(handler).complete(req))
    assert seen["response_format"]["json_schema"]["schema"] == {"type": "object"}


@pytest.mark.parametrize("status", [429, 503])
def test_openai_backend_retryable_status(status):
    with pytest.raises(RetryableError):
        asyncio.run(_backend(lambda _: httpx.Response(status)).complete(request()))


def test_openai_backend_client_error_is_not_retried():
    response = asyncio.run(_backend(lambda _: httpx.Response(400, text="bad")).complete(request()))
    assert response.outcome is Outcome.ERROR


def _model(family):
    return ModelConfig(model_id=family, family=family, provider_class=ProviderClass.OPEN_WEIGHT)


def test_family_separation_enforced():
    good = LLMConfig(
        roles={"generator": "a", "teacher": "b", "judge": "c"},
        models={"a": _model("a"), "b": _model("b"), "c": _model("c")},
    )
    good.check_family_separation()
    bad = good.model_copy(update={"roles": {"generator": "a", "teacher": "a", "judge": "c"}})
    with pytest.raises(ValueError):
        bad.check_family_separation()
