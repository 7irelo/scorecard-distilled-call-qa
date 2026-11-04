"""Checks a real vLLM server for what mocks can't prove. Run this first on any new deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from scorecard.data.models import Speaker, Transcript, Turn
from scorecard.serve.inference import TranscriptTooLong, VLLMBackend

_ENUM_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["pass", "fail", "not_applicable"]}},
    "required": ["status"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


async def _constrained(
    client: httpx.AsyncClient, model: str, thinking_off: bool
) -> tuple[bool, str]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Did the agent verify identity? Answer in JSON."}],
        "max_tokens": 64,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "check", "schema": _ENUM_SCHEMA},
        },
    }
    if thinking_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    reply = await client.post("chat/completions", json=body)
    if reply.status_code != 200:
        return False, f"HTTP {reply.status_code}: {reply.text[:200]}"
    content = reply.json()["choices"][0]["message"].get("content") or ""
    try:
        value = json.loads(content)["status"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, f"output broke the schema: {content[:120]!r}"
    if value not in _ENUM_SCHEMA["properties"]["status"]["enum"]:  # type: ignore[index]
        return False, f"value outside the schema: {value!r}"
    return True, f"returned {content.strip()}"


async def check_server(
    base_url: str,
    student: str = "student",
    base: str | None = "base",
    api_key: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
    overflow_words: int = 200_000,
) -> list[CheckResult]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results: list[CheckResult] = []
    root = base_url.rstrip("/") + "/"
    async with httpx.AsyncClient(
        base_url=root, headers=headers, timeout=300, transport=transport
    ) as client:
        listed = await client.get("models")
        served = (
            {item["id"] for item in listed.json().get("data", [])}
            if listed.status_code == 200
            else set()
        )
        wanted = {student} | ({base} if base else set())
        results.append(
            CheckResult(
                "models_served",
                wanted <= served,
                f"server has {sorted(served)}, need {sorted(wanted)}",
            )
        )
        passed, detail = await _constrained(client, student, thinking_off=True)
        results.append(CheckResult("constrained_with_adapter", passed, detail))
        if base:
            passed, detail = await _constrained(client, base, thinking_off=True)
            results.append(CheckResult("constrained_base_model", passed, detail))

    backend = VLLMBackend(root, student, api_key, transport=transport, max_tokens=16)
    oversized = Transcript(
        scenario_id="check",
        turns=(Turn(id=1, speaker=Speaker.CUSTOMER, text="word " * overflow_words),),
    )
    try:
        await backend.generate(oversized)
    except TranscriptTooLong:
        results.append(CheckResult("overflow_detected", True, "rejected as too long"))
    except Exception as exc:
        results.append(CheckResult("overflow_detected", False, f"got {type(exc).__name__}"))
    else:
        results.append(CheckResult("overflow_detected", False, "server accepted the long call"))
    return results
