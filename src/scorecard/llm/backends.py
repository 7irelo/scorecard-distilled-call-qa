"""Inference backends: OpenAI-compatible hosts and vLLM, and Anthropic for closed baselines."""

from __future__ import annotations

from typing import Any

import httpx

from scorecard.llm.types import LLMRequest, LLMResponse, Outcome, ProviderClass, RetryableError


class OpenAICompatibleBackend:
    """Open-weight models on a hosted endpoint or on our own vLLM server."""

    provider_class = ProviderClass.OPEN_WEIGHT

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._extra_body = extra_body or {}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": r, "content": c} for r, c in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            **self._extra_body,
        }
        if request.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema, "strict": True},
            }
        try:
            reply = await self._client.post("chat/completions", json=body)
        except httpx.TransportError as exc:
            raise RetryableError(str(exc)) from exc
        if reply.status_code == 429 or reply.status_code >= 500:
            raise RetryableError(f"HTTP {reply.status_code}")
        if reply.status_code >= 400:
            return LLMResponse("", Outcome.ERROR, detail=f"HTTP {reply.status_code}: {reply.text}")
        payload = reply.json()
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        finish = choice.get("finish_reason")
        outcome = {"content_filter": Outcome.REFUSED, "length": Outcome.TRUNCATED}.get(
            finish, Outcome.OK
        )
        return LLMResponse(
            text=choice["message"].get("content") or "",
            outcome=outcome,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            detail=str(finish or ""),
        )


class AnthropicBackend:
    """Closed baseline, used through the official SDK. Policy blocks it outside Purpose.BASELINE."""

    provider_class = ProviderClass.CLOSED

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self._client = client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        system = "\n\n".join(c for r, c in request.messages if r == "system")
        messages = [{"role": r, "content": c} for r, c in request.messages if r != "system"]
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if request.json_schema is not None:
            params["output_config"] = {
                "format": {"type": "json_schema", "schema": request.json_schema}
            }
        try:
            message = await self._client.messages.create(**params)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise RetryableError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise RetryableError(str(exc)) from exc
            return LLMResponse("", Outcome.ERROR, detail=str(exc))
        usage = message.usage
        if message.stop_reason == "refusal":
            return LLMResponse(
                "",
                Outcome.REFUSED,
                usage.input_tokens,
                usage.output_tokens,
                detail=str(getattr(message, "stop_details", "") or ""),
            )
        text = "".join(b.text for b in message.content if b.type == "text")
        outcome = Outcome.TRUNCATED if message.stop_reason == "max_tokens" else Outcome.OK
        return LLMResponse(
            text, outcome, usage.input_tokens, usage.output_tokens, detail=message.stop_reason or ""
        )
