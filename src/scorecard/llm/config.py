"""Named models from configs/llm.yaml, with family separation checks."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from scorecard.llm.backends import AnthropicBackend, OpenAICompatibleBackend
from scorecard.llm.client import CostLedger, LLMClient, Price, ResponseCache
from scorecard.llm.types import Backend, ProviderClass


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model_id: str
    family: str
    provider_class: ProviderClass
    base_url: str | None = None
    api_key_env: str | None = None
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roles: dict[str, str]
    models: dict[str, ModelConfig]
    cache_path: Path = Path("runs/llm_cache.sqlite")
    ledger_path: Path = Path("runs/cost_ledger.jsonl")

    def model_for(self, role: str) -> ModelConfig:
        return self.models[self.roles[role]]

    def check_family_separation(self) -> None:
        distinct = [self.model_for(r).family for r in ("generator", "teacher", "judge")]
        if len(set(distinct)) != len(distinct):
            raise ValueError(f"generator, teacher and judge must differ in family: {distinct}")


def load_llm_config(path: Path) -> LLMConfig:
    config = LLMConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    config.check_family_separation()
    return config


def build_backend(model: ModelConfig) -> Backend:
    if model.provider_class is ProviderClass.CLOSED:
        return AnthropicBackend()
    if model.base_url is None:
        raise ValueError(f"{model.model_id} needs a base_url")
    key = os.environ.get(model.api_key_env, "") if model.api_key_env else ""
    return OpenAICompatibleBackend(model.base_url, key)


def build_client(config: LLMConfig, role: str, concurrency: int = 8) -> tuple[LLMClient, str]:
    model = config.model_for(role)
    client = LLMClient(
        build_backend(model),
        cache=ResponseCache(config.cache_path),
        ledger=CostLedger(config.ledger_path),
        price=Price(model.input_per_mtok, model.output_per_mtok),
        concurrency=concurrency,
    )
    return client, model.model_id
