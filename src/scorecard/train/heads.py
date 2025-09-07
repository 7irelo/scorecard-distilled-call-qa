"""Ablation C's third arm: one LoRA backbone with a classification head per categorical field.

Summaries and evidence still need generation, which is the arm's cost and the point of testing it.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from scorecard.data.models import Transcript
from scorecard.io import read_jsonl
from scorecard.schema import (
    CallType,
    ComplianceStatus,
    EscalationReason,
    Scorecard,
    SentimentLabel,
    Trajectory,
    flatten_labels,
)
from scorecard.train.config import TrainConfig
from scorecard.train.format import full_messages, render_prompt, select_training_rows

_SENTIMENT = [s.value for s in SentimentLabel]
_STATUS = [s.value for s in ComplianceStatus]
FIELD_CLASSES: dict[str, list[str]] = {
    "call_type": [c.value for c in CallType],
    "sentiment.opening": _SENTIMENT,
    "sentiment.middle": _SENTIMENT,
    "sentiment.closing": _SENTIMENT,
    "sentiment.trajectory": [t.value for t in Trajectory],
    "escalation.flag": ["false", "true"],
    "escalation.reason": [r.value for r in EscalationReason],
    "compliance.agent_introduced": _STATUS,
    "compliance.recording_disclosed": _STATUS,
    "compliance.identity_verified": _STATUS,
    "compliance.terms_explained": _STATUS,
    "compliance.next_steps_confirmed": _STATUS,
}


class MultiHeadClassifier(nn.Module):  # type: ignore[misc]
    def __init__(self, backbone: Any, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {
                name.replace(".", "__"): nn.Linear(hidden_size, len(classes))
                for name, classes in FIELD_CLASSES.items()
            }
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last = output.hidden_states[-1]
        final_index = attention_mask.sum(dim=1) - 1
        pooled = last[torch.arange(last.size(0), device=last.device), final_index]
        pooled = pooled.to(next(self.heads.parameters()).dtype)
        return {name.replace("__", "."): head(pooled) for name, head in self.heads.items()}


def encode_for_heads(
    tokenizer: Any, row: dict[str, Any], config: TrainConfig
) -> dict[str, Any] | None:
    transcript = Transcript.model_validate(row["transcript"])
    prompt = render_prompt(tokenizer, full_messages(transcript), config.disable_thinking)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > config.max_seq_length:
        return None
    flat = flatten_labels(Scorecard.model_validate(row["card"]))
    return {
        "input_ids": ids,
        "targets": {name: FIELD_CLASSES[name].index(flat[name]) for name in FIELD_CLASSES},
    }


def run_heads_training(config: TrainConfig) -> dict[str, Any]:
    start = time.time()
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision)
    base = AutoModelForCausalLM.from_pretrained(config.base_model, revision=config.revision)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="FEATURE_EXTRACTION",
    )
    hidden = base.config.get_text_config().hidden_size
    model = MultiHeadClassifier(get_peft_model(base, lora), hidden).to(device)
    rows = select_training_rows(list(read_jsonl(config.train_path)), config)
    encoded = [e for e in (encode_for_heads(tokenizer, r, config) for r in rows) if e]
    if not encoded:
        raise ValueError("no examples fit max_seq_length")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rng = random.Random(config.seed)
    epochs = max(1, int(config.epochs))
    steps_per_epoch = len(encoded)
    total_steps = config.max_steps if config.max_steps > 0 else epochs * steps_per_epoch
    step = 0
    last_loss = float("nan")
    model.train()
    while step < total_steps:
        rng.shuffle(encoded)
        for example in encoded:
            if step >= total_steps:
                break
            ids = torch.tensor([example["input_ids"]], device=device)
            logits = model(ids, torch.ones_like(ids))
            loss = sum(
                loss_fn(logits[name], torch.tensor([example["targets"][name]], device=device))
                for name in FIELD_CLASSES
            ) / len(FIELD_CLASSES)
            (loss / config.grad_accum).backward()
            step += 1
            last_loss = float(loss.detach())
            if step % config.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

    config.run_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(config.run_dir / "final_adapter")
    torch.save(model.heads.state_dict(), config.run_dir / "heads.pt")
    hours = (time.time() - start) / 3600
    summary = {
        "name": config.name,
        "mode": "heads",
        "wall_hours": hours,
        "gpu_cost_usd": hours * config.gpu_usd_per_hour,
        "steps": step,
        "last_loss": last_loss,
    }
    (config.run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
