"""Ablation C's third arm: a shared LoRA backbone with a classification head per categorical field.

The backbone also trains on summaries, because classifiers can't write them. Evidence turns aren't
produced at all, which is a real cost of this arm and shows up in its evidence and pass results.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch import nn

from scorecard.data.models import Transcript
from scorecard.io import read_jsonl
from scorecard.schema import (
    COMPLIANCE_CHECKS,
    CallType,
    ComplianceStatus,
    EscalationReason,
    Scorecard,
    SentimentLabel,
    Trajectory,
    flatten_labels,
)
from scorecard.train.config import TrainConfig
from scorecard.train.format import (
    encode_example,
    full_messages,
    per_field_messages,
    per_field_targets,
    render_prompt,
    select_training_rows,
)

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
    **{f"compliance.{check}": _STATUS for check in COMPLIANCE_CHECKS},
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
        parameter = next(self.heads.parameters())
        pooled = pooled.to(dtype=parameter.dtype, device=parameter.device)
        return {name.replace("__", "."): head(pooled) for name, head in self.heads.items()}


def _hidden_size(model: Any) -> int:
    return int(model.config.get_text_config().hidden_size)


def encode_for_heads(
    tokenizer: Any, row: dict[str, Any], config: TrainConfig
) -> dict[str, Any] | None:
    transcript = Transcript.model_validate(row["transcript"])
    card = Scorecard.model_validate(row["card"])
    prompt = render_prompt(tokenizer, full_messages(transcript), config.disable_thinking)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    summary = encode_example(
        tokenizer,
        per_field_messages(transcript, "summary"),
        per_field_targets(card)["summary"],
        config.max_seq_length,
        config.disable_thinking,
    )
    if len(ids) > config.max_seq_length or summary is None:
        return None
    flat = flatten_labels(card)
    return {
        "input_ids": ids,
        "summary": summary,
        "targets": {name: FIELD_CLASSES[name].index(flat[name]) for name in FIELD_CLASSES},
    }


def predict_heads(
    model: MultiHeadClassifier, tokenizer: Any, transcript: Transcript, config: TrainConfig
) -> tuple[Scorecard | None, bool]:
    """Classify the fields, then generate the summary. Cards from this arm cite no evidence."""
    device = next(model.parameters()).device
    prompt = render_prompt(tokenizer, full_messages(transcript), config.disable_thinking)
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)["input_ids"]], device=device)
    model.eval()
    with torch.no_grad():
        logits = model(ids, torch.ones_like(ids))
    flat = {
        name: FIELD_CLASSES[name][int(values.argmax(dim=-1).item())]
        for name, values in logits.items()
    }
    summary_prompt = render_prompt(
        tokenizer, per_field_messages(transcript, "summary"), config.disable_thinking
    )
    inputs = tokenizer(summary_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output = model.backbone.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    try:
        summary = str(json.loads(text.strip())["summary"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, False
    card = Scorecard.model_validate(
        {
            "summary": summary,
            "call_type": flat["call_type"],
            "sentiment": {
                "opening": flat["sentiment.opening"],
                "middle": flat["sentiment.middle"],
                "closing": flat["sentiment.closing"],
                "trajectory": flat["sentiment.trajectory"],
            },
            "compliance": {
                check: {"status": flat[f"compliance.{check}"], "evidence_turns": []}
                for check in COMPLIANCE_CHECKS
            },
            "escalation": {
                "flag": flat["escalation.flag"] == "true",
                "reason": flat["escalation.reason"],
            },
        }
    )
    return card, True


def heads_metrics(
    model: MultiHeadClassifier, tokenizer: Any, rows: list[dict[str, Any]], config: TrainConfig
) -> dict[str, float]:
    valid = agree = fields = 0
    for row in rows:
        transcript = Transcript.model_validate(row["transcript"])
        card, is_valid = predict_heads(model, tokenizer, transcript, config)
        valid += is_valid
        target = flatten_labels(Scorecard.model_validate(row["card"]))
        fields += len(target)
        if card is not None:
            guess = flatten_labels(card)
            agree += sum(guess[name] == value for name, value in target.items())
    return {
        "val/validity": valid / max(1, len(rows)),
        "val/field_agreement": agree / max(1, fields),
        "val/score": agree / max(1, fields),
    }


def run_heads_training(config: TrainConfig) -> dict[str, Any]:
    from scorecard.train.run import load_model_and_tokenizer

    start = time.time()
    torch.manual_seed(config.seed)
    backbone, tokenizer = load_model_and_tokenizer(config)
    device = next(backbone.parameters()).device
    model = MultiHeadClassifier(backbone, _hidden_size(backbone)).to(device)
    rows = select_training_rows(list(read_jsonl(config.train_path)), config)
    encoded = [e for e in (encode_for_heads(tokenizer, r, config) for r in rows) if e]
    if not encoded:
        raise ValueError("no examples fit max_seq_length")
    val_rows = (
        rows[: config.eval_calls]
        if config.val_from_train
        else [r for r in read_jsonl(config.val_path) if r.get("card")][: config.eval_calls]
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rng = random.Random(config.seed)
    epochs = max(1, round(config.epochs))
    total_steps = config.max_steps if config.max_steps > 0 else epochs * len(encoded)
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
            classification = sum(
                loss_fn(logits[name], torch.tensor([example["targets"][name]], device=device))
                for name in FIELD_CLASSES
            ) / len(FIELD_CLASSES)
            summary = example["summary"]
            summary_loss = model.backbone(
                input_ids=torch.tensor([summary["input_ids"]], device=device),
                attention_mask=torch.tensor([summary["attention_mask"]], device=device),
                labels=torch.tensor([summary["labels"]], device=device),
            ).loss
            loss = classification + summary_loss
            (loss / config.grad_accum).backward()
            step += 1
            last_loss = float(loss.detach())
            if step % config.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

    config.run_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(config.run_dir / "final_adapter")
    tokenizer.save_pretrained(config.run_dir / "final_adapter")
    torch.save(model.heads.state_dict(), config.run_dir / "heads.pt")
    final = heads_metrics(model, tokenizer, val_rows, config) if val_rows else {}
    hours = (time.time() - start) / 3600
    summary_row = {
        "name": config.name,
        "mode": "heads",
        "wall_hours": hours,
        "gpu_cost_usd": hours * config.gpu_usd_per_hour,
        "steps": step,
        "last_loss": last_loss,
        "final_metrics": final,
        "config": config.model_dump(mode="json"),
    }
    (config.run_dir / "run_summary.json").write_text(
        json.dumps(summary_row, indent=2), encoding="utf-8"
    )
    return summary_row


def load_heads_model(config: TrainConfig, run_dir: Path | None = None) -> tuple[Any, Any]:
    """Reload a trained arm for prediction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = run_dir or config.run_dir
    tokenizer = AutoTokenizer.from_pretrained(directory / "final_adapter")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config.base_model, revision=config.revision)
    backbone = PeftModel.from_pretrained(base, str(directory / "final_adapter"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiHeadClassifier(backbone, _hidden_size(base)).to(device)
    model.heads.load_state_dict(torch.load(directory / "heads.pt", map_location=device))
    return model, tokenizer
