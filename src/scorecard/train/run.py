"""TRL + PEFT training. Heavy imports stay inside this module, which needs the `train` extra."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from pydantic import ValidationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from scorecard.baselines.llm_system import parse_scorecard
from scorecard.data.models import Transcript
from scorecard.io import read_jsonl, sha256_file
from scorecard.schema import Scorecard, flatten_labels
from scorecard.train.config import OutputMode, Quantization, TrainConfig
from scorecard.train.format import (
    IGNORE_INDEX,
    PER_FIELD_NAMES,
    assemble_fields,
    describe_selection,
    encode_rows,
    full_messages,
    per_field_messages,
    render_prompt,
    select_training_rows,
)


def load_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"revision": config.revision}
    has_gpu = torch.cuda.is_available()
    if config.quantization is Quantization.NF4:
        if not has_gpu:
            raise RuntimeError("nf4 quantization needs a GPU; use quantization: none on CPU")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    elif has_gpu:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    if config.quantization is Quantization.NF4:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), tokenizer


def _generate_text(
    model: Any, tokenizer: Any, messages: list[dict[str, str]], config: TrainConfig
) -> str:
    prompt = render_prompt(tokenizer, messages, config.disable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(
        tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    )


def generate_card(
    model: Any, tokenizer: Any, transcript: Transcript, config: TrainConfig
) -> tuple[Scorecard | None, bool, str]:
    """Generate with the same prompt form the run trained on."""
    if config.output_mode is OutputMode.PER_FIELD:
        outputs = {
            name: _generate_text(model, tokenizer, per_field_messages(transcript, name), config)
            for name in PER_FIELD_NAMES
        }
        text = json.dumps(outputs, ensure_ascii=False)
        try:
            return assemble_fields(outputs), True, text
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            return None, False, text
    text = _generate_text(model, tokenizer, full_messages(transcript), config)
    card, valid = parse_scorecard(text)
    return card, valid, text


def generative_metrics(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], config: TrainConfig
) -> dict[str, float]:
    """Metrics from generated output, since low loss does not guarantee a usable scorecard."""
    valid_count = 0
    agree = fields = 0
    was_training = model.training
    model.eval()
    use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    for row in rows:
        transcript = Transcript.model_validate(row["transcript"])
        card, valid, _ = generate_card(model, tokenizer, transcript, config)
        valid_count += valid
        target = flatten_labels(Scorecard.model_validate(row["card"]))
        fields += len(target)
        if card is not None:
            guess = flatten_labels(card)
            agree += sum(guess[name] == value for name, value in target.items())
    model.config.use_cache = use_cache
    if was_training:
        model.train()
    count = max(1, len(rows))
    return {
        "val/validity": valid_count / count,
        "val/field_agreement": agree / max(1, fields),
        # Invalid cards score zero on every field, so this already includes validity.
        "val/score": agree / max(1, fields),
    }


class GenerativeEvalCallback(TrainerCallback):  # type: ignore[misc]
    """Evaluates generated scorecards, keeps the best adapter and stops when progress ends."""

    def __init__(self, tokenizer: Any, rows: list[dict[str, Any]], config: TrainConfig) -> None:
        self.tokenizer = tokenizer
        self.rows = rows
        self.config = config
        self.best = float("-inf")
        self.best_step = 0
        self.stale = 0
        self.history: list[dict[str, float]] = []

    def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **_: Any) -> Any:
        if not self.rows or state.global_step % self.config.eval_every_steps != 0:
            return control
        metrics = generative_metrics(model, self.tokenizer, self.rows, self.config)
        self.history.append({"step": state.global_step, **metrics})
        if self.config.report_to == "wandb":
            import wandb

            wandb.log(metrics, step=state.global_step)
        if metrics["val/score"] > self.best:
            self.best, self.best_step, self.stale = metrics["val/score"], state.global_step, 0
            model.save_pretrained(self.config.run_dir / "best_adapter")
        else:
            self.stale += 1
            if self.stale >= self.config.early_stopping_patience:
                control.should_training_stop = True
        return control


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_training(config: TrainConfig, resume: bool = False) -> dict[str, Any]:
    if config.output_mode is OutputMode.HEADS:
        from scorecard.train.heads import run_heads_training

        return run_heads_training(config)

    start = time.time()
    rows = select_training_rows(list(read_jsonl(config.train_path)), config)
    val_rows = (
        rows[: config.eval_calls]
        if config.val_from_train
        else [r for r in read_jsonl(config.val_path) if r.get("card")][: config.eval_calls]
    )
    model, tokenizer = load_model_and_tokenizer(config)
    encoded, report = encode_rows(tokenizer, rows, config)
    if not encoded:
        raise ValueError("no examples fit max_seq_length")

    config.run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_sha": git_sha(),
        "train_sha256": sha256_file(config.train_path),
        "selection": describe_selection(rows),
        "encoded": report.encoded,
        "too_long": report.too_long,
        "max_length": report.max_length,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if config.report_to == "wandb":
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.name,
            config={**config.model_dump(mode="json"), **provenance},
        )

    callback = GenerativeEvalCallback(tokenizer, val_rows, config)
    args = SFTConfig(
        output_dir=str(config.run_dir / "checkpoints"),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        lr_scheduler_type="cosine",
        # Transformers 5 reads a value below 1 here as a share of total steps.
        warmup_steps=config.warmup_ratio,
        logging_steps=10,
        save_steps=config.eval_every_steps,
        save_total_limit=2,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_available(),
        report_to=[config.report_to] if config.report_to != "none" else [],
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(encoded),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(config.run_dir / "final_adapter")
    tokenizer.save_pretrained(config.run_dir / "final_adapter")

    final = generative_metrics(model, tokenizer, val_rows, config) if val_rows else {}
    hours = (time.time() - start) / 3600
    summary = {
        "name": config.name,
        "wall_hours": hours,
        "gpu_cost_usd": hours * config.gpu_usd_per_hour,
        "global_step": trainer.state.global_step,
        "best_step": callback.best_step,
        "best_score": callback.best if callback.history else None,
        "final_metrics": final,
        "history": callback.history,
        "provenance": provenance,
        "config": config.model_dump(mode="json"),
    }
    (config.run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if config.report_to == "wandb":
        import wandb

        wandb.summary.update({"gpu_cost_usd": summary["gpu_cost_usd"], "wall_hours": hours})
        wandb.finish()
    return summary


def merge_adapter(
    base_model: str, adapter: Path, output: Path, revision: str | None = None
) -> None:
    """Write merged weights, for comparing against serving the adapter separately in vLLM."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(output)
    AutoTokenizer.from_pretrained(base_model, revision=revision).save_pretrained(output)


def overfit_config(config: TrainConfig, steps: int = 200) -> TrainConfig:
    """Ten examples must reach near-perfect output, which catches template and masking bugs."""
    return config.model_copy(
        update={
            "name": f"{config.name}-overfit",
            "train_size": 10,
            "val_from_train": True,
            "eval_calls": 10,
            "max_steps": steps,
            "grad_accum": 1,
            "warmup_ratio": 0.0,
            "lora_dropout": 0.0,
            "eval_every_steps": steps,
            "early_stopping_patience": 10_000,
            "report_to": "none",
        }
    )


def inspect_example(config: TrainConfig, row: dict[str, Any]) -> dict[str, Any]:
    """Decode one example and its label mask, to read before any long run."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision)
    encoded, report = encode_rows(tokenizer, [row], config)
    if not encoded:
        return {"too_long": report.too_long}
    example = encoded[0]
    trained = [token for token in example["labels"] if token != IGNORE_INDEX]
    masked = len(example["labels"]) - len(trained)
    return {
        "total_tokens": len(example["input_ids"]),
        "masked_tokens": masked,
        "prompt_tail": tokenizer.decode(example["input_ids"][max(0, masked - 40) : masked]),
        "trained_text": tokenizer.decode(trained),
    }
