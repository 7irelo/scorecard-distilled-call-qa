"""Real training on CPU with a tiny random model. Needs the `train` extra and a small download.

This proves the training code runs end to end. It says nothing about model quality.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

from scorecard.io import write_jsonl
from scorecard.schema import Scorecard
from scorecard.train.config import (
    OutputMode,
    Quantization,
    TrainConfig,
)
from tests.fixtures.rubric_calls import FIXTURES

pytestmark = pytest.mark.train
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def rows_path(tmp_path):
    rows = [
        {
            "scenario_id": f"id{index}",
            "transcript": transcript.model_dump(mode="json"),
            "card": card.model_dump(mode="json"),
            "language_mix": "en",
            "qc_clean": True,
        }
        for index, (transcript, card) in enumerate(FIXTURES[:4])
    ]
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    return path


def _config(rows_path, tmp_path, **overrides):
    values = {
        "name": "smoke",
        "base_model": TINY_MODEL,
        "train_path": rows_path,
        "val_path": rows_path,
        "output_dir": tmp_path / "runs",
        "quantization": Quantization.NONE,
        "lora_rank": 2,
        "epochs": 1,
        "max_steps": 2,
        "grad_accum": 1,
        "eval_every_steps": 2,
        "eval_calls": 1,
        "max_new_tokens": 8,
        "report_to": "none",
        "gradient_checkpointing": False,
    }
    return TrainConfig(**{**values, **overrides})


def test_full_json_training_runs_and_saves_an_adapter(rows_path, tmp_path):
    from safetensors.torch import load_file

    from scorecard.train.run import run_training

    config = _config(rows_path, tmp_path)
    summary = run_training(config)
    assert summary["global_step"] == 2
    assert summary["provenance"]["encoded"] == 4
    assert summary["provenance"]["device"] == "cpu"
    assert (config.run_dir / "final_adapter" / "adapter_config.json").exists()
    assert (config.run_dir / "best_adapter" / "adapter_config.json").exists()
    # LoRA's B matrices start at zero, so non-zero values prove gradients reached the adapter.
    weights = load_file(config.run_dir / "final_adapter" / "adapter_model.safetensors")
    trained = [tensor for name, tensor in weights.items() if "lora_B" in name]
    assert trained
    assert any(tensor.abs().sum() > 0 for tensor in trained)
    written = json.loads((config.run_dir / "run_summary.json").read_text())
    assert "val/validity" in written["final_metrics"]


def test_training_continues_from_a_checkpoint(rows_path, tmp_path):
    from scorecard.train.run import run_training

    run_training(_config(rows_path, tmp_path))
    summary = run_training(_config(rows_path, tmp_path, max_steps=3), resume=True)
    assert summary["global_step"] == 3


def test_gradient_checkpointing_runs(rows_path, tmp_path):
    from scorecard.train.run import run_training

    summary = run_training(_config(rows_path, tmp_path, gradient_checkpointing=True))
    assert summary["global_step"] == 2


def test_per_field_training_evaluates_with_per_field_prompts(rows_path, tmp_path):
    from scorecard.train.run import generate_card, load_model_and_tokenizer, run_training

    config = _config(rows_path, tmp_path, output_mode=OutputMode.PER_FIELD, name="field")
    summary = run_training(config)
    assert summary["provenance"]["encoded"] == 4 * 9
    model, tokenizer = load_model_and_tokenizer(config)
    _, valid, text = generate_card(model, tokenizer, FIXTURES[0][0], config)
    assert not valid  # An untrained tiny model can't produce JSON.
    assert set(json.loads(text)) >= {"summary", "compliance.identity_verified"}


def test_heads_training_runs_saves_and_predicts(rows_path, tmp_path):
    from scorecard.train.heads import load_heads_model, predict_heads, run_heads_training

    config = _config(rows_path, tmp_path, output_mode=OutputMode.HEADS, name="heads")
    summary = run_heads_training(config)
    assert summary["steps"] == 2
    assert (config.run_dir / "heads.pt").exists()
    assert "val/validity" in summary["final_metrics"]
    model, tokenizer = load_heads_model(config)
    card, valid = predict_heads(model, tokenizer, FIXTURES[0][0], config)
    assert card is None or isinstance(card, Scorecard)
    assert valid == (card is not None)


def test_adapter_merge_writes_a_loadable_model(rows_path, tmp_path):
    from transformers import AutoModelForCausalLM

    from scorecard.train.run import merge_adapter, run_training

    config = _config(rows_path, tmp_path)
    run_training(config)
    output = tmp_path / "merged"
    merge_adapter(TINY_MODEL, config.run_dir / "final_adapter", output)
    assert AutoModelForCausalLM.from_pretrained(output) is not None


def test_overfit_config_learns_on_the_training_rows(rows_path, tmp_path):
    from scorecard.train.run import overfit_config, run_training

    base = _config(rows_path, tmp_path, train_size=None)
    config = overfit_config(base, steps=2).model_copy(update={"train_size": 4, "eval_calls": 1})
    summary = run_training(config)
    assert config.val_from_train
    assert "val/field_agreement" in summary["final_metrics"]


def test_inspect_shows_the_label_mask(rows_path, tmp_path):
    from scorecard.io import read_jsonl
    from scorecard.train.run import inspect_example

    config = _config(rows_path, tmp_path)
    result = inspect_example(config, next(read_jsonl(rows_path)))
    assert result["masked_tokens"] < result["total_tokens"]
    assert '"schema_version"' in result["trained_text"]
    assert "schema_version" not in result["prompt_tail"]
