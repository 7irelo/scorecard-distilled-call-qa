import json

import pytest
import yaml

from scorecard.data.models import Transcript
from scorecard.train.config import (
    LabelFilter,
    OutputMode,
    TrainConfig,
    ablation_plan,
    load_train_config,
    write_sweep,
)
from scorecard.train.format import (
    IGNORE_INDEX,
    assemble_fields,
    encode_example,
    encode_rows,
    expand_row,
    full_messages,
    per_field_schema,
    per_field_targets,
    render_prompt,
    select_training_rows,
)
from tests.fixtures.rubric_calls import FIXTURES


class CharTokenizer:
    """One token per character, enough to test masking and length rules without a download."""

    eos_token = "<eos>"

    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _row(index=0, mix="en", clean=True):
    transcript, card = FIXTURES[index % len(FIXTURES)]
    return {
        "scenario_id": f"id{index}",
        "transcript": transcript.model_dump(mode="json"),
        "card": card.model_dump(mode="json"),
        "language_mix": mix,
        "qc_clean": clean,
    }


def test_loss_applies_only_to_assistant_tokens():
    tokenizer = CharTokenizer()
    messages = full_messages(FIXTURES[0][0])
    example = encode_example(tokenizer, messages, "TARGET", 100_000)
    prompt_length = len(render_prompt(tokenizer, messages))
    assert example["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    labelled = "".join(chr(i) for i in example["labels"][prompt_length:])
    assert labelled == "TARGET<eos>"
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])


def test_over_long_examples_are_reported_and_never_truncated():
    config = TrainConfig(max_seq_length=50)
    encoded, report = encode_rows(CharTokenizer(), [_row()], config)
    assert encoded == []
    assert report.too_long == 1


def test_thinking_is_disabled_in_the_template():
    tokenizer = CharTokenizer()
    render_prompt(tokenizer, full_messages(FIXTURES[0][0]))
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert tokenizer.template_kwargs["add_generation_prompt"] is True


def test_per_field_targets_rebuild_the_card():
    card = FIXTURES[3][1]
    assert assemble_fields(per_field_targets(card)) == card


def test_per_field_rows_expand_to_every_field():
    config = TrainConfig(output_mode=OutputMode.PER_FIELD)
    pairs = list(expand_row(_row(), config))
    assert len(pairs) == 9
    assert "compliance.identity_verified" in pairs[-3][0][0]["content"]
    assert json.loads(pairs[0][1])["summary"]


@pytest.mark.parametrize(
    "field", ["summary", "call_type", "sentiment", "escalation", "compliance.terms_explained"]
)
def test_per_field_schemas_exist(field):
    assert per_field_schema(field)["type"] == "object"


def test_sizes_are_nested_and_independent_of_seed():
    rows = [_row(i) for i in range(40)]
    small = select_training_rows(rows, TrainConfig(train_size=10))
    large = select_training_rows(rows, TrainConfig(train_size=20, seed=99))
    assert {r["scenario_id"] for r in small} < {r["scenario_id"] for r in large}


def test_asking_for_too_many_rows_fails():
    with pytest.raises(ValueError):
        select_training_rows([_row()], TrainConfig(train_size=5))


def test_clean_filter_and_missing_labels():
    rows = [_row(0, clean=True), _row(1, clean=False), {**_row(2), "card": None}]
    assert len(select_training_rows(rows, TrainConfig())) == 2
    clean = select_training_rows(rows, TrainConfig(label_filter=LabelFilter.CLEAN))
    assert [r["scenario_id"] for r in clean] == ["id0"]


def test_code_switch_share_reduces_only_switched_calls():
    rows = [_row(i, mix="en") for i in range(10)] + [_row(i + 10, mix="en_zu") for i in range(20)]
    picked = select_training_rows(rows, TrainConfig(code_switch_share=0.25))
    assert sum(r["language_mix"] == "en" for r in picked) == 10
    assert sum(r["language_mix"] == "en_zu" for r in picked) == 5


def test_ablation_plan_covers_each_factor_and_gets_unique_names():
    runs = ablation_plan()
    assert {r["factor"] for r in runs} == {
        "A_size",
        "B_rank",
        "C_output",
        "D_quant",
        "E_labels",
        "F_codeswitch",
    }
    names = [r["name"] for r in runs]
    assert len(names) == len(set(names))
    assert len(ablation_plan((1, 2, 3))) == 3 * len(runs)


def test_sweep_files_load_back_as_valid_configs(tmp_path):
    paths = write_sweep(TrainConfig(), tmp_path)
    assert len(paths) == len(ablation_plan())
    loaded = [load_train_config(path) for path in paths]
    assert {c.lora_rank for c in loaded} == {4, 8, 16, 32, 64}
    assert OutputMode.HEADS in {c.output_mode for c in loaded}


def test_alpha_defaults_to_twice_rank():
    assert TrainConfig(lora_rank=8).alpha == 16
    assert TrainConfig(lora_rank=8, lora_alpha=4).alpha == 4


def test_repo_baseline_config_is_valid():
    with open("configs/train/baseline.yaml", encoding="utf-8") as handle:
        assert TrainConfig.model_validate(yaml.safe_load(handle)).quantization.value == "nf4"


def test_transcript_round_trip_through_row():
    row = _row()
    assert Transcript.model_validate(row["transcript"]) == FIXTURES[0][0]


def test_per_field_schemas_are_bounded():
    assert per_field_schema("summary")["properties"]["summary"]["maxLength"] == 1200
    assert "enum" in str(per_field_schema("call_type"))
