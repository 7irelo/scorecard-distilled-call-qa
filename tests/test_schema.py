import json

import pytest
from pydantic import ValidationError

from scorecard.schema import Scorecard, flatten_labels, scorecard_json_schema, semantic_issues


def card(**overrides):
    base = {
        "summary": "Customer disputed a charge and it was reversed.",
        "call_type": "billing_dispute",
        "sentiment": {
            "opening": "negative",
            "middle": "neutral",
            "closing": "positive",
            "trajectory": "improving",
        },
        "compliance": {
            name: {"status": "pass", "evidence_turns": [1]}
            for name in (
                "agent_introduced",
                "recording_disclosed",
                "identity_verified",
                "terms_explained",
                "next_steps_confirmed",
            )
        },
        "escalation": {"flag": False, "reason": "none"},
    }
    base.update(overrides)
    return Scorecard.model_validate(base)


def test_round_trip_is_stable():
    original = card()
    assert Scorecard.model_validate_json(original.canonical_json()) == original
    assert list(json.loads(original.canonical_json()))[:2] == ["schema_version", "summary"]


def test_extra_fields_and_bad_enums_rejected():
    with pytest.raises(ValidationError):
        card(extra="nope")
    with pytest.raises(ValidationError):
        card(call_type="sales")


def test_schema_caps_runaway_output():
    with pytest.raises(ValidationError):
        card(summary="x" * 1201)
    schema = scorecard_json_schema()
    assert schema["properties"]["summary"]["maxLength"] == 1200
    assert schema["$defs"]["ComplianceCheck"]["properties"]["evidence_turns"]["maxItems"] == 10


def test_json_schema_is_closed():
    schema = scorecard_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Compliance"]["additionalProperties"] is False


def test_coherent_card_has_no_issues():
    assert semantic_issues(card(), {1, 2}) == []


@pytest.mark.parametrize(
    ("overrides", "issue"),
    [
        ({"summary": "word " * 81}, "summary_too_long"),
        ({"escalation": {"flag": True, "reason": "none"}}, "escalation_flag_reason_mismatch"),
        (
            {
                "sentiment": {
                    "opening": "positive",
                    "middle": "neutral",
                    "closing": "negative",
                    "trajectory": "improving",
                }
            },
            "trajectory_contradicts_segments",
        ),
    ],
)
def test_semantic_issues(overrides, issue):
    assert issue in semantic_issues(card(**overrides))


def test_evidence_rules():
    compliance = card().compliance.model_dump()
    compliance["identity_verified"] = {"status": "pass", "evidence_turns": []}
    compliance["terms_explained"] = {"status": "not_applicable", "evidence_turns": [1]}
    compliance["agent_introduced"] = {"status": "pass", "evidence_turns": [99]}
    issues = semantic_issues(card(compliance=compliance), {1})
    assert "identity_verified_pass_without_evidence" in issues
    assert "terms_explained_not_applicable_with_evidence" in issues
    assert "agent_introduced_evidence_turn_missing" in issues


def test_flatten_covers_every_categorical_field():
    flat = flatten_labels(card())
    assert flat["compliance.identity_verified"] == "pass"
    assert flat["escalation.flag"] == "false"
    assert len(flat) == 12
