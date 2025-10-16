# Runbook

Steps to take this project from working code to real results, in order. Commands marked **PAID** spend money. Set a spend alert with each provider first, and check `runs/cost_ledger.jsonl` as you go.

## Before you start

```powershell
uv sync --python 3.12
copy .env.example .env   # then fill it in, and load it into your shell
git config core.hooksPath .githooks
uv run pytest -m "not train"
uv run scorecard dry-run # free, proves the plumbing with stub models
```

Close the open items in `docs/adr/0001-licensing.md` before paying for generation. Set the real model IDs and prices in `configs/llm.yaml`.

## Phase 1: Rubric (you and a second person)

1. Have a second rater label the 10 calls in `tests/fixtures/rubric_calls.py` without seeing the labels.
2. Turn each disagreement into a clarification in `docs/RUBRIC.md`.

## Phase 2: Data

```powershell
uv run scorecard data sample --n 5000
uv run scorecard data generate --limit 50   # PAID, about $1
```

**Stop here.** Read all 50 calls. Then get them to a native isiZulu reader and a native Afrikaans reader. This pilot tests the riskiest assumption in the project, so don't scale until mean naturalness is at least 3.5 of 5. Failing that, improve the prompts and pilot again. The response cache means unchanged calls aren't paid for twice.

```powershell
uv run scorecard data round-trip --turns 2000    # PAID
uv run scorecard data calibrate data/asr_pairs.jsonl
uv run scorecard data generate                   # PAID, about $90
uv run scorecard data noise
```

Record 20–30 consented role-play calls, transcribe them with real STT, and keep them out of training.

## Phase 3: Teacher

```powershell
uv run scorecard teacher label               # PAID, about $80
uv run scorecard data split
```

Review `data/teacher_qc.json`. Sample disagreements with planted truth and work out which side is wrong.

## Phase 4: Gold (25–30 hours of your time)

```powershell
uv run scorecard gold sample
uv run scorecard gold label --split dev
uv run scorecard gold label --split test
uv run scorecard gold label --split test --rater second   # about 40 calls
uv run scorecard gold agreement data/gold_test_owner.jsonl data/gold_test_second.jsonl
```

Kappa must reach 0.7 on compliance and escalation. Adjudicate against teacher and planted labels only **after** the blind pass, then seal the set:

```powershell
uv run scorecard gold lock data/gold_test_owner.jsonl
```

Commit `docs/gold_test.lock.json`. Never commit the labels.

## Phases 5–7: Baselines and training (GPU)

On a RunPod Linux GPU, with the repo cloned:

```bash
uv sync --python 3.12 --extra train
uv run pytest -m train
uv run scorecard train run configs/train/baseline.yaml
```

Before any long run, overfit 10 examples by setting `train_size: 10` and checking the outputs. Then:

```bash
uv run scorecard train plan-sweep --seeds 42
for config in configs/sweep/generated/*.yaml; do uv run scorecard train run "$config"; done
```

Pick the final configuration on silver validation and gold dev only. Commit the choice before opening test.

## Phase 8: Final evaluation

Serve the student and base model with vLLM, add them to `configs/llm.yaml` under the `student` and `zero_shot` roles, then:

```powershell
uv run scorecard predict offline rules --source data/gold_pool.jsonl
uv run scorecard predict offline tfidf --source data/gold_pool.jsonl
uv run scorecard predict run student --name student --source data/gold_pool.jsonl
uv run scorecard predict run student --name student_unconstrained --source data/gold_pool.jsonl --no-constrained
uv run scorecard predict run zero_shot --name zero_shot --source data/gold_pool.jsonl
uv run scorecard predict run zero_shot --name few_shot --source data/gold_pool.jsonl --examples data/silver_val.jsonl
uv run scorecard predict run baseline_cheap --name cheap_api --source data/gold_pool.jsonl   # PAID
uv run scorecard predict run baseline_top --name top_api --source data/gold_pool.jsonl       # PAID
```

Add the teacher's cards as `runs/predictions/teacher.jsonl`. Then calibrate the judge on human-labelled claims:

```powershell
uv run scorecard eval calibration-set
# Label each summary's claims by hand, as rows of scenario_id, claim and supported.
uv run scorecard eval calibrate data/judge_claims.jsonl     # PAID
```

**Re-check every price, freeze the student, then open test once:**

```powershell
uv run scorecard eval final data/gold_test_owner.jsonl      # PAID, ONE TIME
```

A second opening requires `--post-hoc-reason`, and the report displays it.

## Phases 9–10: Serving and economics

```bash
export SCORECARD_API_KEY=... VLLM_TAG=... BASE_MODEL=... BASE_REVISION=... LORA_RANK=... ADAPTER_DIR=...
docker compose -f docker/compose.yaml up --build
```

Run the load generator (`scorecard.serve.loadgen`) through saturation on two or three GPU types, then copy measured values into a copy of `benchmarks/costs.example.yaml`, with `measured: true` on each:

```powershell
uv run scorecard bench costs --inputs benchmarks/costs.measured.yaml
```

The report warns until every value is measured.

## Phase 11: Publish

Fill in `docs/DATA_CARD.md` and `docs/MODEL_CARD.md` from generated reports, and put the results table first in the README. Pushing is your decision.
