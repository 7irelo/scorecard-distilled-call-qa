# Runbook

Steps to take this project from working code to real results, in order. Commands marked **PAID** spend money. Set a spend alert with each provider first, and check `runs/cost_ledger.jsonl` as you go.

Each step ends with what must be true before moving on.

## Before you start

```powershell
uv sync --python 3.12
copy .env.example .env   # then fill it in, and load it into your shell
git config core.hooksPath .githooks
uv run pytest -m "not train"
uv run scorecard dry-run # free, proves the plumbing with stub models
```

Close the open items in `docs/adr/0001-licensing.md`, and re-check the IDs and prices in `configs/llm.yaml`.

Generation runs on your own GPU, because Z.ai's terms rule out its API and self-hosting avoids any host's terms. Start it on a rented GPU with about 48 GB and point `qwen_generator.base_url` at it:

```bash
VLLM_TAG=... GENERATOR_REVISION=... docker compose -f docker/compose.generator.yaml up
```

Stop the pod when you aren't generating, as it bills by the hour.

## Phase 1: Rubric (you and a second person)

A second rater labels the 10 edge-case calls without seeing the draft labels. The command then lists every disagreement:

```powershell
uv run scorecard gold rubric-check --rater second
```

Turn each disagreement into a clarification in `docs/RUBRIC.md`, and have a native speaker check the isiZulu and Afrikaans lines in those calls.

**Gate:** most fields agree, and each disagreement has become a rubric clarification.

## Phase 2: Data

```powershell
uv run scorecard data sample --n 5000
uv run scorecard data generate --limit 50          # PAID: GPU hours, plus DeepSeek to verify
uv run scorecard data review-sheet zu --n 25
uv run scorecard data review-sheet af --n 25
```

**Stop here.** Read all 50 calls, then send the CSVs to native isiZulu and Afrikaans readers. This pilot tests the riskiest assumption in the project. When the sheets come back:

```powershell
uv run scorecard data review-score data/review_zu.csv
uv run scorecard data review-score data/review_af.csv
```

**Gate:** mean naturalness of at least 3.5 of 5. If not, improve the prompts and pilot again. The response cache means unchanged calls aren't paid for twice. Scale only after this passes:

```powershell
uv run scorecard data round-trip --turns 2000      # PAID
uv run scorecard data calibrate data/asr_pairs.jsonl
uv run scorecard data generate                     # PAID: GPU hours, plus DeepSeek to verify
uv run scorecard data quality --drop
uv run scorecard data check-noise
uv run scorecard data noise
```

**Gate:** planted-spec adherence of at least 95% and noise WER within 5 points of the measured rate. Repeat the review sheets at full scale, with about 50 calls each.

### Reality-check set

Record 20–30 role-played calls with consenting participants. Keep consent records out of git. Don't use employer call data without written approval. List each recording in a manifest with rows of `file`, `call_id`, `consent_id` and `agent_speaker`, then:

```powershell
uv run scorecard data transcribe-reality recordings/manifest.jsonl   # PAID
```

These calls can't enter training: the training code raises an error if one appears.

## Phase 3: Teacher

Label dev gold first (Phase 4a below), then build 3–5 worked examples with `scorecard gold examples` and compare candidates. Add each candidate as a role in `configs/llm.yaml`, keeping it a different family from the generator:

```powershell
uv run scorecard teacher bakeoff teacher,candidate_b --gold data/gold_dev_owner.jsonl   # PAID
uv run scorecard teacher label --examples data/teacher_examples.jsonl                  # PAID, about $80
uv run scorecard data split
```

Review `data/teacher_qc.json`, and sample disagreements with planted truth to work out which side is wrong.

**Gate:** the teacher meets the floors on dev gold, and its disagreements with planted truth are explained.

## Phase 4: Gold (25–30 hours of your time)

```powershell
uv run scorecard gold sample
uv run scorecard gold label --split dev            # 4a: for calibration and teacher choice
uv run scorecard gold label --split test           # 4b: blind, never showing teacher output
uv run scorecard gold label --split test --rater second --limit 40
uv run scorecard gold agreement data/gold_test_owner.jsonl data/gold_test_second.jsonl
```

**Gate:** kappa of at least 0.7 on compliance and escalation. If it's lower, fix the rubric.

Adjudicate only after the blind pass. The command refuses to start before it's finished:

```powershell
uv run scorecard gold adjudicate data/gold_test_owner.jsonl
uv run scorecard gold finalize data/gold_test_owner.jsonl
uv run scorecard eval label-claims data/teacher.jsonl --split test --output data/teacher_claims.jsonl
uv run scorecard gold audit data/gold_test_final.jsonl --claims data/teacher_claims.jsonl
uv run scorecard gold lock data/gold_test_final.jsonl
```

`runs/audit/TEACHER_AUDIT_test.md` is the teacher's error rate per field, one of the project's main results. Commit `docs/gold_test.lock.json`, and never commit the labels.

Do the same for dev (`adjudicate --split dev`, `finalize --split dev`), without locking it.

## Phases 5–7: Baselines and training (GPU)

Calibrate the judge first, on non-test data:

```powershell
uv run scorecard eval calibration-set
uv run scorecard eval label-claims data/judge_calibration_set.jsonl
uv run scorecard eval calibrate data/judge_claims.jsonl   # PAID
```

On a RunPod Linux GPU, with the repo cloned, or using `docker/train.Dockerfile`:

```bash
uv sync --python 3.12 --extra train
uv run pytest -m train                                  # CPU smoke test on a tiny model

uv run scorecard train bakeoff-plan                     # refuses unverified licences
uv run scorecard train tokens Qwen/Qwen3-4B             # check tokens per word on isiZulu
for config in configs/sweep/bakeoff/*.yaml; do uv run scorecard train run "$config"; done
uv run scorecard train compare --prefix bakeoff-
```

Set the winner in `configs/train/baseline.yaml`, with `revision` pinned to a commit hash. Then, before any long run:

```bash
uv run scorecard train inspect configs/train/baseline.yaml   # read the decoded example and mask
uv run scorecard train overfit configs/train/baseline.yaml   # must reach near-perfect output
uv run scorecard train run configs/train/baseline.yaml
```

**Gate:** the student beats zero-shot on validation, and a run reproduces from a clean pod.

Ablations:

```bash
uv run scorecard train plan-sweep --seeds 42
for config in configs/sweep/generated/*.yaml; do uv run scorecard train run "$config"; done
uv run scorecard train compare
```

Re-run the finalists with `--seeds 42,43,44`. Score systems on dev gold as often as you like:

```bash
uv run scorecard eval select data/gold_dev_final.jsonl
```

Then lock in your choice. `eval final` refuses to run until this file exists:

```bash
uv run scorecard train freeze runs/<chosen-run>
git add configs/train/frozen.yaml && git commit -m "chore: freeze chosen student config"
```

## Phase 8: Final evaluation

Serve the student and base model with vLLM, then produce predictions for every system:

```powershell
uv run scorecard predict teacher
uv run scorecard predict offline rules --source data/gold_pool.jsonl
uv run scorecard predict offline tfidf --source data/gold_pool.jsonl
uv run scorecard predict run student --name student --source data/gold_pool.jsonl
uv run scorecard predict run student --name student_unconstrained --source data/gold_pool.jsonl --no-constrained
uv run scorecard predict run zero_shot --name zero_shot --source data/gold_pool.jsonl
uv run scorecard predict run zero_shot --name few_shot --source data/gold_pool.jsonl --examples data/silver_val.jsonl
uv run scorecard predict run baseline_cheap --name cheap_api --source data/gold_pool.jsonl   # PAID
uv run scorecard predict run baseline_top --name top_api --source data/gold_pool.jsonl       # PAID
```

If the chosen arm is per-field, add `--per-field`. For heads mode, use `predict heads`.

**Re-check every price, then open test once:**

```powershell
uv run scorecard eval final data/gold_test_final.jsonl     # PAID, ONE TIME
```

A second opening requires `--post-hoc-reason`, and the report displays it.

Then:

```powershell
uv run scorecard eval robustness student --gold data/gold_test_final.jsonl
uv run scorecard gold label --reality
uv run scorecard predict run student --name student --source data/reality.jsonl --out runs/predictions_reality
uv run scorecard eval reality data/gold_reality_owner.jsonl
```

Read `runs/final/errors_to_read.jsonl` and fill in each `category`. Real failures belong in the report.

## Phases 9–10: Serving and economics

```bash
export SCORECARD_API_KEY=... VLLM_TAG=... BASE_MODEL=... BASE_REVISION=... LORA_RANK=... ADAPTER_DIR=...
docker compose -f docker/compose.yaml up --build
uv run scorecard bench vllm-check --url http://127.0.0.1:8000/v1   # with vLLM's port exposed
uv run scorecard demo
```

Round `LORA_RANK` up to a value vLLM accepts (a rank 4 adapter uses 8). Compare adapter serving with merged weights by running the same load test against `docker/compose.merged.yaml`. To compare bf16 with 8-bit, quantize the merged model and point `MERGED_DIR` at it.

Run on two or three GPU types:

```bash
uv run scorecard bench load --rates 0.25,0.5,1,2,4 --duration 120
uv run scorecard bench ttft
uv run scorecard bench ledger    # measured tokens and latency for each API
```

Copy the measured values into `benchmarks/costs.measured.yaml`, based on `costs.example.yaml`, with `measured: true` on each option and `quality_retained` from the final report. The report warns until every value is measured.

```powershell
uv run --extra report scorecard bench costs --inputs benchmarks/costs.measured.yaml
uv run --extra report scorecard bench chart --inputs benchmarks/costs.measured.yaml
```

## Phase 11: Publish

Fill in `docs/DATA_CARD.md` and `docs/MODEL_CARD.md` from generated reports, and put the results table first in the README. Check every licence before publishing a dataset sample or adapter. Pushing is your decision.
