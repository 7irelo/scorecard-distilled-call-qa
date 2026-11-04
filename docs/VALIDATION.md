# What has been validated

This records what has actually run, so the code's state isn't confused with real results. Tested 2026-11-04 on a CPU-only Windows machine with Docker.

**None of this measures quality or cost.** The vLLM tests used a tiny model with random weights, so its outputs are nonsense and its timings say nothing about real serving.

## Real software, tiny model

vLLM v0.33.0 (`vllm/vllm-openai-cpu`), serving a 19.7M-parameter random Qwen2 model with a LoRA adapter trained by `scorecard train`:

| Check | Result |
|---|---|
| `bench vllm-check`: adapter and base model served | Pass |
| Schema-constrained output through the adapter and the base model | Pass |
| Over-long call detected from vLLM's real error, returned by the API as 413 | Pass |
| Complete, schema-valid scorecard through client and API | Pass |
| Per-field prediction, assembled into one card | Pass |
| Load generator against the API, 3 of 3 requests | Pass |
| Time-to-first-token streaming | Pass |

## What real vLLM caught that mocks didn't

All fixed:

- Docker's default 64 MB `/dev/shm` stops vLLM from starting, so the compose files set `shm_size`.
- `--max-lora-rank` accepts only 1, 8, 16, 32, 64, 128, 256, 320 or 512, so a rank of 4 must round up to 8.
- `--model` is deprecated in favour of a positional argument.
- Constrained decoding allows endless whitespace, which ran a stuck model to its token limit. The compose files now set `--structured-outputs-config.backend=xgrammar` and `.disable_any_whitespace=true`.
- The schema had no length limits, so a stuck model could run forever inside `summary`. It now has hard caps, including on the per-field schemas.

## Other checks

| Check | Result |
|---|---|
| Training on CPU: full JSON, resume, gradient checkpointing, per-field, heads, merge, overfit, inspect | Pass, and adapter weights change |
| Whole pipeline on stub models | Pass, with reports marked as a dry run |
| API container with rule-based outputs | Pass |
| Tests in a clean install without optional extras | Pass |

## Still unvalidated

| Item | Why | First step |
|---|---|---|
| vLLM on a GPU, with the real student | Needs a GPU | `scorecard bench vllm-check` |
| QLoRA 4-bit training | bitsandbytes needs a GPU | `scorecard train overfit` |
| Training Docker image | The build ran out of memory on the test machine, so it has never been built | `docker build -f docker/train.Dockerfile -t scorecard-train .` |
| DeepSeek, OpenRouter, Azure, Deepgram, Anthropic | Needs API keys. Tested only with mocks | The 50-call pilot |
| Qwen3.8-27B as generator | Needs about 48 GB of GPU memory | The 50-call pilot |

## Real results still needed

Generated data, teacher labels, blind gold labels, native-speaker review, a trained student, the final evaluation and measured costs. These need API keys, a rented GPU and people, as set out in `RUNBOOK.md`. Until they exist, this project makes no claim about quality or cost.
