# Scorecard

Distill a large open LLM into a small model that scores South African contact-centre calls, then measure how good it is, what it costs, and when it's worth running.

**Input:** a call transcript, which may be noisy and mix English, isiZulu and Afrikaans.
**Output:** a QA scorecard with a summary, sentiment trajectory, compliance checks with evidence, and an escalation flag.

## Status

**The code is built. There are no real results yet.** No real data has been generated and no real student has been trained, so this repo doesn't claim any quality or cost figures. Getting them needs API keys, a rented GPU and about 30 hours of human labelling, as set out in [the runbook](docs/RUNBOOK.md).

## Try it

```powershell
uv sync --python 3.12
uv run pytest -m "not train"
uv run scorecard dry-run        # the whole pipeline on stub models: free, and takes seconds
```

The dry run proves the plumbing only. Its reports carry a banner saying so.

To exercise the training code on CPU with a tiny model, which also proves nothing about quality:

```powershell
uv sync --python 3.12 --extra train
uv run pytest -m train
```

To try the API without a GPU:

```powershell
$env:SCORECARD_API_KEY = "local"
docker compose -f docker/compose.fake.yaml up --build
uv run scorecard demo
```

## How it works

1. **Data:** seeded scenarios with planted ground truth, rendered by one model and checked by another, with speech-recognition noise calibrated from a real round trip.
2. **Teacher:** a large open model from a different family labels the calls.
3. **Gold:** a blind human audit measures the teacher's error rate, and its test set is sealed by hash.
4. **Student:** QLoRA on a small open model with ablations for dataset size, LoRA rank and output structure.
5. **Evaluation:** per-field metrics, calibrated summary faithfulness, and comparison with rules, TF-IDF, zero-shot, the teacher and closed APIs.
6. **Serving:** vLLM and FastAPI in Docker, load tests, and a cost model across volume and utilisation.

## Design choices worth reading

- [Licensing ADR](docs/adr/0001-licensing.md): why closed APIs stay out of the training data path, enforced in code
- [Pre-registered metrics](docs/METRICS.md): fixed before training
- [Rubric](docs/RUBRIC.md): the definition of each field

## Limits

The data is synthetic, the compliance checks are illustrative rather than legal, and the code-switched text still needs native-speaker review.

## License

MIT
