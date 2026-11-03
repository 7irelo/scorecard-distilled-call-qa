# Scorecard report

**Status: template. No real results exist yet. Copy every number from a generated file, as listed below, and never type one in from memory.**

## Result

[One sentence: quality retained with its interval, whether every floor held, and the volume at which self-hosting pays off.]

From `runs/final/REPORT.md`:

[Paste the systems table.]

## Data quality: how good were the labels?

From `runs/audit/TEACHER_AUDIT_test.md`:

[Paste the teacher error table and adjudication record. Say where the teacher was weakest, and whether it did worse on code-switched calls.]

## What fine-tuning bought

From `runs/final/REPORT.md` and `scorecard train compare`:

- Student vs zero-shot and few-shot on the same base model: [values]
- Student vs rules and TF-IDF: [values]
- Learning curve: where did more data stop helping? [value]
- LoRA rank, output structure, quantization, label filtering and code-switched share: [findings, with the W&B link]

## JSON validity

- Unconstrained, showing what training achieved: [value]
- Constrained, showing what ships: 100% by construction, so semantic accuracy is the number that matters: [value]

## Summary faithfulness

- Judge calibration: sensitivity [value], specificity [value], on [n] human-labelled claims, including injected errors.
- Corrected estimate per system: [values]
- Judge failures: [counts, or none]

## Code-switching

From the language-mix slice: [English vs isiZulu vs Afrikaans scores, with counts]. Native reviewers' naturalness rating: [values].

## Robustness

From `runs/robustness/student.md`:

- Noise sweep: [how quality falls as WER rises]
- Prompt injection: [fail-to-pass flip rate per attack]
- Bad inputs: [invented passes]

## Sim-to-real

From `runs/final/SIM_TO_REAL.md`: [gap, with n]. Role-played calls are not real customer calls, so this is a warning sign and not a guarantee.

## Serving and cost

From `scorecard bench load`, `bench ledger`, `bench costs` and `bench chart`:

- Throughput, p95 latency and TTFT per GPU type: [values]
- Adapter vs merged weights: [values]
- Break-even volume against each API: [values]

[Include `runs/bench/break_even.png`.]

**The honest comparison:** say plainly where the cheapest capable API wins.

## Where it fails

From `runs/final/errors_to_read.jsonl`, after reading and categorising 50 errors: [categories, with real examples].

## Limits

- Synthetic training data.
- Noise calibrated from TTS audio, which is cleaner than real calls.
- Compliance checks are illustrative, not legal.
- Any post-hoc test-set openings: [list, or none].
