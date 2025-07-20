# Pre-registered metrics

Committed 2026-11-03, before any data generation or student training. Changes after this point must be listed under [Amendments](#amendments), with the date and reason. Changes after the test set is opened must also be disclosed in the report as post-hoc.

The scoring code is `src/scorecard/eval/fields.py`.

## Primary quantities

Scored against **blind human gold** (Phase 4), weighted by each call's design weight.

| Component | Weight | Definition |
|---|---:|---|
| Compliance | 0.40 | Mean macro-F1 across the five checks, over classes present in gold |
| Escalation | 0.20 | F1 of `flag = true` |
| Sentiment | 0.15 | Mean macro-F1 across the opening, middle and closing segments |
| Summary | 0.25 | Mean share of a summary's claims judged supported, corrected for judge error |

```text
Q = 0.40·compliance + 0.20·escalation + 0.15·sentiment + 0.25·summary
quality_retained = Q_student / Q_teacher    (paired bootstrap, 95% CI)
```

An unparseable or missing output counts as wrong on every field. It is never dropped.

## Floors

The headline counts only if the student meets **every** floor.

| Floor | Threshold |
|---|---:|
| Recall on compliance `fail` | ≥ 0.85 |
| Escalation recall | ≥ 0.85 |
| Summary faithfulness | ≥ 0.90 |
| Evidence precision against gold | ≥ 0.90 |

## Reported, without a floor

| Metric | Hypothesis |
|---|---|
| Compliance and sentiment macro-F1 | Within 5 points of the teacher |
| Unconstrained JSON validity | ≥ 95% |
| Constrained JSON validity | 100% (true by construction, so reported only for completeness) |
| English-only score minus code-switched score | Gap ≤ 5 points, per language |
| Sentiment ordinal MAE | Reported only |

## Comparisons

Three separate questions, never merged:

1. **Student vs gold:** quality.
2. **Student vs teacher:** fidelity.
3. **Teacher vs gold:** label quality.

Per-field student vs teacher comparisons use exact McNemar with Holm correction. All slices (language mix, call type, length, noise, reality-check set) are exploratory and are shown with their counts.

## Stop rule

If the best configuration gives `quality_retained` below 0.80, try in order: a 7–9B student, per-field prompts, then a cascade to the teacher. If none works, publish the negative result.

## Gates

| Gate | Threshold |
|---|---:|
| Native-speaker naturalness | Mean ≥ 3.5 of 5 |
| Planted-spec adherence | ≥ 95% |
| Inter-rater kappa on compliance and escalation | ≥ 0.70 |

## Amendments

None yet.
