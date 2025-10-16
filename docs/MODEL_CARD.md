# Model card

**Status: template. No student has been trained on real data yet. Fill in every bracketed value from generated reports, never by hand-typing results.**

| Item | Value |
|---|---|
| Base model | [name, revision hash, licence] |
| Method | QLoRA via TRL and PEFT |
| Adapter | rank [r], alpha [a] |
| Training data | [count], dataset hash [sha256] |
| Teacher | [model, revision] |
| Run | [W&B link] |

## Intended use

Research and portfolio demonstration of model distillation for post-call QA scorecards on South African contact-centre transcripts. A human should review any result before it affects a person.

## Not intended for

- Automated decisions about employees or customers without human review
- Legal or regulatory compliance determinations
- Languages or domains outside the training data

## Results

Copy from the generated `REPORT.md`.

| Metric | Student | Teacher |
|---|---|---|
| Quality retained | [value, CI] | n/a |
| Compliance fail recall | [value] | [value] |
| Escalation recall | [value] | [value] |
| Summary faithfulness | [value] | [value] |
| English vs code-switched gap | [value] | [value] |

## Limitations

- Trained on synthetic data. Reality-check result: [value, n].
- Known failure patterns from error analysis: [list].
- Prompt-injection test result: [value].
- Cost figures are valid only at the stated volume and utilisation.
