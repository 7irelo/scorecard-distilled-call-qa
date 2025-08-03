# Data card

**Status: template. No dataset has been generated yet. Fill in the bracketed values from `data/manifest.json` and `data/teacher_qc.json` after a real run.**

## Summary

Synthetic South African contact-centre call transcripts with structured QA labels. All people, companies and account details are invented. No real customer or employer data is used.

| Item | Value |
|---|---|
| Calls | [count] |
| Call types | Billing dispute, outage, cancellation, other |
| Languages | English, English + isiZulu, English + Afrikaans, all three |
| Generator | [model, revision] |
| Teacher | [model, revision] |
| Noise profile | [calibrated from N turns via STT vendor] |

## How it's made

1. **Scenario sampler.** Seeded Python samples each scenario's planted truth before any model runs. Failures are oversampled, with weights recorded to recover population estimates.
2. **Rendering.** A generator model writes the call.
3. **Verification.** A separate reading checks that the transcript honours its spec. Failures are regenerated once, and then rejected.
4. **Noise.** A deterministic injector applies rates calibrated from a speech round trip.
5. **Labelling.** A teacher model from a different family labels the noisy transcripts.

## Known limitations

- **Synthetic.** Results don't transfer automatically to real calls. The reality-check set gives a small estimate of the gap: [result].
- **Code-switching quality depends on the generator.** Native reviewers rated naturalness at [score]. Their notes: [summary].
- **Noise comes from TTS audio,** which is cleaner than real phone calls.
- **Failure rates are not realistic by design.** Use weights for population estimates.
- **The compliance checks are illustrative,** not a statement of South African law.
- **The lexicon check for language mix is coarse.** It undercounts inflected isiZulu.
- **Company names are invented,** but check them against real SA companies before publishing.

## Review

| Review | Who | Result |
|---|---|---|
| isiZulu naturalness | [reviewer] | [score, n] |
| Afrikaans naturalness | [reviewer] | [score, n] |
| Teacher error rate | Phase 4 audit | [per field, with CI] |
