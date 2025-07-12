# ADR 0001: Licensing and terms for the training data path

**Status:** accepted, 2026-11-03, with one open item before paid generation. This is an engineering reading of published terms, not legal advice.

## Context

Everything that feeds student training has to be something we're allowed to train on. That covers generated transcripts as well as teacher labels, because transcripts are the student's training inputs. Terms apply at two separate levels: the model weights' licence, and the terms of whichever hosted API serves them.

## Decision

1. **Closed APIs never enter the training data path.** `enforce_policy` in `src/scorecard/llm/types.py` raises an error if a closed provider is used for anything but `Purpose.BASELINE`, and tests cover it.
2. **Teacher: DeepSeek V4-Pro.** It has an MIT licence and the only hosted terms found that explicitly permit distillation.
3. **Generator: GLM-5,** a different family from the teacher, so their errors correlate less. **Don't run a paid generation until the open item below is closed.**
4. **Judge: a large Qwen,** a third family, set in `configs/llm.yaml` once chosen.
5. **Closed models run only as evaluation baselines,** to compare quality and cost. No output from them is stored as training data.

## Evidence

Checked 2026-11-03.

| Item | Finding | Source |
|---|---|---|
| Anthropic API | Customer may not "access the Services to build a competing product or service, including to train competing AI models" without approval | [Commercial Terms, D.4](https://www.anthropic.com/legal/commercial-terms) |
| OpenAI API | Reported ban on using output to develop competing models. **Secondary source only** | [ConductAtlas](https://conductatlas.com/platform/openai/openai-business-terms/provision/CA-P-059973/prohibition-on-using-output-to-develop-competing-ai-models/) |
| Gemini API | Reported ban on developing competing models. **Secondary source only** | [ConductAtlas](https://conductatlas.com/platform/google-ai-studio/gemini-api-terms-of-service/provision/CA-P-063159/prohibition-on-developing-competing-models/) |
| DeepSeek V4-Pro weights | MIT. The model card calls this a preview release | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| DeepSeek Open Platform | Section 4.2 assigns output rights to the user, and lists "training other models (such as model distillation)" among permitted uses. Effective 2026-04-29 | [Terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html) |
| GLM-5 weights | MIT | [Hugging Face](https://huggingface.co/zai-org/GLM-5) |
| Z.ai API | **Terms not found.** Secondary pages don't address distillation | Open item |
| Qwen3-4B | Apache-2.0. `enable_thinking=False` disables thinking | [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen3.5-4B | Apache-2.0. Multimodal, needs vLLM's main branch | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B) |
| SmolLM3-3B | Apache-2.0. Natively covers six European languages, none of them isiZulu or Afrikaans | [Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| Phi-4-mini | Reported MIT. **Secondary source only** | Check the model card before using |

## Open items

| Item | Blocks | How to close |
|---|---|---|
| Read Z.ai's API terms, or run GLM-5 weights on a host whose terms you've read | Paid generation | Record the clause and date in this table |
| Choose and check a large Qwen for the judge | Phase 8 | Check licence per model size. Sources say it varies |
| Confirm current model IDs and prices with each host | Paid runs | Secondary sources conflict |
| Check data handling for each host | Paid runs | Transcripts are synthetic, so risk is low, but never send real calls to these hosts |

## Consequences

- The project's story becomes "distilled from a large open model", which is cleaner to publish than the brief's original framing.
- Comparison with closed models stays honest: they appear as baselines and aren't trained on.
- If a provider gives written approval, a closed teacher could be added later as a separately labelled experiment. Enabling that would mean changing `enforce_policy` on purpose and recording the approval here.
