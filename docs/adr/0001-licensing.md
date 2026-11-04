# ADR 0001: Licensing and terms for the training data path

**Status:** accepted, 2026-11-04. Revised after reading Z.ai's terms. This is an engineering reading of published terms, not legal advice.

## Context

Everything that feeds student training has to be something we're allowed to train on. That covers generated transcripts as well as teacher labels, because transcripts are the student's training inputs. Terms apply at two separate levels: the model weights' licence, and the terms of whichever hosted API serves them.

## Decision

1. **Closed APIs never enter the training data path.** `enforce_policy` in `src/scorecard/llm/types.py` raises an error if a closed provider is used for anything but `Purpose.BASELINE`, and tests cover it.
2. **Teacher and verifier: DeepSeek V4-Pro on DeepSeek's API.** It has an MIT licence, and the only hosted terms found that explicitly permit distillation.
3. **Generator: Qwen3.8-27B, self-hosted.** It's Apache-2.0 and a different family from the teacher, so their errors correlate less. Self-hosting means no host's terms apply.
4. **Judge: GLM-5 weights through OpenRouter, with Z.ai excluded.** A third family, used only to evaluate summaries, so its output never trains anything. Z.ai's hosted API is not used at all, because of its terms.
5. **Closed models run only as evaluation baselines,** to compare quality and cost. No output from them is stored as training data.

## Evidence

Checked 2026-11-03 and 2026-11-04.

| Item | Finding | Source |
|---|---|---|
| Anthropic API | Customer may not "access the Services to build a competing product or service, including to train competing AI models" without approval | [Commercial Terms, D.4](https://www.anthropic.com/legal/commercial-terms) |
| OpenAI API | Reported ban on using output to develop competing models. **Secondary source only** | [ConductAtlas](https://conductatlas.com/platform/openai/openai-business-terms/provision/CA-P-059973/prohibition-on-using-output-to-develop-competing-ai-models/) |
| Gemini API | Reported ban on developing competing models. **Secondary source only** | [ConductAtlas](https://conductatlas.com/platform/google-ai-studio/gemini-api-terms-of-service/provision/CA-P-063159/prohibition-on-developing-competing-models/) |
| DeepSeek V4-Pro weights | MIT | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| DeepSeek Open Platform | Section 4.2 assigns output rights to the user, and lists "training other models (such as model distillation)" among permitted uses. Effective 2026-04-29 | [Terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html) |
| DeepSeek API details | Base URL `https://api.deepseek.com`, model `deepseek-v4-pro`, $1.32/$3.96 per MTok at peak and half off-peak. `json_object` output, which can return blank content | [Pricing](https://api-docs.deepseek.com/quick_start/pricing), [JSON mode](https://api-docs.deepseek.com/guides/json_mode) |
| Z.ai API | **Prohibits this use.** Additional Terms 1(xii) bars using its "models, prompts, or model-generated content for the development, training, labeling, fine-tuning" of external models. Updated 2026-04-14 | [Terms](https://docs.z.ai/legal-agreement/terms-of-use) |
| GLM-5 weights | MIT | [Hugging Face](https://huggingface.co/zai-org/GLM-5) |
| OpenRouter | Terms defer to each model's terms (5.1, 5.8), and say nothing directly on training with outputs. Updated 2026-08-31 | [Terms](https://openrouter.ai/terms) |
| Qwen3.8-27B | The model card labels it Apache-2.0, and states no training restriction. Read the full licence file before running | [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Qwen3-4B | Apache-2.0. `enable_thinking=False` disables thinking | [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen3.5-4B | Apache-2.0. Multimodal, needs vLLM's main branch | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B) |
| SmolLM3-3B | Apache-2.0. Natively covers six European languages, none of them isiZulu or Afrikaans | [Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| Phi-4-mini | Reported MIT. **Secondary source only.** Blocked in the bake-off until its card is read | `configs/train/bakeoff.yaml` |

## Open items

| Item | Blocks | How to close |
|---|---|---|
| Confirm the OpenRouter model ID for GLM-5, and that it has a host other than Z.ai | Phase 8 | The ID comes from a third-party listing. If no other host exists, choose another third-family model |
| Re-check prices on the day of each paid run | Paid runs | Prices change |
| Check data handling for each host | Paid runs | Transcripts are synthetic, so risk is low, but never send real calls to these hosts |

## Consequences

- The project's story becomes "distilled from a large open model", which is cleaner to publish than the brief's original framing.
- Generation costs GPU hours in place of API tokens, and needs a GPU with enough memory for a 27B model.
- Comparison with closed models stays honest: they appear as baselines and aren't trained on.
- If a provider gives written approval, a closed teacher could be added later as a separately labelled experiment. Enabling that would mean changing `enforce_policy` on purpose and recording the approval here.
