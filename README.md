---
title: Qwen Image Bench Judge
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
license: apache-2.0
short_description: Judge a generated image with the Qwen-Image-Bench model
---

# Qwen-Image-Bench — Q-Judger

Upload a text-to-image generation together with the prompt that produced it, and
[Q-Judger](https://huggingface.co/Qwen/Qwen-Image-Bench) scores it across the
benchmark's five level-1 dimensions: **Quality**, **Aesthetics**, **Alignment**,
**Real-world Fidelity**, and **Creative Generation**.

Q-Judger is a 27B vision-language judge fine-tuned from Qwen3.6-27B. It reports
Spearman ρ = 0.92 against human expert rankings overall
([paper](https://arxiv.org/abs/2605.28091)).

## How scoring works

The model grades each level-3 criterion on a three-point scale, then scores are
averaged up the hierarchy:

| Verdict | Meaning | Score |
| --- | --- | --- |
| 0 | Fail — clear defect | 0 |
| 1 | Pass — meets baseline | 1 |
| 2 | Excel — concretely excellent | 2 |
| N/A | Criterion doesn't apply | excluded from the average |

Level-3 → level-2 → level-1 → total, each step an arithmetic mean over the
non-N/A children.

Scores are reported on this raw **0–2** scale. The paper maps the same verdicts to
0/60/100, which is *not* a rescale of 0–2 — it is non-linear, so verdicts {1, 2}
average to 1.5 here and to 80 upstream. The paper-scale equivalent is shown beneath
the total so the two stay comparable.

The model runs with thinking enabled, so it reasons before emitting its JSON
verdict; that reasoning is shown in the **Judge reasoning** panel.

## The prompt is required

The judge scores the image *against the prompt that generated it*. Alignment and
Creative Generation compare what was asked for to what was produced, so without a
prompt those dimensions are meaningless. If you don't have the original prompt,
deselect those dimensions and read only Quality and Aesthetics.

## Accuracy caveat

This Space runs the model in **4-bit NF4** so it fits ZeroGPU's 48GB slice — the
full bf16 weights are ~55GB. The vision tower and output head stay in bf16, so
image understanding is unaffected, but quantizing the language model can shift
verdicts near a decision boundary. Scores track the paper's closely; they are not
bit-exact reproductions. For reference numbers, run the
[upstream pipeline](https://github.com/QwenLM/Qwen-Image-Bench) in bf16.

Two other deliberate deviations from a full benchmark run:

- **Free-form images, not benchmark IDs.** The official harness looks up a
  per-prompt subset of dimensions from the benchmark metadata. Here every
  selected dimension is scored, so criteria that don't apply to your image come
  back as N/A rather than being skipped up front.
- **Bounded thinking.** Upstream lets the model think up to 4096 tokens per
  dimension and stop wherever it stops. Here the budget is split: the model reasons
  until its allowance runs out, then `</think>` is forced so there is always room
  left to emit the JSON verdict. Without that, a cutoff lands mid-reasoning and
  takes the scores with it. Dimensions closed this way are marked in the results.

## GPU quota

All five dimensions are decoded as one padded batch. They share a prompt that
differs only in the checklist at the end, and at 4-bit the decode loop is bound
by weight bandwidth rather than arithmetic, so five sequences cost roughly the
wall-clock of one — which is what ZeroGPU bills.

That changes what saves quota. **Deselecting dimensions barely helps**, since the
batch runs at the pace of its slowest row. The thinking budget is the lever, so it
is what the preset control sets:

| Preset | Thinking + answer | Reserves | Prompt cap | Free runs/day |
| --- | --- | --- | --- | --- |
| Fast | 768 + 384 | ~91s | 600 chars | ~2 |
| **Balanced** (default) | 1792 + 384 | ~147s | 1,500 chars | ~1 |
| Fidelity | 3712 + 384 | ~254s | 3,000 chars | PRO |

Fidelity's total matches upstream's 4096 tokens. The budget is a ceiling rather
than a bill — rows that finish early release the GPU immediately — but ZeroGPU
checks admission against the *whole* reservation, so an oversized budget gets a run
refused rather than merely slowed. Free accounts get about 5 minutes a day, PRO 40.

The prompt is capped per preset because it sits in the prefix of every dimension,
so each character is prefilled once per batch row. The field stops at the cap and
the counter says how much is left; nothing is ever silently truncated.

The batch is sized to keep the KV cache inside an 8GB budget, which leaves all five
dimensions in one batch at every preset. Set `JUDGE_BATCH_SIZE=1` to force one
generation per dimension — slower, but each dimension's scores land as soon as that
dimension finishes.

## Running it yourself

```bash
uv sync
uv run python app.py
```

Needs a CUDA GPU with ~20GB free for the 4-bit weights. The first run downloads
~55GB of weights from the Hub before quantizing them.

## Credits

Model, checklists, and scoring methodology by the Qwen team, Apache-2.0.
`src/qwen_image_bench/checklists.py` and `scoring.py` are vendored from
[QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench); the
checklist strings must stay byte-identical to upstream for the scores to mean
what the paper says they mean.

```bibtex
@misc{li2026qwenimagebenchgenerationcreationtexttoimage,
      title={Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation},
      author={Niantong Li and Guangzheng Hu and Weixu Qiao and Ying Ba and others},
      year={2026},
      eprint={2605.28091},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.28091},
}
```
