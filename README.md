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
short_description: Judge a generated image with the Qwen-Image-Bench judge model
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

| Verdict | Meaning | Mapped score |
| --- | --- | --- |
| 0 | Fail — clear defect | 0 |
| 1 | Pass — meets baseline | 60 |
| 2 | Excel — concretely excellent | 100 |
| N/A | Criterion doesn't apply | excluded from the average |

Level-3 → level-2 → level-1 → total, each step an arithmetic mean over the
non-N/A children. The model runs with thinking enabled, so it reasons before
emitting its JSON verdict; that reasoning is shown in the **Judge reasoning**
panel.

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
- **Shorter generations.** The default cap is 1024 new tokens per dimension
  instead of upstream's 4096. Most verdicts finish well inside that; raise the
  slider if you see truncated reasoning.

## GPU quota

Each dimension is a separate generation, so a full five-dimension run takes a
couple of minutes of GPU time. ZeroGPU's daily allowance is 5 minutes for free
accounts and 40 minutes with PRO, which works out to roughly one or two full runs
per day on a free account. Deselect dimensions or lower the token cap to stretch
it further.

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
