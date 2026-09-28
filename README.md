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


<h1> HF Space for Qwen-Image-Bench — Q-Judger</h1>
<div><img alt="Qwen" src="https://img.shields.io/badge/Qwen-6950EF?style=flat&logo=qwen&logoColor=white"/> <img alt="Transformers" src="https://img.shields.io/badge/HuggingFace-%23FFD21E.svg?style=flat&logo=huggingface&logoColor=white"/> <img alt="Pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white"/> </div>

<p><i><a href="https://glarr-qwen-image-bench-classification.hf.space/">Hugging Face Space</a></i></p>

---

Q-Judger is a 27B vision-language judge fine-tuned from Qwen3.6-27B to score TTI output across five different dimensions.
This model is part of Qwen's larger Qwen-Image-Bench collection which they explored in [their paper](https://arxiv.org/abs/2605.28091).

## How scoring works

The model grades each level-3 criterion on a three-point scale.

| Score | Meaning |
| --- | --- | 
| 0 | Fail | 
| 1 | Pass | 
| 2 | Excel| 
| N/A | Criterion doesn't apply |


Raw scores are reported as a .json on this **0–2** scale. This space does not normalize scores to the 0, 60, 100 scale presented in the paper yet.

## Quantization Note

This Space runs the model in **4-bit NF4** to ensure compatibility with HF's ZeroGPU's 48GB. This change alters precision making results not directly comparable to those presented by Qwen's research. For full performance, running the [the official](https://github.com/QwenLM/Qwen-Image-Bench) is highly recommended to ensure accuracy.

Two other deliberate deviations from a full benchmark run:

- **Max Tokens/Chars.** Thinking budgets options are provided with respective token/char cutoffs. Instead of max_tokens, </thinking> is forced into reasoning once condition is reached.
- **Dimension Selection:** Space provides the option to cutoff dimensions that are not needed for the user's goals. 

## GPU quota

| Preset | Thinking + answer | Reserves | Prompt cap | Free runs/day |
| --- | --- | --- | --- | --- |
| Fast | 768 + 384 | ~91s | 600 chars | ~2 |
| **Balanced** (default) | 1792 + 384 | ~147s | 1,500 chars | ~1 |
| Fidelity | 3712 + 384 | ~254s | 3,000 chars | PRO |

Fidelity's total matches upstream's 4096 tokens. 

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

This space is an adaptation of:
[QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench)

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
