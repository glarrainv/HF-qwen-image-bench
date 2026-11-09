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
pinned: true
license: apache-2.0
short_description: Judge a generated image with the Qwen-Image-Bench model
models: [Qwen/Qwen3.6-27B]
library_name: transformers
tags:
  - judge-model
  - text-to-image
  - evaluation
  - benchmark
  - qwen
---


<div align="center">
      
<br>
<br>

<h1> HF Space for Qwen-Image-Bench</h1>
<div>
<a href="https://huggingface.co/spaces/glarr/qwen-image-bench-classification"><img src="https://img.shields.io/badge/-Gradio-f97316?style=flat&logo=gradio&logoColor=white" alt="Space"></a>
<img alt="Qwen" src="https://img.shields.io/badge/Qwen-6950EF?style=flat&logo=qwen&logoColor=white"/>
<img alt="Pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white"/> 
</div>
<div>
<a href="https://huggingface.co/Qwen/Qwen-Image-Bench"><img src="https://img.shields.io/badge/Judge_Model-HuggingFace-ffd21e?logo=huggingface" alt="Model"></a>
<a href="http://arxiv.org/abs/2605.28091"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv" alt="Paper"></a>
</div>

<br>
<br>

</div>


Q-Judger is a 27B vision-language judge fine-tuned from Qwen3.6-27B to score TTI output across five different dimensions.
This model is part of Qwen's larger Qwen-Image-Bench collection which they explored in [their paper](https://arxiv.org/abs/2605.28091).

## Dimensions

#### Quality:
| Realism | Detail | Resolution |
|-------- | ------ | ---------- |

#### Aesthetics:
| Composition | Color Harmony |  Lighting | Anatomical Portraiture | Emotional Expression | Style Control |
| -------- | ------ | ---------- | ---------- | ---------- | ---------- |

#### Alignment:
| Attributes | Actions |  Layout | Relations | Scene | 
| -------- | ------ | ---------- | ---------- | ---------- |

#### Real-world Fidelity:
| Fairness | Safety & Compliance |  World Knowledge |
| -------- | ------ | ---------- | 

#### Creative Generation:

| Imagination | Feature Matchin |  Logical Resolutione | Text Rendering | Design Applications | Visual Storytelling |
| -------- | ------ | ---------- | ---------- | ---------- | ---------- | 




## Scoring

The model grades each criterion on a three-point scale.

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
