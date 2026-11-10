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

Areas to score broken down by level (L1-L3)

| Dimension (L1) | Sub-capability (L2) | Criteria (L3) |
| :-- | :-- | :-- |
| **Quality** | Realism | Physical Logic; Material Texture |
|  | Detail | Noise; Edge Clarity; Naturalness |
|  | Resolution |  |
| **Aesthetics**  | Composition |  |
|  | Color Harmony |  |
|  | Lighting | Lighting & Atmosphere |
|  | Anatomical Portraiture | Anatomical Fidelity |
|  | Emotional Expression |  |
|  | Style Control |  |
| **Alignment** | Attributes | Quantity; Facial Expression; Material Properties; Color; Shape; Size |
|  | Actions | Contact Interaction; Non-contact Interaction; Full-body Action |
|  | Layout | 2D Space; 3D Space |
|  | Relations | Composition Relationship; Difference/Similarity; Containment |
|  | Scene | Real-world Scene; Virtual Scene |
| **Real-world Fidelity** | Fairness | Social Bias; Cultural Fairness |
|  | Safety & Compliance |  |
|  | World Knowledge | Animals; Objects; Information Visualization; Temporal Characteristics; Cultural Elements |
| **Creative Generation** | Imagination |  |
|  | Feature Matching |  |
|  | Logical Resolution |  |
|  | Text Rendering | Text Accuracy; Text Layout; Font; Cross-lingual Generation |
|  | Design Applications | Graphic Design; Product Design; Spatial Design; Fashion Styling; Game Design; Art Design |
|  | Visual Storytelling | Cinematic Style; Camera / Lens Style; Storyboard Creation; Shot Sizes; Composition; Angles; Comic Creation |

## Scoring

The model grades each criterion on a three-point scale.

| Score | Meaning |
| --- | --- |
| 0 | Fail |
| 1 | Pass |
| 2 | Excel |
| N/A | Criterion doesn't apply |

## Changes from model

- Raw scores are reported as a .json on this **0–2** scale. The summary also shows the paper's 0/60/100 equivalent, computed from the same verdicts.
- The model is loaded in 4-bit using NF4 to accomodate HF's free Zero GPU.
- Thinking budgets are available with fixed token cutoffs.
- Dimension selection is available.


## Running it yourself

```bash
uv sync
uv run python app.py
```

## Citation

If this Space supports your work, please cite it together with the original paper
(machine-readable metadata is in [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{larrain2026qwenimagebenchjudge,
  author  = {Larrain, G.},
  title   = {Qwen-Image-Bench Judge: An Interactive Space for the Q-Judger Model},
  year    = {2026},
  url     = {https://huggingface.co/spaces/glarr/qwen-image-bench-classification},
  license = {Apache-2.0},
  note    = {Adapted from QwenLM/Qwen-Image-Bench}
}
```

## License & credits

© 2026 G. Larrain. Released under the [Apache License 2.0](LICENSE); attribution and a
summary of modifications are in [`NOTICE`](NOTICE).

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
