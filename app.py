"""Gradio Space for Qwen/Qwen-Image-Bench (Q-Judger).

Simplifies use of the Qwen-Image-Bench judge model taking a single prompt and image and outputting scores.

"""

import json
import math
import sys
from pathlib import Path

# The Space installs dependencies from requirements.txt and does not install
# this project itself, so make the src/ layout importable either way.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import spaces

from qwen_image_bench.examples import (
    EXAMPLE_JSON,
    EXAMPLE_OUTPUT,
    EXAMPLE_PROMPT,
)
from qwen_image_bench.judge import (
    CHARS_PER_TOKEN,
    DEFAULT_PRESET,
    PRESETS,
    batch_size_for,
    judge_dimensions,
    load_judge,
    prepare_image,
)
from qwen_image_bench.recovery import parse_answer, split_thinking
from qwen_image_bench.report import (
    DIMENSIONS,
    detail_rows,
    format_summary,
)
from qwen_image_bench.scoring import compute_dimension_score, fix_score_json

model, processor = load_judge()

# 4-bit NF4 decoding on a 27B model runs around 18 tok/s whether one sequence or
# five are in flight, so GPU time scales with the number of batches rather than
# the number of dimensions. Re-tune these from the startup log on the real Space.
DECODE_TOKENS_PER_SECOND = 18
PREFILL_SECONDS = 12
PREFILL_TOKENS_PER_SECOND = 400
BASE_SECONDS = 15

# spaces multiplies the requested duration by 1.5 on non-H200 nodes, and admission
# is checked against the full request — so a padded reservation is what gets a run
# refused, not what keeps it safe. Cap ours so even the multiplied figure stays
# inside a plausible free allowance for the cheaper presets.
DURATION_FACTOR_WORST_CASE = 1.5
MAX_RESERVATION_SECONDS = 260

# How much partial reasoning to show for a dimension that is still decoding.
LIVE_TRACE_TAIL = 1200

CITATION = """\
**Citation:** [Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation](https://arxiv.org/abs/2605.28091)
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
"""


def estimate_duration(image, prompt, dimensions, preset_name, *args, **kwargs):
    """Reserve GPU time for the batches we are actually going to run.

    The thinking budget makes the worst case deterministic, so this is a real
    ceiling rather than a guess. It calls the same batch-size helper the judge does,
    so the reservation and the work agree on the batch count, and it prices the
    user's prompt because that text sits in the prefix of every dimension.
    """
    preset = PRESETS.get(preset_name, PRESETS[DEFAULT_PRESET])
    requested = len([d for d in DIMENSIONS if d in (dimensions or [])]) or 1
    batches = math.ceil(requested / batch_size_for(model, preset.total_tokens, requested))

    prompt_tokens = len(prompt or "") / CHARS_PER_TOKEN
    prefill = PREFILL_SECONDS + prompt_tokens / PREFILL_TOKENS_PER_SECOND
    per_batch = prefill + preset.total_tokens / DECODE_TOKENS_PER_SECOND
    return int(min(BASE_SECONDS + batches * per_batch, MAX_RESERVATION_SECONDS))


def _prompt_status(prompt, preset_name):
    """Character counter text, and whether the prompt is over its cap."""
    cap = PRESETS[preset_name].prompt_chars
    used = len(prompt or "")
    over = used > cap
    if over:
        return (
            f"**{used} / {cap} characters. You are {used - cap} characters over the {preset_name} thinking budget limit.** "
            "Shorten the prompt, or select a higher thinking budget.",
            True,
        )
    return f"{used} / {cap} characters (~{int(used / CHARS_PER_TOKEN)} tokens)", False


def _trace_markdown(thinking, raw_outputs, audits, live=None):
    """Reasoning per dimension, plus the raw answer wherever parsing failed."""
    blocks = []
    for dimension, trace in thinking.items():
        audit = audits.get(dimension)
        heading = f"### {dimension}"
        if audit is not None and audit.forced_close:
            heading += " — _thinking hit its budget and was closed early_"
        block = [heading, trace or "_No reasoning emitted._"]
        if audit is not None and audit.status == "failed":
            block += [
                "**Could not parse a score JSON from this answer:**",
                f"```\n{raw_outputs[dimension]}\n```",
            ]
        elif audit is not None and audit.repaired:
            block.append(
                "_The answer was cut off; the scores above were recovered from the "
                "part that arrived._"
            )
        blocks.append("\n\n".join(block))

    for dimension, partial in (live or {}).items():
        tail = (partial or "").strip()[-LIVE_TRACE_TAIL:]
        blocks.append("\n\n".join([f"### {dimension} — thinking…", tail or "_Starting…_"]))
    return "\n\n".join(blocks)


@spaces.GPU(duration=estimate_duration)
def run_judge(image, prompt, dimensions, preset_name):
    """The billed part: decode the batch and hand back raw text, nothing else."""
    yield from judge_dimensions(
        model, processor, image, prompt, dimensions, PRESETS[preset_name]
    )


def judge_and_render(image, prompt, dimensions, preset_name):
    """Validate, run the GPU pass, and format its output. Only the middle is billed."""
    prompt = (prompt or "").strip()
    if image is None:
        raise gr.Error("Upload an image to judge.")
    if not prompt:
        raise gr.Error(
            "Please provide the prompt used to generate this image."
        )
    if not dimensions:
        raise gr.Error("Select at least one evaluation dimension.")

    # The browser's maxlength is a convenience; the Gradio API can be called
    # directly, so this is the guard that actually holds.
    cap = PRESETS[preset_name].prompt_chars
    if len(prompt) > cap:
        raise gr.Error(
            f"Prompt is {len(prompt)} characters; the {preset_name} preset allows "
            f"{cap}. Shorten it or switch preset."
        )

    # Judge in the benchmark's canonical order, not checkbox click order.
    dimensions = [d for d in DIMENSIONS if d in dimensions]
    image = prepare_image(image)

    dim_results, fixed_jsons, raw_outputs, thinking, audits = {}, {}, {}, {}, {}

    def snapshot(live):
        return (
            format_summary(dim_results, dimensions, current=live, audits=audits),
            detail_rows(dim_results, fixed_jsons, audits=audits),
            fixed_jsons,
            _trace_markdown(thinking, raw_outputs, audits, live),
        )

    yield snapshot({})

    for finished, live in run_judge(image, prompt, dimensions, preset_name):
        for dimension, (trace, answer, forced_close) in finished.items():
            if dimension in thinking:
                continue
            thinking[dimension] = trace
            raw_outputs[dimension] = answer

            fixed, audit = parse_answer(
                answer, dimension, fix_score_json, forced_close=forced_close
            )
            audits[dimension] = audit
            if fixed is None:
                dim_results[dimension] = None
            else:
                fixed_jsons[dimension] = fixed
                dim_results[dimension] = compute_dimension_score(fixed)

        yield snapshot(live)


_example_thinking, _example_answer = split_thinking(EXAMPLE_OUTPUT)
EXAMPLE_WALKTHROUGH = f"""\
**1. Prompt**

> {EXAMPLE_PROMPT}

**2. Judge's reasoning**

```
<think>
{_example_thinking[:600]}…
</think>
```

**3. End JSON with scores across dimensions**

```json
{json.dumps(EXAMPLE_JSON, indent=2)[:300]}…
```
"""

CITATION = f"""
Citation\n
If you find this model useful, please cite our paper:\n

@misc{{li2026qwenimagebenchgenerationcreationtexttoimage,\n
      title={{Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation}},\n 
      author={{Niantong Li and Guangzheng Hu and Weixu Qiao and Ying Ba and Qichen Hong and Shijun Shen and Jinlin Wang and Fan Zhou and Jianye Kang and Xin Shang and Ziyi He and Wei Wang and Dalin Li and Jiahao Li and Jie Zhang and Kaiyuan Gao and Kun Yan and Lihan Jiang and Ningyuan Tang and Shengming Yin and Tianhe Wu and Xiao Xu and Xiaoyue Chen and Yuxiang Chen and Yan Shu and Yanran Zhang and Yilei Chen and Yixian Xu and Zekai Zhang and Zhendong Wang and Zihao Liu and Zikai Zhou and Hongzhu Shi and Yi Wang and Bing Zhao and Hu Wei and Lin Qu and Chenfei Wu}},\n
      year={{2026}},\n
      eprint={{2605.28091}},\n
      archivePrefix={{arXiv}},\n
      primaryClass={{cs.CV}},\n
      url={{https://arxiv.org/abs/2605.28091}}, \n
}}\n
License\n
This project is licensed under the Apache License 2.0.
"""

with gr.Blocks(title="Qwen-Image-Bench Judge") as demo:
    gr.Markdown(
        f"""
        # Qwen-Image-Bench — Q-Judger Space

        Scores text-to-image output across the benchmark's five dimensions
        using [Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench).

        Each criterion is graded **0 (Fail), 1 (Pass), 2 (Excel)** or N/A. Scores are reported alongside the paper's 0/60/100 equivalent.
        4 bit quantization is implemented to accomodate for ZeroGPU.
        """
    )

    with gr.Accordion("Example output", open=False):
        gr.Markdown(EXAMPLE_WALKTHROUGH)

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="Generated image", height=340)
            prompt_input = gr.Textbox(
                label="Prompt used to generate the image",
                placeholder=EXAMPLE_PROMPT,
                lines=3,
                max_length=PRESETS[DEFAULT_PRESET].prompt_chars,
            )

            prompt_counter = gr.Markdown(
                _prompt_status(EXAMPLE_PROMPT, DEFAULT_PRESET)[0],
                label="Character count"
            )


            dimension_input = gr.CheckboxGroup(
                choices=DIMENSIONS,
                value=DIMENSIONS,
                label="Evaluation dimensions",
                info="Select dimensions present in JSON output.",
            )
            preset_input = gr.Radio(
                choices=list(PRESETS),
                value=DEFAULT_PRESET,
                label="Thinking budget",
                info="Fast is recommended for HuggingFace free plan. On the other hand, Fidelity matches upstream's 4096 tokens.",
            )
            run_button = gr.Button("Judge image", variant="primary")

        with gr.Column(scale=3):
            summary_output = gr.Markdown("Upload an image and prompt, then press **Judge image**.")
            detail_output = gr.Dataframe(
                headers=["Dimension", "Sub-dimension", "Criterion", "Verdict"],
                label="Per-criterion breakdown",
                wrap=True,
            )
            with gr.Accordion("Parsed scores (JSON)", open=False):
                json_output = gr.JSON()
            with gr.Accordion("Judge reasoning", open=False):
                thinking_output = gr.Markdown()
            with gr.Accordion("Citation", open=False):
                gr.Markdown(CITATION, elem_id="citation")

    def on_prompt_change(prompt, preset_name, dimensions):
        text, over = _prompt_status(prompt, preset_name)
        return (
            gr.update(value=text),
            gr.update(interactive=not over)
        )

    def on_preset_change(prompt, preset_name, dimensions):
        text, over = _prompt_status(prompt, preset_name)
        return (
            gr.update(max_length=PRESETS[preset_name].prompt_chars),
            gr.update(value=text),
            gr.update(interactive=not over)
        )

    prompt_input.change(
        fn=on_prompt_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_counter, run_button],
    )

    # A shorter cap must never silently truncate what someone already typed: the
    # counter says what is wrong and the button locks until they fix it.
    preset_input.change(
        fn=on_preset_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_input, prompt_counter, run_button],
    )

    run_button.click(
        fn=judge_and_render,
        inputs=[image_input, prompt_input, dimension_input, preset_input],
        outputs=[summary_output, detail_output, json_output, thinking_output],
    )

if __name__ == "__main__":
    demo.launch()
