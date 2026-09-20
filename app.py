"""Gradio Space for Qwen/Qwen-Image-Bench (Q-Judger).

Upload a generated image plus the prompt that produced it, and the judge scores
it across the benchmark's five level-1 dimensions.
"""

import sys
from pathlib import Path

# The Space installs dependencies from requirements.txt and does not install
# this project itself, so make the src/ layout importable either way.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import spaces

from qwen_image_bench.judge import judge_dimension, load_judge, prepare_image
from qwen_image_bench.report import DIMENSIONS, detail_rows, format_summary
from qwen_image_bench.scoring import (
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

model, processor = load_judge()

EXAMPLE_PROMPT = (
    "A red ceramic teapot pouring steaming tea into a white cup on a wooden "
    "table, morning light from a window on the left."
)


def estimate_duration(image, prompt, dimensions, max_new_tokens, *args, **kwargs):
    """Reserve GPU time proportional to the work actually requested.

    4-bit NF4 decoding on a 27B model runs around 20 tok/s. Most responses hit
    EOS well before max_new_tokens, and ZeroGPU bills actual usage, so this is
    a ceiling rather than a cost.
    """
    per_dimension = max_new_tokens / 18 + 10
    return int(min(60 + len(dimensions or []) * per_dimension, 600))


def _trace_markdown(thinking, raw_outputs, failed):
    """Reasoning per dimension, plus the raw answer wherever parsing failed."""
    blocks = []
    for dimension, trace in thinking.items():
        block = [f"### {dimension}", trace or "_No reasoning emitted._"]
        if dimension in failed:
            block += [
                "**Could not parse a score JSON from this answer:**",
                f"```\n{raw_outputs[dimension]}\n```",
            ]
        blocks.append("\n\n".join(block))
    return "\n\n".join(blocks)


@spaces.GPU(duration=estimate_duration)
def run_judge(image, prompt, dimensions, max_new_tokens):
    if image is None:
        raise gr.Error("Upload an image to judge.")
    if not (prompt or "").strip():
        raise gr.Error(
            "Enter the text prompt that generated this image — the judge scores "
            "the image against it, and Alignment is meaningless without it."
        )
    if not dimensions:
        raise gr.Error("Select at least one evaluation dimension.")

    # Judge in the benchmark's canonical order, not checkbox click order.
    dimensions = [d for d in DIMENSIONS if d in dimensions]
    image = prepare_image(image)

    dim_results = {}
    fixed_jsons = {}
    raw_outputs = {}
    thinking = {}
    failed = set()

    for index, dimension in enumerate(dimensions):
        yield (
            format_summary(dim_results, index, len(dimensions), current=dimension),
            detail_rows(dim_results, fixed_jsons),
            fixed_jsons,
            _trace_markdown(thinking, raw_outputs, failed),
        )

        trace, answer = judge_dimension(
            model, processor, image, prompt.strip(), dimension, int(max_new_tokens)
        )
        thinking[dimension] = trace
        raw_outputs[dimension] = answer

        score_json = extract_json_from_response(answer)
        if score_json is None:
            dim_results[dimension] = None
            failed.add(dimension)
        else:
            fixed = fix_score_json(score_json, dimension)
            fixed_jsons[dimension] = fixed
            dim_results[dimension] = compute_dimension_score(fixed)

        yield (
            format_summary(dim_results, index + 1, len(dimensions)),
            detail_rows(dim_results, fixed_jsons),
            fixed_jsons,
            _trace_markdown(thinking, raw_outputs, failed),
        )


with gr.Blocks(title="Qwen-Image-Bench Judge") as demo:
    gr.Markdown(
        """
        # Qwen-Image-Bench — Q-Judger

        Scores a text-to-image generation across the benchmark's five dimensions
        using [Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench),
        a 27B judge model fine-tuned from Qwen3.6-27B.

        Each criterion is graded **0 (Fail) · 1 (Pass) · 2 (Excel)** or N/A, mapped
        to **0 / 60 / 100**, then averaged up the level-3 → level-2 → level-1 hierarchy.

        > Running on ZeroGPU in 4-bit NF4 so the model fits a 48GB slice. Scores
        > will track the paper's closely but not reproduce them exactly — use a
        > bf16 GPU if you need reference numbers.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="Generated image", height=340)
            prompt_input = gr.Textbox(
                label="Prompt used to generate the image",
                placeholder=EXAMPLE_PROMPT,
                lines=3,
            )
            dimension_input = gr.CheckboxGroup(
                choices=DIMENSIONS,
                value=DIMENSIONS,
                label="Evaluation dimensions",
                info="Each dimension is a separate generation. Deselect some to save GPU quota.",
            )
            token_input = gr.Slider(
                minimum=256,
                maximum=4096,
                value=1024,
                step=256,
                label="Max new tokens per dimension",
                info="The judge thinks before answering. Upstream uses 4096; 1024 is usually enough.",
            )
            run_button = gr.Button("Judge image", variant="primary")

        with gr.Column(scale=3):
            summary_output = gr.Markdown("Upload an image and prompt, then press **Judge image**.")
            detail_output = gr.Dataframe(
                headers=["Dimension", "Sub-dimension", "Criterion", "Verdict", "Score"],
                label="Per-criterion breakdown",
                wrap=True,
            )
            with gr.Accordion("Parsed scores (JSON)", open=False):
                json_output = gr.JSON()
            with gr.Accordion("Judge reasoning", open=False):
                thinking_output = gr.Markdown()

    run_button.click(
        fn=run_judge,
        inputs=[image_input, prompt_input, dimension_input, token_input],
        outputs=[summary_output, detail_output, json_output, thinking_output],
    )

if __name__ == "__main__":
    demo.launch()
