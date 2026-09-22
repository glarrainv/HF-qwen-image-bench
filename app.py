"""Gradio Space for Qwen/Qwen-Image-Bench (Q-Judger).

Upload a generated image plus the prompt that produced it, and the judge scores
it across the benchmark's five level-1 dimensions.

Only ``run_judge`` runs on the GPU. ZeroGPU bills everything inside that function,
including work that never touches CUDA, so validation, image preparation, parsing
and markdown rendering all live in ``judge_and_render`` around it.
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
    EXAMPLE_DIMENSION,
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
    MAX_SCORE,
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


def _reservation_note(preset_name, prompt="", dimensions=None):
    seconds = estimate_duration(None, prompt, dimensions or DIMENSIONS, preset_name)
    worst = int(seconds * DURATION_FACTOR_WORST_CASE)
    preset = PRESETS[preset_name]
    return (
        f"Reserves **~{seconds}s** of GPU quota (up to ~{worst}s on slower nodes). "
        f"Budget: {preset.think} thinking + {preset.answer} answer tokens per dimension."
    )


def _prompt_status(prompt, preset_name):
    """Character counter text, and whether the prompt is over its cap."""
    cap = PRESETS[preset_name].prompt_chars
    used = len(prompt or "")
    over = used > cap
    if over:
        return (
            f"**{used} / {cap} characters — {used - cap} over the {preset_name} limit.** "
            "Shorten the prompt, or choose a preset with more room.",
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
            "Enter the text prompt that generated this image — the judge scores "
            "the image against it, and Alignment is meaningless without it."
        )
    if not dimensions:
        raise gr.Error("Select at least one evaluation dimension.")

    # The browser's maxlength is a convenience; the Gradio API can be called
    # directly, so this is the guard that actually holds.
    cap = PRESETS[preset_name].prompt_chars
    if len(prompt) > cap:
        raise gr.Error(
            f"Prompt is {len(prompt)} characters; the {preset_name} preset allows "
            f"{cap}. Shorten it or switch preset — the prompt is prefilled once per "
            "dimension, so its length is paid for five times over."
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
**1 · The prompt you supply**

> {EXAMPLE_PROMPT}

**2 · What the judge sends itself** — your prompt, the image, and {EXAMPLE_DIMENSION}'s
full 20-criterion checklist. It reasons first ({len(_example_thinking)} characters here),
then answers:

```
{_example_thinking[:600]}…
</think>
```

**3 · The JSON it must end with** — every criterion in the checklist, scored 0, 1 or 2,
or `"N/A"` where the criterion doesn't apply:

```json
{json.dumps(EXAMPLE_JSON, indent=2)[:700]}…
```

**4 · The score** — six criteria applied, fourteen N/A. Level-3 averages up to
level-2, level-2 up to level-1: **1.33 / {MAX_SCORE}**. N/A criteria leave the mean
entirely; they are not zeros.
"""


with gr.Blocks(title="Qwen-Image-Bench Judge") as demo:
    gr.Markdown(
        f"""
        # Qwen-Image-Bench — Q-Judger

        Scores a text-to-image generation across the benchmark's five dimensions
        using [Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench),
        a 27B judge model fine-tuned from Qwen3.6-27B.

        Each criterion is graded **0 (Fail) · 1 (Pass) · 2 (Excel)** or N/A, then
        averaged up the level-3 → level-2 → level-1 hierarchy. Scores are reported on
        that raw **0–{MAX_SCORE}** scale, with the paper's 0/60/100 equivalent shown
        alongside the total.

        > Running on ZeroGPU in 4-bit NF4 so the model fits a 48GB slice. Scores
        > will track the paper's closely but not reproduce them exactly — use a
        > bf16 GPU if you need reference numbers.
        """
    )

    with gr.Accordion("See a worked example (no GPU used)", open=False):
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
            prompt_counter = gr.Markdown(_prompt_status("", DEFAULT_PRESET)[0])
            gr.Examples(examples=[[EXAMPLE_PROMPT]], inputs=[prompt_input], label="Example prompt")

            dimension_input = gr.CheckboxGroup(
                choices=DIMENSIONS,
                value=DIMENSIONS,
                label="Evaluation dimensions",
                info="All selected dimensions are judged in one batch, so dropping one saves little GPU quota.",
            )
            preset_input = gr.Radio(
                choices=list(PRESETS),
                value=DEFAULT_PRESET,
                label="Thinking budget",
                info="Fast is cheapest and caps the prompt hardest; Fidelity matches upstream's 4096 tokens.",
            )
            reservation_note = gr.Markdown(_reservation_note(DEFAULT_PRESET))
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

    def on_prompt_change(prompt, preset_name, dimensions):
        text, over = _prompt_status(prompt, preset_name)
        return (
            gr.update(value=text),
            gr.update(interactive=not over),
            gr.update(value=_reservation_note(preset_name, prompt, dimensions)),
        )

    def on_preset_change(prompt, preset_name, dimensions):
        text, over = _prompt_status(prompt, preset_name)
        return (
            gr.update(max_length=PRESETS[preset_name].prompt_chars),
            gr.update(value=text),
            gr.update(interactive=not over),
            gr.update(value=_reservation_note(preset_name, prompt, dimensions)),
        )

    prompt_input.change(
        fn=on_prompt_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_counter, run_button, reservation_note],
    )
    # A shorter cap must never silently truncate what someone already typed: the
    # counter says what is wrong and the button locks until they fix it.
    preset_input.change(
        fn=on_preset_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_input, prompt_counter, run_button, reservation_note],
    )
    dimension_input.change(
        fn=lambda prompt, preset_name, dimensions: gr.update(
            value=_reservation_note(preset_name, prompt, dimensions)
        ),
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[reservation_note],
    )

    run_button.click(
        fn=judge_and_render,
        inputs=[image_input, prompt_input, dimension_input, preset_input],
        outputs=[summary_output, detail_output, json_output, thinking_output],
    )

if __name__ == "__main__":
    demo.launch()
