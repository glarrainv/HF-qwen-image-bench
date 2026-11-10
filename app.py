# Copyright 2026 G. Larrain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gradio Space for Qwen/Qwen-Image-Bench (Q-Judger).

Simplifies use of the Qwen-Image-Bench judge model taking a single prompt and image and outputting scores.
Adapted from QwenLM/Qwen-Image-Bench; see NOTICE for attribution.
"""

import math
import sys
import time
from pathlib import Path

# The Space installs dependencies from requirements.txt and does not install
# this project itself, so make the src/ layout importable either way.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import spaces


from qwen_image_bench.judge import (
    CHARS_PER_TOKEN,
    DEFAULT_PRESET,
    PRESETS,
    REFERENCE_MAX_NEW_TOKENS,
    batch_size_for,
    judge_dimensions,
    load_judge,
    prepare_image,
)

from qwen_image_bench.report import DIMENSIONS
from qwen_image_bench.scoring import (
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)
from qwen_image_bench.ui import (
    CSS,
    SPACE_URL,
    breakdown_html,
    fmt_duration,
    footer_html,
    header_html,
    reasoning_html,
    scores_html,
    status_html,
)

model, processor = load_judge()

PROMPT_PLACEHOLDER = "e.g. A red fox curled up in fresh snow at dawn, soft golden light"

DECODE_TOKENS_PER_SECOND = 18
PREFILL_SECONDS = 12
PREFILL_TOKENS_PER_SECOND = 400
BASE_SECONDS = 15

DURATION_FACTOR_WORST_CASE = 1.5
MAX_RESERVATION_SECONDS = 260

SPACE_BIBTEX = f"""\
@software{{larrain2026qwenimagebenchjudge,
  author  = {{Larrain, G.}},
  title   = {{Qwen-Image-Bench Judge: An Interactive Space for the Q-Judger Model}},
  year    = {{2026}},
  url     = {{{SPACE_URL}}},
  license = {{Apache-2.0}},
  note    = {{Adapted from QwenLM/Qwen-Image-Bench}}
}}"""

PAPER_BIBTEX = """\
@misc{li2026qwenimagebenchgenerationcreationtexttoimage,
  title         = {Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation},
  author        = {Niantong Li and Guangzheng Hu and Weixu Qiao and Ying Ba and others},
  year          = {2026},
  eprint        = {2605.28091},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.28091}
}"""

THEME = gr.themes.Base(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)


def _preset_info():
    parts = []
    for name, preset in PRESETS.items():
        if preset.think is None:
            parts.append(f"{name}: uncapped, matches upstream ({REFERENCE_MAX_NEW_TOKENS:,} tokens)")
        else:
            parts.append(f"{name}: {preset.think:,} thinking tokens")
    return " · ".join(parts) + ". Fast is best for free-tier GPU quota."


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


def _dimensions_label(dimensions):
    return f"Evaluation dimensions · {len(dimensions or [])} of {len(DIMENSIONS)} selected"


def _estimate_text(prompt, dimensions, preset_name):
    if not dimensions:
        return "Select at least one dimension to judge."
    seconds = estimate_duration(None, prompt, dimensions, preset_name)
    return (
        f"Reserves up to **{fmt_duration(seconds)}** of GPU time. "
        "Results stream in as the judge reasons."
    )


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

    dim_results, fixed_jsons, raw_outputs, thinking = {}, {}, {}, {}
    rushed = set()
    started = time.monotonic()

    def snapshot(live, state):
        current = [d for d in dimensions if d in live]
        return (
            status_html(
                state,
                done=len(dim_results),
                total=len(dimensions),
                elapsed=time.monotonic() - started,
                current=current,
            ),
            scores_html(dim_results, dimensions, current=current, rushed=rushed),
            breakdown_html(dim_results, fixed_jsons),
            fixed_jsons,
            reasoning_html(thinking, raw_outputs, dim_results, rushed, live),
        )

    # Shown while ZeroGPU allocates a device: this is the longest silent stretch.
    yield snapshot({}, "queued")

    for finished, live in run_judge(image, prompt, dimensions, preset_name):
        for dimension, (trace, answer, forced_close) in finished.items():
            if dimension in thinking:
                continue
            thinking[dimension] = trace
            raw_outputs[dimension] = answer
            if forced_close:
                rushed.add(dimension)

            # Same parse as the reference scorer: a malformed answer is dropped, not repaired.
            score_json = extract_json_from_response(answer)
            if score_json is None:
                dim_results[dimension] = None
            else:
                fixed_jsons[dimension] = fix_score_json(score_json, dimension)
                dim_results[dimension] = compute_dimension_score(fixed_jsons[dimension])

        yield snapshot(live, "running")

    yield snapshot({}, "done")


# --- control state -----------------------------------------------------------------


def on_prompt_change(prompt, preset_name, dimensions):
    text, over = _prompt_status(prompt, preset_name)
    return (
        gr.update(value=text),
        gr.update(interactive=not over),
        gr.update(value=_estimate_text(prompt, dimensions, preset_name)),
    )


def on_preset_change(prompt, preset_name, dimensions):
    text, over = _prompt_status(prompt, preset_name)
    return (
        gr.update(max_length=PRESETS[preset_name].prompt_chars),
        gr.update(value=text),
        gr.update(interactive=not over),
        gr.update(value=_estimate_text(prompt, dimensions, preset_name)),
    )


def lock_controls():
    return gr.update(value="Judging…", interactive=False), gr.update(visible=True)


def unlock_controls(prompt, preset_name):
    _, over = _prompt_status(prompt, preset_name)
    return gr.update(value="Judge image", interactive=not over), gr.update(visible=False)


def stop_run(prompt, preset_name):
    return (*unlock_controls(prompt, preset_name), status_html("stopped"))


def clear_all():
    return (
        None,
        "",
        status_html("idle"),
        scores_html({}, []),
        breakdown_html({}, {}),
        None,
        reasoning_html({}, {}, {}, set()),
    )


# --- layout ------------------------------------------------------------------------

with gr.Blocks(title="Qwen-Image-Bench Judge") as demo:
    gr.HTML(header_html())

    with gr.Row(equal_height=False):
        # Inputs: everything needed for a run, top to bottom, ending in the action.
        with gr.Column(scale=5, min_width=340):
            gr.HTML('<div class="qib-section">Input</div>')
            image_input = gr.Image(
                type="pil",
                label="Generated image",
                height=320,
                sources=["upload", "clipboard"],
            )
            prompt_input = gr.Textbox(
                label="Prompt used to generate the image",
                placeholder=PROMPT_PLACEHOLDER,
                lines=3,
                max_length=PRESETS[DEFAULT_PRESET].prompt_chars,
            )
            prompt_counter = gr.Markdown(
                _prompt_status("", DEFAULT_PRESET)[0], elem_classes=["qib-hint", "qib-counter"]
            )

            preset_input = gr.Radio(
                choices=list(PRESETS),
                value=DEFAULT_PRESET,
                label="Thinking budget",
                info=_preset_info(),
            )
            with gr.Accordion(_dimensions_label(DIMENSIONS), open=False) as dimension_panel:
                dimension_input = gr.CheckboxGroup(
                    choices=DIMENSIONS,
                    value=DIMENSIONS,
                    show_label=False,
                    info="All selected dimensions decode together, so dropping some barely shortens a run.",
                )

            estimate_note = gr.Markdown(
                _estimate_text("", DIMENSIONS, DEFAULT_PRESET), elem_classes="qib-hint"
            )
            with gr.Row():
                run_button = gr.Button("Judge image", variant="primary", scale=3, min_width=120)
                stop_button = gr.Button("Stop", variant="stop", scale=1, min_width=72, visible=False)
                clear_button = gr.Button("Clear", variant="secondary", scale=1, min_width=72)

        # Results: status and headline scores stay pinned above the long-form tabs.
        with gr.Column(scale=7, min_width=420):
            gr.HTML('<div class="qib-section">Results</div>')
            status_output = gr.HTML(status_html("idle"))
            scores_output = gr.HTML(scores_html({}, []))
            with gr.Tabs():
                with gr.Tab("Reasoning"):
                    reasoning_output = gr.HTML(
                        reasoning_html({}, {}, {}, set()),
                        max_height=620,
                        autoscroll=True,
                    )
                with gr.Tab("Breakdown"):
                    breakdown_output = gr.HTML(breakdown_html({}, {}), max_height=620)
                with gr.Tab("JSON"):
                    json_output = gr.JSON(label="Parsed scores (0–2 scale)")

    with gr.Accordion("Citation & license", open=False):
        gr.Markdown(
            "If this Space supports your work, please cite both the Space and the "
            "original Qwen-Image-Bench paper. The Space is released under the "
            "Apache License 2.0; see `LICENSE` and `NOTICE` in the repository."
        )
        with gr.Row():
            gr.Code(SPACE_BIBTEX, label="This Space", interactive=False, wrap_lines=True)
            gr.Code(PAPER_BIBTEX, label="Qwen-Image-Bench paper", interactive=False, wrap_lines=True)

    gr.HTML(footer_html())

    # --- wiring --------------------------------------------------------------------

    prompt_input.change(
        fn=on_prompt_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_counter, run_button, estimate_note],
        show_progress="hidden",
    )
    dimension_input.change(
        fn=on_prompt_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_counter, run_button, estimate_note],
        show_progress="hidden",
    )
    dimension_input.change(
        fn=lambda dims: gr.update(label=_dimensions_label(dims)),
        inputs=[dimension_input],
        outputs=[dimension_panel],
        show_progress="hidden",
        queue=False,
    )

    # A shorter cap must never silently truncate what someone already typed: the
    # counter says what is wrong and the button locks until they fix it.
    preset_input.change(
        fn=on_preset_change,
        inputs=[prompt_input, preset_input, dimension_input],
        outputs=[prompt_input, prompt_counter, run_button, estimate_note],
        show_progress="hidden",
    )

    judge_inputs = [image_input, prompt_input, dimension_input, preset_input]
    judge_outputs = [status_output, scores_output, breakdown_output, json_output, reasoning_output]

    judge_event = run_button.click(
        fn=lock_controls, outputs=[run_button, stop_button], queue=False
    ).then(
        fn=judge_and_render,
        inputs=judge_inputs,
        outputs=judge_outputs,
        show_progress="minimal",
        show_progress_on=[status_output],
    )
    judge_event.then(
        fn=unlock_controls,
        inputs=[prompt_input, preset_input],
        outputs=[run_button, stop_button],
        queue=False,
    )

    stop_button.click(
        fn=stop_run,
        inputs=[prompt_input, preset_input],
        outputs=[run_button, stop_button, status_output],
        cancels=[judge_event],
        queue=False,
    )

    clear_button.click(
        fn=clear_all,
        outputs=[image_input, prompt_input, *judge_outputs],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
