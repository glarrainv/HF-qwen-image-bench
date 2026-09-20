"""Loading and running the Q-Judger model (Qwen/Qwen-Image-Bench).

The model is 27.4B params, ~55GB in bf16, which does not fit ZeroGPU's default
48GB slice. We quantize the language model to 4-bit NF4 (~18GB) but keep the
vision tower and lm_head in bf16 -- the vision tower is only ~0.4B params, so
leaving it unquantized costs little memory and preserves image fidelity, which
is what the judge is actually looking at.
"""

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)

from .checklists import DIM_TO_CHECKLIST, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

MODEL_ID = "Qwen/Qwen-Image-Bench"

# Fixed by the reference implementation for reproducibility. temperature=0 /
# top_k=1 is greedy decoding, which transformers expresses as do_sample=False.
SEED = 42
REPETITION_PENALTY = 1.05

_model = None
_processor = None


def load_judge():
    """Load the judge model onto CUDA. Call this at module level.

    ZeroGPU requires models to be placed on cuda during startup (a CUDA
    emulation layer stands in for the real device outside @spaces.GPU), so
    calling this lazily from inside a GPU function would be much slower.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Keep the vision encoder and output head in bf16.
        llm_int8_skip_modules=["lm_head", "visual"],
    )

    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=quantization_config,
    )
    _model.eval()

    # The published checkpoint has use_cache=False baked into its config, which
    # is fine for training but would force a full recompute of every prefix
    # token at each decoding step. These are long thinking generations, so turn
    # the KV cache back on.
    _model.config.use_cache = True
    if hasattr(_model.config, "text_config"):
        _model.config.text_config.use_cache = True
    _model.generation_config.use_cache = True

    return _model, _processor


def prepare_image(image):
    """Match the reference implementation's preprocessing exactly.

    Note the upstream resize is to a literal 1024x1024 square and does *not*
    preserve aspect ratio. That distortion is part of what the judge was
    calibrated on, so we reproduce it rather than "fixing" it.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > 1024:
        image = image.resize((1024, 1024), Image.LANCZOS)
    return image


def build_messages(prompt, image, dimension):
    """Build the chat messages for one level-1 dimension.

    USER_PROMPT_TEMPLATE carries a literal ``<image>`` marker partway through
    the text (ms-swift's convention). We split on it so the vision tokens land
    at that exact position rather than being prepended to the whole message.
    """
    user_text = USER_PROMPT_TEMPLATE.format(
        prompt=prompt,
        level1_dim=dimension,
        format_checklist=DIM_TO_CHECKLIST[dimension],
    )
    before, after = user_text.split("<image>")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before},
                {"type": "image", "image": image},
                {"type": "text", "text": after},
            ],
        },
    ]


def split_thinking(text):
    """Separate the chain-of-thought from the final answer.

    The chat template emits a dangling ``<think>\\n`` as part of the generation
    prompt, so the model's output starts *inside* the thinking block and only
    ever produces the closing tag.
    """
    end = text.rfind("</think>")
    if end == -1:
        return "", text.strip()
    return text[:end].strip(), text[end + len("</think>") :].strip()


@torch.inference_mode()
def judge_dimension(model, processor, image, prompt, dimension, max_new_tokens):
    """Run the judge for a single level-1 dimension. Returns (thinking, answer)."""
    set_seed(SEED)
    messages = build_messages(prompt, image, dimension)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=REPETITION_PENALTY,
        use_cache=True,
    )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return split_thinking(text)
