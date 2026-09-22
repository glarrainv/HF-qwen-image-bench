"""Loading and running the Q-Judger model (Qwen/Qwen-Image-Bench).

The model is 27.4B params, ~55GB in bf16, which does not fit ZeroGPU's default
48GB slice. We quantize the language model to 4-bit NF4 (~18GB) but keep the
vision tower and lm_head in bf16 -- the vision tower is only ~0.4B params, so
leaving it unquantized costs little memory and preserves image fidelity, which
is what the judge is actually looking at.

The five level-1 dimensions are decoded as one padded batch rather than one
generation each. At 4-bit the decode loop is bound by weight bandwidth, not
arithmetic: pulling 18GB of weights through the cache to produce one token costs
almost exactly what it costs to produce five. ZeroGPU bills wall-clock, so a
full five-dimension run drops from five sequential decodes to one.

Thinking is the only unbounded part of a response and the JSON verdict comes
after it, so a plain max_new_tokens cutoff lands mid-reasoning and destroys the
answer. Instead the budget is split: ``</think>`` is forced once a row has spent
its thinking allowance, leaving a reserved window for the JSON. Worst-case decode
is then deterministic, which is what lets the GPU reservation be tight enough for
ZeroGPU to admit it on free quota.
"""

import os
import queue
import threading
from typing import NamedTuple

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)
from transformers.generation.streamers import BaseStreamer

from .checklists import DIM_TO_CHECKLIST, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .recovery import THINK_END, split_thinking

MODEL_ID = "Qwen/Qwen-Image-Bench"

# Fixed by the reference implementation for reproducibility. temperature=0 /
# top_k=1 is greedy decoding, which transformers expresses as do_sample=False.
SEED = 42
REPETITION_PENALTY = 1.05

# Memory we are willing to spend on KV cache for a batch. The 4-bit weights take
# ~18GB of the 48GB slice and the bf16 vision tower spikes during prefill, so 8GB
# is comfortable headroom rather than a limit worth tuning.
KV_CACHE_BUDGET_BYTES = 8 * 1024**3

# Prompt length is dominated by the image: ~1.4k vision tokens at 1024x1024,
# plus the system prompt and the longest checklist.
PREFILL_TOKEN_ESTIMATE = 2048

# Rough conversion for sizing only -- English prose plus markdown runs a little
# under four characters a token. Used to price the user's prompt, never to cut it.
CHARS_PER_TOKEN = 3.8


class Preset(NamedTuple):
    """A GPU budget, and the prompt length that budget can afford.

    ``think`` bounds the reasoning, ``answer`` is reserved for the JSON that follows
    it -- Creative Generation's 20 criteria need ~340 tokens, so 384 has headroom.
    Total decode is ``think + 1 + answer``; the 1 is the forced ``</think>``.

    ``prompt_chars`` matters because the prompt sits in the prefix of *every*
    dimension, so each character is prefilled once per batch row.
    """

    think: int
    answer: int
    prompt_chars: int

    @property
    def total_tokens(self):
        return self.think + 1 + self.answer


PRESETS = {
    "Fast": Preset(think=768, answer=384, prompt_chars=600),
    "Balanced": Preset(think=1792, answer=384, prompt_chars=1500),
    "Fidelity": Preset(think=3712, answer=384, prompt_chars=3000),
}
DEFAULT_PRESET = "Balanced"

# Set JUDGE_BATCH_SIZE=1 to get one generation per dimension back (slower, but
# it makes a dimension's scores land as soon as that dimension finishes).
BATCH_SIZE_OVERRIDE = int(os.environ.get("JUDGE_BATCH_SIZE", "0"))

# Decoding steps between partial-output updates. Decoding is ~20 tok/s, so this
# refreshes the UI roughly once a second.
STREAM_EVERY_N_STEPS = 24

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

    # Batched generation needs left padding: right padding would leave pad
    # tokens between the prompt and the first generated token.
    _processor.tokenizer.padding_side = "left"
    if _model.generation_config.pad_token_id is None:
        _model.generation_config.pad_token_id = _processor.tokenizer.pad_token_id

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


def _kv_bytes_per_token(model):
    """KV cache cost of one token across every layer, in bytes."""
    config = getattr(model.config, "text_config", model.config)
    layers = config.num_hidden_layers
    heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return 2 * layers * heads * head_dim * 2  # key and value, bf16


def batch_size_for(model, max_new_tokens, requested):
    """How many dimensions to decode at once without overrunning the KV budget.

    Called from app.py's duration estimator as well, so the GPU reservation and
    the work actually done agree on how many batches there will be.
    """
    if BATCH_SIZE_OVERRIDE:
        return max(1, min(requested, BATCH_SIZE_OVERRIDE))
    per_sequence = (PREFILL_TOKEN_ESTIMATE + max_new_tokens) * _kv_bytes_per_token(model)
    return max(1, min(requested, int(KV_CACHE_BUDGET_BYTES // per_sequence)))


class ForceThinkClose(LogitsProcessor):
    """Close the thinking block once a row has spent its budget.

    Thinking is the only unbounded part of the response, and a cutoff lands *inside*
    it -- destroying the JSON that follows and costing the whole dimension. Rather
    than letting max_new_tokens decide where the axe falls, this spends the budget on
    reasoning and then forces ``</think>``, so every row still gets to answer.

    Per-row: rows that closed thinking on their own are untouched, and rows that have
    already finished are immune because generate substitutes pad tokens *after* token
    selection. Construct a fresh instance per generate() call.
    """

    def __init__(self, prompt_length, think_budget, think_end_id):
        self.prompt_length = prompt_length
        self.think_budget = think_budget
        self.think_end_id = think_end_id
        self.forced = None  # per-row, filled on the step the force fires

    def __call__(self, input_ids, scores):
        # Slice the prompt off first. The generation prompt ends with a dangling
        # `<think>`, but with enable_thinking=False a Qwen template emits a complete
        # `<think></think>` pair -- scanning the whole sequence would then read every
        # row as already closed and disable this guard entirely.
        generated = input_ids[:, self.prompt_length :]
        if generated.shape[1] < self.think_budget:
            return scores

        still_thinking = ~(generated == self.think_end_id).any(dim=-1)
        if still_thinking.any():
            if self.forced is None:
                self.forced = still_thinking.clone()
            scores[still_thinking, :] = -float("inf")
            scores[still_thinking, self.think_end_id] = 0.0
        return scores

    def forced_rows(self):
        """Which rows had thinking cut short, so the UI can say so."""
        return [] if self.forced is None else self.forced.tolist()


class StopOnBalancedJson(StoppingCriteria):
    """End a row once the JSON after ``</think>`` closes.

    Without this a row that has already answered keeps decoding until EOS, and the
    whole batch waits for it. Scanning is incremental and string-aware: a brace
    inside a criterion name must not count, and re-decoding each row's full history
    every step would be quadratic.
    """

    def __init__(self, tokenizer, prompt_length, think_end_id, batch_size):
        self.tokenizer = tokenizer
        self.think_end_id = think_end_id
        self.consumed = prompt_length
        self.piece_cache = {}
        self.rows = [
            {"after_think": False, "depth": 0, "in_string": False, "escaped": False, "done": False}
            for _ in range(batch_size)
        ]

    def _piece(self, token):
        if token not in self.piece_cache:
            self.piece_cache[token] = self.tokenizer.decode([token], skip_special_tokens=False)
        return self.piece_cache[token]

    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[:, self.consumed :].tolist()
        self.consumed = input_ids.shape[1]

        for state, tokens in zip(self.rows, new_tokens):
            if state["done"]:
                continue
            for token in tokens:
                if not state["after_think"]:
                    state["after_think"] = token == self.think_end_id
                    continue
                for char in self._piece(token):
                    if state["escaped"]:
                        state["escaped"] = False
                    elif char == "\\":
                        state["escaped"] = True
                    elif char == '"':
                        state["in_string"] = not state["in_string"]
                    elif state["in_string"]:
                        continue
                    elif char == "{":
                        state["depth"] += 1
                    elif char == "}":
                        state["depth"] -= 1
                        if state["depth"] == 0:
                            state["done"] = True
                            break
                if state["done"]:
                    break

        # Contract is a 1-D tensor of shape (batch_size,) -- the base class docstring
        # claims (batch_size, 1), which raises.
        return torch.tensor(
            [state["done"] for state in self.rows],
            dtype=torch.bool,
            device=input_ids.device,
        )


class _BatchStreamer(BaseStreamer):
    """Hand each decoding step's tokens, one per batch row, to a queue.

    ``generate`` pushes the prompt first as a 2-D tensor and then one 1-D tensor
    of shape (batch,) per step; only the steps are of interest here.
    """

    def __init__(self, token_queue):
        self.queue = token_queue
        self.seen_prompt = False

    def put(self, value):
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        self.queue.put(value.reshape(-1).tolist())

    def end(self):
        self.queue.put(None)


def _stop_token_ids(model, processor):
    """Tokens that mean a row is done: EOS, plus the pad that follows it.

    Rows that finish early keep receiving pad tokens until the longest row in
    the batch finishes, and those must not land in the partial output.
    """
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    candidates = list(eos or []) + [model.generation_config.pad_token_id]
    return {token for token in candidates if token is not None}


def _encode(processor, image, prompt, dimensions, device):
    """Tokenize one conversation per dimension into a single left-padded batch."""
    conversations = [build_messages(prompt, image, dimension) for dimension in dimensions]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True, "padding_side": "left"},
    )
    return inputs.to(device)


def think_end_id(processor):
    """Token id of ``</think>``, resolved from the tokenizer rather than hardcoded.

    It is a single token in this checkpoint. Looking it up means a retokenized
    checkpoint fails loudly here instead of silently disabling the budget guard.
    """
    token = processor.tokenizer.convert_tokens_to_ids(THINK_END)
    if token is None or token == processor.tokenizer.unk_token_id:
        raise RuntimeError(
            f"{THINK_END!r} is not a single token in this tokenizer; the thinking "
            "budget cannot be enforced against it."
        )
    return token


def _judge_chunk(model, processor, image, prompt, dimensions, preset):
    """Decode one batch of dimensions, yielding ("live", texts) then ("done", results).

    ``generate`` runs on a worker thread so the caller can surface partial output
    while it decodes; the final text is taken from generate's own return value
    rather than from the stream, which is only ever best-effort.
    """
    set_seed(SEED)
    inputs = _encode(processor, image, prompt, dimensions, model.device)
    prompt_length = inputs["input_ids"].shape[1]
    close_id = think_end_id(processor)

    # Fresh per call: both carry per-row state that would be stale on reuse.
    forcer = ForceThinkClose(prompt_length, preset.think, close_id)
    stopper = StopOnBalancedJson(processor.tokenizer, prompt_length, close_id, len(dimensions))

    token_queue = queue.Queue()
    outcome = {}

    @torch.inference_mode()
    def generate():
        try:
            outcome["output"] = model.generate(
                **inputs,
                max_new_tokens=preset.total_tokens,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
                use_cache=True,
                logits_processor=LogitsProcessorList([forcer]),
                stopping_criteria=StoppingCriteriaList([stopper]),
                streamer=_BatchStreamer(token_queue),
            )
        except BaseException as error:  # re-raised on the consumer's side below
            outcome["error"] = error
            token_queue.put(None)

    worker = threading.Thread(target=generate, daemon=True)
    worker.start()

    stop_tokens = _stop_token_ids(model, processor)
    rows = [[] for _ in dimensions]
    row_done = [False] * len(dimensions)
    texts = [""] * len(dimensions)
    steps = 0

    while True:
        step = token_queue.get()
        if step is None:
            break
        for index, token in enumerate(step):
            if row_done[index]:
                continue
            if token in stop_tokens:
                row_done[index] = True
                continue
            rows[index].append(token)
        steps += 1
        if steps % STREAM_EVERY_N_STEPS == 0:
            # Decode only what is new. Re-decoding each row's full history every
            # refresh is quadratic, and every millisecond here is billed GPU time.
            for index, ids in enumerate(rows):
                if ids:
                    texts[index] += processor.decode(ids, skip_special_tokens=True)
                    ids.clear()
            yield "live", dict(zip(dimensions, texts))

    worker.join()
    if "error" in outcome:
        raise outcome["error"]

    output = outcome["output"]
    forced = forcer.forced_rows()
    results = {}
    for index, dimension in enumerate(dimensions):
        text = processor.decode(output[index, prompt_length:], skip_special_tokens=True)
        thinking, answer = split_thinking(text)
        results[dimension] = (thinking, answer, bool(forced[index]) if forced else False)
    yield "done", results


def judge_dimensions(model, processor, image, prompt, dimensions, preset):
    """Judge several level-1 dimensions, yielding progress as they decode.

    Yields ``(finished, live)`` pairs: ``finished`` maps a dimension to a
    ``(thinking, answer, forced_close)`` triple, ``live`` maps a still-decoding
    dimension to the raw text so far. A dimension appears in ``live`` until its batch
    completes, then moves to ``finished`` and stays there.

    ``forced_close`` records that the row spent its whole thinking budget and had
    ``</think>`` forced on it, so the caller can mark that dimension partial rather
    than presenting a rushed answer as a considered one.
    """
    batch_size = batch_size_for(model, preset.total_tokens, len(dimensions))
    finished = {}

    for start in range(0, len(dimensions), batch_size):
        chunk = dimensions[start : start + batch_size]
        for kind, payload in _judge_chunk(model, processor, image, prompt, chunk, preset):
            if kind == "live":
                yield dict(finished), payload
            else:
                finished.update(payload)
                yield dict(finished), {}
