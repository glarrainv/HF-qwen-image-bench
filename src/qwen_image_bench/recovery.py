"""Recovering and auditing score JSON that the judge truncated or malformed.

Upstream's ``extract_json_from_response`` returns None the moment a response is cut
mid-JSON -- its ``\\{[\\s\\S]*\\}`` regex needs a closing brace -- and the dimension
then vanishes from the average behind a single "failed to parse" cell. That is a
silent bias: a dropped dimension is not a zero, it is an absence.

The JSON tail is only ~230-340 tokens even for Creative Generation's 20 criteria, so
a cutoff usually costs a handful of criteria rather than the whole dimension. This
module closes the braces the model never got to emit, then reports exactly what
survived so the UI can say "partial" instead of pretending nothing happened.

Kept free of torch/transformers imports so the entire parse path is testable on CPU.
"""

import json
from dataclasses import dataclass, field

from .scoring import CHECKLIST_L3_TO_L2

THINK_END = "</think>"

# What the prompt's "# Output Format" block permits: 0 | 1 | 2 | "N/A".
VALID_SCORES = frozenset({0, 1, 2})


def split_thinking(text):
    """Separate the chain-of-thought from the final answer.

    The chat template emits a dangling ``<think>\\n`` as part of the generation
    prompt, so the model's output starts *inside* the thinking block and only ever
    produces the closing tag.
    """
    end = text.rfind(THINK_END)
    if end == -1:
        return "", text.strip()
    return text[:end].strip(), text[end + len(THINK_END) :].strip()


def repair_truncated_json(answer):
    """Parse the score object, closing braces the model never got to emit.

    Returns ``(obj_or_None, repaired)``. ``repaired`` is True only when braces had to
    be synthesised, so callers can flag the dimension rather than trust it silently.

    The walk is string-aware: a ``{`` inside a criterion name must not raise the
    depth, or the recovered object would be truncated in the wrong place.
    """
    start = (answer or "").find("{")
    if start == -1:
        return None, False
    text = answer[start:]

    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    closes = []  # (index just past a '}', nesting depth remaining after it)
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            closes.append((index + 1, depth))
            if depth == 0:
                # Balanced early, with trailing junk after it (prose, a second object).
                try:
                    return json.loads(text[: index + 1]), False
                except json.JSONDecodeError:
                    pass

    # Truncated. Rewind to the last criterion that closed cleanly and shut the rest,
    # which also discards any half-written `"Text Layout": {"sco` fragment.
    for index, remaining in reversed(closes):
        if remaining <= 0:
            continue
        try:
            return json.loads(text[:index] + "}" * remaining), True
        except json.JSONDecodeError:
            continue

    return None, False


def _classify(raw):
    """Bucket one criterion's raw score exactly as ``scoring.map_score`` would treat it.

    Mirroring it matters: a value this calls "scored" must be one the scorer actually
    counts, or the audit would describe a different run than the one being displayed.
    """
    if isinstance(raw, str) and raw.upper() == "N/A":
        return "not_applicable"
    try:
        return "scored" if int(raw) in VALID_SCORES else "invalid"
    except (TypeError, ValueError):
        return "invalid"


@dataclass
class DimensionAudit:
    """What survived of one level-1 dimension's response."""

    status: str = "failed"  # "ok" | "partial" | "failed"
    scored: int = 0
    not_applicable: int = 0
    expected: int = 0
    missing: list = field(default_factory=list)
    invalid: dict = field(default_factory=dict)
    unknown: dict = field(default_factory=dict)
    repaired: bool = False
    forced_close: bool = False

    @property
    def answered(self):
        return self.scored + self.not_applicable

    def note(self):
        """A short parenthetical for the summary table, or '' when nothing is wrong."""
        if self.status == "failed":
            return "no parseable scores"
        parts = []
        if self.answered < self.expected:
            parts.append(f"{self.answered}/{self.expected} criteria")
        if self.repaired:
            parts.append("recovered from a cut-off answer")
        if self.forced_close:
            parts.append("thinking hit its budget")
        if self.invalid:
            parts.append(f"{len(self.invalid)} out-of-range")
        if self.unknown:
            parts.append(f"{len(self.unknown)} unrecognised")
        return ", ".join(parts)


def audit_dimension(fixed_json, dimension, repaired=False, forced_close=False):
    """Check a normalised score object against the dimension's checklist.

    Catches three things the scorer would otherwise swallow: a score outside 0/1/2
    (``map_score`` turns a hallucinated 3 into None, i.e. an N/A that quietly leaves
    the mean), a criterion name that isn't in the checklist, and criteria the model
    never mentioned at all.
    """
    expected = set(CHECKLIST_L3_TO_L2.get(dimension, {}))
    audit = DimensionAudit(expected=len(expected), repaired=repaired, forced_close=forced_close)

    seen = set()
    for level3_dict in (fixed_json or {}).values():
        if not isinstance(level3_dict, dict):
            continue
        for name, score_obj in level3_dict.items():
            raw = score_obj.get("score") if isinstance(score_obj, dict) else score_obj
            seen.add(name)
            if name not in expected:
                audit.unknown[name] = raw
            bucket = _classify(raw)
            if bucket == "scored":
                audit.scored += 1
            elif bucket == "not_applicable":
                audit.not_applicable += 1
            else:
                audit.invalid[name] = raw

    audit.missing = sorted(expected - seen)

    if audit.answered == 0:
        audit.status = "failed"
    elif audit.missing or audit.invalid or audit.unknown or repaired or forced_close:
        audit.status = "partial"
    else:
        audit.status = "ok"
    return audit


def parse_answer(answer, dimension, fix_score_json, forced_close=False):
    """Full recovery path for one dimension: repair, normalise, audit.

    ``fix_score_json`` is injected rather than imported so this module stays a pure
    consumer of the vendored scorer.

    Returns ``(fixed_json_or_None, audit)``.
    """
    score_json, repaired = repair_truncated_json(answer)
    if score_json is None:
        return None, DimensionAudit(
            expected=len(CHECKLIST_L3_TO_L2.get(dimension, {})), forced_close=forced_close
        )
    fixed = fix_score_json(score_json, dimension)
    return fixed, audit_dimension(fixed, dimension, repaired=repaired, forced_close=forced_close)
