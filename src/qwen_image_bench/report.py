"""
Turning judge output into the tables the UI renders.
"""

from .checklists import DIM_TO_CHECKLIST
from .scoring import PAPER_SCALE, aggregate_total_score

DIMENSIONS = list(DIM_TO_CHECKLIST)

MAX_SCORE = 2

VERDICTS = {0: "0; Fail", 1: "1; Pass", 2: "2; Excel"}

RUSHED_NOTE = "! Rushed: thinking hit the budget"
RUSHED_FOOTNOTE = (
    "> **! Rushed:** the preset's thinking budget ran out, so the answer was forced "
    "early. Rushed scores still count toward the total."
)


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def paper_scale_total(dim_results):
    """
    Compute using original PAPER SCALE.
    """

    level1 = []
    for result in dim_results.values():
        if not result:
            continue
        level2 = [
            _mean([PAPER_SCALE.get(v) for v in level3.values() if v is not None])
            for level3 in result["level3_scores"].values()
        ]
        level1.append(_mean(level2))
    return _mean(level1)


def format_summary(dim_results, requested, current=(), rushed=()):
    """Markdown summary: total score plus a per-dimension table.

    ``dim_results`` maps a level-1 dimension to the output of
    compute_dimension_score, or None when the model's JSON could not be parsed.
    ``requested`` is every dimension this run will judge, in canonical order, so
    the table keeps its shape while the batch decodes. ``current`` names the
    dimensions being generated right now — a whole batch of them, normally.
    ``rushed`` names dimensions whose thinking was cut off by the budget. They are
    labelled and still counted in the total.
    """
    scored = {d: r for d, r in dim_results.items() if r and r["level1_score"] is not None}
    total = aggregate_total_score(scored) if scored else None
    done = len(dim_results)

    if total is None:
        lines = ["### Total — pending"]
    elif done < len(requested):
        lines = [f"### Total (partial) — {total:.2f} / {MAX_SCORE}"]
    elif len(scored) < len(DIMENSIONS):
        lines = [f"### Total (subset of dimensions) — {total:.2f} / {MAX_SCORE}"]
    else:
        lines = [f"### Total — {total:.2f} / {MAX_SCORE}"]

    paper = paper_scale_total(scored) if scored else None
    if paper is not None:
        lines += ["", f"_On the paper's 0/60/100 scale: **{paper:.1f} / 100**._"]

    lines += ["", "| Dimension | Score | Notes |", "| --- | --- | --- |"]
    for dimension in requested:
        result = dim_results.get(dimension)
        if dimension not in dim_results:
            value = "_judging…_" if dimension in current else "_pending_"
        elif result is None:
            value = "_failed to parse_"
        elif result["level1_score"] is None:
            value = "N/A"
        else:
            value = f"{result['level1_score']:.2f}"

        note = RUSHED_NOTE if dimension in rushed else ""
        lines.append(f"| {dimension} | {value} | {note} |")

    if done < len(requested):
        lines += ["", f"_Judging… {done}/{len(requested)} dimensions complete._"]

    if any(dimension in rushed for dimension in dim_results):
        lines += ["", RUSHED_FOOTNOTE]
    return "\n".join(lines)


def detail_rows(dim_results, fixed_jsons):
    """Flatten to (L1, L2, L3, verdict) rows."""
    rows = []
    for dimension, result in dim_results.items():
        if result is None:
            continue
        fixed = fixed_jsons.get(dimension) or {}
        for level2, level3_scores in result["level3_scores"].items():
            for level3 in level3_scores:
                entry = fixed.get(level2, {}).get(level3)
                raw = entry.get("score") if isinstance(entry, dict) else entry
                rows.append([dimension, level2, level3, VERDICTS.get(raw, str(raw))])
    return rows
