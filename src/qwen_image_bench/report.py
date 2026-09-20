"""Turning judge output into the tables the UI renders.

Kept free of torch/transformers imports so it can be exercised without a GPU.
"""

from .checklists import DIM_TO_CHECKLIST
from .scoring import aggregate_total_score

DIMENSIONS = list(DIM_TO_CHECKLIST)

VERDICTS = {0: "0 · Fail", 1: "1 · Pass", 2: "2 · Excel"}


def format_summary(dim_results, done, total_requested, current=None):
    """Markdown summary: total score plus a per-dimension table.

    ``dim_results`` maps a level-1 dimension to the output of
    compute_dimension_score, or None when the model's JSON could not be parsed.
    ``current`` names the dimension being generated right now, if any.
    """
    scored = {d: r for d, r in dim_results.items() if r and r["level1_score"] is not None}
    total = aggregate_total_score(scored) if scored else None

    if total is None:
        lines = ["### Total — pending"]
    elif done < total_requested:
        lines = [f"### Total (partial) — {total:.1f} / 100"]
    elif len(scored) < len(DIMENSIONS):
        lines = [f"### Total (subset of dimensions) — {total:.1f} / 100"]
    else:
        lines = [f"### Total — {total:.1f} / 100"]

    lines += ["", "| Dimension | Score |", "| --- | --- |"]
    for dimension, result in dim_results.items():
        if result is None:
            value = "_failed to parse_"
        elif result["level1_score"] is None:
            value = "N/A"
        else:
            value = f"{result['level1_score']:.1f}"
        lines.append(f"| {dimension} | {value} |")

    if current is not None:
        lines += ["", f"_Judging **{current}** — {done}/{total_requested} complete._"]
    elif done < total_requested:
        lines += ["", f"_Judging… {done}/{total_requested} dimensions complete._"]
    return "\n".join(lines)


def detail_rows(dim_results, fixed_jsons):
    """Flatten to (L1, L2, L3, raw verdict, mapped score) rows."""
    rows = []
    for dimension, result in dim_results.items():
        if result is None:
            continue
        fixed = fixed_jsons.get(dimension) or {}
        for level2, level3_scores in result["level3_scores"].items():
            for level3, mapped in level3_scores.items():
                entry = fixed.get(level2, {}).get(level3)
                raw = entry.get("score") if isinstance(entry, dict) else entry
                rows.append([
                    dimension,
                    level2,
                    level3,
                    VERDICTS.get(raw, str(raw)),
                    "N/A" if mapped is None else f"{mapped:.0f}",
                ])
    return rows
