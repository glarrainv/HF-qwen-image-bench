"""Turning judge output into the tables the UI renders.

Kept free of torch/transformers imports so it can be exercised without a GPU.

Scores here are on the raw 0-2 verdict scale, because SCORE_MAP is the identity
(see scoring.py). That is deliberately *not* a rescale of the paper's 0/60/100
numbers -- 0/60/100 is non-linear, so verdicts {1, 2} average to 1.5 here but to 80
upstream. ``paper_scale_total`` recovers the comparable figure.
"""

from .checklists import DIM_TO_CHECKLIST
from .scoring import SCORE_MAP, aggregate_total_score

DIMENSIONS = list(DIM_TO_CHECKLIST)

VERDICTS = {0: "0 · Fail", 1: "1 · Pass", 2: "2 · Excel"}

MAX_SCORE = 2

# Upstream's mapping, kept here rather than in the vendored scorer so the primary
# scale stays the one the rest of the app agrees on.
PAPER_SCALE = {0.0: 0.0, 1.0: 60.0, 2.0: 100.0}

STATUS_MARKS = {"partial": "⚠", "failed": "✕"}


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def paper_scale_total(dim_results):
    """The same verdicts aggregated on upstream's 0/60/100 scale, for comparability.

    Works off ``level3_scores``, which hold the raw verdicts while SCORE_MAP is the
    identity. Returns None if that assumption ever stops holding, rather than quietly
    reporting a number computed twice over.
    """
    if SCORE_MAP != {0: 0.0, 1: 1.0, 2: 2.0}:
        return None

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


def format_summary(dim_results, requested, current=(), audits=None):
    """Markdown summary: total score plus a per-dimension table.

    ``dim_results`` maps a level-1 dimension to the output of
    compute_dimension_score, or None when the model's JSON could not be parsed.
    ``requested`` is every dimension this run will judge, in canonical order, so
    the table keeps its shape while the batch decodes. ``current`` names the
    dimensions being generated right now — a whole batch of them, normally.
    ``audits`` maps a dimension to its DimensionAudit, so a recovered or incomplete
    answer is labelled rather than passed off as a clean one.
    """
    audits = audits or {}
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

        audit = audits.get(dimension)
        note = ""
        if audit is not None and audit.status != "ok":
            note = f"{STATUS_MARKS.get(audit.status, '')} {audit.note()}".strip()
        lines.append(f"| {dimension} | {value} | {note} |")

    if done < len(requested):
        lines += ["", f"_Judging… {done}/{len(requested)} dimensions complete._"]

    if any(a.status == "partial" for a in audits.values()):
        lines += [
            "",
            "> ⚠ A partial dimension is averaged over the criteria that survived, not "
            "over the whole checklist. Missing criteria are absent from the mean, not "
            "counted as zero.",
        ]
    return "\n".join(lines)


def detail_rows(dim_results, fixed_jsons, audits=None):
    """Flatten to (L1, L2, L3, verdict) rows, including criteria the model skipped.

    On the 0-2 scale the mapped score equals the raw verdict, so the separate score
    column is gone. Missing criteria are listed explicitly — an absent row is the one
    thing a reader cannot notice on their own.
    """
    audits = audits or {}
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

        audit = audits.get(dimension)
        for level3 in audit.missing if audit else []:
            rows.append([dimension, "—", level3, "— not answered —"])
    return rows
