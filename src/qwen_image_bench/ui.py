"""HTML rendering and styling for the Gradio front end.

Everything here is pure string-building: no Gradio objects, no model calls, so the
whole presentation layer can be exercised without a GPU. Colours come from Gradio's
own CSS variables wherever possible so the page follows the light/dark setting.
"""

from html import escape

from .report import DIMENSIONS, MAX_SCORE, paper_scale_total
from .scoring import aggregate_total_score

SPACE_URL = "https://huggingface.co/spaces/glarr/qwen-image-bench-classification"
CODE_URL = "https://github.com/glarrainv/qwen-image-bench"
MODEL_URL = "https://huggingface.co/Qwen/Qwen-Image-Bench"
PAPER_URL = "https://arxiv.org/abs/2605.28091"
UPSTREAM_URL = "https://github.com/QwenLM/Qwen-Image-Bench"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"

VERDICT_LABELS = {0: "Fail", 1: "Pass", 2: "Excel"}

LIVE_TRACE_TAIL = 1200


# --- small helpers -------------------------------------------------------------------


def _fmt_elapsed(seconds):
    seconds = int(max(seconds or 0, 0))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def fmt_duration(seconds):
    """Human ceiling for a GPU reservation: '45 s' or '3 min'."""
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    return f"{round(seconds / 60)} min"


def _score_text(value):
    return "—" if value is None else f"{value:.2f}"


def _bar(value, maximum=MAX_SCORE):
    pct = 0 if value is None else max(0.0, min(1.0, value / maximum)) * 100
    return f'<div class="qib-bar"><span style="width:{pct:.1f}%"></span></div>'


def _ordered(mapping):
    """Keys in the benchmark's canonical order, whatever order they finished in."""
    return [d for d in DIMENSIONS if d in mapping] + [d for d in mapping if d not in DIMENSIONS]


def _pill(text, kind):
    return f'<span class="qib-pill qib-pill--{kind}">{escape(text)}</span>'


# --- header / footer -----------------------------------------------------------------


def header_html():
    return f"""
<div class="qib-header">
  <div class="qib-header__text">
    <h1>Qwen-Image-Bench Judge</h1>
    <p>Score a generated image against the prompt that produced it. Q-Judger, a
    27B vision-language judge, grades every criterion of the benchmark's five
    dimensions as <b>0 Fail</b>, <b>1 Pass</b> or <b>2 Excel</b>.</p>
  </div>
  <nav class="qib-links">
    <a href="{MODEL_URL}" target="_blank" rel="noopener">Model</a>
    <a href="{PAPER_URL}" target="_blank" rel="noopener">Paper</a>
    <a href="{CODE_URL}" target="_blank" rel="noopener">Code</a>
  </nav>
</div>
"""


def footer_html():
    return f"""
<div class="qib-footer">
  <span>© 2026 G. Larrain · Licensed under the
  <a href="{LICENSE_URL}" target="_blank" rel="noopener">Apache License 2.0</a></span>
  <span>Adapted from <a href="{UPSTREAM_URL}" target="_blank" rel="noopener">QwenLM/Qwen-Image-Bench</a>.
  Changes: 4-bit NF4 loading, thinking-budget presets, dimension selection, streaming UI.
  Judge model by the Qwen team.</span>
</div>
"""


# --- status bar ----------------------------------------------------------------------


def status_html(state, done=0, total=0, elapsed=None, current=(), message=None):
    """One-line run status. ``state`` is idle | queued | running | done | stopped."""
    titles = {
        "idle": "Ready",
        "queued": "Waiting for a GPU",
        "running": "Judging",
        "done": "Complete",
        "stopped": "Stopped",
    }
    if message is None:
        message = {
            "idle": "Add an image and its prompt, then press <b>Judge image</b>.",
            "queued": "Requesting GPU time. On a busy Space this can take a minute or more.",
            "running": "Now on: " + escape(", ".join(current)) if current else "Parsing answers…",
            "done": "All requested dimensions have been scored.",
            "stopped": "The run was cancelled. Completed dimensions are kept below.",
        }[state]

    meta = []
    if total:
        meta.append(f"{done} / {total} dimensions")
    if elapsed is not None:
        meta.append(_fmt_elapsed(elapsed))

    pct = (done / total * 100) if total else 0
    progress = ""
    if state in ("queued", "running", "done", "stopped") and total:
        progress = (
            f'<div class="qib-progress{" qib-progress--indeterminate" if state == "queued" else ""}">'
            f'<span style="width:{pct:.1f}%"></span></div>'
        )
    return f"""
<div class="qib-status qib-status--{state}">
  <div class="qib-status__row">
    <span class="qib-dot"></span>
    <span class="qib-status__title">{titles[state]}</span>
    <span class="qib-status__msg">{message}</span>
    <span class="qib-status__meta">{" · ".join(meta)}</span>
  </div>
  {progress}
</div>
"""


# --- scores card ---------------------------------------------------------------------


def scores_html(dim_results, requested, current=(), rushed=()):
    """Total score plus one compact card per requested dimension."""
    if not requested:
        return '<div class="qib-empty">Scores will appear here as each dimension finishes.</div>'

    scored = {d: r for d, r in dim_results.items() if r and r["level1_score"] is not None}
    total = aggregate_total_score(scored) if scored else None
    paper = paper_scale_total(scored) if scored else None
    done = len(dim_results)

    if total is None:
        qualifier = "pending"
    elif done < len(requested):
        qualifier = "partial"
    elif len(scored) < len(DIMENSIONS):
        qualifier = "subset of dimensions"
    else:
        qualifier = "all dimensions"

    paper_text = "—" if paper is None else f"{paper:.1f}"
    hero = f"""
<div class="qib-total">
  <div>
    <div class="qib-eyebrow">Overall score <span class="qib-muted">· {qualifier}</span></div>
    <div class="qib-total__value">{_score_text(total)}<small> / {MAX_SCORE}</small></div>
    {_bar(total)}
  </div>
  <div class="qib-total__paper">
    <div class="qib-eyebrow">Paper scale (0/60/100)</div>
    <div class="qib-total__value qib-total__value--sm">{paper_text}<small> / 100</small></div>
  </div>
</div>
"""

    cards = []
    for dimension in requested:
        result = dim_results.get(dimension)
        flag = ""
        if dimension not in dim_results:
            if dimension in current:
                state, value, badge = "live", None, _pill("Judging", "live")
            else:
                state, value, badge = "pending", None, _pill("Queued", "muted")
        elif result is None:
            state, value, badge = "failed", None, _pill("Parse error", "fail")
        elif result["level1_score"] is None:
            state, value, badge = "na", None, _pill("N/A", "muted")
        else:
            state, value, badge = "done", result["level1_score"], ""
        if dimension in rushed:
            flag = _pill("Rushed", "warn")
        cards.append(
            f"""
<div class="qib-card qib-card--{state}">
  <div class="qib-card__head"><span>{escape(dimension)}</span>{badge}{flag}</div>
  <div class="qib-card__value">{_score_text(value)}</div>
  {_bar(value)}
</div>"""
        )

    note = ""
    if any(d in rushed for d in dim_results):
        note = (
            '<p class="qib-note"><b>Rushed</b>: the thinking budget ran out and the '
            "answer was forced early. Rushed scores still count toward the total; "
            "pick a larger budget for a more considered verdict.</p>"
        )
    return hero + f'<div class="qib-cards">{"".join(cards)}</div>' + note


# --- per-criterion breakdown ---------------------------------------------------------


def breakdown_html(dim_results, fixed_jsons):
    """Criterion verdicts grouped by dimension, as a readable table."""
    sections = []
    for dimension in _ordered(dim_results):
        result = dim_results[dimension]
        if result is None:
            sections.append(
                f'<section class="qib-group"><h4>{escape(dimension)}</h4>'
                '<p class="qib-muted">The answer for this dimension could not be parsed. '
                "See the raw output under Reasoning.</p></section>"
            )
            continue
        fixed = fixed_jsons.get(dimension) or {}
        rows = []
        for level2, level3_scores in result["level3_scores"].items():
            first = True
            for level3 in level3_scores:
                entry = fixed.get(level2, {}).get(level3)
                raw = entry.get("score") if isinstance(entry, dict) else entry
                if raw in VERDICT_LABELS:
                    verdict = _pill(f"{raw} · {VERDICT_LABELS[raw]}", f"v{raw}")
                else:
                    verdict = _pill("N/A", "muted")
                l2 = escape(level2) if first else ""
                rows.append(
                    f'<tr><td class="qib-l2">{l2}</td><td>{escape(level3)}</td><td>{verdict}</td></tr>'
                )
                first = False
        score = _score_text(result["level1_score"])
        sections.append(
            f"""
<section class="qib-group">
  <h4>{escape(dimension)} <span class="qib-muted">{score} / {MAX_SCORE}</span></h4>
  <table class="qib-table">
    <thead><tr><th>Sub-capability</th><th>Criterion</th><th>Verdict</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>"""
        )
    if not sections:
        return '<div class="qib-empty">Per-criterion verdicts appear here once a dimension is scored.</div>'
    return "".join(sections)


# --- reasoning stream ----------------------------------------------------------------


def reasoning_html(thinking, raw_outputs, dim_results, rushed, live=None):
    """Finished traces first, then the dimensions still decoding (tail only)."""
    blocks = []
    for dimension in _ordered(thinking):
        trace = thinking[dimension]
        result = dim_results.get(dimension)
        tags = ""
        if result is None:
            tags += _pill("Parse error", "fail")
        elif result["level1_score"] is not None:
            tags += _pill(f"{result['level1_score']:.2f} / {MAX_SCORE}", "done")
        if dimension in rushed:
            tags += _pill("Rushed", "warn")
        body = escape(trace.strip()) if trace and trace.strip() else '<i class="qib-muted">No reasoning emitted.</i>'
        raw = ""
        if result is None:
            raw = (
                '<div class="qib-raw"><div class="qib-eyebrow">Raw answer (no score JSON found)</div>'
                f"<pre>{escape(raw_outputs.get(dimension) or '')}</pre></div>"
            )
        blocks.append(
            f'<article class="qib-trace"><header>{escape(dimension)}{tags}</header>'
            f'<div class="qib-trace__body">{body}</div>{raw}</article>'
        )

    live = live or {}
    for dimension in _ordered(live):
        partial = live[dimension]
        tail = (partial or "").strip()
        clipped = len(tail) > LIVE_TRACE_TAIL
        tail = tail[-LIVE_TRACE_TAIL:]
        body = escape(tail) if tail else '<i class="qib-muted">Starting…</i>'
        lead = '<span class="qib-muted">… </span>' if clipped else ""
        blocks.append(
            f'<article class="qib-trace qib-trace--live"><header>{escape(dimension)}'
            f'{_pill("Thinking", "live")}</header>'
            f'<div class="qib-trace__body">{lead}{body}<span class="qib-caret"></span></div></article>'
        )

    if not blocks:
        return (
            '<div class="qib-empty">The judge reasons before it scores. Its thinking '
            "streams here live, one block per dimension.</div>"
        )
    return "".join(blocks)


# --- stylesheet ----------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }

/* header */
.qib-header { display:flex; gap:24px; align-items:flex-end; justify-content:space-between;
  flex-wrap:wrap; padding:8px 2px 18px; border-bottom:1px solid var(--border-color-primary);
  margin-bottom:6px; }
.qib-header h1 { font-size:1.6rem; font-weight:650; letter-spacing:-0.01em; margin:0 0 6px; }
.qib-header p { margin:0; max-width:720px; color:var(--body-text-color-subdued); line-height:1.5; }
.qib-links { display:flex; gap:8px; }
.qib-links a { font-size:.85rem; font-weight:500; padding:6px 12px; border-radius:999px;
  border:1px solid var(--border-color-primary); color:var(--body-text-color);
  text-decoration:none; background:var(--background-fill-secondary); }
.qib-links a:hover { border-color:var(--color-accent); color:var(--color-accent); }

/* section labels */
.qib-section { font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:var(--body-text-color-subdued); margin:4px 0 -4px; }
.qib-hint, .qib-hint p { font-size:.82rem !important; color:var(--body-text-color-subdued) !important; margin:0 !important; }
.qib-hint strong { color:var(--body-text-color) !important; }
.qib-counter strong { color:var(--error-text-color, #dc2626) !important; }

/* status bar */
.qib-status { border:1px solid var(--border-color-primary); border-radius:10px;
  padding:12px 14px; background:var(--background-fill-secondary); }
.qib-status__row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.qib-status__title { font-weight:600; }
.qib-status__msg { color:var(--body-text-color-subdued); flex:1 1 220px; font-size:.9rem; }
.qib-status__meta { font-variant-numeric:tabular-nums; font-size:.85rem; color:var(--body-text-color-subdued); }
.qib-dot { width:9px; height:9px; border-radius:50%; background:var(--neutral-400); flex:none; }
.qib-status--queued .qib-dot, .qib-status--running .qib-dot { background:var(--color-accent); animation:qib-pulse 1.4s ease-in-out infinite; }
.qib-status--done .qib-dot { background:#16a34a; }
.qib-status--stopped .qib-dot { background:#d97706; }
.qib-progress { height:4px; border-radius:4px; background:var(--border-color-primary); margin-top:10px; overflow:hidden; }
.qib-progress span { display:block; height:100%; background:var(--color-accent); transition:width .4s ease; }
.qib-progress--indeterminate span { width:30% !important; animation:qib-slide 1.6s ease-in-out infinite; }
@keyframes qib-pulse { 0%,100% { opacity:1 } 50% { opacity:.35 } }
@keyframes qib-slide { 0% { transform:translateX(-100%) } 100% { transform:translateX(340%) } }

/* totals + cards */
.qib-total { display:flex; gap:24px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;
  padding:4px 2px 14px; }
.qib-total > div:first-child { flex:1 1 260px; }
.qib-total__value { font-size:2.4rem; font-weight:650; line-height:1.1; font-variant-numeric:tabular-nums; margin:2px 0 8px; }
.qib-total__value small { font-size:1rem; font-weight:500; color:var(--body-text-color-subdued); }
.qib-total__value--sm { font-size:1.6rem; margin-bottom:8px; }
.qib-total__paper { text-align:right; }
@media (max-width: 640px) { .qib-total__paper { text-align:left; } .qib-total { gap:12px; } }
.qib-eyebrow { font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--body-text-color-subdued); }
.qib-muted { color:var(--body-text-color-subdued); font-weight:400; text-transform:none; letter-spacing:0; }
.qib-bar { height:6px; border-radius:6px; background:var(--border-color-primary); overflow:hidden; }
.qib-bar span { display:block; height:100%; border-radius:6px; background:var(--color-accent); transition:width .5s ease; }
.qib-cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(118px, 1fr)); gap:10px; }
.qib-card { border:1px solid var(--border-color-primary); border-radius:10px; padding:12px; background:var(--background-fill-primary); }
.qib-card__head { display:flex; align-items:center; gap:6px; flex-wrap:wrap; font-size:.85rem; font-weight:600; min-height:22px; }
.qib-card__head > span:first-child { margin-right:auto; }
.qib-card__value { font-size:1.5rem; font-weight:650; font-variant-numeric:tabular-nums; margin:6px 0 8px; }
.qib-card--pending .qib-card__value, .qib-card--na .qib-card__value, .qib-card--failed .qib-card__value { color:var(--body-text-color-subdued); }
.qib-card--live { border-color:var(--color-accent); }
.qib-note { font-size:.82rem; color:var(--body-text-color-subdued); margin:12px 0 0; }

/* pills */
.qib-pill { display:inline-block; font-size:.7rem; font-weight:600; line-height:1; padding:4px 7px;
  border-radius:999px; white-space:nowrap; border:1px solid transparent; }
.qib-pill--muted { color:var(--body-text-color-subdued); border-color:var(--border-color-primary); }
.qib-pill--live { color:var(--color-accent); border-color:var(--color-accent); }
.qib-pill--done { color:var(--body-text-color); background:var(--background-fill-secondary); border-color:var(--border-color-primary); }
.qib-pill--warn, .qib-pill--v1 { color:#b45309; background:rgba(217,119,6,.12); }
.qib-pill--fail, .qib-pill--v0 { color:#dc2626; background:rgba(220,38,38,.12); }
.qib-pill--v2 { color:#15803d; background:rgba(22,163,74,.14); }
.dark .qib-pill--warn, .dark .qib-pill--v1 { color:#fbbf24; }
.dark .qib-pill--fail, .dark .qib-pill--v0 { color:#f87171; }
.dark .qib-pill--v2 { color:#4ade80; }

/* breakdown */
.qib-group { margin:0 0 18px; }
.qib-group h4 { margin:0 0 8px; font-size:.95rem; font-weight:600; }
.qib-table { width:100%; border-collapse:collapse; font-size:.86rem; }
.qib-table th { border-bottom:1px solid var(--border-color-primary) !important; text-align:left; font-weight:600; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--body-text-color-subdued); padding:6px 10px; border-bottom:1px solid var(--border-color-primary); }
.qib-table tr, .qib-table thead, .qib-table tbody { border:none !important; background:none !important; }
.qib-table, .qib-table th, .qib-table td { border-left:none !important; border-right:none !important; border-top:none !important; background:none !important; }
.qib-table td { padding:7px 10px; border-bottom:1px solid var(--border-color-primary) !important; vertical-align:top; }
.qib-table td:last-child { width:1%; white-space:nowrap; }
.qib-l2 { font-weight:500; width:34%; }

/* reasoning */
.qib-trace { border:1px solid var(--border-color-primary); border-radius:10px; margin:0 0 10px; overflow:hidden; }
.qib-trace header { display:flex; align-items:center; gap:8px; font-weight:600; font-size:.88rem;
  padding:8px 12px; background:var(--background-fill-secondary); border-bottom:1px solid var(--border-color-primary); }
.qib-trace__body { padding:10px 12px; white-space:pre-wrap; word-break:break-word; font-size:.84rem; line-height:1.55;
  color:var(--body-text-color); }
.qib-trace--live { border-color:var(--color-accent); }
.qib-caret { display:inline-block; width:7px; height:1em; vertical-align:text-bottom; margin-left:2px;
  background:var(--color-accent); animation:qib-pulse 1s steps(2) infinite; }
.qib-raw { padding:0 12px 12px; }
.qib-raw pre { white-space:pre-wrap; font-size:.8rem; margin:6px 0 0; padding:8px; border-radius:6px; background:var(--background-fill-secondary); }

.qib-empty { color:var(--body-text-color-subdued); font-size:.9rem; padding:28px 12px; text-align:center;
  border:1px dashed var(--border-color-primary); border-radius:10px; }

/* footer */
.qib-footer { display:flex; flex-direction:column; gap:4px; font-size:.78rem; color:var(--body-text-color-subdued);
  border-top:1px solid var(--border-color-primary); padding-top:14px; margin-top:4px; }
.qib-footer a { color:inherit !important; text-decoration:underline; text-underline-offset:2px;
  background:none !important; border:none !important; padding:0 !important; }
"""
