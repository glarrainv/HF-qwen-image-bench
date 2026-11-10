import json
import re

from .checklists import L3_TO_L2

SCORE_MAP = {0: 0.0, 1: 1.0, 2: 2.0}
PAPER_SCALE = {0: 0.0, 1: 60.0, 2: 100.0}


def extract_json_from_response(response_text):
    """Extract JSON score object from model output (skip <think> section)."""
    text = response_text
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def map_score(raw_score):
    """Map raw score to final score: 0→0, 1→1, 2→2, 'N/A'→None."""
    if isinstance(raw_score, str) and raw_score.upper() == "N/A":
        return None
    try:
        return SCORE_MAP[int(raw_score)]
    except (KeyError, ValueError, TypeError):
        return None


def mean_non_none(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def compute_dimension_score(score_json):
    """
    Compute aggregated score for a single level-1 dimension.

    Input: {"Realism": {"Physical Logic": {"score": 0}, "Material Properties": {"score": 1}}, ...}
    Output: {
        "level1_score": float | None,
        "level2_scores": {"Realism": float | None, ...},
        "level3_scores": {"Realism": {"Physical Logic": 0.0, ...}, ...}
    }
    """
    level2_scores = {}
    level3_scores = {}

    for level2_name, level3_dict in score_json.items():
        level3_scores[level2_name] = {}
        level3_mapped = []

        for level3_name, score_obj in level3_dict.items():
            raw = score_obj.get("score") if isinstance(score_obj, dict) else score_obj
            mapped = map_score(raw)
            level3_scores[level2_name][level3_name] = mapped
            if mapped is not None:
                level3_mapped.append(mapped)

        level2_scores[level2_name] = mean_non_none(level3_mapped)

    level1_score = mean_non_none(list(level2_scores.values()))

    return {
        "level1_score": level1_score,
        "level2_scores": level2_scores,
        "level3_scores": level3_scores,
    }


L3_RENAME = {
    "Creative Generation": {"Feature Mapping": "Feature Matching"},
}


def fix_score_json(score_json, l1_dim):
    """Fix flat structure, L3 misplacement, and L3 typos based on checklists.py hierarchy."""
    if not score_json:
        return score_json

    mapping = L3_TO_L2.get(l1_dim, {})
    rename = L3_RENAME.get(l1_dim, {})

    first_val = next(iter(score_json.values()), None)
    if isinstance(first_val, dict) and "score" in first_val:
        result = {}
        for l3_name, score_obj in score_json.items():
            l3_name = rename.get(l3_name, l3_name)
            l2_name = mapping.get(l3_name, l3_name)
            result.setdefault(l2_name, {})[l3_name] = score_obj
        return result

    result = {}
    for l2_key, l3_dict in score_json.items():
        if not isinstance(l3_dict, dict):
            continue
        for l3_name, score_obj in l3_dict.items():
            l3_name = rename.get(l3_name, l3_name)
            correct_l2 = mapping.get(l3_name, l2_key)
            result.setdefault(correct_l2, {})[l3_name] = score_obj
    return result


def aggregate_total_score(dim_results):
    """
    Aggregate across all level-1 dimensions to total score.

    Input: {"Quality": {"level1_score": 60.0, ...}, "Aesthetics": {"level1_score": 80.0, ...}, ...}
    Output: float | None
    """
    level1_scores = [
        r["level1_score"] for r in dim_results.values()
        if r is not None and r.get("level1_score") is not None
    ]
    return mean_non_none(level1_scores)