from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from agent1_scrubber import NLP, frame_sentence, scrub_and_mask, scrub_sentence
from category_dimensions import RACE_ETHNICITY_DIMENSION, fairness_dimension_for_category


def _compute_weighted_bias_penalty(
    masked_items: List[Dict[str, Any]],
    *,
    base_points: int = 5,
    race_ethnicity_extra: int = 1,
    cap: int = 30,
) -> Tuple[int, Dict[str, Any]]:
    """
    Weighted penalty per cue: race/ethnicity cues get a slightly higher weight
    for audit visibility. Total penalty is capped.
    """
    by_dim: Dict[str, List[Dict[str, Any]]] = {}
    for item in masked_items:
        dim = fairness_dimension_for_category(item.get("category"))
        by_dim.setdefault(dim, []).append(item)

    total_penalty = 0
    penalty_by_dim: Dict[str, int] = {}
    for dim, items in by_dim.items():
        if dim == RACE_ETHNICITY_DIMENSION:
            p = (base_points + race_ethnicity_extra) * len(items)
        else:
            p = base_points * len(items)
        penalty_by_dim[dim] = p
        total_penalty += p

    total_penalty = min(cap, total_penalty)

    breakdown: Dict[str, Any] = {
        "weights": {
            "base_points_per_cue": base_points,
            "race_ethnicity_extra_per_cue": race_ethnicity_extra,
            "penalty_cap": cap,
        },
        "dimensions": {
            dim: {
                "count": len(items),
                "penalty_points": penalty_by_dim.get(dim, 0),
                "items": items[:15],
            }
            for dim, items in sorted(by_dim.items(), key=lambda x: x[0])
        },
        "race_ethnicity": {
            "count": len(by_dim.get(RACE_ETHNICITY_DIMENSION, [])),
            "penalty_points": penalty_by_dim.get(RACE_ETHNICITY_DIMENSION, 0),
            "items": by_dim.get(RACE_ETHNICITY_DIMENSION, [])[:15],
            "note": (
                "Cues mapped via category_dimensions.CATEGORY_TO_FAIRNESS_DIMENSION "
                f"(dimension key {RACE_ETHNICITY_DIMENSION!r}). "
                "Scrubbing these should reduce this component of the simulated bias penalty."
            ),
        },
    }
    return total_penalty, breakdown


@dataclass(frozen=True, slots=True)
class GradeResult:
    score_0_100: int
    breakdown: Dict[str, Any]
    masked_text: str


def _split_into_sentences(text: str) -> List[str]:
    doc = NLP(text)
    return [s.text.strip() for s in doc.sents if s.text and s.text.strip()]


def _score_frame(frame: Any) -> Tuple[int, Dict[str, Any]]:
    points = 0
    details: Dict[str, Any] = {}

    verb = (frame.action_verb or "").strip()
    if verb:
        points += 15
        details["action_verb"] = {"value": verb, "points": 15}
    else:
        details["action_verb"] = {"value": None, "points": 0}

    role = (frame.role or "").strip()
    if role and not role.startswith("["):
        points += 15
        details["role"] = {"value": role, "points": 15}
    else:
        details["role"] = {"value": role or None, "points": 0}

    skills = [s for s in (frame.technical_skills or []) if s and s.strip()]
    skill_points = min(30, 10 * len(skills))
    points += skill_points
    details["technical_skills"] = {"value": skills, "points": skill_points}

    impact = (frame.impact_metric or "").strip()
    if impact:
        points += 20
        details["impact_metric"] = {"value": impact, "points": 20}
    else:
        details["impact_metric"] = {"value": None, "points": 0}

    present = sum(bool(x) for x in [verb, skills, impact])
    bonus = 10 if present >= 2 else 0
    points += bonus
    details["completeness_bonus"] = {"value": present, "points": bonus}

    return points, details


def grade_resume(text: str) -> GradeResult:
    """Default: score scrubbed text, no simulated bias penalty."""
    return grade_resume_mode(text=text, scrub=True, apply_bias_penalty=False)


def grade_resume_mode(*, text: str, scrub: bool, apply_bias_penalty: bool) -> GradeResult:
    scored_text = scrub_and_mask(text) if scrub else text
    sentences = _split_into_sentences(scored_text)

    per_sentence: List[Dict[str, Any]] = []
    total = 0
    for sent in sentences:
        frame = frame_sentence(sent)
        s_points, s_details = _score_frame(frame)
        total += s_points
        per_sentence.append(
            {
                "sentence": sent,
                "frame": {
                    "action_verb": frame.action_verb,
                    "role": frame.role,
                    "technical_skills": frame.technical_skills,
                    "impact_metric": frame.impact_metric,
                },
                "points": s_points,
                "breakdown": s_details,
            }
        )

    bias_penalty = 0
    bias_findings: Dict[str, Any] = {"enabled": bool(apply_bias_penalty), "penalty_points": 0}
    if apply_bias_penalty:
        _masked_preview, payload = scrub_sentence(text)
        masked_items = list(payload.get("masked_items") or [])
        bias_penalty, dim_breakdown = _compute_weighted_bias_penalty(masked_items)
        total -= bias_penalty
        bias_findings = {
            "enabled": True,
            "penalty_points": bias_penalty,
            "sensitive_items_count": len(masked_items),
            "sensitive_items": masked_items[:20],
            "note": (
                "Penalty is an intentional bias simulation for counterfactual fairness experiments. "
                "Includes explicit breakdown for race/ethnicity-related cues (RACE_ETHNICITY_CULTURE)."
            ),
            **dim_breakdown,
        }

    score = max(0, min(100, int(round(total))))
    section_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    breakdown = {
        "raw_points": total,
        "bias_penalty": bias_findings,
        "sentences": per_sentence,
        "document": {
            "sentence_count": len(sentences),
            "paragraph_or_section_count": len(section_chunks),
            "character_count": len(text.strip()),
            "scored_character_count": len(scored_text),
        },
        "scoring_note": "Rule-based heuristic score for fairness auditing; not a hiring decision.",
    }
    return GradeResult(score_0_100=score, breakdown=breakdown, masked_text=scored_text)


if __name__ == "__main__":
    sample = (
        "Sarah served as President of the Women in Computing society. "
        "Optimized the MIPS assembly pipeline to reduce execution time by 15%."
    )
    result = grade_resume_mode(text=sample, scrub=False, apply_bias_penalty=True)
    print("SCORE:", result.score_0_100)
    print("MASKED_TEXT:", result.masked_text)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
