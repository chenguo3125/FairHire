from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from agent1_scrubber import scrub_and_mask
from agent2_grader import GradeResult, grade_resume_mode


@dataclass(frozen=True, slots=True)
class FairnessAudit:
    original_score: int
    scrubbed_score: int
    fairness_delta: int
    original_masked_preview: str
    scrubbed_masked_preview: str
    notes: Dict[str, Any]


def audit_counterfactual_fairness(resume_text: str) -> FairnessAudit:
    """
    Agent 3: Counterfactual fairness auditor.

    - Baseline: score raw text with simulated bias penalty (detects cues via Agent 1).
    - Scrubbed: score scrubbed text without bias penalty.
    Fairness delta = scrubbed_score - original_score.
    """
    baseline: GradeResult = grade_resume_mode(
        text=resume_text, scrub=False, apply_bias_penalty=True
    )
    scrubbed_text = scrub_and_mask(resume_text)
    scrubbed: GradeResult = grade_resume_mode(
        text=scrubbed_text, scrub=False, apply_bias_penalty=False
    )

    delta = scrubbed.score_0_100 - baseline.score_0_100

    notes: Dict[str, Any] = {
        "interpretation": (
            "If delta != 0, the scoring function is sensitive to terms removed/masked by Agent 1. "
            "Goal is to minimize delta while preserving technical content."
        ),
        "baseline_raw_points": baseline.breakdown.get("raw_points"),
        "scrubbed_raw_points": scrubbed.breakdown.get("raw_points"),
        "baseline_bias_penalty": baseline.breakdown.get("bias_penalty"),
    }

    return FairnessAudit(
        original_score=baseline.score_0_100,
        scrubbed_score=scrubbed.score_0_100,
        fairness_delta=delta,
        original_masked_preview=baseline.masked_text,
        scrubbed_masked_preview=scrubbed.masked_text,
        notes=notes,
    )


if __name__ == "__main__":
    text = (
        "Sarah served as President of the Women in Computing society and optimized the MIPS assembly code "
        "to reduce execution time by 15%."
    )
    audit = audit_counterfactual_fairness(text)
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))
