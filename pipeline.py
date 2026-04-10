"""
pipeline.py — Orchestrator / Facade for the FairHire multi-agent system.

This module is the ONLY entry point the UI layer (Streamlit) should import.
It enforces strict separation of concerns: the frontend calls
``run_fairhire_evaluation()`` and receives a single standardised dictionary.
It never needs to know how the agents communicate internally.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from agent1_scrubber import scrub_sentence
from agent2_grader import grade_resume_mode
from agent3_auditor import audit_counterfactual_fairness


def run_fairhire_evaluation(raw_resume_text: str) -> Dict[str, Any]:
    """
    Execute the full FairHire pipeline and return a standardised result dict.

    Pipeline:
        1. Agent 1 — scrub / mask the raw resume.
        2. Agent 2 — score the raw resume WITH simulated bias (baseline).
        3. Agent 2 — score the scrubbed resume WITHOUT bias (fair score).
        4. Agent 3 — compute the counterfactual fairness delta.
    """

    # --- Agent 1: Scrub & extract -----------------------------------------
    masked_text, extracted_data = scrub_sentence(raw_resume_text)

    # --- Agent 2: Baseline score (raw text, bias penalty ON) --------------
    baseline = grade_resume_mode(
        text=raw_resume_text, scrub=False, apply_bias_penalty=True,
    )

    # --- Agent 2: Fair score (scrubbed text, bias penalty OFF) ------------
    fair = grade_resume_mode(
        text=masked_text, scrub=False, apply_bias_penalty=False,
    )

    # --- Agent 3: Counterfactual fairness audit ---------------------------
    audit = audit_counterfactual_fairness(raw_resume_text)

    # --- Build per-sentence technical justification -----------------------
    justification_lines = []
    for entry in fair.breakdown.get("sentences", []):
        frame = entry.get("frame", {})
        skills = frame.get("technical_skills") or []
        verb = frame.get("action_verb") or ""
        impact = frame.get("impact_metric") or ""
        parts = []
        if verb:
            parts.append(f"action: {verb}")
        if skills:
            parts.append(f"skills: {', '.join(skills)}")
        if impact:
            parts.append(f"impact: {impact}")
        if parts:
            justification_lines.append(" | ".join(parts))

    justification = (
        "; ".join(justification_lines)
        if justification_lines
        else "No strong technical signals detected."
    )

    return {
        "original_text": raw_resume_text,
        "masked_text": masked_text,
        "extracted_data": extracted_data,
        "baseline_score": baseline.score_0_100,
        "fair_score": fair.score_0_100,
        "justification": justification,
        "fairness_delta": audit.fairness_delta,
        "baseline_breakdown": baseline.breakdown,
        "fair_breakdown": fair.breakdown,
        "audit": asdict(audit),
    }
