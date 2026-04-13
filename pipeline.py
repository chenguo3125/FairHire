"""
pipeline.py — Orchestrator / Facade for the FairHire multi-agent system.

This module is the ONLY entry point the UI layer (Streamlit) should import.
It enforces strict separation of concerns: the frontend calls
``run_fairhire_evaluation()`` and receives a single standardised dictionary.
It never needs to know how the agents communicate internally.
"""

from __future__ import annotations

from typing import Any, Dict

from agent1_scrubber import scrub_sentence
from agent2_llm import evaluate_technical_merit
from resume_format import bullets_to_prose


def run_fairhire_evaluation(
    raw_resume_text: str,
    *,
    normalize_bullets_to_prose: bool = False,
) -> Dict[str, Any]:
    """
    Execute the full FairHire pipeline and return a standardised result dict.

    Pipeline:
        1. Optional — bullet lists → paragraph prose when ``normalize_bullets_to_prose``.
        2. Agent 1 — scrub / mask the effective resume text.
        3. Agent 2 — LLM evaluates that text before masking (baseline).
        4. Agent 2 — LLM evaluates the scrubbed text (fair).
        5. Fairness delta = fair_score - baseline_score.

    ``original_text`` is always the pasted input. Agents use the normalized form
    when the option is enabled and bullets were present.
    """

    effective = raw_resume_text
    normalized_text: str | None = None
    if normalize_bullets_to_prose:
        nt = bullets_to_prose(raw_resume_text)
        if nt.strip():
            normalized_text = nt
            effective = nt

    # --- Agent 1: Scrub & extract -----------------------------------------
    masked_text, extracted_data = scrub_sentence(effective)

    # --- Agent 2: Baseline (LLM sees text with demographic cues) ----------
    baseline_result = evaluate_technical_merit(effective)

    # --- Agent 2: Fair (LLM sees scrubbed text, no demographic cues) ------
    fair_result = evaluate_technical_merit(masked_text)

    baseline_score = baseline_result.get("score_0_100", 0)
    fair_score = fair_result.get("score_0_100", 0)
    fairness_delta = fair_score - baseline_score

    applied = (
        normalize_bullets_to_prose
        and normalized_text is not None
        and normalized_text.strip() != raw_resume_text.strip()
    )

    return {
        "original_text": raw_resume_text,
        "normalized_text": normalized_text,
        "normalized_bullets_applied": applied,
        "masked_text": masked_text,
        "extracted_data": extracted_data,
        "baseline_score": baseline_score,
        "baseline_justification": baseline_result.get("justification", ""),
        "baseline_strengths": baseline_result.get("strengths", []),
        "baseline_weaknesses": baseline_result.get("weaknesses", []),
        "fair_score": fair_score,
        "fair_justification": fair_result.get("justification", "No justification provided."),
        "fair_strengths": fair_result.get("strengths", []),
        "fair_weaknesses": fair_result.get("weaknesses", []),
        "fairness_delta": fairness_delta,
    }
