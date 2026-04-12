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


def run_fairhire_evaluation(raw_resume_text: str) -> Dict[str, Any]:
    """
    Execute the full FairHire pipeline and return a standardised result dict.

    Pipeline:
        1. Agent 1 — scrub / mask the raw resume.
        2. Agent 2 — LLM evaluates the RAW text (baseline, demographic cues visible).
        3. Agent 2 — LLM evaluates the SCRUBBED text (fair, de-biased).
        4. Fairness delta = fair_score - baseline_score.
    """

    # --- Agent 1: Scrub & extract -----------------------------------------
    masked_text, extracted_data = scrub_sentence(raw_resume_text)

    # --- Agent 2: Baseline (LLM sees raw text with demographic cues) ------
    baseline_result = evaluate_technical_merit(raw_resume_text)

    # --- Agent 2: Fair (LLM sees scrubbed text, no demographic cues) ------
    fair_result = evaluate_technical_merit(masked_text)

    baseline_score = baseline_result.get("score_0_100", 0)
    fair_score = fair_result.get("score_0_100", 0)
    fairness_delta = fair_score - baseline_score

    return {
        "original_text": raw_resume_text,
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
