"""
FairHire Streamlit Frontend.

The ONLY backend import is ``run_fairhire_evaluation`` from pipeline.py.
This file knows nothing about agents, ontologies, or spaCy internals.
"""

import json

import streamlit as st

from pipeline import run_fairhire_evaluation

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FairHire",
    page_icon=":briefcase:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("FairHire: Unbiased Technical Evaluator")
st.caption(
    "A multi-agent AI system that evaluates entry-level backend & systems "
    "engineering resumes with **Responsible AI by Design**."
)

st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

resume_text = st.text_area(
    "Paste a resume or bullet points below:",
    height=220,
    placeholder=(
        "e.g.  Sarah served as President of the Women in Computing society "
        "and optimized the MIPS assembly pipeline to reduce execution time by 15%."
    ),
)

run_button = st.button("Run Evaluation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

if run_button:
    if not resume_text.strip():
        st.warning("Please paste some resume text before running the evaluation.")
        st.stop()

    with st.spinner("Running the FairHire pipeline..."):
        result = run_fairhire_evaluation(resume_text.strip())

    st.divider()

    # ── Section 1: Data Transformation ────────────────────────────────────
    st.header("1 — Data Transformation (Agent 1: Scrubber)")

    col_orig, col_mask = st.columns(2)

    with col_orig:
        st.subheader("Original Text")
        st.text(result["original_text"])

    with col_mask:
        st.subheader("Masked Text")
        st.text(result["masked_text"])

    with st.expander("Extracted JSON payload (masked items & metadata)"):
        st.json(result["extracted_data"])

    st.divider()

    # ── Section 2: AI Evaluation ──────────────────────────────────────────
    st.header("2 — AI Evaluation (Agent 2: Grader)")

    score_col, just_col = st.columns([1, 2])

    with score_col:
        st.metric(
            label="Fair Score",
            value=f"{result['fair_score']} / 100",
            delta=f"{result['fair_score'] - result['baseline_score']:+d} vs baseline",
        )
        st.caption(
            f"Baseline score (with simulated bias): **{result['baseline_score']}** / 100"
        )

    with just_col:
        st.subheader("Technical Justification")
        st.info(result["justification"])

    with st.expander("Fair score breakdown (per-sentence detail)"):
        st.json(result["fair_breakdown"])

    with st.expander("Baseline score breakdown (simulated bias detail)"):
        st.json(result["baseline_breakdown"])

    st.divider()

    # ── Section 3: Auditor Report ─────────────────────────────────────────
    st.header("3 — Fairness Audit (Agent 3: Auditor)")

    delta = result["fairness_delta"]

    if delta == 0:
        st.success(
            "**Fairness Delta = 0** — The resume score is identical before and "
            "after scrubbing. No demographic bias detected in the evaluation."
        )
    elif delta > 0:
        st.warning(
            f"**Fairness Delta = +{delta}** — The scrubbed resume scored higher "
            f"than the raw resume. Agent 1 removed demographic cues that were "
            f"penalised by the simulated bias model."
        )
    else:
        st.error(
            f"**Fairness Delta = {delta}** — Unexpected: the scrubbed resume scored "
            f"lower. This may indicate that masking removed content the grader valued."
        )

    with st.expander("Full audit report"):
        st.json(result["audit"])
