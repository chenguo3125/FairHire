"""
FairHire Streamlit Frontend.

The ONLY backend import is ``run_fairhire_evaluation`` from pipeline.py.
This file knows nothing about agents, ontologies, or spaCy internals.
"""

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

    # ── Section 2: AI Evaluation (LLM) ───────────────────────────────────
    st.header("2 — AI Evaluation (Agent 2: LLM Evaluator)")

    score_col, just_col = st.columns([1, 2])

    with score_col:
        st.metric(
            label="Fair Score (de-biased)",
            value=f"{result['fair_score']} / 100",
            delta=f"{result['fair_score'] - result['baseline_score']:+d} vs baseline",
        )
        st.caption(
            f"Baseline (LLM on raw text): **{result['baseline_score']}** / 100"
        )

    with just_col:
        st.subheader("Technical Justification")
        st.info(result["fair_justification"])

    fair_strengths = result.get("fair_strengths", [])
    fair_weaknesses = result.get("fair_weaknesses", [])

    if fair_strengths or fair_weaknesses:
        str_col, weak_col = st.columns(2)
        with str_col:
            st.subheader("Strengths")
            for s in fair_strengths:
                st.markdown(f"- {s}")
        with weak_col:
            st.subheader("Weaknesses")
            for w in fair_weaknesses:
                st.markdown(f"- {w}")

    with st.expander("Baseline evaluation (LLM on raw text)"):
        st.markdown(f"**Score:** {result['baseline_score']} / 100")
        st.markdown(f"**Justification:** {result.get('baseline_justification', '')}")
        bl_str = result.get("baseline_strengths", [])
        bl_weak = result.get("baseline_weaknesses", [])
        if bl_str:
            st.markdown("**Strengths:** " + ", ".join(bl_str))
        if bl_weak:
            st.markdown("**Weaknesses:** " + ", ".join(bl_weak))

    st.divider()

    # ── Section 3: Fairness Audit ─────────────────────────────────────────
    st.header("3 — Fairness Audit (Agent 3: Counterfactual)")

    delta = result["fairness_delta"]

    if delta == 0:
        st.success(
            "**Fairness Delta = 0** — The same LLM scored the resume identically "
            "before and after scrubbing. No demographic bias detected."
        )
    elif delta > 0:
        st.warning(
            f"**Fairness Delta = +{delta}** — The scrubbed resume scored higher. "
            f"The LLM may have been influenced by demographic cues in the raw text."
        )
    else:
        st.error(
            f"**Fairness Delta = {delta}** — The scrubbed resume scored lower. "
            f"Masking may have removed content the LLM valued, or the raw text "
            f"received a demographic bonus."
        )
