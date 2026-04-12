"""
agent2_llm.py — Agent 2: LLM-based Technical Merit Evaluator.

Takes de-biased ``masked_text`` from Agent 1 and sends it to the OpenAI API
(gpt-4o-mini) for semantic evaluation.  Output is a structured JSON object
with score, justification, strengths, and weaknesses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

# Always load project-root .env. Bare load_dotenv() uses find_dotenv(), which
# falls back to os.getcwd() under a debugger (sys.gettrace()) or some REPLs, so
# .env next to this file can be missed when cwd is not the repo root.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _get_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, "
            "or run: export OPENAI_API_KEY=... (never commit secrets)."
        )
    return openai.OpenAI(api_key=api_key)

SYSTEM_PROMPT = """\
You are a strict, highly technical Senior Backend Engineering Manager evaluating resume bullet points for an entry-level systems engineer.
Your task is to evaluate the technical competency, complexity, and impact of the candidate's work based strictly on the provided text.

CRITICAL INSTRUCTIONS:
1. The text you receive has been passed through a de-biasing scrubber. You will see bracketed tags like [CANDIDATE_NAME] or [STEM_AFFINITY_GROUP]. 
2. DO NOT penalize the text for awkward grammar resulting from these tags. Treat them as neutral placeholders.
3. Base your score out of 100 on three pillars:
   - Action & Ownership (e.g., "Architected" > "Helped with")
   - Technical Complexity (e.g., "PostgreSQL indexing" > "Data entry")
   - Quantifiable Impact (e.g., "Reduced latency by 15%")

You MUST return a valid JSON object with the following schema:
{
  "score_0_100": int,
  "justification": "A 2-3 sentence technical explanation of the score.",
  "strengths": ["list of 1-3 technical strengths"],
  "weaknesses": ["list of 1-3 areas lacking context or complexity"]
}"""


def evaluate_technical_merit(masked_text: str) -> dict:
    """Send *masked_text* to gpt-4o-mini and return a structured evaluation dict."""
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": masked_text},
        ],
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    test_input = (
        "[CANDIDATE_NAME] served as President of the [DEMOGRAPHIC] society "
        "and optimized the MIPS assembly pipeline to reduce execution time by 15%."
    )
    print("INPUT:", test_input)
    print()
    result = evaluate_technical_merit(test_input)
    print(json.dumps(result, indent=2))
