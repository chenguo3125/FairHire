from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import spacy

from frames import ResumeAchievement

BASE_DIR = Path(__file__).resolve().parent

MASK_BY_DEMOGRAPHIC_SECTION = {
    "gendered_terms": "[DEMOGRAPHIC]",
    "socio_economic_indicators": "[SOCIO_ECONOMIC_STATUS]",
    "affinity_groups": "[AFFINITY_GROUP]",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)


def _build_demographic_term_map(demo_payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Build explicit term -> mask mapping from demographics ontology.
    """
    term_to_mask: Dict[str, str] = {}
    for top_section, raw in demo_payload.items():
        mask = MASK_BY_DEMOGRAPHIC_SECTION.get(top_section, "[DEMOGRAPHIC]")
        for term in _iter_strings(raw):
            normalized = _normalize(term)
            if normalized:
                term_to_mask[normalized] = mask
    return term_to_mask


def _extract_terms(payload: Dict[str, Any]) -> Set[str]:
    return {_normalize(s) for s in _iter_strings(payload) if s and s.strip()}


NLP = spacy.load("en_core_web_sm")
DEMOGRAPHICS = _load_json(BASE_DIR / "demographics.json")
LEADERSHIP = _load_json(BASE_DIR / "leadership.json")
TECHNICAL_DOMAIN = _load_json(BASE_DIR / "technical_domain.json")
DEMOGRAPHIC_TERM_TO_MASK = _build_demographic_term_map(DEMOGRAPHICS)
MAX_DEMOGRAPHIC_PHRASE_LEN = max(len(t.split()) for t in DEMOGRAPHIC_TERM_TO_MASK)
LEADERSHIP_TITLES = {_normalize(t) for t in LEADERSHIP.get("titles", [])}
LEADERSHIP_ACTIONS = {_normalize(v) for v in LEADERSHIP.get("action_verbs", [])}
TECHNICAL_TERMS = _extract_terms(TECHNICAL_DOMAIN)
LEADERSHIP_TERMS = _extract_terms(LEADERSHIP)
SAFE_TERMS = TECHNICAL_TERMS | LEADERSHIP_TERMS


def _phrase_forms(tokens: List[Any]) -> Tuple[str, str]:
    text_form = _normalize(" ".join(tok.text for tok in tokens))
    lemma_form = _normalize(" ".join(tok.lemma_ for tok in tokens))
    return text_form, lemma_form


def mask_sentence(sentence: str) -> str:
    """
    Replace demographic tokens/phrases with categorical masks while preserving fluency.

    Rules:
    - If token or multi-word phrase appears in demographics ontology, replace with
      category mask (do not delete surrounding grammar/punctuation).
    - If spaCy NER tags PERSON, replace the full entity span with [CANDIDATE_NAME]
      unless already masked by ontology phrase matching.
    """
    doc = NLP(sentence)
    replacements: Dict[int, Tuple[int, str]] = {}
    token_replacements: Dict[int, str] = {}
    occupied_token_ids = set()

    # Pass 1: Longest demographic phrase matching.
    for window in range(min(MAX_DEMOGRAPHIC_PHRASE_LEN, len(doc)), 0, -1):
        for start in range(0, len(doc) - window + 1):
            end = start + window
            if any(i in occupied_token_ids for i in range(start, end)):
                continue

            span_tokens = [doc[i] for i in range(start, end)]
            text_form, lemma_form = _phrase_forms(span_tokens)
            mask = DEMOGRAPHIC_TERM_TO_MASK.get(text_form) or DEMOGRAPHIC_TERM_TO_MASK.get(lemma_form)
            if not mask:
                continue

            # Preserve technical/leadership terms inside affinity/demographic phrases
            # (e.g., "women who code" keeps "code"; "women engineers" keeps "engineers").
            safe_token_ids = set()
            for tok in span_tokens:
                tok_forms = {_normalize(tok.text), _normalize(tok.lemma_)}
                if any(form in SAFE_TERMS for form in tok_forms):
                    safe_token_ids.add(tok.i)

            if safe_token_ids:
                for tok in span_tokens:
                    if tok.i in safe_token_ids or tok.is_punct:
                        continue
                    token_replacements[tok.i] = mask
                    occupied_token_ids.add(tok.i)
            else:
                replacements[start] = (end, mask)
                occupied_token_ids.update(range(start, end))

    # Pass 2: PERSON entity masking (bonus), as long as span is not already occupied.
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        if any(i in occupied_token_ids for i in range(ent.start, ent.end)):
            continue
        replacements[ent.start] = (ent.end, "[CANDIDATE_NAME]")
        occupied_token_ids.update(range(ent.start, ent.end))

    # Reconstruct with original whitespace and punctuation preserved.
    parts: List[str] = []
    i = 0
    while i < len(doc):
        if i in token_replacements:
            parts.append(token_replacements[i] + doc[i].whitespace_)
            i += 1
        elif i in replacements:
            end, mask = replacements[i]
            trailing_ws = doc[end - 1].whitespace_ if end > 0 else ""
            parts.append(mask + trailing_ws)
            i = end
        else:
            parts.append(doc[i].text_with_ws)
            i += 1

    masked = "".join(parts)
    # Improve fluency by collapsing repeated adjacent masks:
    # "[AFFINITY_GROUP] [AFFINITY_GROUP] code" -> "[AFFINITY_GROUP] code"
    masked = re.sub(r"(\[[A-Z_]+\])(?:\s+\1)+", r"\1", masked)
    return masked


def _token_forms(token: Any) -> Set[str]:
    return {_normalize(token.text), _normalize(token.lemma_)}


def _span_forms(tokens: List[Any]) -> Set[str]:
    return {
        _normalize(" ".join(tok.text for tok in tokens)),
        _normalize(" ".join(tok.lemma_ for tok in tokens)),
    }


def _extract_impact_metric(sentence: str) -> str | None:
    patterns = [
        r"\bby\s+\d+(\.\d+)?\s*%\b",
        r"\b\d+(\.\d+)?\s*%\b",
        r"\bO\([^)]+\)\b",
        r"\b\d+(\.\d+)?\s*(ms|s|sec|seconds|minutes|min|hrs|hours)\b",
        r"\b\d+(\.\d+)?x\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def scrub_and_mask(sentence: str) -> str:
    """
    Primary Phase-2 API: returns fluent masked text.
    """
    return mask_sentence(sentence)


def scrub_and_frame(sentence: str) -> ResumeAchievement:
    """
    Backward-compatible API: returns ResumeAchievement so existing
    call-sites using `.to_neutral_string()` still work.
    """
    masked = mask_sentence(sentence)
    doc = NLP(masked)

    action_verb = None
    role = None
    for tok in doc:
        forms = _token_forms(tok)
        if action_verb is None and tok.pos_ == "VERB":
            if any(form in LEADERSHIP_ACTIONS for form in forms):
                action_verb = tok.lemma_.capitalize()
        if role is None and tok.pos_ in {"NOUN", "PROPN"}:
            if any(form in LEADERSHIP_TITLES for form in forms):
                role = tok.text

    if action_verb is None:
        for tok in doc:
            if tok.pos_ == "VERB":
                action_verb = tok.lemma_.capitalize()
                break

    if role is None:
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"} and not tok.text.startswith("["):
                role = tok.text
                break

    technical_skills: List[str] = []
    used_ranges: List[Tuple[int, int]] = []
    for window in range(5, 0, -1):
        for start in range(0, len(doc) - window + 1):
            end = start + window
            if any(not (start >= r_end or end <= r_start) for r_start, r_end in used_ranges):
                continue
            span_tokens = [doc[i] for i in range(start, end)]
            if any(tok.text.startswith("[") and tok.text.endswith("]") for tok in span_tokens):
                continue
            if any(sf in TECHNICAL_TERMS for sf in _span_forms(span_tokens)):
                technical_skills.append(" ".join(tok.text for tok in span_tokens))
                used_ranges.append((start, end))

    impact_metric = _extract_impact_metric(masked)
    return ResumeAchievement(
        action_verb=action_verb,
        role=role,
        technical_skills=technical_skills,
        impact_metric=impact_metric,
    )


if __name__ == "__main__":
    test_sentences = [
        "Sarah served as President of the Women in Computing society.",
        "Optimized the MIPS assembly pipeline to reduce execution time by 15%.",
        "As a Chinese scholar who has been in SG for 6 years, I managed the backend development using C++ and PostgreSQL.",
        "Led a diverse team of developers to build a REST API in Docker.",
    ]

    for sentence in test_sentences:
        frame = scrub_and_frame(sentence)
        print(f"INPUT:   {sentence}")
        print(f"MASKED:  {scrub_and_mask(sentence)}")
        print(f"NEUTRAL: {frame.to_neutral_string()}")
        print("-" * 80)