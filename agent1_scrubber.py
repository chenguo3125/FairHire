from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import spacy
from spacy.matcher import PhraseMatcher

from frames import ResumeAchievement

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Declarative WSD Rule Table
#
# Each entry defines how to disambiguate a polysemous demographic term using
# spaCy's dependency graph.  The inference engine reads this table at runtime;
# adding a new ambiguous word requires NO code changes — just a new entry here.
#
# Schema per entry:
#   "term"          – the ambiguous lemma
#   "mask"          – the bracket mask to apply when the demographic sense wins
#   "preserve_if"   – list of graph conditions under which the term is SAFE:
#       "head_in_technical"    – head token's lemma ∈ TECHNICAL_TERMS | SAFE_TERMS
#       "child_in_technical"   – any child's lemma ∈ TECHNICAL_TERMS | SAFE_TERMS
#       "head_in_leadership"   – head token's lemma ∈ LEADERSHIP_TERMS
#       "child_in_leadership"  – any child's lemma ∈ LEADERSHIP_TERMS
#       "head_in_demographic"  – head token's lemma ∈ DEMOGRAPHIC terms (bias context)
#       "child_text_in"        – any child's lowered text is in supplied set
#       "head_text_in"         – head's lowered text is in supplied set
#       "always"               – never mask this term regardless of context
#   "mask_if"        – list of graph conditions that CONFIRM the bias sense:
#       same condition names as preserve_if, but trigger masking
#   "default"        – "mask" | "preserve" — what to do if no rule fires
# ---------------------------------------------------------------------------

WSD_RULES: List[Dict[str, Any]] = [
    # --- Always-preserve ---
    {
        "term": "master",
        "mask": "[TITLE_DEMOGRAPHIC]",
        "preserve_if": ["always"],
        "mask_if": [],
        "default": "preserve",
    },
    # --- Graph + ontology disambiguated ---
    {
        "term": "legacy",
        "mask": "[SOCIO_ECONOMIC_STATUS]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "mask",
    },
    {
        "term": "free",
        "mask": "[SOCIO_ECONOMIC_STATUS]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "preserve",
    },
    {
        "term": "single",
        "mask": "[TITLE_DEMOGRAPHIC]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic", "dep_is_acomp"],
        "default": "preserve",
    },
    {
        "term": "public",
        "mask": "[SOCIO_ECONOMIC_STATUS]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "preserve",
    },
    # --- Syntactic-signal rules ---
    {
        "term": "miss",
        "mask": "[TITLE_DEMOGRAPHIC]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["next_is_propn"],
        "default": "preserve",
    },
    {
        "term": "disabled",
        "mask": "[DISABILITY_NEURODIVERSITY]",
        "preserve_if": ["head_in_safe", "child_in_safe", "is_sent_start"],
        "mask_if": [],
        "default": "preserve",
    },
    # --- Standard graph rules ---
    {
        "term": "poor",
        "mask": "[SOCIO_ECONOMIC_STATUS]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "preserve",
    },
    {
        "term": "rich",
        "mask": "[SOCIO_ECONOMIC_STATUS]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "preserve",
    },
    {
        "term": "blind",
        "mask": "[DISABILITY_NEURODIVERSITY]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "preserve",
    },
    {
        "term": "elite",
        "mask": "[SOCIO_ECONOMIC_STATUS]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "mask",
    },
    {
        "term": "steward",
        "mask": "[TITLE_DEMOGRAPHIC]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["head_in_demographic"],
        "default": "preserve",
    },
    # --- Deep-traversal rule ---
    {
        "term": "veteran",
        "mask": "[VETERAN_MILITARY_AFFILIATION]",
        "preserve_if": ["head_in_safe", "child_in_safe"],
        "mask_if": ["child_in_demographic", "descendant_in_demographic"],
        "default": "preserve",
    },
]

WSD_RULE_INDEX: Dict[str, Dict[str, Any]] = {rule["term"]: rule for rule in WSD_RULES}


# ---------------------------------------------------------------------------
# Ontology loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _iter_terms_from_any(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_terms_from_any(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_terms_from_any(item)


def _extract_terms(payload: Dict[str, Any]) -> Set[str]:
    return {_normalize(s) for s in _iter_terms_from_any(payload) if s and s.strip()}


def _to_category_mask(category: str) -> str:
    return f"[{category}]"


def _normalize_demographics_ontology(demo_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    looks_hierarchical = all(isinstance(v, list) for v in demo_payload.values())
    if looks_hierarchical:
        normalized: Dict[str, List[str]] = {}
        for k, terms in demo_payload.items():
            if not isinstance(terms, list):
                continue
            normalized[str(k)] = [str(t) for t in terms if isinstance(t, str) and t.strip()]
        return normalized

    legacy_map = {
        "gendered_terms": "TITLE_DEMOGRAPHIC",
        "socio_economic_indicators": "SOCIO_ECONOMIC_STATUS",
        "affinity_groups": "AFFINITY_GROUP",
        "gender_and_women_in_stem": "STEM_AFFINITY_GROUP",
        "race_ethnicity_culture": "RACE_ETHNICITY_CULTURE",
        "religion_faith": "RELIGION_FAITH",
        "sexual_orientation_and_gender_identity": "GENDER_IDENTITY_ORIENTATION",
        "disability_and_neurodiversity": "DISABILITY_NEURODIVERSITY",
        "veterans_and_military_affiliations": "VETERAN_MILITARY_AFFILIATION",
    }
    out: Dict[str, List[str]] = {}
    for top_key, value in demo_payload.items():
        category = legacy_map.get(top_key, str(top_key).upper())
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                nested_category = legacy_map.get(nested_key, category)
                terms = [t for t in _iter_terms_from_any(nested_value) if t and t.strip()]
                out.setdefault(nested_category, []).extend(terms)
        else:
            terms = [t for t in _iter_terms_from_any(value) if t and t.strip()]
            out.setdefault(category, []).extend(terms)
    return out


def _build_term_category_map(demo_ontology: Dict[str, List[str]]) -> Dict[str, str]:
    term_to_category: Dict[str, str] = {}
    for category, terms in demo_ontology.items():
        for term in terms:
            key = _normalize(term)
            if key:
                term_to_category[key] = category
    return term_to_category


def _build_phrase_matcher(nlp_model: Any, term_to_category: Dict[str, str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp_model.vocab, attr="LOWER")
    by_category: Dict[str, List[Any]] = {}
    for term, category in term_to_category.items():
        doc = nlp_model.make_doc(term)
        if len(doc) < 2:
            continue
        by_category.setdefault(category, []).append(doc)
    for category, patterns in by_category.items():
        if patterns:
            matcher.add(category, patterns)
    return matcher


# ---------------------------------------------------------------------------
# Stateless helpers (no ontology state required)
# ---------------------------------------------------------------------------

def _token_in_set(token: Any, term_set: Set[str]) -> bool:
    return _normalize(token.text) in term_set or _normalize(token.lemma_) in term_set


def _token_forms(token: Any) -> Set[str]:
    return {_normalize(token.text), _normalize(token.lemma_)}


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


_MASK_PLACEHOLDERS: Dict[str, str] = {
    "[CANDIDATE_NAME]": "Candidate",
    "[STEM_AFFINITY_GROUP]": "",
    "[TITLE_DEMOGRAPHIC]": "They",
    "[AFFINITY_GROUP]": "",
    "[SOCIO_ECONOMIC_STATUS]": "",
    "[RACE_ETHNICITY_CULTURE]": "",
    "[RELIGION_FAITH]": "",
    "[GENDER_IDENTITY_ORIENTATION]": "",
    "[DISABILITY_NEURODIVERSITY]": "",
    "[VETERAN_MILITARY_AFFILIATION]": "",
}

_MASK_PATTERN = re.compile(r"\[[A-Z_]+\]")


def _normalize_masks_for_parsing(text: str) -> str:
    """Replace bracket masks with neutral words so spaCy POS-tags reliably."""
    def _replace(m: re.Match) -> str:
        return _MASK_PLACEHOLDERS.get(m.group(0), "")
    result = _MASK_PATTERN.sub(_replace, text)
    return re.sub(r"  +", " ", result).strip()


# ---------------------------------------------------------------------------
# FairHireScrubber — encapsulates all stateful NLP resources
# ---------------------------------------------------------------------------

class FairHireScrubber:
    """
    Encapsulates the spaCy model, ontology data, and matchers for the
    FairHire Agent 1 scrubbing and framing pipeline.

    Designed for use with ``@st.cache_resource`` in Streamlit.
    """

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

        self.demographics_raw = _load_json(BASE_DIR / "demographics.json")
        self.demographics = _normalize_demographics_ontology(self.demographics_raw)
        self.leadership = _load_json(BASE_DIR / "leadership.json")
        self.technical_domain = _load_json(BASE_DIR / "technical_domain.json")

        self.demographic_term_to_category = _build_term_category_map(self.demographics)
        self.demographic_terms = set(self.demographic_term_to_category.keys())
        self.leadership_titles = {_normalize(t) for t in self.leadership.get("titles", [])}
        self.leadership_actions = {_normalize(v) for v in self.leadership.get("action_verbs", [])}
        self.technical_terms = _extract_terms(self.technical_domain)
        self.leadership_terms = _extract_terms(self.leadership)
        self.safe_terms = self.technical_terms | self.leadership_terms

        self.phrase_matcher = _build_phrase_matcher(
            self.nlp, self.demographic_term_to_category
        )
        self.technical_matcher = _build_phrase_matcher(
            self.nlp, {term: "TECH" for term in self.technical_terms}
        )

    # -------------------------------------------------------------------
    # Graph-based WSD engine (purely ontology-driven)
    # -------------------------------------------------------------------

    def _check_head_with_compounds(
        self, token: Any, term_set: Set[str]
    ) -> bool:
        """
        Check token.head against *term_set*.  If the head itself is not
        in the set, walk its children for ``compound`` dependents — this
        catches compound nouns like "SQL database" where the modifier
        carries the technical signal rather than the syntactic head.
        """
        if _token_in_set(token.head, term_set):
            return True
        return any(
            child.dep_ == "compound" and _token_in_set(child, term_set)
            for child in token.head.children
        )

    @staticmethod
    def _iter_descendants(token: Any) -> Iterable[Any]:
        """Yield every token in the subtree rooted at *token* (excluding itself)."""
        for child in token.children:
            yield child
            yield from FairHireScrubber._iter_descendants(child)

    def _eval_wsd_condition(self, condition: str, token: Any) -> bool:
        """
        Evaluate a single declarative WSD condition against the dependency
        graph rooted at *token*.  Every condition is resolved against the
        loaded ontologies — no hardcoded word lists.

        Supported conditions:
          Graph (ontology-driven):
            always, head_in_technical, child_in_technical,
            head_in_leadership, child_in_leadership,
            head_in_demographic, child_in_demographic,
            head_in_safe, child_in_safe,
            descendant_in_demographic
          Syntactic (POS / dependency):
            next_is_propn, is_sent_start, dep_is_acomp, dep_is_compound
        """
        if condition == "always":
            return True

        # -- Graph: ontology checks on head / children / descendants ------

        if condition == "head_in_technical":
            return self._check_head_with_compounds(token, self.technical_terms)

        if condition == "child_in_technical":
            return any(
                _token_in_set(child, self.technical_terms)
                for child in token.children
            )

        if condition == "head_in_leadership":
            return self._check_head_with_compounds(token, self.leadership_terms)

        if condition == "child_in_leadership":
            return any(
                _token_in_set(child, self.leadership_terms)
                for child in token.children
            )

        if condition == "head_in_demographic":
            return _token_in_set(token.head, self.demographic_terms)

        if condition == "child_in_demographic":
            return any(
                _token_in_set(child, self.demographic_terms)
                for child in token.children
            )

        if condition == "head_in_safe":
            return self._check_head_with_compounds(token, self.safe_terms)

        if condition == "child_in_safe":
            return any(
                _token_in_set(child, self.safe_terms)
                for child in token.children
            )

        if condition == "descendant_in_demographic":
            return any(
                _token_in_set(desc, self.demographic_terms)
                for desc in self._iter_descendants(token)
            )

        # -- Syntactic: POS tags and dependency labels --------------------

        if condition == "next_is_propn":
            if token.i + 1 < len(token.doc):
                return token.nbor(1).pos_ == "PROPN"
            return False

        if condition == "is_sent_start":
            return token.i == 0 or bool(token.is_sent_start)

        if condition == "dep_is_acomp":
            return token.dep_ == "acomp"

        if condition == "dep_is_compound":
            return token.dep_ == "compound"

        return False

    def _wsd_should_mask(self, token: Any) -> Tuple[bool, str | None]:
        """
        Unified WSD dispatcher.  Looks up the token's lemma in
        WSD_RULE_INDEX, then evaluates the declarative conditions against
        the dependency graph.

        Returns (should_mask, mask_string | None).
        """
        lemma = _normalize(token.lemma_)
        rule = WSD_RULE_INDEX.get(lemma) or WSD_RULE_INDEX.get(
            _normalize(token.text)
        )
        if rule is None:
            return False, None

        for cond in rule.get("preserve_if", []):
            if self._eval_wsd_condition(cond, token):
                return False, None

        for cond in rule.get("mask_if", []):
            if self._eval_wsd_condition(cond, token):
                return True, rule["mask"]

        if rule.get("default") == "mask":
            return True, rule["mask"]

        return False, None

    # -------------------------------------------------------------------
    # Core scrubbing pipeline
    # -------------------------------------------------------------------

    def scrub_sentence(self, sentence: str) -> Tuple[str, Dict[str, Any]]:
        """
        Rule-based expert system pipeline:
        1) PhraseMatcher for multi-word ontology phrases.
        2) Token-level: WSD via graph traversal for ambiguous terms,
           then direct ontology category lookup for unambiguous terms.
        3) spaCy NER PERSON masking.
        4) Fluent reconstruction with mask collapsing.
        """
        doc = self.nlp(sentence)
        replacements: Dict[int, Tuple[int, str]] = {}
        token_replacements: Dict[int, str] = {}
        occupied_token_ids: Set[int] = set()
        masked_items: List[Dict[str, Any]] = []

        # Pass 1: multi-word phrase matching via ontology-built PhraseMatcher.
        phrase_matches = sorted(
            self.phrase_matcher(doc), key=lambda m: (m[1], -(m[2] - m[1]))
        )
        for match_id, start, end in phrase_matches:
            if any(i in occupied_token_ids for i in range(start, end)):
                continue
            category = self.nlp.vocab.strings[match_id]
            mask = _to_category_mask(category)
            span = doc[start:end]

            safe_token_ids = set()
            for tok in span:
                if _token_in_set(tok, self.safe_terms):
                    safe_token_ids.add(tok.i)

            if safe_token_ids:
                for tok in span:
                    if tok.i in safe_token_ids or tok.is_punct:
                        continue
                    token_replacements[tok.i] = mask
                    occupied_token_ids.add(tok.i)
                    masked_items.append(
                        {"text": tok.text, "mask": mask, "category": category, "rule": "phrase_partial"}
                    )
            else:
                replacements[start] = (end, mask)
                occupied_token_ids.update(range(start, end))
                masked_items.append(
                    {"text": span.text, "mask": mask, "category": category, "rule": "phrase_full"}
                )

        # Pass 2: token-level — WSD via graph traversal, then ontology lookup.
        for tok in doc:
            if tok.i in occupied_token_ids:
                continue

            low_text = _normalize(tok.text)
            low_lemma = _normalize(tok.lemma_)

            # 2a: if token has a WSD rule, delegate to the graph-based engine.
            if low_lemma in WSD_RULE_INDEX or low_text in WSD_RULE_INDEX:
                should_mask, mask = self._wsd_should_mask(tok)
                if should_mask and mask:
                    token_replacements[tok.i] = mask
                    occupied_token_ids.add(tok.i)
                    rule_entry = WSD_RULE_INDEX.get(low_lemma) or WSD_RULE_INDEX[low_text]
                    masked_items.append(
                        {"text": tok.text, "mask": mask, "category": rule_entry["term"], "rule": "wsd_graph"}
                    )
                continue

            # 2b: direct ontology category lookup (unambiguous terms).
            # Safe-terms override: if the token is also a known technical/leadership
            # term, preserve it (same principle as the NER override in Pass 3).
            if _token_in_set(tok, self.safe_terms):
                continue
            category = (
                self.demographic_term_to_category.get(low_text)
                or self.demographic_term_to_category.get(low_lemma)
            )
            if category:
                mask = _to_category_mask(category)
                token_replacements[tok.i] = mask
                occupied_token_ids.add(tok.i)
                masked_items.append(
                    {"text": tok.text, "mask": mask, "category": category, "rule": "token_lookup"}
                )

        # Pass 3: PERSON NER masking.
        # Ontology overrides NER: if spaCy thinks a technical term is a person
        # (e.g. "Docker", "Redis"), our knowledge base takes priority.
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            if any(i in occupied_token_ids for i in range(ent.start, ent.end)):
                continue
            ent_is_safe = any(
                _token_in_set(doc[i], self.safe_terms)
                for i in range(ent.start, ent.end)
            )
            if ent_is_safe:
                continue
            replacements[ent.start] = (ent.end, "[CANDIDATE_NAME]")
            occupied_token_ids.update(range(ent.start, ent.end))
            masked_items.append(
                {"text": ent.text, "mask": "[CANDIDATE_NAME]", "category": "PERSON", "rule": "ner_person"}
            )

        # Reconstruct preserving original whitespace/punctuation.
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
        masked = re.sub(r"(\[[A-Z_/]+\])(?:\s+\1)+", r"\1", masked)
        payload = {
            "original_text": sentence,
            "masked_text": masked,
            "masked_items": masked_items,
        }
        return masked, payload

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def scrub_and_mask(self, sentence: str) -> str:
        masked, _ = self.scrub_sentence(sentence)
        return masked

    def scrub_resume_document(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scrub a full resume: multiple sentences and paragraphs in one string.

        Implementation is the same as ``scrub_sentence``; the name makes
        document-level use explicit for Agent 3 and evaluation pipelines.
        """
        return self.scrub_sentence(text)

    def frame_sentence(self, sentence: str) -> ResumeAchievement:
        """
        Extract a ResumeAchievement frame from text WITHOUT scrubbing.
        Used by Agent 2 when scoring raw (baseline) text.
        """
        parseable = _normalize_masks_for_parsing(sentence)
        doc = self.nlp(parseable)

        action_verb = None
        role = None
        for tok in doc:
            forms = _token_forms(tok)
            if action_verb is None and tok.pos_ in {"VERB", "AUX"}:
                if any(form in self.leadership_actions for form in forms):
                    action_verb = tok.lemma_.capitalize()
            if role is None and tok.pos_ in {"NOUN", "PROPN"}:
                if any(form in self.leadership_titles for form in forms):
                    role = tok.text

        if action_verb is None:
            for tok in doc:
                if tok.pos_ in {"VERB", "AUX"}:
                    action_verb = tok.lemma_.capitalize()
                    break

        if role is None:
            for tok in doc:
                if tok.pos_ in {"NOUN", "PROPN"} and not tok.text.startswith("["):
                    role = tok.text
                    break

        # Technical skills via PhraseMatcher (multi-word) + token lookup (single-word).
        technical_skills: List[str] = []
        occupied: Set[int] = set()

        matches = sorted(
            self.technical_matcher(doc), key=lambda m: (m[1], -(m[2] - m[1]))
        )
        for _match_id, start, end in matches:
            if any(i in occupied for i in range(start, end)):
                continue
            span_tokens = [doc[i] for i in range(start, end)]
            if any(
                tok.text.startswith("[") and tok.text.endswith("]")
                for tok in span_tokens
            ):
                continue
            technical_skills.append(" ".join(tok.text for tok in span_tokens))
            occupied.update(range(start, end))

        for tok in doc:
            if tok.i in occupied:
                continue
            if tok.text.startswith("[") and tok.text.endswith("]"):
                continue
            if any(f in self.technical_terms for f in _token_forms(tok)):
                technical_skills.append(tok.text)
                occupied.add(tok.i)

        if not technical_skills and "[STEM_AFFINITY_GROUP]" in sentence:
            technical_skills.append("STEM engagement")

        impact_metric = _extract_impact_metric(sentence)
        return ResumeAchievement(
            action_verb=action_verb,
            role=role,
            technical_skills=technical_skills,
            impact_metric=impact_metric,
        )

    def scrub_and_frame(self, sentence: str) -> ResumeAchievement:
        masked, _ = self.scrub_sentence(sentence)
        return self.frame_sentence(masked)


# ---------------------------------------------------------------------------
# Backward-compatible module-level API
#
# External modules (agent2_grader, agent3_auditor, pipeline, test_fairhire)
# import names like ``NLP``, ``DEMOGRAPHICS``, ``scrub_sentence``, etc.
# A default singleton satisfies those imports without any caller changes.
# ---------------------------------------------------------------------------

_default_scrubber = FairHireScrubber()

NLP = _default_scrubber.nlp
DEMOGRAPHICS = _default_scrubber.demographics


def scrub_sentence(sentence: str) -> Tuple[str, Dict[str, Any]]:
    return _default_scrubber.scrub_sentence(sentence)


def scrub_and_mask(sentence: str) -> str:
    return _default_scrubber.scrub_and_mask(sentence)


def scrub_resume_document(text: str) -> Tuple[str, Dict[str, Any]]:
    return _default_scrubber.scrub_resume_document(text)


def frame_sentence(sentence: str) -> ResumeAchievement:
    return _default_scrubber.frame_sentence(sentence)


def scrub_and_frame(sentence: str) -> ResumeAchievement:
    return _default_scrubber.scrub_and_frame(sentence)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scrubber = FairHireScrubber()
    test_sentences = [
        # WSD: "legacy" in technical context -> preserve
        "I optimized the legacy Master database architecture.",
        # WSD: "legacy" in academic context -> mask
        "As a legacy student, I was admitted to the university.",
        # WSD: "legacy" in technical context -> preserve
        "I migrated the legacy backend system to a microservice architecture.",
        # WSD: "master" -> always preserve (policy)
        "I completed my Master degree in Computer Science.",
        # Phrase match + safe-token preservation
        "I received a scholarship from the Society of Women Engineers.",
        # Mixed: demographic masking + technical preservation
        "As a first-generation college student, I built REST APIs with PostgreSQL and Docker.",
        # WSD: "free" in technical context -> preserve
        "The module provides a lock-free concurrent hash map implementation.",
        # WSD: "single" in technical context -> preserve
        "Designed a single point of failure detection system using Redis.",
        # Compound noun: "SQL" is compound modifier of "database" -> preserve "legacy"
        "I optimized the legacy SQL database.",
    ]

    for sentence in test_sentences:
        masked, payload = scrubber.scrub_sentence(sentence)
        print(f"INPUT:   {sentence}")
        print(f"MASKED:  {masked}")
        print("PAYLOAD:")
        print(json.dumps(payload, indent=2))
        print("-" * 80)
