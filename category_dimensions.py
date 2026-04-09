"""
Single source of truth: map Agent 1 ontology `category` strings (from scrub payloads)
to coarse **fairness dimensions** used by Agent 2 bias reporting.

Agent 1 categories come from:
  - `demographics.json` after `_normalize_demographics_ontology` (keys like RACE_ETHNICITY_CULTURE)
  - WSD rule lemmas in masked_items (`master`, `legacy`, …)
  - NER: PERSON

When you add a new demographic category in `demographics.json`, add a row here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# Ontology tag (as in scrub `masked_items[].category`) -> audit bucket name.
CATEGORY_TO_FAIRNESS_DIMENSION: Dict[str, str] = {
    # Core ontology tags (see agent1_scrubber._normalize_demographics_ontology legacy_map)
    "RACE_ETHNICITY_CULTURE": "race_ethnicity",
    "SOCIO_ECONOMIC_STATUS": "socioeconomic",
    "STEM_AFFINITY_GROUP": "gender_affinity",
    "AFFINITY_GROUP": "gender_affinity",
    "TITLE_DEMOGRAPHIC": "gender_affinity",
    "RELIGION_FAITH": "religion",
    "GENDER_IDENTITY_ORIENTATION": "gender_affinity",
    "DISABILITY_NEURODIVERSITY": "disability",
    "VETERAN_MILITARY_AFFILIATION": "veteran",
    # NER
    "PERSON": "person_name",
    # WSD graph rules store the ambiguous lemma as category (agent1 scrub_sentence)
    "master": "ambiguous_demographic_wsd",
    "legacy": "ambiguous_demographic_wsd",
    "free": "ambiguous_demographic_wsd",
    "single": "ambiguous_demographic_wsd",
    "public": "ambiguous_demographic_wsd",
}

# Dimension that receives an extra penalty weight in Agent 2 (must match value above).
RACE_ETHNICITY_DIMENSION = "race_ethnicity"


def fairness_dimension_for_category(category: Any) -> str:
    """
    Map a scrub payload `category` field to a coarse fairness dimension.

    Unknown categories fall back to ``\"other\"`` so new ontology tags fail soft
    until this table is updated.
    """
    if category is None:
        return "unknown"
    key = str(category).strip()
    return CATEGORY_TO_FAIRNESS_DIMENSION.get(key, "other")


def missing_mappings_for_ontology_keys(ontology_category_keys: Iterable[str]) -> List[str]:
    """Return demographic category keys that have no row in CATEGORY_TO_FAIRNESS_DIMENSION."""
    keys = {str(k).strip() for k in ontology_category_keys if k is not None}
    return sorted(k for k in keys if k not in CATEGORY_TO_FAIRNESS_DIMENSION)
