from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class ResumeAchievement:
    """
    Frame-Based Representation: ResumeAchievement

    This class is a classical AI *frame* used for deterministic knowledge representation
    of a single resume “achievement” clause/sentence after parsing.

    Key properties:
    - GOFAI-first: This structure is intentionally symbolic and does not rely on
      machine learning, embeddings, or probabilistic inference.
    - Ontology-aligned: Slots are designed to map directly to explicit vocabularies
      (e.g., leadership action verbs/titles and technical-domain terms).
    - Deterministic reconstruction: `to_neutral_string()` converts the frame back into
      a grammatically-correct, anonymized sentence that preserves technical meaning
      while omitting demographic or socio-economic identifiers.
    """

    action_verb: Optional[str] = None
    role: Optional[str] = None
    technical_skills: List[str] = field(default_factory=list)
    impact_metric: Optional[str] = None

    def to_neutral_string(self) -> str:
        """
        Reconstruct this frame into a neutral, anonymized sentence.

        The output intentionally avoids:
        - personal identifiers (names, pronouns, affiliations)
        - demographic or socio-economic descriptors

        And intentionally preserves:
        - leadership/role context (when provided)
        - technical skills and domains
        - measurable impact (percentages, latency, complexity, throughput, etc.)
        """

        verb = (self.action_verb or "Contributed").strip()
        role = (self.role or "a role").strip()

        skills = [s.strip() for s in self.technical_skills if s and s.strip()]
        skills_phrase = ""
        if skills:
            if len(skills) == 1:
                skills_phrase = f" and utilized {skills[0]}"
            elif len(skills) == 2:
                skills_phrase = f" and utilized {skills[0]} and {skills[1]}"
            else:
                skills_phrase = " and utilized " + ", ".join(skills[:-1]) + f", and {skills[-1]}"

        impact = (self.impact_metric or "").strip()
        impact_phrase = f" to achieve {impact}" if impact else ""

        return f"{verb} as {role}{skills_phrase}{impact_phrase}."
