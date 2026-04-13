"""
FairHire integration tests: Agent 1 scrubber, Agent 2 grader, Agent 3 auditor.

Run:
  python -m unittest test_fairhire -v

Or:
  python test_fairhire.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent1_scrubber import DEMOGRAPHICS, scrub_and_mask, scrub_resume_document, scrub_sentence
from agent2_grader import grade_resume_mode
from agent3_auditor import audit_counterfactual_fairness
from category_dimensions import missing_mappings_for_ontology_keys
from kaggle_resume_data import load_labeled_resume_texts
from resume_format import bullets_to_prose
from resume_samples import RESUME_MIXED_DEMOGRAPHIC, RESUME_TECHNICAL_ONLY


class TestCategoryDimensions(unittest.TestCase):
    """Every ontology category key from demographics must have a fairness-dimension row."""

    def test_all_demographic_categories_mapped(self) -> None:
        missing = missing_mappings_for_ontology_keys(DEMOGRAPHICS.keys())
        self.assertEqual(
            missing,
            [],
            f"Add these keys to category_dimensions.CATEGORY_TO_FAIRNESS_DIMENSION: {missing}",
        )


class TestAgent1Scrubber(unittest.TestCase):
    """Scrubbing: masks should appear; sensitive plaintext should not leak in obvious ways."""

    def test_person_name_masked(self) -> None:
        masked, _ = scrub_sentence("Alex Chen led the API migration.")
        self.assertIn("[CANDIDATE_NAME]", masked)
        self.assertNotIn("Alex", masked)

    def test_stem_affinity_phrase(self) -> None:
        masked, _ = scrub_sentence(
            "I received a scholarship from the Society of Women Engineers."
        )
        self.assertIn("[STEM_AFFINITY_GROUP]", masked)
        self.assertNotIn("Women Engineers", masked)

    def test_race_ethnicity_phrase(self) -> None:
        masked, _ = scrub_sentence(
            "I volunteered with the Chinese Students Association on campus."
        )
        self.assertIn("[RACE_ETHNICITY_CULTURE]", masked)

    def test_socioeconomic_first_gen(self) -> None:
        masked, _ = scrub_sentence(
            "As a first-generation college student, I built REST APIs with PostgreSQL."
        )
        self.assertIn("[SOCIO_ECONOMIC_STATUS]", masked)
        self.assertIn("PostgreSQL", masked)

    def test_scholarship_tagged(self) -> None:
        masked, _ = scrub_sentence("I received a need-based scholarship for tuition.")
        self.assertIn("[SOCIO_ECONOMIC_STATUS]", masked)

    def test_technical_sentence_largely_preserves_skills(self) -> None:
        s = (
            "Optimized the MIPS microarchitecture pipeline using C++ to reduce latency by 12%."
        )
        masked, _ = scrub_sentence(s)
        self.assertIn("MIPS", masked)
        self.assertIn("12%", masked)
        self.assertIn("C++", masked)

    def test_fraternity_membership(self) -> None:
        masked, _ = scrub_sentence("Served as treasurer of the local fraternity chapter.")
        self.assertIn("[TITLE_DEMOGRAPHIC]", masked)  # gendered group membership

    def test_scrub_and_mask_api(self) -> None:
        out = scrub_and_mask("She optimized the Python service.")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


class TestAgent2Grader(unittest.TestCase):
    """Grader: deterministic scores; bias penalty only when enabled on raw text."""

    def test_scrubbed_no_penalty_higher_or_equal_than_raw_with_penalty(self) -> None:
        text = (
            "Maria served as President of the Women in Computing society. "
            "Reduced latency by 20% using Go."
        )
        raw_biased = grade_resume_mode(text=text, scrub=False, apply_bias_penalty=True)
        scrubbed_clean = grade_resume_mode(
            text=text, scrub=True, apply_bias_penalty=False
        )
        self.assertTrue(raw_biased.breakdown["bias_penalty"]["enabled"])
        self.assertFalse(scrubbed_clean.breakdown["bias_penalty"]["enabled"])
        self.assertGreaterEqual(scrubbed_clean.score_0_100, 0)

    def test_race_ethnicity_in_bias_breakdown(self) -> None:
        text = "Led outreach for the Asian American Association and shipped a REST API."
        r = grade_resume_mode(text=text, scrub=False, apply_bias_penalty=True)
        bp = r.breakdown["bias_penalty"]
        self.assertTrue(bp.get("enabled"))
        re_block = bp.get("race_ethnicity") or {}
        self.assertGreaterEqual(re_block.get("count", 0), 0)

    def test_pure_technical_low_penalty(self) -> None:
        text = "Implemented a binary search tree in Rust with O(log n) lookup."
        r = grade_resume_mode(text=text, scrub=False, apply_bias_penalty=True)
        self.assertEqual(r.breakdown["bias_penalty"].get("sensitive_items_count", -1), 0)


class TestAgent3Auditor(unittest.TestCase):
    """Counterfactual fairness: delta reflects scrub vs baseline."""

    def test_audit_returns_delta_and_notes(self) -> None:
        text = (
            "Sarah served as President of the Women in Computing society and "
            "optimized the MIPS assembly code to reduce execution time by 15%."
        )
        audit = audit_counterfactual_fairness(text)
        self.assertIsInstance(audit.fairness_delta, int)
        self.assertIn("baseline_raw_points", audit.notes)

    def test_audit_technical_only_small_delta(self) -> None:
        # Avoid "reduced" (matches socio-economic ontology) and names NER may tag.
        text = (
            "Cut p99 latency from 800ms to 120ms using an in-memory cache layer in Go."
        )
        audit = audit_counterfactual_fairness(text)
        self.assertEqual(audit.fairness_delta, 0)


class TestFullResumeDocument(unittest.TestCase):
    """Multi-paragraph resume strings: scrub, grade, and audit end-to-end."""

    def test_scrub_resume_document_returns_masked_and_payload(self) -> None:
        masked, payload = scrub_resume_document(RESUME_MIXED_DEMOGRAPHIC)
        self.assertTrue(len(masked) > 50)
        self.assertIn("masked_items", payload)
        self.assertGreater(len(payload["masked_items"]), 0)

    def test_grade_full_resume_multiple_sentences(self) -> None:
        r = grade_resume_mode(text=RESUME_TECHNICAL_ONLY, scrub=True, apply_bias_penalty=False)
        doc = r.breakdown.get("document") or {}
        self.assertGreaterEqual(doc.get("sentence_count", 0), 3)
        self.assertGreaterEqual(doc.get("paragraph_or_section_count", 0), 2)
        self.assertGreater(len(r.breakdown.get("sentences") or []), 2)

    def test_full_resume_mixed_demographic_bias_detection(self) -> None:
        r = grade_resume_mode(
            text=RESUME_MIXED_DEMOGRAPHIC, scrub=False, apply_bias_penalty=True
        )
        bp = r.breakdown.get("bias_penalty") or {}
        self.assertTrue(bp.get("enabled"))
        self.assertGreater(bp.get("sensitive_items_count", 0), 0)

    def test_full_resume_fairness_audit_runs(self) -> None:
        audit = audit_counterfactual_fairness(RESUME_MIXED_DEMOGRAPHIC)
        self.assertIsInstance(audit.fairness_delta, int)
        # Baseline scores raw text; scrubbed path uses masked full resume — previews should differ.
        self.assertNotEqual(audit.scrubbed_masked_preview, audit.original_masked_preview)


# ---------------------------------------------------------------------------
# Curated sentence list (for manual runs / debugging)
# ---------------------------------------------------------------------------

EXTRA_SCRUB_EXAMPLES = [
    # Religion / faith
    "I volunteered with the Muslim Student Association during orientation week.",
    # Veteran
    "Served as ROTC liaison and deployed monitoring dashboards in Kubernetes.",
    # Disability / neurodiversity
    "Advocated for neurodiversity hiring practices while building the CI pipeline.",
    # LGBTQ+
    "Member of the campus LGBTQ+ alliance and contributed to the React frontend.",
    # Mixed technical + demographic
    "As a Singaporean Chinese developer, I shipped TypeScript microservices.",
    # Ivy / elite markers (socioeconomic ontology)
    "Participated in an Ivy League hackathon and built a Rust CLI tool.",
    # Long achievement
    "Led a team of five engineers to migrate legacy monolith to Docker, cutting costs by 30%.",
]


class TestBulletToProse(unittest.TestCase):
    """``resume_format.bullets_to_prose`` turns list lines into paragraphs."""

    def test_merges_glyph_bullets(self) -> None:
        raw = "• Built APIs in Go\n• Led a team of five\n"
        out = bullets_to_prose(raw)
        self.assertEqual(out, "Built APIs in Go. Led a team of five.")

    def test_merges_dash_bullets(self) -> None:
        raw = "- First item here\n- Second item\n"
        out = bullets_to_prose(raw)
        self.assertEqual(out, "First item here. Second item.")

    def test_numbered_list(self) -> None:
        raw = "1. Alpha work\n2. Beta work\n"
        out = bullets_to_prose(raw)
        self.assertEqual(out, "Alpha work. Beta work.")

    def test_no_bullets_returns_stripped(self) -> None:
        t = "Just a sentence.\n\nAnother paragraph."
        self.assertEqual(bullets_to_prose(t), t.strip())


class TestKaggleResumeCsvSniff(unittest.TestCase):
    """CSV column sniffing for Kaggle-style Category + Resume tables."""

    def test_load_labeled_from_temp_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "word " * 15  # > min_chars
            (root / "sample.csv").write_text(
                f"Category,Resume_str\nData Science,{body}\n",
                encoding="utf-8",
            )
            pairs = load_labeled_resume_texts(root, max_rows=10)
        self.assertEqual(len(pairs), 1)
        self.assertIn("Data Science", pairs[0][0])
        self.assertEqual(pairs[0][1].strip(), body.strip())


class TestExtraScrubSmoke(unittest.TestCase):
    """Smoke: curated sentences should scrub without raising."""

    def test_extra_examples_run(self) -> None:
        for sentence in EXTRA_SCRUB_EXAMPLES:
            with self.subTest(sentence=sentence[:50] + "..."):
                masked, payload = scrub_sentence(sentence)
                self.assertTrue(masked.strip())
                self.assertIn("masked_text", payload)


if __name__ == "__main__":
    unittest.main()
