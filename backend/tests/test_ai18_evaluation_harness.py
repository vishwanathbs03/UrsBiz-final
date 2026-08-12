"""SPRINT AI-18 — Universal AI Evaluation Harness.

End-to-end test suite for the evaluation harness.

The harness has 7 fixture modules + a runner + a metrics
calculator. The tests exercise:

  1. Question bank structure (PART 1)
  2. Golden set structure (PART 7)
  3. Follow-up scripts structure (PART 2)
  4. Adversarial cases structure (PART 3)
  5. Provider failure scenarios structure (PART 4)
  6. Data quality profiles structure (PART 5)
  7. Runner + production path (PART 8)
  8. Metrics calculator (PART 6)

The runner is intentionally fixture-based: it does NOT touch
the DB. The metrics calculator is pure. The fixtures are
build-only.

Every assertion measures behaviour. The suite NEVER claims
"100% accuracy" — only the measured value.
"""
from __future__ import annotations

import pytest

from app.services.ai.evaluation import (
    AdversarialCase,
    AdversarialKind,
    DataQualityProfile,
    EvaluationResult,
    EvaluationRunner,
    FollowUpScript,
    GoldenCase,
    MetricsCalculator,
    MetricsReport,
    QuestionEntry,
    all_adversarial_cases,
    all_failure_scenarios,
    all_golden_cases,
    all_profiles,
    all_questions,
    all_scripts,
    category_coverage,
    category_vocabulary,
    failure_kinds,
    get_case,
    get_profile,
    get_script,
    questions_by_category,
)
from app.services.ai.evaluation.runner import (
    _BUSINESS_TOOLS,
    _GENERAL_TOOLS,
)


# --------------------------------------------------------------------------- #
# 1 — Question bank
# --------------------------------------------------------------------------- #


class TestQuestionBank:
    """PART 1: ≥100 prompts across ≥16 categories."""

    def test_total_prompts_meet_brief(self):
        entries = all_questions()
        assert len(entries) >= 100, (
            f"question bank has {len(entries)} prompts; brief requires >=100"
        )

    def test_categories_meet_brief(self):
        cats = category_vocabulary()
        assert len(cats) >= 16, (
            f"category vocabulary has {len(cats)} entries; brief requires >=16"
        )

    def test_each_category_has_prompts(self):
        cats = category_vocabulary()
        for cat in cats:
            in_cat = questions_by_category(cat)
            assert in_cat, f"category {cat!r} has zero prompts"
            assert all(isinstance(q, QuestionEntry) for q in in_cat)
            assert all(q.category == cat for q in in_cat)

    def test_category_coverage_returns_counts(self):
        coverage = category_coverage()
        assert isinstance(coverage, dict)
        cats = category_vocabulary()
        assert set(coverage.keys()) == set(cats)
        # Every category has >= 1 prompt.
        for cat, count in coverage.items():
            assert count >= 1, f"category {cat!r} has zero prompts"
        # Brief requires the bank to total >= 100.
        assert sum(coverage.values()) >= 100

    def test_no_duplicate_prompts(self):
        prompts = [q.prompt for q in all_questions()]
        dups = {p for p in prompts if prompts.count(p) > 1}
        assert not dups, f"duplicate prompts detected: {sorted(dups)[:3]}"

    def test_prompts_non_empty(self):
        for q in all_questions():
            assert q.prompt.strip(), f"empty prompt in category {q.category}"


# --------------------------------------------------------------------------- #
# 2 — Golden set
# --------------------------------------------------------------------------- #


class TestGoldenSet:
    """PART 7: 11 immutable cases covering the brief's capabilities."""

    def test_golden_set_count(self):
        cases = all_golden_cases()
        assert len(cases) == 11, f"golden set has {len(cases)} cases; expected 11"

    def test_each_golden_case_is_well_formed(self):
        for c in all_golden_cases():
            assert isinstance(c, GoldenCase)
            assert c.case_id, "case_id required"
            assert c.prompt.strip(), "prompt required"
            assert c.category, "category required"

    def test_golden_set_spans_all_brief_categories(self):
        cats = {c.category for c in all_golden_cases()}
        # The 11 golden cases cover these 11 brief categories.
        # (Names follow the question-bank / evaluation
        # vocabulary.)
        required = {
            "general_knowledge",
            "business_fact",
            "calculation",
            "risk",
            "recommendation",
            "scenario",
            "comparison",
            "government_scheme",
            "export",
            "financial",  # the missing-data case lives here
            "mixed",
        }
        missing = required - cats
        assert not missing, f"golden set missing categories: {missing}"
        # And the golden set has 11 cases total.
        assert len(all_golden_cases()) == 11

    def test_get_case_round_trip(self):
        for c in all_golden_cases():
            assert get_case(c.case_id) is c


# --------------------------------------------------------------------------- #
# 3 — Follow-up scripts
# --------------------------------------------------------------------------- #


class TestFollowUpScripts:
    """PART 2: multi-turn scripts with context retention."""

    def test_scripts_non_empty(self):
        scripts = all_scripts()
        assert len(scripts) >= 3, (
            f"only {len(scripts)} scripts; brief requires multi-turn coverage"
        )

    def test_each_script_has_turns(self):
        for s in all_scripts():
            assert isinstance(s, FollowUpScript)
            assert s.script_id, "script_id required"
            assert s.turns, f"script {s.script_id} has zero turns"
            assert len(s.turns) >= 2, (
                f"script {s.script_id} has only {len(s.turns)} turns; "
                "follow-up scripts must be multi-turn"
            )

    def test_each_turn_has_user_message(self):
        for s in all_scripts():
            for t in s.turns:
                assert t.user.strip(), (
                    f"empty user turn in script {s.script_id}"
                )

    def test_get_script_round_trip(self):
        for s in all_scripts():
            assert get_script(s.script_id) is s


# --------------------------------------------------------------------------- #
# 4 — Adversarial cases
# --------------------------------------------------------------------------- #


class TestAdversarialCases:
    """PART 3: ≥10 adversarial fixtures preserving server authority."""

    def test_adversarial_count_meets_brief(self):
        cases = all_adversarial_cases()
        assert len(cases) >= 10, (
            f"adversarial fixtures: {len(cases)}; brief requires >=10"
        )

    def test_adversarial_kind_vocabulary(self):
        kinds = {c.kind for c in all_adversarial_cases()}
        required = {
            AdversarialKind.PROMPT_INJECTION,
            AdversarialKind.FAKE_EVIDENCE_ID,
            AdversarialKind.USER_FALSE_FACT,
            AdversarialKind.CONFLICTING_NUMBERS,
            AdversarialKind.IMPOSSIBLE_REQUEST,
            AdversarialKind.UNSUPPORTED_GUARANTEE,
            AdversarialKind.MALICIOUS_INSTRUCTION,
            AdversarialKind.EVIDENCE_OVERRIDE,
            AdversarialKind.COT_REQUEST,
            AdversarialKind.ELIGIBILITY_CLAIM,
        }
        missing = required - kinds
        assert not missing, f"adversarial kinds missing: {missing}"

    def test_adversarial_case_shape(self):
        for c in all_adversarial_cases():
            assert isinstance(c, AdversarialCase)
            assert c.case_id, "case_id required"
            assert c.prompt.strip(), "prompt required"
            assert c.kind, "kind required"


# --------------------------------------------------------------------------- #
# 5 — Provider failure scenarios
# --------------------------------------------------------------------------- #


class TestFailureScenarios:
    """PART 4: 9 provider failure modes the harness must exercise."""

    def test_failure_kinds_vocabulary(self):
        kinds = failure_kinds()
        assert len(kinds) == 9, (
            f"failure vocabulary has {len(kinds)} kinds; expected 9"
        )

    def test_failure_scenarios_have_prompts(self):
        for s in all_failure_scenarios():
            assert s.scenario_id, "scenario_id required"
            assert s.prompt.strip(), "prompt required"
            assert s.expected_outcome is not None


# --------------------------------------------------------------------------- #
# 6 — Data quality profiles
# --------------------------------------------------------------------------- #


class TestDataQualityProfiles:
    """PART 5: 8 data-quality profiles covering complete → adversarial."""

    def test_profile_count(self):
        profiles = all_profiles()
        assert len(profiles) == 8, f"got {len(profiles)} profiles; expected 8"

    def test_profile_shape(self):
        for p in all_profiles():
            assert isinstance(p, DataQualityProfile)
            assert p.profile_id, "profile_id required"
            assert p.label, "label required"
            assert p.factory, "factory required"
            assert p.quality_notes, "quality_notes required"

    def test_profiles_build_real_assistant_context(self):
        for p in all_profiles():
            ctx = p.build()
            # Real dataclass instance — NOT a SimpleNamespace.
            assert type(ctx).__name__ == "AssistantContext", (
                f"profile {p.profile_id} produced {type(ctx).__name__}; "
                "must be AssistantContext"
            )
            # Required production-path fields are populated.
            assert ctx.business_id > 0
            assert ctx.overall_business_score >= 0
            assert ctx.band, "band required for AI-1 layers"

    def test_get_profile_round_trip(self):
        for p in all_profiles():
            assert get_profile(p.profile_id) is p


# --------------------------------------------------------------------------- #
# 7 — Runner + production path
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def runner() -> EvaluationRunner:
    """Module-scoped runner with the complete Acme profile."""
    return EvaluationRunner(context=all_profiles()[0].build())


class TestRunner:
    """PART 8: runner drives prompts through the production path."""

    def test_runner_instantiates(self, runner):
        assert isinstance(runner, EvaluationRunner)

    def test_run_prompt_returns_evaluation_result(self, runner):
        r = runner.run_prompt(
            "What is our current revenue?", case_id="run_smoke"
        )
        assert isinstance(r, EvaluationResult)
        assert r.case_id == "run_smoke"
        assert r.prompt == "What is our current revenue?"
        assert r.production_path is True

    def test_run_prompt_body_non_empty(self, runner):
        r = runner.run_prompt(
            "What is our current revenue?", case_id="run_body"
        )
        assert r.body.strip(), "production path returned an empty body"
        assert r.success is True
        assert len(r.body) >= 100

    def test_run_prompt_captures_generation_metadata(self, runner):
        r = runner.run_prompt(
            "What is our biggest risk?", case_id="run_meta"
        )
        assert r.generation is not None
        gen = r.generation
        # Server-owned confidence MUST be present.
        assert hasattr(gen, "server_confidence")
        assert gen.server_confidence is not None
        # Answer mode MUST be populated.
        assert gen.answer_mode in (
            "general_knowledge",
            "business_fact",
            "business_analysis",
            "calculation",
            "scenario",
            "comparison",
            "forecast",
            "financial",
            "operational",
            "risk",
            "scheme",
            "export",
            "roadmap",
            "mixed",
        ), f"unexpected answer_mode={gen.answer_mode!r}"

    def test_run_prompt_captures_latency(self, runner):
        r = runner.run_prompt(
            "What is EBITDA?", case_id="run_latency"
        )
        assert r.latency_ms > 0, "latency must be positive"

    def test_extract_tools_used_static_helper(self):
        # Build a synthetic result that has at least one tool.
        from types import SimpleNamespace
        gen = SimpleNamespace(
            deterministic_services_used=("score",),
            structured_tool_envelopes=(),
        )
        result = SimpleNamespace(generation=gen)
        tools = EvaluationRunner.extract_tools_used(result)
        assert "score" in tools

    def test_extract_numbers_static_helper(self):
        nums = EvaluationRunner.extract_numbers(
            "We need ₹30,00,000 to reach ₹3 Cr. Gap = ₹12,00,000."
        )
        assert 3000000.0 in nums or 30_000_000.0 in nums
        assert 1200000.0 in nums

    def test_body_has_cot_marker_detects_leakage(self):
        assert EvaluationRunner.body_has_cot_marker(
            "Let me think step by step. First, let me reason."
        )
        assert not EvaluationRunner.body_has_cot_marker(
            "Your revenue is ₹1.8 Cr. Consider hiring."
        )


# --------------------------------------------------------------------------- #
# 8 — Metrics calculator
# --------------------------------------------------------------------------- #


class TestMetricsCalculator:
    """PART 6: 14 metrics, every value a measured number."""

    def test_compute_with_empty_inputs_returns_report(self):
        report = MetricsCalculator().compute()
        assert isinstance(report, MetricsReport)
        d = report.to_dict()
        # Every metric key is present.
        for key in (
            "question_coverage", "evidence_correctness",
            "numeric_correctness", "calculation_correctness",
            "unsupported_claim_rate", "missing_data_correctness",
            "contradiction_handling", "tool_selection_precision",
            "unnecessary_tool_execution", "confidence_calibration",
            "answer_completeness", "actionability",
            "response_latency_p50_ms", "response_latency_p95_ms",
            "response_latency_max_ms", "fallback_correctness",
            "production_path_fraction", "total_cases",
            "successful_cases",
        ):
            assert key in d, f"missing metric key {key}"

    def test_compute_total_cases_matches_inputs(self, runner):
        # Drive 5 prompts through the runner and feed the
        # calculator.
        results = tuple(
            runner.run_prompt(p, case_id=f"m_{i}")
            for i, p in enumerate(
                [
                    "What is our current revenue?",
                    "What is EBITDA?",
                    "What is our biggest risk?",
                    "Recommend 3 actions.",
                    "What is a sole proprietorship?",
                ]
            )
        )
        report = MetricsCalculator().compute(
            question_bank_results=results,
        )
        assert report.total_cases == 5
        assert report.production_path_fraction == 1.0

    def test_to_dict_is_json_safe(self):
        report = MetricsCalculator().compute()
        import json
        # to_dict must serialise cleanly (no dataclass leakage).
        json.dumps(report.to_dict())

    def test_business_and_general_tool_sets_disjoint(self):
        # The tool sets must NOT overlap; otherwise precision
        # and unnecessary-execution metrics collide.
        assert not (_BUSINESS_TOOLS & _GENERAL_TOOLS), (
            f"overlap: {_BUSINESS_TOOLS & _GENERAL_TOOLS}"
        )


# --------------------------------------------------------------------------- #
# 9 — End-to-end integration
# --------------------------------------------------------------------------- #


class TestEndToEndIntegration:
    """Drive the full harness and assert the report is well-formed."""

    def test_full_harness_drives_through_production_path(self, runner):
        # Drive a representative slice (5 prompts).
        prompts = (
            "What is our current revenue?",
            "What is EBITDA?",
            "What is our biggest risk?",
            "Recommend 3 actions.",
            "What is a sole proprietorship?",
        )
        results = tuple(
            runner.run_prompt(p, case_id=f"e2e_{i}")
            for i, p in enumerate(prompts)
        )
        # Every prompt reached the production path.
        assert all(r.production_path for r in results)
        # Every prompt returned a non-empty body.
        assert all(r.body.strip() for r in results)
        # Every prompt captured a generation.
        assert all(r.generation is not None for r in results)
        # The metrics calculator consumes them without error.
        report = MetricsCalculator().compute(
            question_bank_results=results,
        )
        assert isinstance(report, MetricsReport)
        # production_path_fraction MUST be 1.0 here (every prompt
        # went through the service).
        assert report.production_path_fraction == 1.0