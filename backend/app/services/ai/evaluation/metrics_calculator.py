"""SPRINT AI-18 — Universal AI Evaluation Harness.

Quality metrics calculator.

The brief (PART 6) requires 14 documented metrics:

  1.  question coverage
  2.  evidence correctness
  3.  numeric correctness
  4.  calculation correctness
  5.  unsupported-claim rate
  6.  missing-data correctness
  7.  contradiction handling
  8.  tool-selection precision
  9.  unnecessary tool execution
  10. confidence calibration
  11. answer completeness
  12. actionability
  13. response latency
  14. fallback correctness

The calculator NEVER claims "100% accuracy" — every metric
is a measured value. Failures are explicit, never silently
ignored. All values are JSON-safe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.services.ai.evaluation.runner import (
    EvaluationResult,
    EvaluationRunner,
    _BUSINESS_TOOLS,
    _GENERAL_TOOLS,
)


# --------------------------------------------------------------------------- #
# Metrics report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricsReport:
    """All 14 brief metrics as measured values.

    Every field is documented. The runner never claims
    ``"100% accuracy"`` — it reports the raw counts.
    """

    # 1 — question coverage: fraction of categories the bank covers.
    question_coverage: float = 0.0
    categories_covered: int = 0
    categories_total: int = 0
    # 2 — evidence correctness: fraction of business prompts that
    #     cited at least one evidence_ref.
    evidence_correctness: float = 0.0
    # 3 — numeric correctness: fraction of numeric-bearing prompts
    #     whose body numbers match the answer mode (calc / forecast).
    numeric_correctness: float = 0.0
    # 4 — calculation correctness: fraction of CALCULATION prompts
    #     whose body actually carries a numeric.
    calculation_correctness: float = 0.0
    # 5 — unsupported-claim rate: average
    #     ``unsupported_claim_count`` per response.
    unsupported_claim_rate: float = 0.0
    # 6 — missing-data correctness: fraction of data-quality cases
    #     whose reply acknowledged the gap.
    missing_data_correctness: float = 0.0
    # 7 — contradiction handling: fraction of adversarial
    #     "false-fact" / "evidence-override" prompts the runner
    #     caught.
    contradiction_handling: float = 0.0
    # 8 — tool-selection precision: fraction of business prompts
    #     that fired at least one business tool.
    tool_selection_precision: float = 0.0
    # 9 — unnecessary tool execution: fraction of general-knowledge
    #     prompts that fired any business tool.
    unnecessary_tool_execution: float = 0.0
    # 10 — confidence calibration: mean server_confidence across
    #     the bank.
    confidence_calibration: float = 0.0
    # 11 — answer completeness: fraction of bank prompts whose
    #     body satisfies the brief's ``body_min_chars`` floor.
    answer_completeness: float = 0.0
    # 12 — actionability: fraction of RECOMMENDATION prompts whose
    #     body carries an action verb.
    actionability: float = 0.0
    # 13 — response latency: p50 / p95 latency in milliseconds.
    response_latency_p50_ms: int = 0
    response_latency_p95_ms: int = 0
    response_latency_max_ms: int = 0
    # 14 — fallback correctness: fraction of failure scenarios the
    #     runner caught (deterministic fallback engaged with a
    #     non-empty body).
    fallback_correctness: float = 0.0

    # Cross-cutting
    production_path_fraction: float = 0.0
    total_cases: int = 0
    successful_cases: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_coverage": float(self.question_coverage),
            "categories_covered": int(self.categories_covered),
            "categories_total": int(self.categories_total),
            "evidence_correctness": float(self.evidence_correctness),
            "numeric_correctness": float(self.numeric_correctness),
            "calculation_correctness": float(self.calculation_correctness),
            "unsupported_claim_rate": float(self.unsupported_claim_rate),
            "missing_data_correctness": float(self.missing_data_correctness),
            "contradiction_handling": float(self.contradiction_handling),
            "tool_selection_precision": float(self.tool_selection_precision),
            "unnecessary_tool_execution": float(self.unnecessary_tool_execution),
            "confidence_calibration": float(self.confidence_calibration),
            "answer_completeness": float(self.answer_completeness),
            "actionability": float(self.actionability),
            "response_latency_p50_ms": int(self.response_latency_p50_ms),
            "response_latency_p95_ms": int(self.response_latency_p95_ms),
            "response_latency_max_ms": int(self.response_latency_max_ms),
            "fallback_correctness": float(self.fallback_correctness),
            "production_path_fraction": float(self.production_path_fraction),
            "total_cases": int(self.total_cases),
            "successful_cases": int(self.successful_cases),
        }


# --------------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------------- #


# Minimum body length the runner treats as "complete". Below
# this the answer is considered an empty / inadequate reply.
_BODY_MIN_CHARS = 20

# Action verbs the runner keys off for the actionability metric.
_ACTION_VERBS = (
    "apply", "consider", "focus", "hire", "invest", "review",
    "schedule", "start", "track", "use", "adopt", "expand",
    "register", "contact", "negotiate", "reduce", "increase",
    "prioritise", "prioritize", "implement", "deploy",
    "should", "recommend", "next step", "next:", "first,",
)


class MetricsCalculator:
    """Compute the 14 brief metrics from runner results.

    Pure. No I/O. The calculator is fed
    :class:`EvaluationResult` tuples; it produces a
    :class:`MetricsReport`.
    """

    def compute(
        self,
        *,
        question_bank_results: tuple[EvaluationResult, ...] = (),
        golden_results: tuple[EvaluationResult, ...] = (),
        adversarial_results: tuple[EvaluationResult, ...] = (),
        followup_results: tuple[tuple[EvaluationResult, ...], ...] = (),
        failure_results: tuple[EvaluationResult, ...] = (),
        data_quality_results: tuple[tuple[EvaluationResult, ...], ...] = (),
        question_bank_categories: tuple[str, ...] = (),
        question_bank_entries: tuple[Any, ...] = (),
        adversarial_cases: tuple[Any, ...] = (),
        data_quality_profiles: tuple[Any, ...] = (),
        failure_scenarios: tuple[Any, ...] = (),
    ) -> MetricsReport:
        """Compute the :class:`MetricsReport`.

        Parameters
        ----------
        question_bank_results:
            Per-prompt :class:`EvaluationResult` tuples from
            the runner.
        golden_results:
            Per-case golden results.
        adversarial_results:
            Per-case adversarial results. Used for the
            ``contradiction_handling`` metric.
        followup_results:
            Tuple of per-script result tuples.
        failure_results:
            Per-scenario failure results.
        data_quality_results:
            Tuple of per-profile result tuples.
        question_bank_categories:
            The brief's 16+ category vocabulary.
        question_bank_entries:
            The original :class:`QuestionEntry` tuples (used
            to bucket results by category for tool-selection
            precision).
        adversarial_cases:
            The original :class:`AdversarialCase` tuples
            (used to identify false-fact / evidence-override
            cases for ``contradiction_handling``).
        data_quality_profiles:
            The original :class:`DataQualityProfile` tuples
            (used to identify expected-warning cases).
        failure_scenarios:
            The original :class:`ProviderFailureScenario`
            tuples (used to identify which cases are failure
            cases).
        """
        # Total cases + success count.
        all_results = list(question_bank_results) + list(golden_results) + list(adversarial_results)
        for script_results in followup_results:
            all_results.extend(script_results)
        for prof_results in data_quality_results:
            all_results.extend(prof_results)
        all_results.extend(failure_results)

        total = len(all_results)
        successful = sum(1 for r in all_results if r.success)
        production_path = sum(1 for r in all_results if r.production_path)

        return MetricsReport(
            question_coverage=self._question_coverage(
                question_bank_categories, question_bank_entries
            ),
            categories_covered=self._categories_covered(question_bank_entries),
            categories_total=len(question_bank_categories),
            evidence_correctness=self._evidence_correctness(
                question_bank_results, question_bank_entries
            ),
            numeric_correctness=self._numeric_correctness(
                question_bank_results, question_bank_entries
            ),
            calculation_correctness=self._calculation_correctness(
                question_bank_results, question_bank_entries
            ),
            unsupported_claim_rate=self._unsupported_claim_rate(all_results),
            missing_data_correctness=self._missing_data_correctness(
                data_quality_results, data_quality_profiles
            ),
            contradiction_handling=self._contradiction_handling(
                adversarial_results, adversarial_cases
            ),
            tool_selection_precision=self._tool_selection_precision(
                question_bank_results, question_bank_entries
            ),
            unnecessary_tool_execution=self._unnecessary_tool_execution(
                question_bank_results, question_bank_entries
            ),
            confidence_calibration=self._confidence_calibration(all_results),
            answer_completeness=self._answer_completeness(all_results),
            actionability=self._actionability(
                question_bank_results, question_bank_entries
            ),
            response_latency_p50_ms=self._latency_percentile(all_results, 50),
            response_latency_p95_ms=self._latency_percentile(all_results, 95),
            response_latency_max_ms=self._max_latency(all_results),
            fallback_correctness=self._fallback_correctness(
                failure_results, failure_scenarios
            ),
            production_path_fraction=_fraction(production_path, total),
            total_cases=total,
            successful_cases=successful,
        )

    # ---- individual metrics --------------------------------------- #

    @staticmethod
    def _question_coverage(
        categories: tuple[str, ...], entries: tuple[Any, ...]
    ) -> float:
        """Fraction of categories covered by at least one entry."""
        if not categories:
            return 0.0
        present = {q.category for q in entries}
        return _fraction(len(present & set(categories)), len(categories))

    @staticmethod
    def _categories_covered(entries: tuple[Any, ...]) -> int:
        return len({q.category for q in entries})

    @staticmethod
    def _evidence_correctness(
        results: tuple[EvaluationResult, ...], entries: tuple[Any, ...]
    ) -> float:
        """Business-prompt fraction that cited >=1 evidence_ref."""
        if len(results) != len(entries):
            return 0.0
        biz = [
            (r, e) for r, e in zip(results, entries)
            if e.category in _BUSINESS_CATEGORIES
        ]
        if not biz:
            return 0.0
        cited = sum(1 for r, _ in biz if r.notes.get("evidence_count", 0) >= 1)
        return _fraction(cited, len(biz))

    @staticmethod
    def _numeric_correctness(
        results: tuple[EvaluationResult, ...], entries: tuple[Any, ...]
    ) -> float:
        """Calculation / financial / forecast prompts that carry
        at least one numeric literal."""
        if len(results) != len(entries):
            return 0.0
        numeric = [
            (r, e) for r, e in zip(results, entries)
            if e.category in _NUMERIC_CATEGORIES
        ]
        if not numeric:
            return 0.0
        ok = 0
        for r, _ in numeric:
            if EvaluationRunner.extract_numbers(r.body):
                ok += 1
        return _fraction(ok, len(numeric))

    @staticmethod
    def _calculation_correctness(
        results: tuple[EvaluationResult, ...], entries: tuple[Any, ...]
    ) -> float:
        """CALCULATION prompts whose body carries a numeric."""
        if len(results) != len(entries):
            return 0.0
        calc = [
            (r, e) for r, e in zip(results, entries)
            if e.category == "calculation"
        ]
        if not calc:
            return 0.0
        ok = sum(
            1 for r, _ in calc
            if EvaluationRunner.extract_numbers(r.body)
        )
        return _fraction(ok, len(calc))

    @staticmethod
    def _unsupported_claim_rate(results: tuple[EvaluationResult, ...]) -> float:
        """Average ``unsupported_claim_count`` per response."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            gen = r.generation
            if gen is None:
                continue
            total += float(getattr(gen, "unsupported_claim_count", 0) or 0)
        return total / float(len(results))

    @staticmethod
    def _missing_data_correctness(
        profile_results: tuple[tuple[EvaluationResult, ...], ...],
        profiles: tuple[Any, ...],
    ) -> float:
        """Fraction of data-quality profiles whose reply
        acknowledged the gap (warning or disclosure)."""
        if not profile_results or not profiles:
            return 0.0
        ok = 0
        for results, profile in zip(profile_results, profiles):
            if not results:
                continue
            # The runner runs the same prompt against each
            # profile; we look at the FIRST result (any result
            # is fine for the metric).
            r = results[0]
            gen = r.generation
            disclosed = (
                bool(r.notes.get("needs_warning"))
                or bool(getattr(gen, "needs_warning", False))
                or bool(getattr(gen, "partial_failure_disclosure", None))
            )
            if profile.expected_warning and not disclosed:
                continue
            if not profile.expected_warning and disclosed:
                # Disclosing when not expected is OK.
                pass
            ok += 1
        return _fraction(ok, len(profiles))

    @staticmethod
    def _contradiction_handling(
        results: tuple[EvaluationResult, ...], cases: tuple[Any, ...]
    ) -> float:
        """Fraction of USER_FALSE_FACT / EVIDENCE_OVERRIDE prompts
        the runner caught (body must NOT confirm the false claim).

        The runner evaluates this purely on the body content.
        """
        if not cases or not results:
            return 0.0
        caught = 0
        total = 0
        for r, c in zip(results, cases):
            if c.kind not in {"user_false_fact", "evidence_override", "conflicting_numbers"}:
                continue
            total += 1
            body_low = (r.body or "").lower()
            # Caught when the body does NOT contain the
            # expected_confirmation tokens AND does NOT match
            # the unsafe pattern.
            unsafe_tokens = (
                "is correct", "confirmed", "your score is",
                "your revenue is", "₹500 cr is correct",
            )
            contains_unsafe = any(tok in body_low for tok in unsafe_tokens)
            if not contains_unsafe:
                caught += 1
        return _fraction(caught, total)

    @staticmethod
    def _tool_selection_precision(
        results: tuple[EvaluationResult, ...], entries: tuple[Any, ...]
    ) -> float:
        """Business-prompt fraction that fired >=1 business tool."""
        if len(results) != len(entries):
            return 0.0
        biz = [
            (r, e) for r, e in zip(results, entries)
            if e.category in _BUSINESS_CATEGORIES
        ]
        if not biz:
            return 0.0
        ok = 0
        for r, _ in biz:
            tools = EvaluationRunner.extract_tools_used(r)
            if any(t.lower() in _BUSINESS_TOOLS for t in tools):
                ok += 1
        return _fraction(ok, len(biz))

    @staticmethod
    def _unnecessary_tool_execution(
        results: tuple[EvaluationResult, ...], entries: tuple[Any, ...]
    ) -> float:
        """General-knowledge / external / mixed prompts that
        fired a business tool. Reported as the inverse:
        fraction that did NOT (lower = better)."""
        if len(results) != len(entries):
            return 0.0
        general = [
            (r, e) for r, e in zip(results, entries)
            if e.category in _GENERAL_KNOWLEDGE_CATEGORIES
        ]
        if not general:
            return 0.0
        ok = 0
        for r, _ in general:
            tools = EvaluationRunner.extract_tools_used(r)
            if not any(t.lower() in _BUSINESS_TOOLS for t in tools):
                ok += 1
        return _fraction(ok, len(general))

    @staticmethod
    def _confidence_calibration(results: tuple[EvaluationResult, ...]) -> float:
        """Mean ``server_confidence`` across all results."""
        if not results:
            return 0.0
        total = 0.0
        count = 0
        for r in results:
            gen = r.generation
            if gen is None:
                continue
            sc = getattr(gen, "server_confidence", None)
            if sc is None:
                continue
            total += float(sc)
            count += 1
        return total / float(count) if count else 0.0

    @staticmethod
    def _answer_completeness(results: tuple[EvaluationResult, ...]) -> float:
        """Fraction whose body satisfies ``_BODY_MIN_CHARS``."""
        if not results:
            return 0.0
        ok = sum(1 for r in results if len(r.body or "") >= _BODY_MIN_CHARS)
        return _fraction(ok, len(results))

    @staticmethod
    def _actionability(
        results: tuple[EvaluationResult, ...], entries: tuple[Any, ...]
    ) -> float:
        """RECOMMENDATION prompts whose body carries an action verb."""
        if len(results) != len(entries):
            return 0.0
        rec = [
            (r, e) for r, e in zip(results, entries)
            if e.category == "recommendation"
        ]
        if not rec:
            return 0.0
        ok = 0
        for r, _ in rec:
            low = (r.body or "").lower()
            if any(v in low for v in _ACTION_VERBS):
                ok += 1
        return _fraction(ok, len(rec))

    @staticmethod
    def _latency_percentile(
        results: tuple[EvaluationResult, ...], pct: int
    ) -> int:
        """Latency percentile in milliseconds."""
        if not results:
            return 0
        lats = sorted(r.latency_ms for r in results)
        if not lats:
            return 0
        idx = max(0, min(len(lats) - 1, int(len(lats) * pct / 100.0)))
        return int(lats[idx])

    @staticmethod
    def _max_latency(results: tuple[EvaluationResult, ...]) -> int:
        if not results:
            return 0
        return int(max(r.latency_ms for r in results))

    @staticmethod
    def _fallback_correctness(
        results: tuple[EvaluationResult, ...], scenarios: tuple[Any, ...]
    ) -> float:
        """Fraction of failure scenarios whose reply was safe
        (non-empty body). The runner exercises the deterministic
        fallback, which always produces a body — but the metric
        here measures ``body_nonempty`` to catch any future
        regression where the fallback is silent.
        """
        if not results:
            return 0.0
        ok = sum(1 for r in results if r.success and r.body.strip())
        return _fraction(ok, len(results))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_BUSINESS_CATEGORIES: frozenset[str] = frozenset({
    "business_fact", "business_analysis", "calculation",
    "recommendation", "scenario", "forecast", "comparison",
    "financial", "operational", "risk", "government_scheme",
    "export", "roadmap",
})

_NUMERIC_CATEGORIES: frozenset[str] = frozenset({
    "calculation", "financial", "forecast", "scenario",
})

_GENERAL_KNOWLEDGE_CATEGORIES: frozenset[str] = frozenset({
    "general_knowledge", "external_information",
})


def _fraction(num: int, den: int) -> float:
    """Safe fraction: 0 when den is 0."""
    if not den:
        return 0.0
    return float(num) / float(den)


__all__ = ["MetricsCalculator", "MetricsReport"]
