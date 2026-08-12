"""SPRINT AI-18 — Universal AI Evaluation Harness.

Driver: drive the full harness fixture suite through the
production path and emit the measured metrics.

Run from repo root:

    python scripts/debug/run_ai18_harness.py

Output is JSON on stdout. The script is a debug driver,
not a test; the real assertions live in
``backend/tests/test_ai18_evaluation_harness.py``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make ``backend`` importable when running from repo root.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.services.ai.evaluation import (  # noqa: E402
    EvaluationRunner,
    MetricsCalculator,
    all_adversarial_cases,
    all_failure_scenarios,
    all_golden_cases,
    all_profiles,
    all_questions,
    all_scripts,
    category_vocabulary,
)


def _drive_all() -> dict:
    """Drive every fixture set through the production path."""
    runner = EvaluationRunner(context=all_profiles()[0].build())

    # 1) question bank
    bank = all_questions()
    bank_results = runner.run_question_bank(bank)

    # 2) golden set
    golden_results = runner.run_golden_set(all_golden_cases())

    # 3) adversarial
    adversarial_results = runner.run_adversarial(all_adversarial_cases())

    # 4) follow-up scripts
    followup_results = []
    for script in all_scripts():
        followup_results.append(runner.run_followup(script))

    # 5) failure scenarios
    failure_results = runner.run_prompt.__func__  # not used; we run them manually
    # The runner doesn't auto-inject failures; we drive the
    # base provider (which always takes the deterministic
    # fallback) and report the fallback correctness for the
    # base path. The metrics calculator reports this as
    # fallback_correctness based on body non-empty.
    scenarios = all_failure_scenarios()
    failure_results = tuple(
        runner.run_prompt(s.prompt, case_id=s.scenario_id)
        for s in scenarios
    )

    # 6) data quality profiles
    data_quality_results = []
    for prof in all_profiles():
        prof_runner = EvaluationRunner(context=prof.build())
        one_prompt = "What is our current revenue and what should we do?"
        data_quality_results.append(
            prof_runner.run_prompt(one_prompt, case_id=prof.profile_id)
        )
    data_quality_results = tuple(
        (r,) for r in data_quality_results
    )

    # Compute metrics.
    report = MetricsCalculator().compute(
        question_bank_results=bank_results,
        golden_results=golden_results,
        adversarial_results=adversarial_results,
        followup_results=tuple(followup_results),
        failure_results=failure_results,
        data_quality_results=data_quality_results,
        question_bank_categories=category_vocabulary(),
        question_bank_entries=bank,
        adversarial_cases=all_adversarial_cases(),
        data_quality_profiles=all_profiles(),
        failure_scenarios=scenarios,
    )

    return {
        "metrics": report.to_dict(),
        "fixture_counts": {
            "question_bank": len(bank),
            "categories": len(category_vocabulary()),
            "golden_set": len(all_golden_cases()),
            "adversarial": len(all_adversarial_cases()),
            "followup_scripts": len(all_scripts()),
            "followup_turns": sum(len(s.turns) for s in all_scripts()),
            "failure_scenarios": len(scenarios),
            "data_quality_profiles": len(all_profiles()),
        },
        "wall_clock_seconds": None,
    }


def main() -> None:
    t0 = time.perf_counter()
    out = _drive_all()
    out["wall_clock_seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
