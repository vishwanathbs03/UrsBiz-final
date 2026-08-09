"""AdaptiveAnswer — SPRINT AI-1 Stage 8.

The legacy assistant renders the same 10-section consultant
framing for every prompt. AI-1 keeps that as the default
(``"expanded"`` shell) but adds three more shells:

  * ``"executive"`` — short, 3-section shell for simple
    questions. Saves tokens on trivial lookups.
  * ``"scenario"`` — Assumptions / Estimated outcome / Risks /
    What would change the result. Used for "what if" prompts.
  * ``"missing_info"`` — What I can determine / What is
    missing / Why it matters / What to provide next. Used when
    the context is missing fields the answer requires.

The composer does NOT overwrite the LLM's prose. It returns
metadata only (``AdaptiveAnswer``) — the service uses it to
stamp ``GenerationMeta.possible_answer_structure`` and to
enrich the audit trail.

Backward compatibility
-----------------------

The composer is invoked AFTER the LLM has produced its
response. The LLM prose stays unchanged. Only the metadata
envelope changes. The fallback path (no LLM) returns a
default :class:`AdaptiveAnswer` with
``mode_used="expanded"`` so the deterministic path's audit
trail is also uniformly tagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AnswerShell = Literal["executive", "expanded", "scenario", "missing_info"]


@dataclass(frozen=True)
class AdaptiveAnswer:
    """The structured shell the answer composer picks.

    Attributes
    ----------
    mode_used
        One of the :data:`AnswerShell` literals. Records which
        shell the composer selected for the audit trail.
    sections
        Ordered list of section titles the chosen shell uses.
        The renderer surfaces these in the response envelope.
    executive_summary
        A one-paragraph summary derived from the parsed
        response (the legacy ``ExecutiveSummary.text`` for the
        grounded path, the first paragraph of the raw body for
        the open path). Empty when no LLM response was
        available.
    key_findings
        Top-N bullets the composer extracted from the parsed
        response. Empty when the parsed response has none.
    recommendations
        Recommendation titles the composer surfaced. Empty
        when none.
    assumptions
        Assumptions declared by the composer shell. Three max.
    limitations
        Limitations declared by the composer shell. Three max.
    """

    mode_used: AnswerShell = "expanded"
    sections: tuple[str, ...] = field(default_factory=tuple)
    executive_summary: str = ""
    key_findings: tuple[str, ...] = field(default_factory=tuple)
    recommendations: tuple[str, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Shell templates
# --------------------------------------------------------------------------- #


_SHELL_EXECUTIVE: tuple[str, ...] = (
    "1. EXECUTIVE SUMMARY",
    "2. KEY FINDINGS",
    "3. NEXT ACTION",
)

_SHELL_EXPANDED: tuple[str, ...] = (
    "1. EXECUTIVE SUMMARY",
    "2. KEY FINDINGS",
    "3. GROWTH STRATEGY",
    "4. QUARTER-WISE ROADMAP",
    "5. FUNDING & PROJECTIONS",
    "6. KEY RISKS",
    "7. SCHEMES",
    "8. KPIs TO TRACK",
    "9. NEXT ACTIONS",
    "10. ASSUMPTIONS / LIMITATIONS",
)

_SHELL_SCENARIO: tuple[str, ...] = (
    "1. SCENARIO ASSUMPTIONS",
    "2. ESTIMATED OUTCOME",
    "3. KEY RISKS UNDER THIS SCENARIO",
    "4. WHAT WOULD CHANGE THE RESULT",
)

_SHELL_MISSING_INFO: tuple[str, ...] = (
    "1. WHAT I CAN DETERMINE",
    "2. WHAT IS MISSING",
    "3. WHY IT MATTERS",
    "4. WHAT TO PROVIDE NEXT",
)


# --------------------------------------------------------------------------- #
# Composer
# --------------------------------------------------------------------------- #


def compose_adaptive_answer(
    *,
    parsed: Any,
    question_understanding: Any,
    reasoning_plan: Any,
    tool_results: tuple[Any, ...] = (),
    context: Any = None,
) -> AdaptiveAnswer:
    """Return an :class:`AdaptiveAnswer` describing the chosen shell.

    Parameters
    ----------
    parsed
        The parsed response (e.g. ``GroundedResponse`` for the
        grounded path, ``OpenResponse`` for the open path).
        May be ``None`` when the deterministic fallback
        produced the body — the composer then uses the plan's
        ``possible_answer_structure`` field.
    question_understanding
        The :class:`QuestionUnderstanding` from Stage 1.
    reasoning_plan
        The :class:`ReasoningPlan` from Stage 4. The plan's
        :attr:`possible_answer_structure` field is the
        primary driver of the shell choice.
    tool_results
        The :class:`ToolResult` tuple from Stage 5.
        Currently informational only — the composer never
        emits a tool-specific shell.
    context
        The :class:`AssistantContext`. Informational only.
    """
    # 1. Pick the shell.
    shell = _pick_shell(parsed, question_understanding, reasoning_plan)

    # 2. Map the shell to its section list.
    sections = _SECTIONS_BY_SHELL[shell]

    # 3. Pull findings / recommendations from the parsed response.
    executive_summary, key_findings, recommendations = _extract_summary(
        parsed
    )

    # 4. Add shell-specific assumptions / limitations.
    assumptions, limitations = _assumptions_for_shell(
        shell, question_understanding, reasoning_plan, tool_results
    )

    return AdaptiveAnswer(
        mode_used=shell,
        sections=sections,
        executive_summary=executive_summary,
        key_findings=key_findings,
        recommendations=recommendations,
        assumptions=assumptions,
        limitations=limitations,
    )


_SECTIONS_BY_SHELL: dict[str, tuple[str, ...]] = {
    "executive": _SHELL_EXECUTIVE,
    "expanded": _SHELL_EXPANDED,
    "scenario": _SHELL_SCENARIO,
    "missing_info": _SHELL_MISSING_INFO,
}


def _pick_shell(
    parsed: Any,
    question_understanding: Any,
    reasoning_plan: Any,
) -> AnswerShell:
    """Pick the answer shell.

    Priority order:

      1. Plan's ``possible_answer_structure`` (when set) wins.
      2. Understanding's ``unknowns`` (when non-empty) flips
         to ``"missing_info"``.
      3. Understanding's ``complexity``:

         * ``"simple"`` → ``"executive"``
         * ``"scenario"`` → ``"scenario"``
         * ``"moderate"`` / ``"strategic"`` → ``"expanded"``

      4. Default ``"expanded"``.
    """
    plan_value = getattr(reasoning_plan, "possible_answer_structure", "") or ""
    if plan_value in _SECTIONS_BY_SHELL:
        return plan_value  # type: ignore[return-value]

    unknowns = getattr(question_understanding, "unknowns", ()) or ()
    if unknowns:
        return "missing_info"

    complexity = getattr(question_understanding, "complexity", "moderate")
    if complexity == "scenario":
        return "scenario"
    if complexity == "simple":
        return "executive"
    return "expanded"


def _extract_summary(parsed: Any) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Pull summary text + findings + recommendations from ``parsed``."""
    if parsed is None:
        return "", (), ()
    # ``GroundedResponse`` has a string executive_summary.
    if isinstance(getattr(parsed, "executive_summary", None), str):
        exec_sum = parsed.executive_summary
    else:
        exec_sum = (
            getattr(getattr(parsed, "executive_summary", None), "text", "") or ""
        )
    findings: list[str] = []
    for kf in getattr(parsed, "key_findings", ()) or ():
        if isinstance(kf, str):
            findings.append(kf)
        else:
            text = (
                getattr(kf, "statement", "") or getattr(kf, "text", "") or ""
            )
            if text:
                findings.append(text)
    recs: list[str] = []
    for rec in getattr(parsed, "recommendations", ()) or ():
        if isinstance(rec, str):
            recs.append(rec)
        else:
            title = (
                getattr(rec, "title", "") or getattr(rec, "action", "") or ""
            )
            if title:
                recs.append(title)
    return exec_sum, tuple(findings[:5]), tuple(recs[:3])


def _assumptions_for_shell(
    shell: AnswerShell,
    question_understanding: Any,
    reasoning_plan: Any,
    tool_results: tuple[Any, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (assumptions, limitations) for the chosen shell."""
    if shell == "scenario":
        return (
            (
                "Scenario outputs are estimates, not predictions.",
                "Confidence reflects data completeness, not outcome certainty.",
            ),
            (
                "Sensitivities to cost / demand changes may be larger than shown.",
            ),
        )
    if shell == "missing_info":
        unknowns = getattr(question_understanding, "unknowns", ()) or ()
        limitations = tuple(
            f"Missing: {u}" for u in unknowns[:3]
        ) or ("Add profile data to unlock a fuller answer.",)
        return (
            (
                "Answer is partial because the context is missing fields.",
            ),
            limitations,
        )
    if shell == "executive":
        return (
            ("Answer is a short summary; ask a follow-up for depth.",),
            (),
        )
    # expanded (default)
    return (
        (
            "Every fact in the response traces back to an evidence registry entry.",
        ),
        (
            "Detailed numbers depend on the freshness of your business profile.",
        ),
    )