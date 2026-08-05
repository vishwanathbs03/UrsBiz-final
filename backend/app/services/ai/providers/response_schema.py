"""H7.3 — Docx Prompt 3 Part 3 response schema and validator.

The docx requires the generative response to follow this structure:

    {
      "executive_summary": "",
      "key_findings": [],
      "recommendations": [],
      "thirty_day_plan": [],
      "assumptions": [],
      "limitations": [],
      "confidence": 0,
      "evidence_references": []
    }

This module:

  1. Defines the schema as dataclasses (no Pydantic dep on the critical
     path; the protocol layer already uses dataclasses).
  2. Validates an arbitrary dict against that schema. Validation never
     raises — it returns a ``ValidationResult`` so the caller can choose
     whether to fall back to the deterministic provider, surface the
     partial response, or log and move on.
  3. Provides a normalization step that tolerates common LLM deviations
     (extra whitespace, Markdown fences around JSON, missing optional
     fields, slightly out-of-range confidence, etc.).
  4. Renders a validated response back to plain text suitable for the
     assistant chat UI — the UI now shows the structured response with
     a "Generated explanation" trust label rather than free-form prose.

Per docx P3 Part 3: "Validate the model response. When validation fails,
use the existing deterministic consultant response." That decision lives
in ``AssistantProviderService.generate`` — this module is the validator
it consults.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Schema shape
# --------------------------------------------------------------------------- #

# Hard caps that protect the API from runaway model outputs (docx P3 Part 6:
# "Very long prompts"). Every list field honours them on parse.
_MAX_LIST_LEN = 16
_MAX_TEXT_LEN = 2000
_MAX_PROMPT_LEN = 4000  # input prompt cap on the way IN


@dataclass(frozen=True)
class ExecutiveSummary:
    """The single-paragraph headline the UI renders as the assistant
    reply. Empty string is permitted (the fallback path uses the
    deterministic body)."""

    text: str


@dataclass(frozen=True)
class KeyFinding:
    """One bullet under "key findings". ``evidence_refs`` lists the
    ``evidence_references.id`` values that justify the finding."""

    statement: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Recommendation:
    """One action item. ``rationale`` MUST cite an evidence_reference.

    ``priority`` is constrained to a small set so the UI can colour-code.
    ``score_gain`` is an integer 0..100, never a percent."""

    title: str
    rationale: str
    priority: str  # "Critical" | "High" | "Medium" | "Low"
    score_gain: int  # 0..100


@dataclass(frozen=True)
class PlanItem:
    """One week-by-week task inside the 30-day plan."""

    week: int  # 1..4
    task: str


@dataclass(frozen=True)
class EvidenceReference:
    """Pointer back to the upstream service that produced a value.

    The dataclass is opaque to the model — the prompt builder injects a
    numbered list of available references, and the model references them
    by ``id``. We never leak the upstream payload to the response."""

    id: str
    kind: str  # "score" | "recommendation" | "rule" | "scheme" | "forecast" | "action" | "dna"
    label: str


@dataclass(frozen=True)
class GroundedResponse:
    """The fully validated response envelope."""

    executive_summary: str
    key_findings: tuple[KeyFinding, ...]
    recommendations: tuple[Recommendation, ...]
    thirty_day_plan: tuple[PlanItem, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    confidence: int  # 0..100
    evidence_references: tuple[EvidenceReference, ...]

    # ----- convenience: render for the assistant chat UI -----

    def to_chat_body(self) -> str:
        """Render the structured response as a readable markdown-ish
        blob suitable for the assistant chat surface. The trust label
        "Generated explanation" is appended at the bottom by the UI —
        this method renders only the content."""

        lines: list[str] = []
        if self.executive_summary:
            lines.append(self.executive_summary)
            lines.append("")

        if self.key_findings:
            lines.append("Key findings")
            for i, kf in enumerate(self.key_findings, start=1):
                lines.append(
                    f"  {i}. {kf.statement}"
                    + (f" (evidence: {', '.join(kf.evidence_refs)})" if kf.evidence_refs else "")
                )
            lines.append("")

        if self.recommendations:
            lines.append("Recommended next actions")
            for i, r in enumerate(self.recommendations, start=1):
                lines.append(
                    f"  {i}. [{r.priority} +{r.score_gain} score] "
                    f"{r.title} — {r.rationale}"
                )
            lines.append("")

        if self.thirty_day_plan:
            lines.append("30-day plan")
            for p in sorted(self.thirty_day_plan, key=lambda x: x.week):
                lines.append(f"  Week {p.week}: {p.task}")
            lines.append("")

        if self.assumptions:
            lines.append("Assumptions")
            for a in self.assumptions:
                lines.append(f"  - {a}")
            lines.append("")

        if self.limitations:
            lines.append("Limitations")
            for l in self.limitations:
                lines.append(f"  - {l}")
            lines.append("")

        lines.append(f"Model confidence: {self.confidence}/100")
        return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------- #
# Validation result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a model output against the schema.

    ``response`` is non-None only when validation passed well enough to
    produce a usable :class:`GroundedResponse`. ``errors`` lists every
    reason validation failed; an empty list means success."""

    response: GroundedResponse | None
    errors: tuple[str, ...] = field(default_factory=tuple)
    raw_text: str = ""

    @property
    def ok(self) -> bool:
        return self.response is not None


# --------------------------------------------------------------------------- #
# Allowed enums
# --------------------------------------------------------------------------- #

_ALLOWED_PRIORITIES = {"Critical", "High", "Medium", "Low"}
_ALLOWED_REF_KINDS = {
    "score", "recommendation", "rule", "scheme", "forecast", "action", "dna",
}

# Markdown fences many small models add around JSON output.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Public validator
# --------------------------------------------------------------------------- #


def parse_model_output(raw_text: str) -> ValidationResult:
    """Validate ``raw_text`` against the docx P3 Part 3 schema.

    Tolerates common LLM deviations:

      * ```json ... ``` fences (stripped).
      * Prose before / after the JSON object (first balanced ``{...}``
        extracted).
      * Slightly out-of-range confidence / score_gain (clamped).
      * Missing optional fields (empty defaults).
      * Extra unknown fields (ignored).
      * Strings that exceed ``_MAX_TEXT_LEN`` (truncated with ellipsis).

    Returns a :class:`ValidationResult` whose ``ok`` indicates whether
    the response is usable. The caller (the service) uses ``ok`` to
    decide whether to surface the response or fall back to the
    deterministic provider.
    """
    if not raw_text or not raw_text.strip():
        return ValidationResult(
            response=None,
            errors=("empty model output",),
            raw_text=raw_text or "",
        )

    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        return ValidationResult(
            response=None,
            errors=("model output is not a JSON object",),
            raw_text=raw_text,
        )

    errors: list[str] = []

    executive_summary = _clamp_str(
        parsed.get("executive_summary"), "executive_summary", errors,
    )

    key_findings_raw = parsed.get("key_findings") or []
    if not isinstance(key_findings_raw, list):
        errors.append("key_findings is not a list")
        key_findings_raw = []
    key_findings: list[KeyFinding] = []
    for item in key_findings_raw[:_MAX_LIST_LEN]:
        if not isinstance(item, dict):
            continue
        statement = _clamp_str(
            item.get("statement"), "key_finding.statement", errors,
        )
        if not statement:
            continue
        refs = item.get("evidence_refs") or []
        if not isinstance(refs, list):
            refs = []
        key_findings.append(KeyFinding(
            statement=statement,
            evidence_refs=tuple(str(r) for r in refs[:_MAX_LIST_LEN]),
        ))

    recommendations_raw = parsed.get("recommendations") or []
    if not isinstance(recommendations_raw, list):
        errors.append("recommendations is not a list")
        recommendations_raw = []
    recommendations: list[Recommendation] = []
    for item in recommendations_raw[:_MAX_LIST_LEN]:
        if not isinstance(item, dict):
            continue
        title = _clamp_str(item.get("title"), "recommendation.title", errors)
        rationale = _clamp_str(item.get("rationale"), "recommendation.rationale", errors)
        if not title:
            continue
        priority = str(item.get("priority", "Medium") or "Medium")
        if priority not in _ALLOWED_PRIORITIES:
            errors.append(f"recommendation.priority not in allowed set: {priority!r}")
            priority = "Medium"
        score_gain = _clamp_int(
            item.get("score_gain"), 0, 100, "recommendation.score_gain", errors,
        )
        recommendations.append(Recommendation(
            title=title,
            rationale=rationale,
            priority=priority,
            score_gain=score_gain,
        ))

    plan_raw = parsed.get("thirty_day_plan") or []
    if not isinstance(plan_raw, list):
        errors.append("thirty_day_plan is not a list")
        plan_raw = []
    plan: list[PlanItem] = []
    for item in plan_raw[:_MAX_LIST_LEN]:
        if not isinstance(item, dict):
            continue
        week = _clamp_int(item.get("week"), 1, 4, "plan.week", errors)
        task = _clamp_str(item.get("task"), "plan.task", errors)
        if not task:
            continue
        plan.append(PlanItem(week=week, task=task))

    assumptions = _string_list(
        parsed.get("assumptions"), "assumptions", errors,
    )
    limitations = _string_list(
        parsed.get("limitations"), "limitations", errors,
    )
    confidence = _clamp_int(
        parsed.get("confidence"), 0, 100, "confidence", errors,
    )

    refs_raw = parsed.get("evidence_references") or []
    if not isinstance(refs_raw, list):
        errors.append("evidence_references is not a list")
        refs_raw = []
    references: list[EvidenceReference] = []
    for item in refs_raw[:_MAX_LIST_LEN]:
        if not isinstance(item, dict):
            continue
        rid = _clamp_str(item.get("id"), "evidence.id", errors)
        kind = str(item.get("kind", "score") or "score")
        if kind not in _ALLOWED_REF_KINDS:
            kind = "score"
        label = _clamp_str(item.get("label"), "evidence.label", errors)
        if not rid:
            continue
        references.append(EvidenceReference(id=rid, kind=kind, label=label))

    # Validation is *strict enough to fall back* if the core fields are
    # missing. The structure exists, but the response must include at
    # least an executive_summary OR at least one recommendation to be
    # considered useful. Otherwise we treat the model output as unusable
    # and surface the deterministic body instead.
    if not executive_summary and not recommendations:
        errors.append("response is empty: no executive_summary and no recommendations")
        return ValidationResult(
            response=None,
            errors=tuple(errors),
            raw_text=raw_text,
        )

    response = GroundedResponse(
        executive_summary=executive_summary,
        key_findings=tuple(key_findings),
        recommendations=tuple(recommendations),
        thirty_day_plan=tuple(plan),
        assumptions=assumptions,
        limitations=limitations,
        confidence=confidence,
        evidence_references=tuple(references),
    )
    return ValidationResult(
        response=response,
        errors=tuple(errors),
        raw_text=raw_text,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> Any:
    """Return the first balanced top-level JSON object in ``text``.

    LLMs commonly wrap JSON in prose ("Sure, here is the answer:\n```json\n{...}\n```")
    or emit prose with a stray trailing closing. ``json.loads`` on the
    entire string fails; we find the first ``{`` and walk the depth.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    # Fast path — the whole string is JSON.
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        pass

    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except (ValueError, TypeError):
                    return None
    return None


def _clamp_str(value: Any, field_name: str, errors: list[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{field_name} must be a string, got {type(value).__name__}")
        return str(value)[:_MAX_TEXT_LEN]
    if len(value) > _MAX_TEXT_LEN:
        errors.append(f"{field_name} truncated to {_MAX_TEXT_LEN} chars")
        return value[:_MAX_TEXT_LEN - 1].rstrip() + "…"
    return value


def _clamp_int(
    value: Any, lo: int, hi: int, field_name: str, errors: list[str],
) -> int:
    if value is None:
        return lo
    try:
        n = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} not an integer: {value!r}")
        return lo
    if n < lo:
        errors.append(f"{field_name} clamped up to {lo}")
        return lo
    if n > hi:
        errors.append(f"{field_name} clamped down to {hi}")
        return hi
    return n


def _string_list(
    value: Any, field_name: str, errors: list[str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        errors.append(f"{field_name} is not a list")
        return ()
    out: list[str] = []
    for item in value[:_MAX_LIST_LEN]:
        s = _clamp_str(item, field_name, errors)
        if s:
            out.append(s)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Input-time guards (called by the service before calling the LLM)
# --------------------------------------------------------------------------- #


def cap_user_prompt(text: str) -> tuple[str, bool]:
    """Return ``(clipped, was_truncated)``.

    Docx P3 Part 6: "Very long prompts" — refuse to send the model a
    runaway prompt. Truncate at ``_MAX_PROMPT_LEN`` and signal that the
    caller should inform the user.
    """
    if not text:
        return text, False
    if len(text) <= _MAX_PROMPT_LEN:
        return text, False
    return text[:_MAX_PROMPT_LEN - 1].rstrip() + "…", True
