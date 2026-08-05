"""H7.3 — Docx Prompt 3 Part 3 response schema and validator.

The docx requires the generative response to follow this structure:

    {
      "executive_summary": "",
      "key_findings": [],
      "recommendations": [],
      "thirty_day_plan": [],
      "scheme_matches": [],
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

H7.8C changes
-------------

  * ``Recommendation`` no longer carries model-authored
    ``priority`` / ``score_gain`` / ``title``. The model now
    cites a real recommendation ID; the registry supplies the
    authoritative fields at enrichment time
    (see ``enrich_with_resolved_fields``).
  * ``Recommendation`` carries ``evidence_refs`` so the
    :class:`GroundingValidator` can verify each citation.
  * ``PlanItem`` carries ``recommendation_ref`` + ``evidence_refs``.
  * ``SchemeMatch`` is a new section the model can populate
    with ``scheme_ref`` (a real scheme ID from the registry)
    plus an explanatory paragraph.
  * The whole ``GroundedResponse`` is now designed to be
    validated by ``app.services.ai.providers.grounding_validator``.
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
    """One action item — H7.8C ID-referenced.

    H7.8C removes model authority over ``priority`` /
    ``score_gain`` / ``title`` / ``timeline``. The model
    cites a real recommendation ID; the registry supplies
    the authoritative fields at enrichment time. The
    dataclass still keeps ``title`` and ``rationale`` for
    backwards compatibility (the model will still author
    those — they are descriptive, not authoritative), but
    the validator does NOT use them to score the response.
    Only ``recommendation_id`` + ``evidence_refs`` are
    checked.

    ``recommendation_id`` MUST resolve to a registry entry
    of kind ``recommendation`` (the validator enforces
    this). ``evidence_refs`` MUST contain at least one ID
    that resolves.
    """

    recommendation_id: str
    title: str
    rationale: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlanItem:
    """One week-by-week task inside the 30-day plan.

    H7.8C adds ``recommendation_ref`` (optional — a registry
    entry of kind ``recommendation`` the task implements)
    and ``evidence_refs`` (registry IDs that justify the
    task). Tasks without a ``recommendation_ref`` are still
    permitted; the validator only checks them when present.
    """

    week: int  # 1..4
    task: str
    recommendation_ref: str | None = None
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SchemeMatch:
    """One scheme profile match the model surfaced.

    The model explains why the scheme is a match for the
    user's profile. The match score and eligibility
    determination are NOT authored by the model — the
    registry's ``profile_match_score`` is canonical and is
    surfaced in the rendered payload without ever being
    rewritten.
    """

    scheme_ref: str
    match_explanation: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)


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
    """The fully validated response envelope.

    H7.8C additions over H7.3:

      * ``scheme_matches`` — populated by the model when it
        wants to flag a real scheme the user should look up.
        The validator confirms each ``scheme_ref`` resolves.
      * ``server_grounding_score`` (0..100) — computed by
        :class:`GroundingValidator` after the model output
        is parsed. The wire envelope carries the score so
        the UI can display confidence as the lower of the
        model-reported ``confidence`` and the server score.
    """

    executive_summary: str
    key_findings: tuple[KeyFinding, ...] = field(default_factory=tuple)
    recommendations: tuple[Recommendation, ...] = field(default_factory=tuple)
    thirty_day_plan: tuple[PlanItem, ...] = field(default_factory=tuple)
    scheme_matches: tuple[SchemeMatch, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    confidence: int = 0
    evidence_references: tuple[EvidenceReference, ...] = field(default_factory=tuple)
    server_grounding_score: int | None = None

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
                ref = f" [{r.recommendation_id}]" if r.recommendation_id else ""
                lines.append(
                    f"  {i}.{ref} {r.title} — {r.rationale}"
                )
            lines.append("")

        if self.thirty_day_plan:
            lines.append("30-day plan")
            for p in sorted(self.thirty_day_plan, key=lambda x: x.week):
                ref = f" [{p.recommendation_ref}]" if p.recommendation_ref else ""
                lines.append(f"  Week {p.week}{ref}: {p.task}")
            lines.append("")

        if self.scheme_matches:
            lines.append("Scheme profile matches")
            for i, sm in enumerate(self.scheme_matches, start=1):
                lines.append(
                    f"  {i}. [{sm.scheme_ref}] {sm.match_explanation}"
                )
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

        if self.server_grounding_score is not None:
            lines.append(
                f"Server grounding score: {self.server_grounding_score}/100"
            )
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
        # H7.8C — primary identifier is recommendation_id.
        # We still accept legacy ``id`` for backward
        # compatibility with H7.3 outputs.
        rec_id = _clamp_str(
            item.get("recommendation_id")
            or item.get("id"),
            "recommendation.id",
            errors,
        )
        title = _clamp_str(item.get("title"), "recommendation.title", errors)
        rationale = _clamp_str(item.get("rationale"), "recommendation.rationale", errors)
        refs_raw = item.get("evidence_refs") or []
        if not isinstance(refs_raw, list):
            refs_raw = []
        evidence_refs = tuple(str(r) for r in refs_raw[:_MAX_LIST_LEN])
        # Drop the row when neither an ID nor a title is
        # present — the legacy parser dropped only on empty
        # title. H7.8C drops on empty id AND empty title so
        # we don't keep a free-text recommendation that
        # bypasses the registry.
        if not rec_id and not title:
            continue
        recommendations.append(Recommendation(
            recommendation_id=rec_id,
            title=title,
            rationale=rationale,
            evidence_refs=evidence_refs,
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
        rec_ref = _clamp_str(
            item.get("recommendation_ref"),
            "plan.recommendation_ref",
            errors,
        )
        refs_raw = item.get("evidence_refs") or []
        if not isinstance(refs_raw, list):
            refs_raw = []
        plan.append(PlanItem(
            week=week,
            task=task,
            recommendation_ref=rec_ref or None,
            evidence_refs=tuple(str(r) for r in refs_raw[:_MAX_LIST_LEN]),
        ))

    scheme_raw = parsed.get("scheme_matches") or []
    if not isinstance(scheme_raw, list):
        errors.append("scheme_matches is not a list")
        scheme_raw = []
    scheme_matches: list[SchemeMatch] = []
    for item in scheme_raw[:_MAX_LIST_LEN]:
        if not isinstance(item, dict):
            continue
        sref = _clamp_str(
            item.get("scheme_ref") or item.get("id"),
            "scheme.scheme_ref",
            errors,
        )
        if not sref:
            continue
        explanation = _clamp_str(
            item.get("match_explanation")
            or item.get("explanation")
            or item.get("rationale"),
            "scheme.match_explanation",
            errors,
        )
        refs_raw = item.get("evidence_refs") or []
        if not isinstance(refs_raw, list):
            refs_raw = []
        scheme_matches.append(SchemeMatch(
            scheme_ref=sref,
            match_explanation=explanation,
            evidence_refs=tuple(str(r) for r in refs_raw[:_MAX_LIST_LEN]),
        ))

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
        scheme_matches=tuple(scheme_matches),
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
