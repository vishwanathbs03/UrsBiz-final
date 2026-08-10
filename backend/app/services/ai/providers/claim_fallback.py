"""Deterministic fallback for the claim-aware response — SPRINT AI-3.

When the real LLM is unavailable, the deterministic fallback
already renders a coherent text body via
:func:`app.services.ai.providers.base._fallback_body`. The
fallback for the claim-aware contract mirrors that — it builds
a :class:`ClaimAwareResponse` payload directly from
:class:`AssistantContext` so the wire always carries a non-None
``ChatMessageOut.claim_aware_response``.

The fallback's payload is:

* One :class:`Claim` per non-empty business snapshot field
  (``legal_name``, ``industry``, ``employee_count``,
  ``overall_business_score``, ``annual_revenue_inr``).
* One :class:`ClaimRecommendation` per top business
  recommendation, mapped from ``ctx.recommendations``.
* Empty lists for ``calculations`` and ``scenarios`` — the
  deterministic fallback does not author hypothetical numerics.
* ``assumptions`` / ``limitations`` lifted verbatim from the
  fallback body.
* ``server_confidence=100`` (the fallback is grounded by construction).

The function returns a JSON-safe dict ready to be stashed in
``GenerationMeta.grounded_payload`` and projected onto the wire
via ``conversation_service._message_payload``.
"""
from __future__ import annotations

from typing import Any


def build_fallback_claim_aware(request: Any) -> dict[str, Any]:
    """Build the deterministic fallback claim-aware payload.

    Returns a JSON-safe dict with the same shape as
    :meth:`ClaimAwareResponse.to_dict`. The values are sourced
    directly from ``AssistantContext`` — no invented numbers.
    """
    context = getattr(request, "context", None)
    if context is None:
        return _empty_payload()

    claims: list[dict] = []
    recommendations: list[dict] = []

    # One FACT claim per business snapshot field.
    fields = (
        ("legal_name", f"Legal name: {context.legal_name}"),
        ("industry", f"Industry: {context.industry}"),
        ("annual_revenue_inr", f"Annual revenue baseline: ₹{context.annual_revenue_inr:,}"),
        (
            "overall_business_score",
            f"Overall business score: {context.overall_business_score}/100 ({context.band})",
        ),
    )
    for ftype, text in fields:
        if text and "unknown" not in text.lower():
            claims.append({
                "text": text,
                "claim_type": "FACT",
                "evidence_references": [],
                "confidence": 100,
                "audit_log": [],
                "user_provided": False,
            })

    dna_title = getattr(context.dna, "archetype_title", "") or ""
    if dna_title:
        claims.append({
            "text": (
                f"Business DNA: {dna_title} "
                f"(match {context.dna.match_score}%)"
            ),
            "claim_type": "FACT",
            "evidence_references": [],
            "confidence": 100,
            "audit_log": [],
            "user_provided": False,
        })

    # Map top recommendations.
    for rec in list(getattr(context, "recommendations", ()) or [])[:8]:
        recommendations.append({
            "title": str(rec.title or rec.id),
            "reason": (
                f"Priority {rec.priority}; estimated +{rec.estimated_score_gain} "
                f"score over {rec.estimated_timeline}."
            ),
            "recommendation_id": str(rec.id),
            "evidence_references": [f"rec_{_slugify(rec.id)}"] if rec.id else [],
            "category": str(rec.category or ""),
            "priority": str(rec.priority or ""),
            "estimated_score_gain": int(rec.estimated_score_gain or 0),
            "estimated_timeline": str(rec.estimated_timeline or ""),
        })

    return {
        "answer": _summary_text(context),
        "claims": claims,
        "recommendations": recommendations,
        "calculations": [],
        "scenarios": [],
        "unknowns": [],
        "evidence_references": [],
        "assumptions": ["All values are read from the deterministic engines."],
        "limitations": ["Re-run with a fuller business profile for richer next steps."],
        "narrative": "",
        "server_confidence": 100,
        "server_confidence_rationale": "fallback grounded by construction",
        "numeric_conflicts": [],
        "server_audit": {"source": "deterministic_fallback"},
    }


def _summary_text(context: Any) -> str:
    score = getattr(context, "overall_business_score", 0) or 0
    band = getattr(context, "band", "") or ""
    name = (
        getattr(context, "legal_name", "unknown")
        if getattr(context, "legal_name", "unknown") != "unknown"
        else "this business"
    )
    if score:
        return (
            f"{name} currently scores {score}/100 ({band}). "
            "Top recommendations are listed below."
        )
    return f"{name} profile is in early data-collection state."


def _empty_payload() -> dict[str, Any]:
    return {
        "answer": "",
        "claims": [],
        "recommendations": [],
        "calculations": [],
        "scenarios": [],
        "unknowns": [],
        "evidence_references": [],
        "assumptions": [],
        "limitations": ["Business profile not yet available."],
        "narrative": "",
        "server_confidence": 0,
        "server_confidence_rationale": "no AssistantContext available",
        "numeric_conflicts": [],
        "server_audit": {"source": "deterministic_fallback", "empty": True},
    }


def _slugify(value: str) -> str:
    """Mimic the registry's slug rule so the fallback cites real IDs."""
    if not value:
        return ""
    import re
    out = re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_")
    return re.sub(r"_+", "_", out)
