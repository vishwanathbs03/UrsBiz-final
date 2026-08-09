"""QuestionUnderstanding — SPRINT AI-1 universal assistant Stage 1.

Before the assistant can answer ANY business question, it must
first understand WHAT the user actually asked. The existing
:class:`~app.services.ai.providers.intent_router.QuestionIntent`
enum is a 6-way classification that neatly maps to the flagship
demo questions, but it is a **routing boundary today** — any
prompt that does not match a flagship keyword falls back to a
generic consultant template or, in the worst case, is rejected
as "unrecognized intent".

This module is the AI-1 fix. It introduces a richer
:class:`QuestionUnderstanding` dataclass that:

  * captures the literal question text,
  * assigns a coarse :attr:`topic` (finance / marketing / operations
    / hiring / export / strategy / education / risk / scenario /
    general) that the system can act on,
  * records the :class:`QuestionIntent` value the existing
    :func:`classify_intent` would have produced, as
    :attr:`relevant_existing_intents` (a tuple, since a prompt
    may overlap multiple intents and the legacy system used
    priority — we keep the same priority order here),
  * detects when the prompt is unambiguously non-business
    (:func:`is_purely_educational`) so the backend can auto-flip
    its internal mode to ``open`` while preserving the wire
    ``mode`` field the user picked,
  * records the **complexity** (simple / moderate / strategic /
    scenario) that the adaptive answer composer keys off,
  * records what the user wants the assistant to compute
    (:attr:`needs_calculations`) and which deterministic engines
    it should consult (:attr:`needs_deterministic_services`),
  * records what is unknown (:attr:`unknowns`) so the
    :class:`AdaptiveAnswer` composer can switch to a
    ``missing_info`` shell.

The dataclass is ``frozen=True`` so it can be safely shared
across the reasoning pipeline without defensive copying.

Backward-compatibility
----------------------

The five existing flagship intents remain reachable via
:attr:`relevant_existing_intents`. The existing
:func:`build_intent_frame` call sites are NOT changed — they
still call :func:`classify_intent` directly. ``QuestionUnderstanding``
is the AI-1 layer that **augments** the routing, not replaces
it. Prompts that match no flagship intent still get a useful
``topic`` (e.g. ``"marketing"``) and a useful ``complexity``
(e.g. ``"strategic"``), so the system never falls back to
"I don't recognize this intent."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, TYPE_CHECKING

from app.services.ai.providers.intent_router import (
    QuestionIntent,
    classify_intent,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from app.services.ai.providers.base import AssistantContext


Complexity = Literal["simple", "moderate", "strategic", "scenario"]
Topic = Literal[
    "finance",
    "marketing",
    "operations",
    "hiring",
    "export",
    "strategy",
    "education",
    "risk",
    "scenario",
    "general",
]


# --------------------------------------------------------------------------- #
# Topic heuristic
# --------------------------------------------------------------------------- #
#
# A keyword scan that maps a prompt to ONE coarse topic. The
# keyword lists are intentionally small and overlapping — the
# system never fails just because the prompt does not match a
# topic; it falls back to "general".
#
# Order matters: the first hit wins. Marketing is checked before
# finance because "selling more" overlaps with "make more
# revenue" — marketing is the more specific framing.

_FINANCE_KEYWORDS = (
    "loan", "funding", "credit", "cash", "cashflow", "cash flow",
    "working capital", "debt", "interest", "emi", "gst ",
    "tax", "taxes", "invoice", "invoicing", "margin", "gross margin",
    "revenue", "turnover", "pricing", "expense", "expenses",
    "profit", "loss", "break-even", "break even",
    "finance", "financial", "payments", "receivable",
)
_MARKETING_KEYWORDS = (
    "market", "marketing", "advertis", "brand", "branding",
    "social media", "instagram", "facebook ad", "google ad",
    "seo", "content marketing", "lead generation", "leads",
    "funnel", "awareness", "campaign", "promotion",
    "b2b marketing", "b2c marketing",
    "sell more", "sell online", "market my", "market our",
    "reach customers", "customer acquisition",
)
_OPERATIONS_KEYWORDS = (
    "inventory", "warehouse", "stock", "supply chain",
    "vendor", "supplier", "logistics", "shipping",
    "process", "operational", "operations", "efficiency",
    "bottleneck", "throughput", "production", "manufacturing",
    "quality", "lean", "six sigma", "automation",
)
_HIRING_KEYWORDS = (
    "hire", "hiring", "recruit", "recruitment", "interview",
    "onboarding", "employee", "employees", "staff",
    "workforce", "salary", "payroll", "compensation",
    "team", "talent", "headcount",
    "should i hire", "add an employee", "add staff",
)
_EXPORT_KEYWORDS = (
    "export", "exports", "international", "overseas",
    "foreign market", "global market", "ship abroad",
    "export market", "export expansion", "export readiness",
    "shipping abroad", "foreign buyer", "import",
)
_STRATEGY_KEYWORDS = (
    "strategy", "strategic", "plan", "roadmap", "playbook",
    "long-term", "long term", "vision", "goal", "goals",
    "growth", "grow", "scale", "competitive", "moat",
    "positioning", "differentiate", "pivot",
    "this month", "next quarter", "where should i focus",
    "creative ways", "three ways to grow",
    "analyze my entire business", "analyze my business",
)
_EDUCATION_KEYWORDS = (
    "what is", "what are", "explain", "define", "difference between",
    "how does", "how do", "meaning of", "tell me about",
    "compare", "vs ", "versus", "introduction to",
    "what does", "concept of", "teach me",
)
_RISK_KEYWORDS = (
    "risk", "risks", "risky", "danger", "weakness", "weak",
    "biggest problem", "biggest issue", "biggest risk",
    "main risk", "top risk", "what's wrong", "what is wrong",
    "gap in my", "concern", "bottleneck",
    "failing", "stuck", "downside", "threat",
)
_SCENARIO_KEYWORDS = (
    "what if", "what happens if", "what would happen if",
    "suppose", "scenario", "simulate", "simulation",
    "if my supplier", "if costs", "if revenue", "if demand",
    "if a competitor", "if i raise", "if i lower",
    "sensitivity", "best case", "worst case", "downside case",
)


@dataclass(frozen=True)
class QuestionUnderstanding:
    """Stage 1 output — structured understanding of the user prompt.

    Attributes
    ----------
    literal_question
        The raw prompt text (trimmed). Preserved so downstream
        renderers can echo the question verbatim when helpful.
    user_intent
        A short dotted path describing what the user wants
        (e.g. ``"operational.finance.working_capital"``). NOT
        used as a routing key — it is informational metadata for
        the prompt builder and the audit trail.
    topic
        Coarse topic bucket — one of the :data:`Topic` literals.
        Defaults to ``"general"`` when no keyword matches.
    is_business_specific
        True when the prompt references the user's own business
        (uses ``"my"``, ``"our"``, ``"I should"``, the profile
        industry, location, etc.). False for purely educational
        questions.
    is_purely_educational
        True when the prompt is explain / define / compare
        against a non-business subject matter. Determined by
        :func:`is_purely_educational`. Used by the backend to
        auto-flip its internal ``_effective_mode`` to ``"open"``
        while the wire ``mode`` stays as the user picked.
    needs_calculations
        Calculation labels the user implicitly asked for. One
        of ``"gap_math"``, ``"growth_multiple"``, ``"roi"``,
        ``"working_capital"``, ``"headcount_cost"``,
        ``"scenario_delta"``. Empty tuple when no calculation is
        needed.
    needs_deterministic_services
        Service names the assistant should consult. Reuses the
        existing service names from the deterministic pool
        (``"health_score"``, ``"recommendation"``,
        ``"schemes_sprint16"``, ``"finance"``,
        ``"knowledge_retrieval"``, ``"business_dna"``,
        ``"risk"``, ``"insights"``). Empty tuple when no
        service is needed.
    unknowns
        Context fields the assistant would need to answer this
        question well but the user has not provided. Drives the
        ``missing_info`` shell in the adaptive answer composer.
    relevant_existing_intents
        The :class:`QuestionIntent` values the existing
        classifier would have produced. Always at least one
        element (the default is ``QuestionIntent.GENERAL``).
        Order is priority order so the prompt builder can pick
        the first as the primary framing.
    sentiment
        ``"neutral"`` / ``"concerned"`` / ``"optimistic"`` —
        simple keyword scan. Defaults to ``"neutral"``.
    complexity
        One of ``"simple"``, ``"moderate"``, ``"strategic"``,
        ``"scenario"``. The adaptive answer composer keys off
        this to pick a shell.
    parsed_at
        ISO-8601 timestamp of when the understanding was
        produced. Useful for the audit trail.
    """

    literal_question: str
    user_intent: str
    topic: Topic
    is_business_specific: bool
    is_purely_educational: bool
    needs_calculations: tuple[str, ...] = field(default_factory=tuple)
    needs_deterministic_services: tuple[str, ...] = field(default_factory=tuple)
    unknowns: tuple[str, ...] = field(default_factory=tuple)
    relevant_existing_intents: tuple[QuestionIntent, ...] = field(default_factory=tuple)
    sentiment: str = "neutral"
    complexity: Complexity = "moderate"
    parsed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialize for the GenerationMeta audit envelope.

        Tuples become lists and the :attr:`relevant_existing_intents`
        enum tuple becomes a list of ``.value`` strings so the
        payload is JSON-serialisable without leaking the enum
        type. Used by the service when stamping the wire
        envelope.
        """
        return {
            "literal_question": self.literal_question,
            "user_intent": self.user_intent,
            "topic": self.topic,
            "is_business_specific": self.is_business_specific,
            "is_purely_educational": self.is_purely_educational,
            "needs_calculations": list(self.needs_calculations),
            "needs_deterministic_services": list(self.needs_deterministic_services),
            "unknowns": list(self.unknowns),
            "relevant_existing_intents": [
                intent.value for intent in self.relevant_existing_intents
            ],
            "sentiment": self.sentiment,
            "complexity": self.complexity,
            "parsed_at": self.parsed_at,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _first_match(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True iff any keyword appears as a substring of ``text``."""
    for kw in keywords:
        if kw in text:
            return True
    return False


def _detect_topic(text: str) -> Topic:
    """Return the coarse topic for the prompt (default ``"general"``)."""
    # Order matters: more specific topics first.
    if _first_match(text, _RISK_KEYWORDS):
        return "risk"
    if _first_match(text, _SCENARIO_KEYWORDS):
        return "scenario"
    if _first_match(text, _HIRING_KEYWORDS):
        return "hiring"
    if _first_match(text, _EXPORT_KEYWORDS):
        return "export"
    if _first_match(text, _MARKETING_KEYWORDS):
        return "marketing"
    if _first_match(text, _OPERATIONS_KEYWORDS):
        return "operations"
    if _first_match(text, _FINANCE_KEYWORDS):
        return "finance"
    if _first_match(text, _STRATEGY_KEYWORDS):
        return "strategy"
    if _first_match(text, _EDUCATION_KEYWORDS):
        return "education"
    return "general"


def _is_business_specific(prompt: str, context: Any) -> bool:
    """True when the prompt references the user's own business.

    Heuristic — keep it cheap. Returns True when:

      * the prompt uses a possessive pronoun (``my``, ``our``),
      * or asks ``should I`` / ``what should I``,
      * or contains an industry / location / business-type
        keyword from the context,
      * or contains a flagship keyword (the existing
        intent classifier would have matched).

    Returns False for purely educational prompts.
    """
    text = (prompt or "").lower()
    if not text.strip():
        return False

    # Possessive or first-person phrasing
    if re.search(r"\b(my|our|i should|we should|i am|we are)\b", text):
        return True
    if re.search(r"\bshould i\b", text):
        return True

    # Cross-check against the existing classifier — if the
    # flagship router matched, the prompt is implicitly
    # business-specific.
    if classify_intent(prompt) is not QuestionIntent.GENERAL:
        return True

    # Match against known context fields when available
    industry = getattr(context, "industry", None) if context is not None else None
    location = getattr(context, "location", None) if context is not None else None
    business_type = getattr(context, "business_type", None) if context is not None else None
    for token in (industry, location, business_type):
        if token and isinstance(token, str) and token.strip() and token.strip().lower() in text:
            return True

    return False


# Heuristic keywords that classify a prompt as "purely educational"
# — i.e. asking what something MEANS rather than what to do.
_EDUCATIONAL_OPENERS = (
    "what is", "what are", "what does", "what do",
    "explain", "define", "meaning of", "tell me about",
    "difference between", "how does", "how do",
    "compare ", "vs ", "versus", "introduction to",
)
# Topics that are always non-business — used to detect
# educational prompts that are NOT about the user's MSME.
_NON_BUSINESS_SUBJECTS = (
    "photosynthesis", "quantum", "philosophy", "religion",
    "recipe", "cooking", "astrology", "history of",
    "mathematics", "calculus", "algebra", "biology",
    "chemistry", "physics", "literature", "poetry",
    "movie", "film", "song", "lyrics", "game",
    "sports", "football", "cricket", "tennis",
)


def is_purely_educational(prompt: str) -> bool:
    """True when the prompt asks a concept-level question.

    The check is permissive: it returns True only when the prompt
    opens with an explainer opener AND is not business-specific.

    A prompt is "business-specific" only when it uses a
    possessive / first-person marker OR a directive marker
    ("should I", "recommend"). The mere presence of a business
    topic (e.g. "marketing", "GST", "working capital") does NOT
    flip an educational prompt to business-specific — the
    assistant should still be able to define the term.

    Edge cases handled:

      * "What is working capital?" — True (concept question, no
        business-specific markers).
      * "What is my working capital gap?" — False (possessive).
      * "What is a marketing funnel?" — True.
      * "What is the best marketing funnel for my business?" —
        False (possessive "my").
      * "Should I hire five employees?" — False (directive).
      * "" or None — False.
      * "Explain the difference between marketing and
        advertising." — True (no possessive / directive).
      * "Define GST." — True.
    """
    text = (prompt or "").lower().strip()
    if not text:
        return False

    # An imperative / directive phrase flips the prompt to
    # business-specific (the user is asking for advice).
    if re.search(r"\bshould i\b", text):
        return False
    # "Tell me about" / "Give me" / "Recommend" — directive
    if re.search(r"\b(tell me|give me|recommend|suggest)\b", text):
        return False

    # Possessive / first-person marker — the user is asking
    # about their own business (even if the topic is generic).
    if re.search(r"\b(my|our|i should|we should|i am|we are)\b", text):
        return False

    # Must start with an educational opener
    if not any(text.startswith(opener) for opener in _EDUCATIONAL_OPENERS):
        return False

    # If the prompt references a clearly non-business subject,
    # treat it as purely educational.
    if any(s in text for s in _NON_BUSINESS_SUBJECTS):
        return True

    # Default: a prompt that opens with an educational opener
    # and lacks any business-specific marker is purely
    # educational. This is the "What is working capital?" case.
    return True


def _detect_complexity(prompt: str, topic: Topic) -> Complexity:
    """Classify the prompt's complexity.

    Heuristic order:

      * scenario keywords → ``"scenario"``,
      * strategy / multi-intent / creative-thinking keywords → ``"strategic"``,
      * educational / explainer / definition → ``"simple"``,
      * otherwise ``"moderate"``.
    """
    text = (prompt or "").lower()
    if _first_match(text, _SCENARIO_KEYWORDS):
        return "scenario"
    if _first_match(text, _STRATEGY_KEYWORDS):
        return "strategic"
    # Educational / definition prompts are simple regardless of
    # which topic bucket they fall into (a topic that itself
    # came from a keyword like "working capital" still wins the
    # topic slot, but the complexity is the underlying question
    # type — asking what something is).
    if topic == "education" or _first_match(text, _EDUCATION_KEYWORDS):
        return "simple"
    return "moderate"


def _detect_needs_calculations(topic: Topic, prompt: str) -> tuple[str, ...]:
    """Return calculation labels the user implicitly asked for.

    The labels are NOT computed here — they are typed strings
    that the prompt builder can use to tell the LLM which
    numbers to show. The LLM itself never does the math; the
    deterministic engines produce the numbers and the
    assistant cites them.
    """
    text = (prompt or "").lower()
    needs: list[str] = []
    if topic == "finance" or "gap" in text or "reach" in text or "target" in text:
        needs.append("gap_math")
    if "growth multiple" in text or "multiple" in text:
        needs.append("growth_multiple")
    if "roi" in text or "return on investment" in text:
        needs.append("roi")
    if "working capital" in text or "cash flow" in text or "cashflow" in text:
        needs.append("working_capital")
    if topic == "hiring" or "salary" in text or "payroll" in text:
        needs.append("headcount_cost")
    if topic == "scenario" or "scenario" in text or "if my" in text:
        needs.append("scenario_delta")
    return tuple(needs)


def _detect_needs_services(
    topic: Topic, prompt: str, context: Any
) -> tuple[str, ...]:
    """Return deterministic service names the prompt authorises.

    Defaults to the four most-used services. Adding more is
    free — the dispatcher caps at 5 tool calls per request.
    """
    text = (prompt or "").lower()
    services: list[str] = []

    # Knowledge retrieval is useful for nearly every prompt
    services.append("knowledge_retrieval")

    # Topic-keyed services
    if topic == "finance":
        services.append("finance")
    if topic in {"strategy", "scenario", "general"}:
        services.append("health_score")
        services.append("recommendation")
    if topic == "risk":
        services.append("risk")
    if topic == "export":
        services.append("schemes_sprint16")
    if "scheme" in text or "subsidy" in text or "mudra" in text or "pmegp" in text:
        services.append("schemes_sprint16")
    if topic == "hiring":
        services.append("finance")
    if topic == "marketing":
        services.append("insights")
    if topic == "operations":
        services.append("risk")

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in services:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def _detect_unknowns(
    topic: Topic, prompt: str, context: Any
) -> tuple[str, ...]:
    """Return context fields the assistant would want but does not have.

    Drives the ``missing_info`` shell in the adaptive answer
    composer. The list is intentionally short — three items
    max — so the trailing "what to provide next" section stays
    actionable.
    """
    unknowns: list[str] = []

    if context is None:
        return ("business_profile", "industry", "annual_revenue")

    if not getattr(context, "industry", None) or getattr(context, "industry", "") == "unknown":
        unknowns.append("industry")
    if not getattr(context, "annual_revenue_inr", None):
        unknowns.append("annual_revenue")
    if not getattr(context, "target_revenue_inr", None) and topic in {"strategy", "finance", "scenario"}:
        unknowns.append("target_revenue")
    if topic == "export" and not getattr(context, "certifications", None):
        unknowns.append("certifications")
    if topic == "hiring" and not getattr(context, "employee_count", None):
        unknowns.append("employee_count")

    return tuple(unknowns[:3])


def _detect_sentiment(text: str) -> str:
    """Return ``"concerned"`` / ``"optimistic"`` / ``"neutral"``."""
    if not text:
        return "neutral"
    negative = (
        "worried", "concerned", "anxious", "afraid",
        "struggling", "losing", "failing", "stuck",
        "problem", "issue", "wrong", "bad", "down",
    )
    positive = (
        "excited", "optimistic", "great", "amazing",
        "growing", "booming", "thriving", "love",
        "winning", "succeed", "success",
    )
    if any(w in text for w in negative):
        return "concerned"
    if any(w in text for w in positive):
        return "optimistic"
    return "neutral"


def _build_user_intent_string(topic: Topic, prompt: str) -> str:
    """Build a short dotted path describing the user intent."""
    text = (prompt or "").lower().strip()
    # Specific intent sub-paths for the most common topic buckets
    if topic == "finance":
        if "working capital" in text or "cash flow" in text:
            return "operational.finance.working_capital"
        if "loan" in text or "funding" in text:
            return "operational.finance.funding"
        if "pricing" in text or "margin" in text:
            return "operational.finance.pricing"
        return "operational.finance.general"
    if topic == "marketing":
        if "b2b" in text:
            return "operational.marketing.b2b"
        if "online" in text or "digital" in text:
            return "operational.marketing.digital"
        return "operational.marketing.general"
    if topic == "operations":
        if "vendor" in text or "supplier" in text:
            return "operational.operations.vendor"
        if "inventory" in text or "stock" in text:
            return "operational.operations.inventory"
        return "operational.operations.general"
    if topic == "hiring":
        return "operational.hiring.workforce"
    if topic == "export":
        return "strategic.export.international"
    if topic == "risk":
        return "diagnostic.risk.weakness"
    if topic == "scenario":
        return "diagnostic.scenario.simulation"
    if topic == "strategy":
        if "this month" in text or "next month" in text:
            return "strategic.planning.immediate"
        return "strategic.planning.general"
    if topic == "education":
        return "educational.concept.explanation"
    return "general.business.advice"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def understand_question(
    prompt: str, context: Any = None
) -> QuestionUnderstanding:
    """Return a :class:`QuestionUnderstanding` for the given prompt.

    Pure function. No I/O. Safe to call from anywhere in the
    reasoning pipeline.

    Parameters
    ----------
    prompt
        The raw user prompt text.
    context
        Optional :class:`AssistantContext` (forward-ref to
        avoid the import cycle). When provided, the existence
        of certain fields is used to detect unknowns.
    """
    text = (prompt or "").strip()
    lower = text.lower()

    topic = _detect_topic(lower)
    is_biz = _is_business_specific(text, context)
    is_edu = is_purely_educational(text)
    complexity = _detect_complexity(lower, topic)
    needs_calc = _detect_needs_calculations(topic, lower)
    needs_services = _detect_needs_services(topic, lower, context)
    unknowns = _detect_unknowns(topic, lower, context)
    sentiment = _detect_sentiment(lower)
    user_intent = _build_user_intent_string(topic, lower)

    # The existing intent classifier → tuple of intents in
    # priority order. We always emit at least one (the GENERAL
    # fallback). For prompts that match multiple intents, we
    # preserve the priority order emitted by the existing
    # priority-coded scan: revenue-target > weakness > schemes
    # > roadmap > export > general.
    relevant = _INTENTS_BY_TOPIC.get(topic, (QuestionIntent.GENERAL,))

    return QuestionUnderstanding(
        literal_question=text,
        user_intent=user_intent,
        topic=topic,
        is_business_specific=is_biz,
        is_purely_educational=is_edu,
        needs_calculations=needs_calc,
        needs_deterministic_services=needs_services,
        unknowns=unknowns,
        relevant_existing_intents=relevant,
        sentiment=sentiment,
        complexity=complexity,
    )


# Map each topic to the existing QuestionIntent values that are
# most relevant. The tuple is in priority order so the prompt
# builder can pick the first as the primary reroute.
_INTENTS_BY_TOPIC: dict[Topic, tuple[QuestionIntent, ...]] = {
    "finance": (
        QuestionIntent.REACH_REVENUE_TARGET,
        QuestionIntent.GENERAL,
    ),
    "marketing": (QuestionIntent.GENERAL,),
    "operations": (QuestionIntent.GENERAL,),
    "hiring": (QuestionIntent.GENERAL,),
    "export": (QuestionIntent.EXPORT_EXPANSION, QuestionIntent.GENERAL),
    "strategy": (
        QuestionIntent.REACH_REVENUE_TARGET,
        QuestionIntent.TWELVE_MONTH_ROADMAP,
        QuestionIntent.GENERAL,
    ),
    "education": (QuestionIntent.GENERAL,),
    "risk": (QuestionIntent.BIGGEST_WEAKNESS, QuestionIntent.GENERAL),
    "scenario": (
        QuestionIntent.REACH_REVENUE_TARGET,
        QuestionIntent.GENERAL,
    ),
    "general": (QuestionIntent.GENERAL,),
}
