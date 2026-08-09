"""Pydantic schemas for the chat persistence endpoints (Sprint 7 Part 3).

The endpoint surface is:

  POST   /api/v1/chat                 create a new conversation
  GET    /api/v1/chat                 list the user's conversations
  GET    /api/v1/chat/{id}            fetch a conversation + messages
  DELETE /api/v1/chat/{id}            delete a conversation
  POST   /api/v1/chat/{id}/message    append a user message, get a reply

All response models use :class:`pydantic.BaseModel` with
``model_config = ConfigDict(extra="forbid")`` so an upstream
refactor that adds a new field fails loudly at the API boundary
instead of silently shipping a shape the UI does not know how to
render.

Every schema is **owner-scoped** at the endpoint layer — a
conversation belongs to the authenticated user, and a request for
another user's conversation returns 404, never 403, so the
resource's existence is not leaked.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Conversation source — what the assistant reply drew on
# --------------------------------------------------------------------------- #


class ChatSource(BaseModel):
    """One source the assistant reply leaned on."""

    model_config = ConfigDict(extra="forbid")

    topic: Literal[
        "Twin",
        "Recommendations",
        "Roadmap",
        "Insights",
        "Rules",
        "Business DNA",
        "Export",
        # Sprint 7 Part 4 — knowledge retrieval sources.
        "Knowledge",
        "Rule",
        "Recommendation",
        "GovernmentScheme",
        "Glossary",
    ]
    detail: str = Field(min_length=1, max_length=500)


# --------------------------------------------------------------------------- #
# H7.8C — provenance envelope
# --------------------------------------------------------------------------- #


class ChatEvidenceReference(BaseModel):
    """One pointer back to an upstream service that produced a fact.

    The model cites these IDs in its response; the registry
    resolves them against the live upstream payloads.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    kind: Literal[
        "score",
        "recommendation",
        "rule",
        "insight",
        "scheme",
        "forecast",
        "action",
        "dna",
    ] = "score"
    label: str = Field(default="", max_length=200)


class ChatGroundedFinding(BaseModel):
    """One short bullet the model surfaced."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(default="", max_length=400)
    evidence_refs: list[str] = Field(default_factory=list)


class ChatGroundedRecommendation(BaseModel):
    """A recommendation the model authored in grounded mode.

    H7.8C — ``recommendation_id`` MUST resolve to an
    ``EvidenceKind.RECOMMENDATION`` entry in the registry.
    The legacy ``priority`` / ``score_gain`` fields are kept
    as the *resolved* values from the registry — the model
    never authors them.
    """

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(min_length=1, max_length=120)
    title: str = Field(default="", max_length=200)
    rationale: str = Field(default="", max_length=500)
    priority: Literal["Critical", "High", "Medium", "Low"] | None = None
    score_gain: int | None = Field(default=None, ge=0, le=100)
    evidence_refs: list[str] = Field(default_factory=list)


class ChatGroundedPlanItem(BaseModel):
    """A week-by-week task inside the 30-day plan."""

    model_config = ConfigDict(extra="forbid")

    week: Literal[1, 2, 3, 4]
    task: str = Field(min_length=1, max_length=240)
    recommendation_ref: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class ChatGroundedSchemeMatch(BaseModel):
    """A scheme the model flagged as a profile match.

    The match score and eligibility determination are NEVER
    authored by the model. They come from the registry.
    """

    model_config = ConfigDict(extra="forbid")

    scheme_ref: str = Field(min_length=1, max_length=120)
    match_explanation: str = Field(default="", max_length=400)
    profile_match_score: int | None = Field(default=None, ge=0, le=100)
    authority: str = Field(default="", max_length=200)
    evidence_refs: list[str] = Field(default_factory=list)


class ChatGroundedResponse(BaseModel):
    """The structured payload from a real grounded-mode LLM call."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(default="", max_length=600)
    key_findings: list[ChatGroundedFinding] = Field(default_factory=list)
    recommendations: list[ChatGroundedRecommendation] = Field(default_factory=list)
    thirty_day_plan: list[ChatGroundedPlanItem] = Field(default_factory=list)
    scheme_matches: list[ChatGroundedSchemeMatch] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: int = Field(ge=0, le=100, default=0)
    server_grounding_score: int = Field(ge=0, le=100, default=0)
    evidence_references: list[ChatEvidenceReference] = Field(default_factory=list)


class ChatGenerationMeta(BaseModel):
    """The full provenance envelope persisted with every assistant turn.

    See :class:`app.services.ai.providers.base.GenerationMeta`
    for the canonical definition. This is the wire mirror.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=120)
    runtime_provider: str = Field(default="", max_length=80)
    """H7.8C — the runtime provider that actually answered the
    request. Equal to ``provider`` for the deterministic
    fallback path; equal to ``provider`` for any real call.
    The field is differentiated from ``provider`` so the wire
    payload can carry it even when the configured provider name
    and the runtime provider name diverge. Defaults to ``""``
    for legacy rows that pre-date H7.8C."""
    mode: Literal["grounded", "open"] = "grounded"
    fallback_used: bool
    fallback_reason: Literal[
        "provider_unavailable",
        "timeout",
        "rate_limited",
        "provider_error",
        "http_4xx",
        "http_5xx",
        "malformed_response",
        "empty_response",
        "schema_invalid",
        "grounding_invalid",
        "not_configured",
        "open_mode_provider_failure",
    ] | None = None
    generation_method: Literal["generative", "deterministic"]
    schema_validated: bool = False
    grounding_validated: bool = False
    server_grounding_score: int = Field(ge=0, le=100, default=0)
    evidence_count: int = Field(ge=0, default=0)
    confidence: int | None = Field(default=None, ge=0, le=100)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    generated_at: str = Field(min_length=1, max_length=80)
    prompt_truncated: bool = False
    provider_latency_ms: int | None = Field(default=None, ge=0)
    grounded_payload: ChatGroundedResponse | None = None
    business_evidence_validated: bool = False
    context_manifest: dict | None = None

    # AI-1 — universal-assistant audit trail fields. Mirrors
    # of the GenerationMeta dataclass fields. Each new field
    # has a safe default so legacy rows deserialize cleanly.
    # The Pydantic ``extra="forbid"`` config would otherwise
    # reject these when the persistence layer emits them.
    deterministic_services_used: list[str] = Field(default_factory=list)
    calculations_used: list[str] = Field(default_factory=list)
    question_understanding: dict | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    claim_categories_used: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


class ChatMessageOut(BaseModel):
    """One message in a conversation. Returned by GET /chat/{id}.

    H7.8C — assistant responses now expose the full provenance
    envelope at the TOP level of the message payload, not only
    nested inside ``generation``. Every brief-mandated field
    (provider, model, runtime provider, grounding score,
    evidence references, assumptions, limitations, fallback
    active, mode, confidence) is now a top-level field on this
    model so the frontend can render the trust disclosure without
    drilling into ``generation.*``.

    Backward compatibility
    -----------------------

    * All new fields are **optional** (defaulting to safe
      empties) so older clients that only read ``id``,
      ``role``, ``content`` still parse the response.
    * The ``generation`` block is preserved unchanged for any
      client that already reads it.
    * User turns and legacy rows that pre-date the
      ``generation_meta_json`` column return the new fields as
      empty / ``None`` — exactly the same contract v1 had.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    role: Literal["user", "assistant"]
    kind: str = ""
    content: str = Field(min_length=1)
    sources: list[ChatSource] = Field(default_factory=list)
    created_at: datetime

    # H7.8A P2 — per-message fallback flag. The frontend uses this to
    # decide whether to render the bubble with the
    # "Calculated by UrsBiz rule engine" trust label (True) or the
    # "Generated explanation" label (False, only when a real LLM
    # answered). Defaults to False so user messages and older rows
    # remain valid.
    fallback_used: bool = False

    # H7.8C — full provenance envelope for assistant turns. None
    # on user turns and on legacy rows that pre-date the
    # ``generation_meta_json`` column.
    generation: ChatGenerationMeta | None = None

    # ---- H7.8C — top-level provenance fields ------------------------- #
    #
    # Every field below is a flat mirror of the matching
    # ``generation.*`` value. The frontend trust disclosure
    # (provider name, model, grounding score, evidence count,
    # assumptions, limitations, mode, confidence, fallback
    # active) is now reachable without parsing the
    # ``generation`` block.
    #
    # The mirrors are NEVER derived from a hidden source — they
    # are read from the same ``GenerationMeta`` the assistant
    # layer stamps on every reply. If a mirror disagrees with
    # ``generation.*`` the renderer should prefer ``generation``
    # (the structured envelope is the authoritative source).

    provider: str = Field(default="", max_length=80)
    """The model-side ``provider`` field. ``"deterministic-fallback"``
    when the deterministic fallback answered; ``"openai_compatible"``
    / ``"ollama"`` when a real provider answered."""

    model: str = Field(default="", max_length=120)
    """The model identifier the provider stamped on the response."""

    runtime_provider: str = Field(default="", max_length=80)
    """H7.8C — the runtime provider that actually answered the
    request. Equal to ``provider`` for the deterministic
    fallback path; equal to ``provider`` for any real call
    (the value is differentiated from ``provider`` so the
    frontend can render the "trusted runtime" badge
    separately from the "configured provider" label)."""

    grounding_score: int = Field(default=0, ge=0, le=100)
    """The server-computed grounding score (0–100). Mirrors
    ``generation.server_grounding_score``. The deterministic
    fallback always reports 100 (grounded by construction);
    a real provider's score reflects how many data sources
    the answer cited."""

    evidence_references: list[str] = Field(default_factory=list)
    """Mirrors ``generation.evidence_references`` — the list
    of Evidence Registry IDs the answer cited."""

    assumptions: list[str] = Field(default_factory=list)
    """Mirrors ``generation.assumptions`` — the assumptions the
    provider / fallback surfaced with the answer."""

    limitations: list[str] = Field(default_factory=list)
    """Mirrors ``generation.limitations`` — the limitations the
    provider / fallback surfaced with the answer."""

    fallback_active: bool = False
    """H7.8C — same semantic as ``fallback_used`` but exposed
    under the brief-mandated name. Always equals ``fallback_used``
    on this message (kept as a separate field so the frontend
    can branch on either name)."""

    mode: Literal["grounded", "open"] | None = None
    """The mode the assistant ran under. ``None`` for user turns
    and for legacy rows that pre-date the column."""

    confidence: int | None = Field(default=None, ge=0, le=100)
    """Mirrors ``generation.confidence`` — the provider's
    self-reported confidence (0–100). ``None`` when the
    provider did not emit one (the deterministic fallback
    fills it with the configured value)."""

    # AI-1 — universal-assistant audit trail mirrors. Each
    # field mirrors the matching ``generation.*`` value so
    # the frontend trust disclosure can render the dispatch
    # + claim-category + understanding audit without parsing
    # the structured envelope. All default to safe empties.
    deterministic_services_used: list[str] = Field(default_factory=list)
    """Mirrors ``generation.deterministic_services_used`` —
    the deterministic engines the ToolDispatcher invoked
    during this turn."""

    calculations_used: list[str] = Field(default_factory=list)
    """Mirrors ``generation.calculations_used`` — the
    deterministic calc names whose authoritative output the
    LLM was shown."""

    question_understanding: dict | None = None
    """Mirrors ``generation.question_understanding`` — the
    Stage 1 QuestionUnderstanding dict the universal-assistant
    layer produced for this turn. ``None`` when no
    understanding was produced (legacy callers)."""

    tool_calls: list[dict] = Field(default_factory=list)
    """Mirrors ``generation.tool_calls`` — the ToolCall
    tuples the dispatcher selected, as a list of dicts."""

    claim_categories_used: list[str] = Field(default_factory=list)
    """Mirrors ``generation.claim_categories_used`` — the
    claim-category labels the validator observed on the
    LLM's prose (FACT/CALCULATION/INFERENCE/RECOMMENDATION/
    SCENARIO/EXTERNAL_FACT/UNKNOWN)."""


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #


class ChatSessionSummary(BaseModel):
    """A conversation as it appears in the sidebar / list endpoint."""

    model_config = ConfigDict(extra="forbid")

    id: int
    title: str
    summary: str
    message_count: int = Field(ge=0)
    last_model: str = ""
    fallback_used: bool
    created_at: datetime
    updated_at: datetime


class ChatSessionDetail(BaseModel):
    """A conversation with every message inline."""

    model_config = ConfigDict(extra="forbid")

    id: int
    title: str
    summary: str
    message_count: int = Field(ge=0)
    last_model: str = ""
    fallback_used: bool
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageOut] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Request envelopes
# --------------------------------------------------------------------------- #


class ChatSessionCreateRequest(BaseModel):
    """Body for POST /chat (create a new conversation)."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=120)


class ChatMessageCreateRequest(BaseModel):
    """Body for POST /chat/{id}/message (append a user message)."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=4000)
    # H7.8C — the hybrid mode. ``grounded`` is the default
    # (evidence-bounded); ``open`` is permissive.
    mode: Literal["grounded", "open"] = "grounded"


# --------------------------------------------------------------------------- #
# Response envelopes
# --------------------------------------------------------------------------- #


class ChatMessageAppendResponse(BaseModel):
    """Reply body for POST /chat/{id}/message."""

    model_config = ConfigDict(extra="forbid")

    user_message: ChatMessageOut
    assistant_message: ChatMessageOut
    session: ChatSessionDetail


class ChatSessionListResponse(BaseModel):
    """Reply body for GET /chat."""

    model_config = ConfigDict(extra="forbid")

    sessions: list[ChatSessionSummary]
    count: int = Field(ge=0)


class ChatDeleteResponse(BaseModel):
    """Reply body for DELETE /chat/{id}."""

    model_config = ConfigDict(extra="forbid")

    deleted: bool
    id: int


# --------------------------------------------------------------------------- #
# H7.8C — provider status endpoint
# --------------------------------------------------------------------------- #


class ChatProviderStatusResponse(BaseModel):
    """Reply body for GET /chat/provider-status.

    The endpoint never exposes the API key, the auth header,
    or the full upstream base URL. The renderer only needs the
    provider *name*, the configured *model*, whether the
    provider is reachable *now*, and the list of supported
    *modes* — enough for the header dot and the mode toggle.

    H7.9R+ — ``reason`` carries a frontend-safe string that
    explains *why* the boolean is what it is. The frontend
    branches on it to render "Provider unavailable" vs
    "Missing API key" vs "Reachable" without parsing logs.
    """

    model_config = ConfigDict(extra="forbid")

    configured_provider: str = Field(min_length=1, max_length=80)
    runtime_provider: str = Field(min_length=1, max_length=80)
    model: str = Field(default="", max_length=120)
    available: bool
    schema_required: bool
    fallback_active: bool
    reason: Literal[
        "reachable",
        "missing_api_key",
        "missing_base_url",
        "ping_failed",
        "placeholder",
        "provider_unconfigured",
    ] = "provider_unconfigured"
    modes: list[Literal["grounded", "open"]] = Field(default_factory=list)
    default_mode: Literal["grounded", "open"] = "grounded"