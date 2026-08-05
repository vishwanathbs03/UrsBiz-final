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


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


class ChatMessageOut(BaseModel):
    """One message in a conversation. Returned by GET /chat/{id}."""

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
    """

    model_config = ConfigDict(extra="forbid")

    configured_provider: str = Field(min_length=1, max_length=80)
    runtime_provider: str = Field(min_length=1, max_length=80)
    model: str = Field(default="", max_length=120)
    available: bool
    schema_required: bool
    fallback_active: bool
    modes: list[Literal["grounded", "open"]] = Field(default_factory=list)
    default_mode: Literal["grounded", "open"] = "grounded"