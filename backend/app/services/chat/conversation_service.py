"""ConversationService — Sprint 7 Part 3.

The service composes the chat persistence layer with the
Sprint 7 Part 2 AI provider layer:

    ChatSessionRepository     <— storage
            ^
            |
    ConversationService       <— this file
            |
            v
    AssistantProviderService   <— Sprint 7 Part 2
    AssistantContextBuilder    <— Sprint 7 Part 2

The service owns three pieces of *session-level* state:

  * **Rolling context** — the last N turns fed back into the
    assistant provider as the ``history`` field of the
    request. N is small (default 8) so the prompt stays
    bounded.
  * **Conversation summary** — a short deterministic string
    rebuilt every time a new message is appended. Capped at
    500 chars so the sidebar can render it without joining
    ``chat_messages``. The summary is *derived from the
    existing assistant context* (Twin + Recommendations +
    Roadmap + Rules + Insights) — no new business logic.
  * **Title** — auto-derived from the first user message
    ("First 80 chars of 'How can I improve my business?'")
    when the caller does not pass an explicit one.

What this service is NOT
------------------------

  * It does NOT re-implement intent classification. The
    ``kind`` field on the message is whatever the caller
    classifies — the assistant provider does not require a
    kind to answer.
  * It does NOT store vector embeddings. RAG and semantic
    search are out of scope.
  * It does NOT modify the Business Digital Twin. The
    session is a user-level artefact.
  * It does NOT modify the assistant provider's
    response shape. The provider already returns a
    deterministic body + sources + model + fallback_used;
    the service persists them verbatim.

Determinism contract
--------------------

When the configured provider is the deterministic fallback
(default), two appends of the same user message to the same
session produce the same assistant body (sans the
``generated_at`` / ``updated_at`` timestamps). When Ollama
is configured and reachable, the assistant body is whatever
the model returns — non-deterministic by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from app.repositories.chat_session_repository import (
    ChatSessionNotFound,
    ChatSessionRepository,
)
from app.services.ai.providers.base import (
    AssistantContext,
    AssistantTurn,
    ProviderUnavailableError,
    ProviderTimeoutError,
)
from app.services.ai.providers.context_builder import AssistantContextBuilder
from app.services.ai.providers.factory import ProviderFactory
from app.services.ai.providers.ollama import OllamaProvider
from app.services.ai.providers.service import AssistantProviderService
from app.services.knowledge_retrieval.service import KnowledgeRetrievalService


# Default number of recent turns replayed into the
# provider's history. Keep small so the prompt stays
# bounded; raise in a later milestone if needed.
_ROLLING_CONTEXT_TURNS = 8


# Summary cap. Matches the column width on chat_sessions.
_SUMMARY_CAP = 480

# Title cap. Matches the column width on chat_sessions.
_TITLE_CAP = 80


@dataclass(frozen=True)
class AppendResult:
    """The result of an append-message call.

    The service returns this envelope so the endpoint can
    render both messages + a refreshed session detail in a
    single Pydantic response.
    """

    user_message: dict
    assistant_message: dict
    session: dict


class ConversationService:
    """The façade the chat endpoint depends on."""

    def __init__(
        self,
        repo: ChatSessionRepository,
        *,
        assistant_service: AssistantProviderService | None = None,
        rolling_context_turns: int = _ROLLING_CONTEXT_TURNS,
        knowledge_retriever: KnowledgeRetrievalService | None = None,
    ) -> None:
        self._repo = repo
        self._assistant = assistant_service or self._default_assistant_service()
        self._rolling_context_turns = max(0, int(rolling_context_turns))
        # Optional Sprint 7 Part 4 retrieval layer. When None
        # the chat service is identical to Part 3 — no
        # retrieval, no citations.
        self._knowledge = knowledge_retriever

    # ---- CRUD -------------------------------------------------------- #

    def list_sessions(self, *, owner_id: int) -> list[dict]:
        sessions = self._repo.list_sessions(owner_id=owner_id)
        return [_session_summary(s) for s in sessions]

    def get_session(self, *, owner_id: int, session_id: int) -> dict:
        session = self._repo.get_session(owner_id=owner_id, session_id=session_id)
        if session is None:
            raise ChatSessionNotFound(
                f"Session {session_id} not found for owner {owner_id}"
            )
        return _session_detail(session)

    def create_session(
        self, *, owner_id: int, title: str = ""
    ) -> dict:
        session = self._repo.create_session(owner_id=owner_id, title=title)
        return _session_detail(session)

    def delete_session(self, *, owner_id: int, session_id: int) -> bool:
        deleted = self._repo.delete_session(
            owner_id=owner_id, session_id=session_id
        )
        if not deleted:
            raise ChatSessionNotFound(
                f"Session {session_id} not found for owner {owner_id}"
            )
        return True

    def append_message(
        self,
        *,
        owner_id: int,
        session_id: int,
        content: str,
        mode: str = "grounded",
    ) -> AppendResult:
        """Append a user message + the assistant's reply.

        1. Load the session (404 if not found).
        2. Insert the user message.
        3. Compose the rolling context from the last N turns.
        4. Build the AssistantContext via the existing
           Part 2 context builder.
        5. Call the AssistantProviderService.
        6. Insert the assistant message.
        7. Refresh the session title / summary / last_model /
           fallback_used / message_count.

        ``mode`` (H7.8C) selects the hybrid assistant mode.
        ``"grounded"`` is the default — evidence-bounded.
        ``"open"`` is the permissive general-purpose mode.
        The mode is forwarded to the provider so the prompt
        builder, grounding validator, and response schema all
        line up.
        """
        session = self._repo.get_session(owner_id=owner_id, session_id=session_id)
        if session is None:
            raise ChatSessionNotFound(
                f"Session {session_id} not found for owner {owner_id}"
            )

        # 1. Insert the user message.
        user_msg = self._repo.add_message(
            session=session,
            role="user",
            content=content,
            kind="",
        )

        # 2. Compose rolling history. Use every existing
        #    message *before* the user message we just added.
        history = self._build_history(session, exclude_message_id=user_msg.id)

        # 3. Build the assistant context via Part 2.
        context = self._assistant.build_context(owner_id=owner_id)

        # 4. Sprint 7 Part 4 — retrieve, rank, build
        #    citations. Bind Owner_id to the assistant's
        #    context shape so the per-business boost fires
        #    when the layer has access to the data.
        knowledge_ctx = None
        if self._knowledge is not None:
            owner_context = self._build_owner_context(context)
            knowledge_ctx = self._knowledge.retrieve(
                query=content,
                owner_context=owner_context,
            )

        # 5. Call the provider. The provider already
        #    catches ProviderUnavailableError /
        #    ProviderTimeoutError and falls back; any
        #    other AIProviderError propagates so the
        #    endpoint can surface a 502.
        assistant_resp = self._assistant.generate(
            owner_id=owner_id,
            user_prompt=content,
            history=history,
            knowledge=knowledge_ctx,
            mode=mode,
        )

        # 6. Persist the assistant reply. Sources are the
        #    union of the provider's own sources and the
        #    knowledge retrieval's citations. The chat
        #    schema is the same either way.
        provider_sources = _sources_to_payload(
            getattr(assistant_resp, "sources", ()) or ()
        )
        citation_sources = (
            _citations_to_sources(knowledge_ctx)
            if knowledge_ctx is not None else []
        )
        sources = provider_sources + citation_sources
        # H7.8C — serialise the GenerationMeta into the
        # per-message ``generation_meta_json`` column so the
        # frontend can render the provenance disclosure on
        # refresh (the trust label persists across reloads).
        generation_payload = _generation_meta_to_payload(
            getattr(assistant_resp, "generation", None)
        )
        assistant_msg = self._repo.add_message(
            session=session,
            role="assistant",
            content=assistant_resp.body,
            kind="",
            sources=sources,
            # H7.8A P2 — persist the per-message fallback flag so the
            # frontend MessageBubble can render the right trust label.
            # ``assistant_resp.fallback_used`` is True when the
            # deterministic placeholder / safe provider answered, False
            # when a real OpenAI-compatible or Ollama response was
            # produced.
            fallback_used=assistant_resp.fallback_used,
            # H7.8C — provenance envelope persists on the message
            # row. The frontend re-reads it after page reload so the
            # "Generated by ollama:llama3.1" disclosure is stable.
            generation_meta=generation_payload,
        )

        # 6. Refresh the session meta. Only auto-derive
        #    the title from the first user message when
        #    the caller did not pass one — otherwise the
        #    caller's explicit title is preserved verbatim
        #    across every subsequent append.
        new_count = len(self._repo.get_messages(session=session))
        title = session.title or _derive_title(content)
        summary = _derive_summary(
            session_summary=session.summary,
            context=context,
            latest_user=content,
            latest_assistant=assistant_resp.body,
        )
        self._repo.touch_session(
            session=session,
            title=title,
            summary=summary,
            last_model=assistant_resp.model,
            fallback_used=assistant_resp.fallback_used,
            message_count=new_count,
        )

        return AppendResult(
            user_message=_message_payload(user_msg),
            assistant_message=_message_payload(assistant_msg),
            session=_session_detail(session),
        )

    # ---- internals --------------------------------------------------- #

    def _build_history(
        self,
        session,
        *,
        exclude_message_id: int | None = None,
    ) -> tuple[AssistantTurn, ...]:
        """Return the last N turns as AssistantTurn records.

        ``exclude_message_id`` is the id of the user message
        that was just appended. We must skip it because the
        provider is about to receive it as the live ``user_prompt``
        — repassing it as history would echo the same text back
        into the model and inflate the prompt with duplicate
        content. (H7.8C P3 — the previous filter compared
        ``m.id`` to ``session.id`` which almost never matched,
        so the just-inserted user message leaked into the
        rolling context.)
        """
        messages = self._repo.get_messages(session=session)
        # Exclude the message we just added (it's the user
        # message we're about to send as the prompt).
        prior = [
            m
            for m in messages
            if m.role in ("user", "assistant")
            and (exclude_message_id is None or m.id != exclude_message_id)
        ]
        if self._rolling_context_turns <= 0:
            return ()
        prior = prior[-self._rolling_context_turns:]
        return tuple(
            AssistantTurn(role=m.role, content=m.content) for m in prior
        )

    def _build_owner_context(self, context) -> dict:
        """Project the assembled AssistantContext into the
        per-business boost payload the retriever consumes.

        The projection is a degeneralization: the
        :class:`AssistantContext` is already a narrow
        projection of Twin + Recommendations, so we only
        forward the keys the retriever understands.

        Two calls with the same upstream state produce
        the same boost payload.
        """
        try:
            from app.services.knowledge_retrieval.service import (
                KnowledgeRetrievalService,
            )
        except ImportError:
            return {}
        low_score_keys: list[str] = []
        for score in getattr(context, "scores", ()) or ():
            try:
                if str(score.level).lower() == "low":
                    low_score_keys.append(str(score.key).lower())
            except AttributeError:
                continue
        rec_categories: list[str] = []
        for rec in getattr(context, "recommendations", ()) or ():
            try:
                cat = rec.category
                if isinstance(cat, str) and cat:
                    rec_categories.append(cat.lower())
            except AttributeError:
                continue
        return {
            "low_score_keys": tuple(sorted(set(low_score_keys))),
            "recommendation_categories": tuple(sorted(set(rec_categories))),
        }

    def _default_assistant_service(self) -> AssistantProviderService:
        """Build the default AssistantProviderService.

        Wires the Sprint 7 Part 2 context builder to call
        the five upstream engines (Twin, Recommendations,
        Roadmap, Rules, Insights) via a callable bridge that
        the endpoint will close over with the request's
        ``BusinessRepository``.
        """
        from app.config.settings import get_settings
        from app.repositories.business_repository import BusinessRepository
        from app.services.intelligence import IntelligenceService
        from app.services.recommendations import RecommendationService
        from app.services.roadmap import RoadmapService
        from app.services.rules import RuleEngineService
        from app.services.twin import TwinService
        from app.services.ai import AIDecisionService

        # NOTE: a real endpoint that owns a session
        #       will construct its own AssistantProviderService
        #       with a closure over BusinessRepository.
        #       This default is only used by callers that
        #       want a service for tests / scripts.
        raise RuntimeError(
            "ConversationService requires an explicit assistant_service. "
            "Build one with AssistantContextBuilder bound to the request's "
            "BusinessRepository and pass it in."
        )


# --------------------------------------------------------------------------- #
# Helpers — payload projection
# --------------------------------------------------------------------------- #


def _message_payload(msg) -> dict:
    try:
        sources_raw = json.loads(msg.sources_json or "[]")
    except json.JSONDecodeError:
        sources_raw = []
    sources: list[dict] = []
    for s in sources_raw:
        if not isinstance(s, dict):
            continue
        topic = str(s.get("topic", ""))
        detail = str(s.get("detail", ""))
        if topic and detail:
            sources.append({"topic": topic, "detail": detail})
    # H7.8C — re-hydrate the GenerationMeta envelope from the
    # ``generation_meta_json`` column so the frontend can render
    # the disclosure without a second fetch.
    generation = None
    try:
        raw = getattr(msg, "generation_meta_json", "") or ""
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                generation = parsed
    except (json.JSONDecodeError, ValueError):
        generation = None

    # H7.8C — flatten the brief-mandated provenance fields to
    # the TOP level of the message payload. The frontend trust
    # disclosure (provider, model, runtime provider, grounding
    # score, evidence references, assumptions, limitations,
    # fallback active, mode, confidence) is now reachable
    # without parsing the ``generation`` block. All fields are
    # safe defaults so user turns and legacy rows remain
    # parseable.
    #
    # Source of truth priority: the live ``generation`` block
    # is authoritative. The pydantic ``ChatMessageOut`` schema
    # accepts the top-level fields verbatim — the brief value
    # is propagated and the ``generation`` block is kept for
    # any client that already reads it.
    fallback_used_flag = bool(getattr(msg, "fallback_used", False))
    gen_provider = str((generation or {}).get("provider") or "")
    gen_model = str((generation or {}).get("model") or "")
    gen_runtime_provider = str((generation or {}).get("runtime_provider") or gen_provider)
    gen_mode = (generation or {}).get("mode")
    gen_grounding_score = int((generation or {}).get("server_grounding_score") or 0)
    gen_evidence_refs = list((generation or {}).get("evidence_references") or [])
    gen_assumptions = list((generation or {}).get("assumptions") or [])
    gen_limitations = list((generation or {}).get("limitations") or [])
    gen_confidence = (generation or {}).get("confidence")
    # AI-1 — universal-assistant audit-trail fields surfaced
    # at the top level for the frontend trust disclosure. Each
    # default is a safe empty so user turns + legacy rows remain
    # parseable.
    gen_deterministic_services = list(
        (generation or {}).get("deterministic_services_used") or []
    )
    gen_calculations_used = list(
        (generation or {}).get("calculations_used") or []
    )
    gen_question_understanding = (generation or {}).get("question_understanding")
    gen_tool_calls = list((generation or {}).get("tool_calls") or [])
    gen_claim_categories = list(
        (generation or {}).get("claim_categories_used") or []
    )
    # AI-3 — claim-aware response contract mirrors. The
    # claim_aware_response is sourced from GenerationMeta
    # ``grounded_payload["claim_aware"]`` (set by the AI-3
    # pipeline in service.py) and projected to the top level
    # so the frontend can render the structured panel without
    # drilling into ``generation.*``.
    gen_grounded_payload = (
        (generation or {}).get("grounded_payload")
        if isinstance((generation or {}).get("grounded_payload"), dict)
        else None
    )
    gen_claim_aware = (
        gen_grounded_payload.get("claim_aware")
        if gen_grounded_payload is not None
        else None
    )
    gen_claim_aware_validated = bool(
        (generation or {}).get("claim_aware_validated") or False
    )
    gen_numeric_conflicts_count = int(
        (generation or {}).get("numeric_conflicts_count") or 0
    )
    gen_server_confidence = (generation or {}).get("server_confidence")
    gen_server_confidence_rationale = str(
        (generation or {}).get("server_confidence_rationale") or ""
    )
    # AI-4 — server-side claim auditor mirrors. The compact
    # claim trace is stashed on GenerationMeta.grounded_payload
    # by the AI-4 pipeline (service.py) and mirrored to the
    # wire as a top-level ``claim_audit`` field so the
    # frontend's "Why am I seeing this?" disclosure panel
    # renders without drilling into ``generation.*``.
    gen_claim_audit = (
        gen_grounded_payload.get("claim_audit")
        if gen_grounded_payload is not None
        and isinstance(gen_grounded_payload.get("claim_audit"), dict)
        else None
    )
    gen_claim_audit_rejected = bool(
        (generation or {}).get("claim_audit_rejected") or False
    )
    gen_claim_audit_soft_corrections = int(
        (generation or {}).get("claim_audit_soft_corrections") or 0
    )

    payload = {
        "id": int(msg.id),
        "role": str(msg.role),
        "kind": str(msg.kind or ""),
        "content": str(msg.content),
        "sources": sources,
        "created_at": _iso(msg.created_at),
        # H7.8A P2 — per-message fallback flag surfaced to the
        # frontend so MessageBubble can render the right trust label.
        "fallback_used": fallback_used_flag,
        # H7.8C — full provenance envelope. Always present on
        # assistant turns; None for user turns (the user has no
        # generation metadata).
        "generation": generation,
        # H7.8C — flat mirrors of the provenance envelope.
        # The frontend can render the trust disclosure from
        # these top-level fields without drilling into
        # ``generation.*``. Backward-compatible: every
        # field has a safe default so old clients still
        # parse.
        "provider": gen_provider,
        "model": gen_model,
        "runtime_provider": gen_runtime_provider,
        "grounding_score": gen_grounding_score,
        "evidence_references": gen_evidence_refs,
        "assumptions": gen_assumptions,
        "limitations": gen_limitations,
        "fallback_active": fallback_used_flag,
        "mode": gen_mode,
        "confidence": gen_confidence,
        # AI-1 — universal-assistant audit-trail mirrors at the
        # top level of the wire payload.
        "deterministic_services_used": gen_deterministic_services,
        "calculations_used": gen_calculations_used,
        "question_understanding": gen_question_understanding,
        "tool_calls": gen_tool_calls,
        "claim_categories_used": gen_claim_categories,
        # AI-3 — claim-aware contract mirrors. Backward-compatible:
        # claim_aware_response is None when the LLM did not fill the
        # schema or when this is a legacy row; the frontend renders
        # the old path. server_confidence is always populated when
        # the meta carries it (None on legacy rows).
        "claim_aware_response": gen_claim_aware,
        "claim_aware_validated": gen_claim_aware_validated,
        "numeric_conflicts_count": gen_numeric_conflicts_count,
        "server_confidence": gen_server_confidence,
        "server_confidence_rationale": gen_server_confidence_rationale,
        # AI-4 — claim-auditor mirrors. The compact trace
        # (``gen_claim_audit``) is what the "Why am I seeing
        # this?" disclosure panel renders; the two boolean
        # companions let the frontend stamp a top-level trust
        # badge without parsing the trace. Backward-compatible:
        # all three default to None / False / 0 on legacy rows.
        "claim_audit": gen_claim_audit,
        "claim_audit_rejected": gen_claim_audit_rejected,
        "claim_audit_soft_corrections": gen_claim_audit_soft_corrections,
    }
    # H7.8C — leak guard. The serializer must never emit a
    # field name from the brief-mandated secret set
    # (``api_key``, ``authorization``, ``base_url``, etc.).
    # The audit fix changed the provider layer to never
    # include them; this guard catches any future regression
    # at the projection boundary.
    _assert_no_leaked_secrets(payload, where="_message_payload")
    return payload


def _session_summary(session) -> dict:
    return {
        "id": int(session.id),
        "title": str(session.title or ""),
        "summary": str(session.summary or ""),
        "message_count": int(session.message_count or 0),
        "last_model": str(session.last_model or ""),
        "fallback_used": bool(session.fallback_used),
        "created_at": _iso(session.created_at),
        "updated_at": _iso(session.updated_at),
    }


def _session_detail(session) -> dict:
    out = _session_summary(session)
    out["messages"] = [_message_payload(m) for m in session.messages]
    return out


def _sources_to_payload(sources) -> list[dict]:
    """Map provider sources (ChatSource dataclasses) to dict payload."""
    out: list[dict] = []
    for s in sources:
        topic = getattr(s, "topic", None)
        detail = getattr(s, "detail", None)
        if topic and detail:
            out.append({"topic": str(topic), "detail": str(detail)})
    return out


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


# H7.8C — the brief-mandated list of fields that must NEVER
# appear in a wire payload. The service layer asserts this set
# is disjoint from every payload it emits. Adding a new
# sensitive key here is a non-breaking change (it just adds
# another field to the leak guard). Removing a key is a
# breaking change for the audit trail.
_LEAKED_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "auth_header",
        "base_url",
        "upstream_url",
        "secret",
        "bearer",
        "access_token",
    }
)


def _assert_no_leaked_secrets(payload: dict | None, *, where: str) -> None:
    """Raise ``ValueError`` if ``payload`` contains any key in
    :data:`_LEAKED_FIELDS`.

    H7.8C — the brief mandates that the assistant response
    NEVER exposes API keys, authorization headers, or base
    URLs. The audit-fixed provider layer never includes these
    fields, but a future refactor could accidentally paste a
    config dict into the payload. This guard catches that
    regression at the projection boundary, in the same place
    that fixes the brief, so any future change is forced to
    think about the leak surface.

    Parameters
    ----------
    payload:
        The dict the serializer is about to emit (either the
        ``generation`` envelope or the top-level message
        payload). ``None`` is a no-op.
    where:
        Short label used in the error message so the audit
        log knows which surface leaked.
    """
    if not payload:
        return
    leaked = set(payload.keys()) & _LEAKED_FIELDS
    if leaked:
        raise ValueError(
            f"H7.8C leak guard tripped at {where}: "
            f"refusing to serialise payload containing "
            f"secrets: {sorted(leaked)!r}"
        )


def _generation_meta_to_payload(meta) -> dict | None:
    """Project a :class:`GenerationMeta` dataclass into a JSON-safe dict.

    Returns ``None`` when the provider did not stamp a
    ``GenerationMeta`` (e.g. the legacy mock-provider path).
    Tuples become lists so ``json.dumps`` round-trips cleanly.

    H7.8C — ``runtime_provider`` is filled from ``provider``
    when the dataclass did not stamp it explicitly. The
    short-circuit keeps the field always-present on the wire
    so the frontend can rely on it without checking for
    ``None``. The leak guard runs before returning so any
    future regression that pulls a config dict into the
    envelope is caught at the boundary.
    """
    if meta is None:
        return None
    try:
        from dataclasses import asdict
        out = asdict(meta)
    except Exception:
        return None
    # Tuples → lists for JSON.
    for key, value in list(out.items()):
        if isinstance(value, tuple):
            out[key] = list(value)
    # H7.8C — default runtime_provider to provider when the
    # upstream didn't set it. A real provider path always
    # sets it explicitly; the deterministic fallback path
    # leaves it empty so this default kicks in.
    if not out.get("runtime_provider"):
        out["runtime_provider"] = out.get("provider") or ""
    _assert_no_leaked_secrets(out, where="_generation_meta_to_payload")
    return out


def _derive_title(content: str) -> str:
    """First 80 chars of the user message, single-line, trimmed."""
    flat = " ".join((content or "").split())
    if not flat:
        return "New conversation"
    return flat[:_TITLE_CAP]


def _derive_summary(
    *,
    session_summary: str,
    context: AssistantContext,
    latest_user: str,
    latest_assistant: str,
) -> str:
    """Compose the conversation summary.

    The summary is a single line that names the score band +
    the DNA archetype + a short hint of the latest exchange.
    It is rebuilt on every append so the sidebar always
    shows the freshest state.
    """
    bits: list[str] = []
    bits.append(
        f"Score {context.overall_business_score}/100 ({context.band})"
    )
    if context.dna.archetype_title:
        bits.append(
            f"DNA: {context.dna.archetype_title} ({context.dna.match_score}%)"
        )
    rec = (latest_user or "").strip()
    if rec:
        flat_user = " ".join(rec.split())[:80]
        bits.append(f"Latest: {flat_user}")
    summary = " · ".join(bits)
    if len(summary) > _SUMMARY_CAP:
        summary = summary[: _SUMMARY_CAP - 1] + "…"
    return summary


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _citations_to_sources(knowledge_ctx) -> list[dict]:
    """Translate :class:`Citation` rows into the
    ``ChatSource`` payload the chat schema stores.

    The schema is the same shape the existing providers
    use:

      {
        "topic": <SourceCategory-as-string>,
        "detail": <Citation.detail>,
      }

    Citation ``article_id`` is preserved in the detail
    string so the UI can deep-link to the knowledge
    article once such a route exists.
    """
    if knowledge_ctx is None:
        return []
    out: list[dict] = []
    for c in getattr(knowledge_ctx, "citations", ()) or ():
        topic = getattr(c, "source_category", None) or "Knowledge"
        detail = getattr(c, "detail", None) or ""
        if detail:
            out.append({
                "topic": str(topic),
                "detail": f"{detail} (article {c.article_id})",
            })
    return out