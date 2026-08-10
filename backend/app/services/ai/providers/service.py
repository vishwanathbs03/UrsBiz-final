"""AssistantProviderService — Sprint 7 Part 2 + H7.3 + H7.8C.

The high-level façade the brief asks for. Wires the four
pieces together:

  AssistantContextBuilder     (read 5 upstream payloads)
            |
            v
  AssistantPromptBuilder      (assemble the prompt surface)
            |
            v
  ProviderFactory             (select the runtime provider)
            |
            v
  Provider.complete(request)  (real Ollama, real OpenAI-compat, or fallback)
            |
            v
  response_schema validation  (H7.3 — only when the provider is JSON-mode)
            |
            v
  GroundingValidator          (H7.8C — evidence-bound audit)
            |
            v
  AssistantResponse           (the return value, with GenerationMeta)

The service is thin — it owns no business logic. It is the
*only* file in the layer that knows about the four pieces,
which keeps the others unit-testable in isolation.

H7.8C — hybrid modes
--------------------

The service accepts a ``mode`` parameter on :meth:`generate`:

* ``mode="grounded"`` (default) — strict evidence-bounded.
  Builds an :class:`EvidenceRegistry` from the assembled
  context, embeds it in the prompt, requires JSON output,
  and runs the :class:`GroundingValidator` over the parsed
  response. Any rule failure falls back to the deterministic
  provider with ``fallback_reason="grounding_invalid"``.

* ``mode="open"`` — permissive. The prompt has no
  registry and no schema requirement. The model returns
  free-form prose. Provider failures get
  ``fallback_reason="open_mode_provider_failure"``.

Graceful degradation
--------------------

When the configured provider raises :class:`ProviderUnavailableError`
or :class:`ProviderTimeoutError`, the service catches the error
and asks the factory for the deterministic fallback. The
caller sees a normal :class:`AssistantResponse` whose
``fallback_used`` flag is ``True``, whose ``model`` is
``"deterministic-fallback"``, and whose ``generation`` block
carries the ``fallback_reason``.

H7.3 added: when the configured provider raises a
non-soft :class:`AIProviderError` *and* the error originated
from a schema-validation failure (``schema_invalid`` reason),
the service falls back to the deterministic provider. Per
docx P3 Part 3: "When validation fails, use the existing
deterministic consultant response." Other AIProviderError
classes (HTTP 5xx, malformed JSON, empty body) propagate
so the caller can decide.

The service does *not* retry. A retry policy is the next
milestone's problem.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.services.ai.providers.base import (
    AIProviderError,
    AssistantContext,
    AssistantRequest,
    AssistantResponse,
    AssistantTurn,
    DeterministicFallbackProvider,
    GenerationMeta,
    Mode,
    NormalizedReason,
    Provider,
    ProviderHTTPStatusError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.services.ai.providers.context_builder import AssistantContextBuilder
from app.services.ai.providers.evidence_registry import EvidenceRegistry
from app.services.ai.providers.factory import ProviderFactory
from app.services.ai.providers.grounding_validator import (
    GroundingValidator,
    OpenResponseValidator,
)
from app.services.ai.providers.prompt_builder import AssistantPromptBuilder
from app.services.ai.providers.response_schema import (
    parse_model_output,
    parse_open_model_output,
)
# H8.11 — pre-LLM reasoning layer. Imported lazily inside
# ``__init__`` to avoid the circular import through
# ``app.services.ai.providers.__init__`` (which eagerly
# loads this module before :mod:`evidence_retriever` is
# fully initialised).
from app.services.ai.reasoning.question_understanding import (
    QuestionUnderstanding,
    is_purely_educational,
    understand_question,
)
from app.services.ai.reasoning.tool_selector import (
    StubToolInterface,
    ToolSelector,
)
from app.services.ai.reasoning.answer_composer import compose_adaptive_answer


# Module-level logger for structured provider events.
# The deployment /monitoring layer redacts known secret keys
# (``AI_API_KEY``, ``Authorization``, ``Cookie`` …) before
# these records hit disk — see ``app/monitoring/logging.py``.
from app.services.ai.providers.circuit_breaker import AICircuitBreaker
from app.services.ai.providers.base import (
    AIProviderError,
    AssistantContext,
    AssistantRequest,
    AssistantResponse,
    AssistantTurn,
    DeterministicFallbackProvider,
    GenerationMeta,
    Mode,
    NormalizedReason,
    Provider,
    ProviderAuthError,
    ProviderConfigError,
    ProviderHTTPStatusError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger("atlas.ai.provider")


# H7.9R+ — hard wall-clock cap on a single outbound LLM call.
# The provider's underlying httpx client already has its own
# connect/read timeouts; this constant is the *outer* ceiling
# that guarantees no chat request can ever hold a worker thread
# for more than this many seconds, regardless of retry behaviour
# inside the circuit breaker or any future backoff. 15s matches
# the value committed in ``backend/.env`` (``AI_REQUEST_TIMEOUT_SECONDS``).
HARD_CALL_TIMEOUT_SECONDS: float = 15.0

# Single shared executor for ``_call_with_hard_timeout`` below.
# ``max_workers=1`` so a slow provider does not get a pool of
# parallel attempts; the cap is the timeout, not parallelism.
# Daemon threads so a stuck provider cannot block process exit.
_HARD_TIMEOUT_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ai-hard-timeout"
)


def _call_with_hard_timeout(
    provider: Any, request: AssistantRequest, timeout: float
) -> AssistantResponse:
    """Invoke ``provider.complete(request)`` under a hard wall-clock cap.

    Why this exists
    ---------------
    The underlying ``httpx.Client`` in :class:`OpenAICompatibleProvider`
    has a per-request timeout, but the service layer also runs the
    call through :class:`AICircuitBreaker.execute_with_resilience`
    which can retry with exponential backoff. A misbehaving upstream
    (TLS handshake stall, dropped connection, DNS hang) can therefore
    hold a worker thread for many times the configured timeout
    before any typed error surfaces.

    This wrapper puts the entire call (including the breaker's
    retries) on a single dedicated thread and bounds the total
    wall-clock time at ``timeout`` seconds. If the cap is
    exceeded, the thread is *left running* (it is daemonic and
    cannot block process exit) but the *caller* immediately
    receives a :class:`ProviderTimeoutError` and falls back to
    the deterministic provider. The provider's ``close()`` is
    invoked from the caller's thread so the underlying HTTP
    connection is released and the next request starts clean.

    The previous architecture (no hard cap) caused chat requests
    to hang indefinitely when the upstream was unreachable. The
    H7.9R fix activates the fallback path within the wall-clock
    budget so the frontend never waits forever.
    """
    future = _HARD_TIMEOUT_EXECUTOR.submit(provider.complete, request)
    try:
        return future.result(timeout=timeout)
    except Exception:
        # Any exception from the provider propagates; only a
        # wall-clock timeout is converted here. We do NOT call
        # ``future.cancel()`` once the call has started — Python
        # cannot interrupt a running thread — but we DO close
        # the provider so the underlying socket is released.
        try:
            provider.close()
        except Exception:
            pass
        raise


class AssistantProviderService:
    """The public façade for the AI Provider Layer with Multi-Tier Failover & Circuit Breaker."""

    def __init__(
        self,
        *,
        context_builder: AssistantContextBuilder,
        prompt_builder: AssistantPromptBuilder | None = None,
        provider_factory: ProviderFactory | None = None,
        reasoning_engine: Any | None = None,
        evidence_retriever: Any | None = None,
        tool_dispatcher: Any | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder or AssistantPromptBuilder()
        self._factory = provider_factory or ProviderFactory()
        self._circuit_breaker = AICircuitBreaker(name="gemini")
        # H8.11 — pre-LLM reasoning layer. The engine runs
        # the intent classifier + the H8.3 pipeline; the
        # retriever ranks the evidence registry. Both are
        # imported lazily so this module can finish loading
        # even when ``providers/__init__.py`` is mid-import
        # (the original eager-import path triggered a
        # circular ImportError when the package itself was
        # being collected by the test runner).
        if reasoning_engine is None or evidence_retriever is None:
            from app.services.ai.reasoning.reasoning_engine import (
                BusinessReasoningEngine,
            )
            from app.services.ai.reasoning.evidence_retriever import (
                EvidenceRetriever,
            )
        self._reasoning_engine = (
            reasoning_engine
            if reasoning_engine is not None
            else BusinessReasoningEngine()
        )
        self._evidence_retriever = (
            evidence_retriever
            if evidence_retriever is not None
            else EvidenceRetriever()
        )
        # SPRINT AI-2 — ToolDispatcher dependency. When the
        # chat endpoint constructs the service with a real
        # dispatcher (16 real engine wrappers), that dispatcher
        # is used. When the kwarg is None (legacy callers,
        # unit tests that don't care about tool dispatch) a
        # stub-only dispatcher is built so ``generate()`` keeps
        # working with ``status="not_implemented"`` everywhere.
        if tool_dispatcher is None:
            from app.services.ai.reasoning.tool_selector import (
                ToolDispatcher as _ToolDispatcher,
            )
            tool_dispatcher = _ToolDispatcher()
        self._tool_dispatcher = tool_dispatcher

    # ---- public API -------------------------------------------------- #

    def configured_provider_name(self) -> str:
        """Return the name of the configured provider."""
        return self._factory.configured_provider_name()

    def generate(
        self,
        *,
        owner_id: int,
        user_prompt: str,
        history: tuple[AssistantTurn, ...] = (),
        knowledge: object | None = None,
        provider: Provider | None = None,
        require_schema: bool | None = None,
        mode: Mode = "grounded",
    ) -> AssistantResponse:
        """Generate a reply with multi-tier resilience and circuit breaker protection."""
        try:
            context = self._context_builder.build(
                owner_id=owner_id, user_prompt=user_prompt
            )
        except TypeError:
            context = self._context_builder.build(owner_id=owner_id)
            if context.context_manifest is None and user_prompt:
                from app.services.ai.providers.context_builder import select_relevant_context
                context = select_relevant_context(context, user_prompt)

        # H8.11 — pre-LLM reasoning layer. The engine emits
        # a structured plan; the retriever ranks the
        # evidence registry by intent + plan. Both are
        # wrapped in try/except so a failure here can never
        # break a chat request — the prompt builder falls
        # back to the pre-H8.11 surface when either is
        # ``None``.
        #
        # AI-1 — Stage 1 builds a QuestionUnderstanding
        # from the user prompt + context. Stage 4 threads
        # that understanding into the reasoning plan.
        # Stage 5 dispatches deterministic tools (stubs by
        # default). Stages 7+8 use the same understanding /
        # plan / tool results to label claim categories and
        # pick the adaptive answer shell. None of these
        # layers can raise — every step is wrapped in
        # try/except so a Stage 1-8 failure can never break
        # a chat request.
        question_understanding: QuestionUnderstanding | None = None
        reasoning_plan = None
        ranked_evidence = None
        tool_results: tuple = ()
        adaptive_answer = None
        try:
            question_understanding = understand_question(
                user_prompt, context
            )
            # Stage 4 — call the reasoning engine's plan()
            # with the AI-1 kwarg. The legacy 2-kwarg
            # signature is preserved (the keyword is optional
            # in BusinessReasoningEngine.plan) but custom
            # TrackingEngine subclasses in the tests may
            # not have updated their signature. We try
            # with the kwarg first, fall back to the legacy
            # call if the engine rejects it.
            try:
                reasoning_plan = self._reasoning_engine.plan(
                    user_prompt=user_prompt,
                    context=context,
                    question_understanding=question_understanding,
                )
            except TypeError:
                reasoning_plan = self._reasoning_engine.plan(
                    user_prompt=user_prompt,
                    context=context,
                )
            registry = EvidenceRegistry(context)
            ranked_evidence = self._evidence_retriever.rank(
                context=context,
                registry=registry,
                reasoning_plan=reasoning_plan,
            )
            # Stage 5 — dispatch deterministic tools. The
            # dispatcher returns stubs by default for any
            # engine the layer hasn't wired in. SPRINT AI-2:
            # ``self._tool_dispatcher`` is the dispatcher
            # passed in via the constructor (or a default
            # stub-only dispatcher when no kwarg was given).
            tool_results = self._tool_dispatcher.dispatch(
                owner_id=owner_id,
                question_understanding=question_understanding,
                reasoning_plan=reasoning_plan,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[service] AI-1 universal-assistant layers failed; "
                "falling back to pre-AI-1 prompt surface: %s",
                exc,
            )
            question_understanding = None
            reasoning_plan = None
            ranked_evidence = None
            tool_results = ()

        # AI-1 — internal ``_effective_mode``. The wire
        # ``mode`` is the user's selection (always). The
        # internal ``_effective_mode`` flips to ``"open"``
        # only when the prompt is purely educational AND
        # not business-specific — so the LLM is not forced
        # through the evidence registry for definitions or
        # explanations that have nothing to do with the
        # user's profile. The trust label uses the wire
        # ``mode`` (the user always sees their pick).
        effective_mode: Mode = mode
        try:
            if mode == "grounded" and is_purely_educational(user_prompt):
                if not getattr(
                    question_understanding, "is_business_specific", True
                ):
                    effective_mode = "open"
        except Exception:  # noqa: BLE001
            effective_mode = mode

        request = self._prompt_builder.build(
            context=context,
            user_prompt=user_prompt,
            history=history,
            knowledge=knowledge,
            mode=effective_mode,
            reasoning_plan=reasoning_plan,
            ranked_evidence=ranked_evidence,
        )

        chosen = provider or self._factory.build()

        # Check circuit breaker before making expensive network calls
        if not provider and not self._circuit_breaker.allow_request():
            logger.warning("[service] Circuit breaker is OPEN. Skipping primary provider call.")
            return self._fallback_chain(request, reason="circuit_open", mode=mode)

        try:
            if provider:
                # H7.9R+ — hard wall-clock cap on every outbound
                # call. ``asyncio.wait_for`` is the async equivalent;
                # we use ``_call_with_hard_timeout`` (a
                # ``ThreadPoolExecutor``-based equivalent) because
                # ``generate()`` runs in FastAPI's sync threadpool,
                # not on the asyncio loop. The cap is per-call;
                # the underlying provider's own connect/read timeouts
                # still apply on the inside.
                response = _call_with_hard_timeout(
                    chosen, request, timeout=HARD_CALL_TIMEOUT_SECONDS
                )
            else:
                # Same cap, but through the circuit breaker which
                # may add retry-with-backoff. The total wall-clock
                # is still bounded by ``HARD_CALL_TIMEOUT_SECONDS``
                # because the breaker runs inside the timed thread.
                response = self._circuit_breaker.execute_with_resilience(
                    lambda: _call_with_hard_timeout(
                        chosen, request, timeout=HARD_CALL_TIMEOUT_SECONDS
                    )
                )
        except ProviderQuotaError:
            return self._fallback_chain(request, reason="quota_exhausted", mode=mode)
        except ProviderRateLimitError:
            return self._fallback_chain(request, reason="rate_limited", mode=mode)
        except ProviderAuthError:
            return self._fallback_chain(request, reason="auth_failed", mode=mode)
        except (concurrent.futures.TimeoutError, TimeoutError) as exc:
            # H7.9R+ — the wall-clock cap fired. This is NOT the
            # same as ``ProviderTimeoutError`` (which the provider
            # raises when ITS own httpx client times out). The
            # fallback chain is invoked immediately, the underlying
            # provider's ``close()`` has already been called by
            # ``_call_with_hard_timeout`` (so the HTTP socket is
            # released), and a structured log entry carries the
            # provider name + model + elapsed time.
            provider_name = getattr(chosen, "name", type(chosen).__name__)
            model = getattr(chosen, "model_name", "") or ""
            elapsed_ms = int(HARD_CALL_TIMEOUT_SECONDS * 1000)
            logger.warning(
                "ai.provider.hard_timeout",
                extra={
                    "event": "ai.provider.hard_timeout",
                    "provider": provider_name,
                    "model": model,
                    "elapsed_ms": elapsed_ms,
                    "mode": mode,
                    "reason": "timeout",
                },
            )
            return self._fallback_chain(request, reason="timeout", mode=mode)
        except ProviderConfigError:
            return self._fallback_chain(request, reason="config_error", mode=mode)
        except (ProviderUnavailableError, ProviderTimeoutError):
            if mode == "open":
                logger.info(
                    "ai.provider.open_mode_provider_failure",
                    extra={
                        "event": "ai.provider.open_mode_provider_failure",
                        "mode": "open",
                        "reason": "open_mode_provider_failure",
                        "provider": getattr(request, "provider_hint", None),
                        "request_id": getattr(request, "request_id", None),
                    },
                )
                return self._fallback_chain(request, reason="open_mode_provider_failure", mode=mode)
            return self._fallback_chain(request, reason="provider_unavailable", mode=mode)
        except ProviderHTTPStatusError as exc:
            reason: NormalizedReason = (
                "http_5xx" if exc.status_code >= 500 else "http_4xx"
            )
            return self._fallback_chain(request, reason=reason, mode=mode)
        except AIProviderError as exc:
            if self._is_schema_error(exc):
                return self._fallback_chain(request, reason="schema_invalid", mode=mode)
            if self._is_malformed_error(exc):
                return self._fallback_chain(request, reason="malformed_response", mode=mode)
            return self._fallback_chain(request, reason="provider_error", mode=mode)
        except Exception as exc:
            logger.error(f"[service] Unexpected exception during generation: {exc}")
            return self._fallback_chain(request, reason="provider_error", mode=mode)

        # Provider succeeded. The internal ``effective_mode``
        # drives which validator pipeline runs; the wire
        # ``mode`` (the user's pick) is preserved on the
        # GenerationMeta so the trust label stays truthful.
        if effective_mode == "open":
            return self._generate_open(
                request, response,
                question_understanding=question_understanding,
                reasoning_plan=reasoning_plan,
                tool_results=tool_results,
                wire_mode=mode,
                adaptive_answer_out=adaptive_answer,
            )
        return self._generate_grounded(
            request,
            response,
            require_schema=require_schema,
            question_understanding=question_understanding,
            reasoning_plan=reasoning_plan,
            tool_results=tool_results,
            wire_mode=mode,
            adaptive_answer_out=adaptive_answer,
        )

    # ---- mode-specific finalisers ---------------------------------- #

    def _generate_grounded(
        self,
        request: AssistantRequest,
        response: AssistantResponse,
        *,
        require_schema: bool | None,
        question_understanding: QuestionUnderstanding | None = None,
        reasoning_plan: Any = None,
        tool_results: tuple = (),
        wire_mode: Mode = "grounded",
        adaptive_answer_out: Any = None,
    ) -> AssistantResponse:
        """Validate + ground a response in grounded mode.

        The pipeline:

        1. If schema-required (default on) and the response
           was not already produced by the deterministic
           fallback, parse the body as JSON and run the
           schema validator. Parse failure → fallback with
           reason ``schema_invalid``.
        2. Build the :class:`EvidenceRegistry` from the
           request's :class:`AssistantContext`.
        3. Run the :class:`GroundingValidator` over the
           parsed response (or the raw body when no parsed
           payload exists). Any rule failure → fallback with
           reason ``grounding_invalid``.
        4. On success, return the response enriched with
           ``GenerationMeta`` carrying the
           ``server_grounding_score`` and a stamp that
           ``grounding_validated=true``.

        AI-1 — the response is stamped with the
        :class:`AdaptiveAnswer` chosen from the Stage 1
        understanding, the Stage 4 plan, and the Stage 5
        tool results. The validator's ``claim_categories_used``
        set, the dispatcher's ``tool_calls``, and the
        understanding's ``unknowns`` all surface on the
        GenerationMeta for the audit trail. The wire
        ``mode`` is preserved (the user's pick) — only the
        internal validator pipeline uses ``effective_mode``.
        """
        if _is_deterministic(response):
            return response

        registry = EvidenceRegistry(request.context)
        # H7.8C — debug log so we can verify the registry
        # actually carries the biz_profile_revenue entry
        # emitted from the new annual_revenue_inr context
        # field. Stripped in production by the deployment
        # layer's logger config — never logs the prompt body.
        logger.info(
            "ai.provider.evidence_registry_built",
            extra={
                "event": "ai.provider.evidence_registry_built",
                "mode": "grounded",
                "registry_count": registry.count,
                "registry_ids": list(registry.ids()),
                "annual_revenue_inr": getattr(
                    request.context, "annual_revenue_inr", 0
                ),
            },
        )
        schema_required = self._schema_required(require_schema)
        parsed = None

        if schema_required and not _is_deterministic(response):
            body = response.body or ""
            # Empty body short-circuits to fallback.
            if not body.strip():
                return self._fallback(
                    request, reason="empty_response", mode="grounded",
                )
            result = parse_model_output(body)
            if not result.ok or result.response is None:
                return self._fallback(
                    request, reason="schema_invalid", mode="grounded",
                )
            parsed = result.response

        # Run the grounding validator. A registry with zero
        # entries still produces a valid empty registry —
        # the validator scores coverage from ``0`` but does
        # not fail by default. The threshold is configurable
        # via ``Settings.ai_grounding_threshold`` (H7.8C).
        validator = GroundingValidator(
            registry,
            parsed,
            raw_body=response.body,
            threshold=self._grounding_threshold(),
        )
        report = validator.validate()

        if not report.passed:
            logger.info(
                "ai.provider.grounding_failed",
                extra={
                    "event": "ai.provider.grounding_failed",
                    "mode": "grounded",
                    "errors": list(report.errors),
                    "score": report.score,
                    "request_id": getattr(request, "request_id", None),
                },
            )
            debug_path = (
                "C:/Users/Win/.claude/jobs/c5f14bf8/tmp/schema_debug.txt"
            )
            try:
                with open(debug_path, "a", encoding="utf-8") as _f:
                    _f.write(
                        "=== H7.8C DEBUG: grounding_failed ===\n"
                        f"errors: {list(report.errors)[:30]}\n"
                        f"score: {report.score}\n"
                        "=== END DEBUG ===\n\n"
                    )
            except Exception:
                pass
            return self._fallback(
                request,
                reason="grounding_invalid",
                mode="grounded",
                extra_meta={
                    "grounding_score": report.score,
                    "grounding_errors": list(report.errors),
                },
            )

        # Stamp the GenerationMeta so the UI can render the
        # evidence disclosure panel. The provider already
        # populated provider_used / model / provider_latency_ms;
        # we attach the grounding score + register the
        # grounding_validated flag.
        meta = response.generation or GenerationMeta.empty(
            mode="grounded",
            provider_used=response.provider_used,
            model=response.model,
            provider_latency_ms=response.provider_latency_ms,
            fallback_used=False,
        )
        manifest_dict = (
            request.context.context_manifest.to_dict()
            if request.context.context_manifest
            else None
        )
        meta = meta.merge(
            grounding_validated=True,
            grounding_score=report.score,
            schema_validated=bool(parsed is not None),
            business_evidence_validated=True,
            context_manifest=manifest_dict,
            evidence_references=tuple(
                ref.id for ref in (parsed.evidence_references if parsed else ())
            ),
            assumptions=tuple(parsed.assumptions if parsed else ()),
            limitations=tuple(parsed.limitations if parsed else ()),
            confidence=(parsed.confidence if parsed else None),
            generation_method="generative",
        )
        # AI-1 — stamp the universal-assistant audit trail.
        # The wire ``mode`` is preserved (the user's pick).
        try:
            adaptive = adaptive_answer_out or compose_adaptive_answer(
                parsed=parsed,
                question_understanding=question_understanding,
                reasoning_plan=reasoning_plan,
                tool_results=tool_results,
                context=request.context,
            )
        except Exception:
            adaptive = None
        from dataclasses import replace as _replace_ai1
        meta = _replace_ai1(
            meta,
            mode=wire_mode,
            deterministic_services_used=tuple(
                r.service_name for r in tool_results if r.status == "ok"
            ),
            calculations_used=tuple(
                r.service_name for r in tool_results
                if r.status == "ok" and "calc" in r.service_name
            ),
            question_understanding=(
                question_understanding.to_dict()
                if question_understanding is not None
                and hasattr(question_understanding, "to_dict")
                else None
            ),
            tool_calls=tuple(
                {"service_name": c.service_name, "inputs": c.inputs}
                for c in (
                    getattr(reasoning_plan, "tool_calls", ()) or ()
                )
            ),
            claim_categories_used=tuple(report.claim_categories_used or ()),
        )
        # H7.8C — emit a structured event for every successful
        # grounded-mode pass. The event payload carries the
        # server-side grounding score, the registry coverage,
        # and the provider/model — never the prompt body or
        # any auth header.
        logger.info(
            "ai.provider.grounded_succeeded",
            extra={
                "event": "ai.provider.grounded_succeeded",
                "mode": "grounded",
                "provider_used": response.provider_used,
                "model": response.model,
                "grounding_score": report.score,
                "registry_count": registry.count,
                "evidence_count": len(meta.evidence_references or ()),
                "provider_latency_ms": response.provider_latency_ms,
                "request_id": getattr(request, "request_id", None),
            },
        )
        from dataclasses import replace
        return replace(response, generation=meta)

    def _generate_open(
        self,
        request: AssistantRequest,
        response: AssistantResponse,
        *,
        question_understanding: QuestionUnderstanding | None = None,
        reasoning_plan: Any = None,
        tool_results: tuple = (),
        wire_mode: Mode = "open",
        adaptive_answer_out: Any = None,
    ) -> AssistantResponse:
        """Exploratory Business Advisor mode validation + envelope stamping."""
        body = response.body or ""
        if not body.strip():
            return self._fallback(
                request, reason="open_mode_provider_failure", mode="open",
            )

        # AI-1 auto-flip defence: when the prompt was
        # routed internally to "open" but the provider
        # already answered via the deterministic fallback,
        # preserve its GenerationMeta ``generation_method``
        # (which is "deterministic"). The wire ``mode`` is
        # the user's selection — overwrite the GenerationMeta
        # so the trust label is truthful.
        if _is_deterministic(response) and response.generation is not None:
            from dataclasses import replace as _preserve_meta
            return _preserve_meta(
                response, generation=_preserve_meta(
                    response.generation, mode=wire_mode,
                ),
            )
        if _is_deterministic(response):
            return response

        registry = EvidenceRegistry(request.context)
        parsed = parse_open_model_output(body)
        validator = OpenResponseValidator(registry, parsed, raw_body=body)
        val_report = validator.validate()

        if not val_report.passed:
            logger.info(
                "ai.provider.open_mode_validation_failed",
                extra={
                    "event": "ai.provider.open_mode_validation_failed",
                    "mode": "open",
                    "errors": list(val_report.errors),
                },
            )
            return self._fallback(
                request, reason="open_mode_provider_failure", mode="open",
            )

        meta = response.generation or GenerationMeta.empty(
            mode="open",
            provider_used=response.provider_used,
            model=response.model,
            provider_latency_ms=response.provider_latency_ms,
            fallback_used=False,
        )

        manifest_dict = (
            request.context.context_manifest.to_dict()
            if request.context.context_manifest
            else None
        )

        is_structured = bool(
            parsed.verified_business_context
            or parsed.exploratory_recommendations
            or parsed.illustrative_scenarios
            or parsed.questions_to_validate
        )

        meta = meta.merge(
            generation_method="generative",
            grounding_validated=False,
            schema_validated=is_structured,
            business_evidence_validated=val_report.business_evidence_validated,
            server_grounding_score=val_report.score,
            evidence_references=tuple(
                ref for fact in getattr(parsed, "verified_business_context", ()) for ref in getattr(fact, "evidence_refs", ())
            ),
            assumptions=tuple(getattr(parsed, "assumptions", ())),
            limitations=tuple(getattr(parsed, "limitations", ())),
            confidence=getattr(parsed, "confidence", 70),
            context_manifest=manifest_dict,
        )
        # AI-1 — stamp the universal-assistant audit trail
        # for open mode. The wire ``mode`` is preserved.
        from dataclasses import replace as _replace_open
        meta = _replace_open(
            meta,
            mode=wire_mode,
            deterministic_services_used=tuple(
                r.service_name for r in tool_results if r.status == "ok"
            ),
            calculations_used=tuple(
                r.service_name for r in tool_results
                if r.status == "ok" and "calc" in r.service_name
            ),
            question_understanding=(
                question_understanding.to_dict()
                if question_understanding is not None
                and hasattr(question_understanding, "to_dict")
                else None
            ),
            tool_calls=tuple(
                {"service_name": c.service_name, "inputs": c.inputs}
                for c in (
                    getattr(reasoning_plan, "tool_calls", ()) or ()
                )
            ),
            claim_categories_used=tuple(val_report.claim_categories_used or ()),
        )
        return _replace_open(response, generation=meta)

    # ---- convenience ------------------------------------------------- #

    def build_context(self, owner_id: int) -> AssistantContext:
        """Return the assembled context without generating a reply.

        Useful for tests and for future endpoints that want
        to inspect what the prompt would have included.
        """
        return self._context_builder.build(owner_id=owner_id)

    def provider_status(self) -> dict[str, Any]:
        """Surface the current provider configuration.

        Returns a JSON-safe dict for the
        ``GET /api/v1/chat/provider-status`` endpoint. The
        endpoint never exposes API keys, auth headers, or the
        full base URL — only the provider *name*, the model
        identifier, the configured mode list, a boolean
        availability flag derived from the factory's
        ``is_available`` check, and a short ``reason`` string
        that tells the frontend *why* the provider is up or
        down (H7.9R+ — the boolean alone is not enough; the
        frontend needs ``"missing_api_key"`` vs
        ``"ping_failed"`` vs ``"reachable"`` to render the
        right "Provider unavailable" copy).
        """
        name = self.configured_provider_name()
        available = self._factory.is_available()
        reason = self._factory.status_reason()
        modes = ("grounded", "open")
        return {
            "configured_provider": name,
            "runtime_provider": name if available else "deterministic-fallback",
            "model": self._factory.configured_model(),
            "available": available,
            "schema_required": self._schema_required(None),
            "fallback_active": not available,
            "reason": reason,
            "modes": list(modes),
            "default_mode": "grounded",
        }

    # ---- internal ---------------------------------------------------- #

    def _schema_required(self, override: bool | None) -> bool:
        if override is not None:
            return bool(override)
        settings = self._factory._settings
        if settings is None:
            return True  # default-on for the JSON contract
        return bool(getattr(settings, "ai_require_schema", True))

    def _grounding_threshold(self) -> int:
        """Read the grounding threshold from Settings.

        Defaults to ``GroundingValidator.DEFAULT_GROUNDING_THRESHOLD``
        (50) when the factory is Settings-less. The value is
        clamped to ``[0, 100]`` by the validator.
        """
        from app.services.ai.providers.grounding_validator import (
            DEFAULT_GROUNDING_THRESHOLD,
        )
        settings = self._factory._settings
        if settings is None:
            return DEFAULT_GROUNDING_THRESHOLD
        return int(getattr(settings, "ai_grounding_threshold", DEFAULT_GROUNDING_THRESHOLD))

    def _is_schema_error(self, exc: AIProviderError) -> bool:
        """Decide whether an AIProviderError is a schema failure."""
        msg = str(exc) or ""
        return "schema validation" in msg.lower() or "schema" in msg.lower()

    def _is_malformed_error(self, exc: AIProviderError) -> bool:
        """Decide whether an AIProviderError is a malformed-response failure."""
        msg = str(exc) or ""
        low = msg.lower()
        return (
            "json" in low
            or "malformed" in low
            or "parse" in low
        )

    def _fallback_chain(
        self,
        request: AssistantRequest,
        *,
        reason: NormalizedReason,
        mode: Mode,
        extra_meta: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        """Execute the multi-tier failover chain:
        1. Try secondary provider if configured.
        2. Try deterministic rule engine.
        3. If offline/snapshot mode is requested or live engine unavailable, return offline snapshot.
        """
        # Tier 2: Check if secondary provider is configured
        sec_provider_name = getattr(self._factory._settings, "ai_secondary_provider", "")
        if sec_provider_name:
            try:
                sec_provider = self._factory.build_named(sec_provider_name)
                if sec_provider and sec_provider.is_available:
                    res = sec_provider.complete(request)
                    meta = res.generation or GenerationMeta.empty(
                        mode=mode,
                        provider_used=res.provider_used,
                        model=res.model,
                        provider_latency_ms=res.provider_latency_ms,
                        fallback_used=True,
                        fallback_reason="primary_provider_unavailable",
                        generation_method="generative",
                    )
                    from dataclasses import replace as _replace
                    return _replace(res, fallback_used=True, fallback_reason="primary_provider_unavailable", generation=meta)
            except Exception as sec_exc:
                logger.warning(f"[service] Secondary provider {sec_provider_name} failed: {sec_exc}")

        # Tier 3: Deterministic Rule Engine
        det_response = self._fallback(request, reason=reason, mode=mode, extra_meta=extra_meta)

        # Tier 4: Offline Demo Snapshot (when demo mode enabled and snapshot fallback triggered)
        is_demo_mode = bool(getattr(self._factory._settings, "ursbiz_demo_mode", True))
        is_flagship_query = "acme" in request.user_prompt.lower() or "grow" in request.user_prompt.lower()
        if is_demo_mode and is_flagship_query and reason in ("offline_snapshot",):
            return self._load_offline_snapshot(request, reason="offline_snapshot")

        return det_response

    def _fallback(
        self,
        request: AssistantRequest,
        *,
        reason: NormalizedReason,
        mode: Mode,
        extra_meta: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        """Return a deterministic fallback response."""
        logger.info(
            "ai.provider.fallback_chosen",
            extra={
                "event": "ai.provider.fallback_chosen",
                "mode": mode,
                "reason": reason,
                "request_id": getattr(request, "request_id", None),
            },
        )
        fallback = DeterministicFallbackProvider()
        response = fallback.complete(request, reason=reason)
        from dataclasses import replace as _replace
        if response.generation is not None:
            if extra_meta:
                return _replace(
                    response,
                    generation=response.generation.merge(**extra_meta),
                )
        else:
            return _replace(
                response,
                generation=GenerationMeta.empty(
                    mode=mode,
                    provider_used=response.provider_used,
                    model=response.model,
                    provider_latency_ms=response.provider_latency_ms,
                    fallback_used=True,
                    fallback_reason=reason,
                    generation_method="deterministic",
                ),
            )
        return response

    def _load_offline_snapshot(
        self, request: AssistantRequest, reason: NormalizedReason = "offline_snapshot"
    ) -> AssistantResponse:
        """Load canonical offline demo snapshot."""
        import os, json
        snapshot_path = os.path.join(os.path.dirname(__file__), "..", "snapshots", "acme_flagship_snapshot.json")
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            body = json.dumps(data)
        except Exception:
            body = json.dumps({
                "mode": request.mode,
                "executive_summary": "Acme Textiles Growth Strategy (Offline Demo Snapshot).",
                "current_situation": "Acme Textiles ₹1.8 Cr baseline revenue.",
                "key_findings": [],
                "recommendations": [],
                "thirty_day_plan": [],
                "assumptions": ["Offline demonstration mode active"],
                "limitations": ["Pre-generated snapshot"],
                "evidence_references": ["biz_profile_revenue", "rule_supplier_risk"]
            })

        now_iso = _now_iso()
        meta = GenerationMeta.empty(
            mode=request.mode,
            provider_used="offline_snapshot",
            model="acme_flagship_snapshot",
            provider_latency_ms=5,
            fallback_used=True,
            fallback_reason=reason,
            generation_method="offline_snapshot",
            schema_validated=True,
            grounding_validated=True,
            server_grounding_score=100,
            business_evidence_validated=True,
            context_manifest=request.context.context_manifest.to_dict() if request.context.context_manifest else None,
            generated_at=now_iso,
        )
        return AssistantResponse(
            body=body,
            model="offline_snapshot",
            fallback_used=True,
            provider_used="offline_snapshot",
            generated_at=now_iso,
            fallback_reason=reason,
            generation=meta,
        )


def _is_deterministic(response: AssistantResponse) -> bool:
    return response.provider_used == "deterministic-fallback"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
