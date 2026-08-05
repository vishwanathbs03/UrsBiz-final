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

import logging
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
from app.services.ai.providers.grounding_validator import GroundingValidator
from app.services.ai.providers.prompt_builder import AssistantPromptBuilder
from app.services.ai.providers.response_schema import parse_model_output


# Module-level logger for structured provider events.
# The deployment /monitoring layer redacts known secret keys
# (``AI_API_KEY``, ``Authorization``, ``Cookie`` …) before
# these records hit disk — see ``app/monitoring/logging.py``.
logger = logging.getLogger("atlas.ai.provider")


class AssistantProviderService:
    """The public façade for the AI Provider Layer."""

    def __init__(
        self,
        *,
        context_builder: AssistantContextBuilder,
        prompt_builder: AssistantPromptBuilder | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder or AssistantPromptBuilder()
        # The factory defaults to a Settings-less instance,
        # which always returns the deterministic fallback.
        # Callers that want a real provider must pass a
        # factory built from a Settings (or stub).
        self._factory = provider_factory or ProviderFactory()

    # ---- public API -------------------------------------------------- #

    def configured_provider_name(self) -> str:
        """Return the name of the configured provider.

        The actual runtime provider may be the fallback even
        when the configured provider is "ollama" — Ollama can
        be offline at request time. Use
        :attr:`AssistantResponse.provider_used` for the
        ground truth.
        """
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
        """Generate a reply for the user's prompt.

        ``owner_id`` is the Business owner (the same value
        the upstream services use). ``user_prompt`` is the
        literal text the user typed in the assistant.
        ``history`` is the prior conversation (caller-owned).
        ``provider`` is an optional override — pass a stub
        from a test, or pass the fallback explicitly to
        bypass the factory's runtime check.
        ``require_schema`` is an optional override for the
        H7.3 JSON-mode validation — when True, the service
        parses the body via ``response_schema.parse_model_output``
        and falls back to deterministic on validation failure.
        When None (default), the service uses
        ``Settings.ai_require_schema``.
        ``mode`` selects the hybrid mode. ``"grounded"`` is
        the default; ``"open"`` bypasses the registry and the
        schema validator.

        Returns an :class:`AssistantResponse`. The
        ``fallback_used`` flag tells the caller whether the
        body came from a real LLM or the deterministic
        fallback; ``generation.fallback_reason`` tells the
        caller *why* the fallback was chosen.
        """
        context = self._context_builder.build(owner_id=owner_id)
        request = self._prompt_builder.build(
            context=context,
            user_prompt=user_prompt,
            history=history,
            knowledge=knowledge,
            mode=mode,
        )
        chosen = provider or self._factory.build()
        try:
            response = chosen.complete(request)
        except ProviderRateLimitError:
            # Rate-limited: still soft-fail — the next request
            # may be allowed. We treat this the same as a
            # generic provider_unavailable for the fallback
            # contract, but the fallback_reason is "rate_limited"
            # so the verifier can prove what happened.
            return self._fallback(
                request, reason="rate_limited", mode=mode,
            )
        except ProviderHTTPStatusError as exc:
            # Map 4xx → http_4xx, 5xx → http_5xx.
            reason: NormalizedReason = (
                "http_5xx" if exc.status_code >= 500 else "http_4xx"
            )
            return self._fallback(request, reason=reason, mode=mode)
        except (ProviderUnavailableError, ProviderTimeoutError):
            # Open-mode provider failure uses a dedicated reason
            # so the UI can label the message differently —
            # "Open-domain LLM is unavailable" vs the grounded
            # "Provider unavailable".
            if mode == "open":
                return self._fallback(
                    request, reason="open_mode_provider_failure", mode=mode,
                )
            return self._fallback(
                request, reason="provider_unavailable", mode=mode,
            )
        except AIProviderError as exc:
            # H7.3: schema-validation failures get graceful
            # treatment. Other AIProviderError propagate.
            if self._is_schema_error(exc):
                return self._fallback(
                    request, reason="schema_invalid", mode=mode,
                )
            if self._is_malformed_error(exc):
                return self._fallback(
                    request, reason="malformed_response", mode=mode,
                )
            raise

        # Provider returned. Now run the mode-specific validation
        # pipeline. Open mode passes the raw body through with no
        # further checks; grounded mode validates + grounds.
        if mode == "open":
            return self._generate_open(request, response)
        return self._generate_grounded(
            request,
            response,
            require_schema=require_schema,
        )

    # ---- mode-specific finalisers ---------------------------------- #

    def _generate_grounded(
        self,
        request: AssistantRequest,
        response: AssistantResponse,
        *,
        require_schema: bool | None,
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
        """
        registry = EvidenceRegistry(request.context)
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
        # not fail by default.
        validator = GroundingValidator(registry, parsed, raw_body=response.body)
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
        meta = meta.merge(
            grounding_validated=True,
            grounding_score=report.score,
            schema_validated=bool(parsed is not None),
            evidence_refs=tuple(
                ref.id for ref in (parsed.evidence_references if parsed else ())
            ),
            assumptions=tuple(parsed.assumptions if parsed else ()),
            limitations=tuple(parsed.limitations if parsed else ()),
            confidence=(parsed.confidence if parsed else None),
            generation_method="generative",
        )
        # ``AssistantResponse`` is a frozen dataclass; we
        # use ``dataclasses.replace`` to clone it with the
        # new ``generation`` envelope attached.
        from dataclasses import replace
        return replace(response, generation=meta)

    def _generate_open(
        self,
        request: AssistantRequest,
        response: AssistantResponse,
    ) -> AssistantResponse:
        """Pass-through for open mode.

        Open mode has no schema and no registry. The provider's
        body is the answer. We still stamp a ``GenerationMeta``
        so the UI can render the ``open_domain`` trust badge
        with provider/model disclosure.
        """
        body = response.body or ""
        if not body.strip():
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
        meta = meta.merge(
            generation_method="generative",
            grounding_validated=False,
            schema_validated=False,
            evidence_refs=(),
            assumptions=(),
            limitations=(),
        )
        from dataclasses import replace
        return replace(response, generation=meta)

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
        identifier, the configured mode list, and a boolean
        availability flag derived from the factory's
        ``is_available`` check.
        """
        name = self.configured_provider_name()
        available = self._factory.is_available()
        modes = ("grounded", "open")
        return {
            "configured_provider": name,
            "runtime_provider": name if available else "deterministic-fallback",
            "model": self._factory.configured_model(),
            "available": available,
            "schema_required": self._schema_required(None),
            "fallback_active": not available,
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

    def _fallback(
        self,
        request: AssistantRequest,
        *,
        reason: NormalizedReason,
        mode: Mode,
        extra_meta: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        """Return a deterministic fallback response.

        ``reason`` is stamped on the response via the
        ``fallback_reason`` field *and* on the
        ``generation.fallback_reason`` field, so the verifier
        and the frontend can prove the graceful-degradation
        contract end-to-end.
        """
        fallback = DeterministicFallbackProvider()
        response = fallback.complete(request, reason=reason)
        # The deterministic fallback stamps its own
        # GenerationMeta. We add the requested extras here
        # so the audit trail captures the precise reason the
        # fallback was chosen.
        from dataclasses import replace as _replace
        if response.generation is not None:
            if extra_meta:
                return _replace(
                    response,
                    generation=response.generation.merge(**extra_meta),
                )
        else:
            # Defensive: build a meta if the fallback
            # provider did not produce one.
            return _replace(
                response,
                generation=GenerationMeta.empty(
                    mode=mode,
                    provider_used=response.provider_used,
                    model=response.model,
                    provider_latency_ms=response.provider_latency_ms,
                    fallback_used=True,
                    fallback_reason=reason,
                ),
            )
        return response


def _is_deterministic(response: AssistantResponse) -> bool:
    return response.provider_used == "deterministic-fallback"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
