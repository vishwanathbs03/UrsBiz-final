"""ProviderFactory — Sprint 7 Part 2 + H7.3.

Selects the concrete provider at runtime from
``Settings.ai_provider``. Always returns a usable provider —
when the configured provider is unavailable, the factory
returns the deterministic fallback rather than raising.

Selection rules
---------------

  * ``ai_provider == "ollama"``             -> construct :class:`OllamaProvider`
                                                and ping it. If the ping fails,
                                                return the deterministic
                                                fallback instead.
  * ``ai_provider == "openai_compatible"``  -> construct
                                                :class:`OpenAICompatibleProvider`
                                                (the H7.3 generic path that
                                                covers OpenAI, OpenRouter,
                                                Together, Groq, vLLM, llama.cpp
                                                server mode, and Ollama's
                                                ``/v1/chat`` adapter). If the
                                                ping fails, return the
                                                deterministic fallback.
  * anything else (including ``"placeholder"``, ``"disabled"``,
    ``""``, ``None``)                      -> return the deterministic
                                                fallback directly.

The factory is the *only* place in the layer that decides
which provider to use. The :class:`AssistantProviderService`
takes the factory's output and treats it as an opaque
:class:`Provider` — it never asks which concrete class it got.
That separation is what keeps the verifier simple: it asks
the factory for "the configured provider or fallback", looks at
``provider_used`` on the response, and the rest is opaque.
"""
from __future__ import annotations

from typing import Any

from app.services.ai.providers.base import (
    AIProviderError,
    AssistantContext,
    AssistantRequest,
    AssistantResponse,
    DeterministicFallbackProvider,
    Provider,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.services.ai.providers.ollama import OllamaProvider
from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider


class ProviderFactory:
    """Construct a :class:`Provider` from a Settings-like object.

    The factory accepts any object with the relevant attributes
    (``ai_provider``, ``ollama_base_url``, ``ollama_model``,
    ``ai_base_url``, ``ai_model``, ``ai_api_key``,
    ``ai_request_timeout_seconds``, ``ai_require_schema``) — this is
    the same shape :class:`app.config.settings.Settings` exposes.
    Tests can pass a tiny stub.
    """

    def __init__(self, settings: Any | None = None) -> None:
        self._settings = settings

    # ---- public API -------------------------------------------------- #

    def build(self) -> Provider:
        """Build the configured provider, falling back on failure."""
        provider_name = self._provider_name()
        if provider_name == "ollama":
            return self._build_ollama_or_fallback()
        if provider_name == "openai_compatible":
            return self._build_openai_compatible_or_fallback()
        # Any other value: deterministic fallback only.
        return DeterministicFallbackProvider()

    def fallback(self) -> Provider:
        """Return the deterministic fallback unconditionally."""
        return DeterministicFallbackProvider()

    # ---- internal ---------------------------------------------------- #

    def _provider_name(self) -> str:
        if self._settings is None:
            return ""
        return str(getattr(self._settings, "ai_provider", "") or "").strip().lower()

    def _build_ollama_or_fallback(self) -> Provider:
        base_url = str(
            getattr(self._settings, "ollama_base_url", "") or ""
        ).strip()
        model = str(
            getattr(self._settings, "ollama_model", "") or "llama3.1"
        ).strip()
        timeout = float(
            getattr(self._settings, "ai_request_timeout_seconds", 60.0) or 60.0
        )
        provider = OllamaProvider(
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        # A successful ping means Ollama is reachable; the
        # factory hands the real provider to the service. A
        # failed ping is *not* an error — we just fall back.
        if provider.ping():
            return provider
        # Make sure the unused client is closed so the test
        # runner doesn't leak sockets.
        provider.close()
        return DeterministicFallbackProvider()

    def _build_openai_compatible_or_fallback(self) -> Provider:
        """Build the OpenAI-compatible provider, falling back on failure.

        The settings consumed are the four H7.3 keys:

          * ``ai_base_url``  — required for the real provider.
          * ``ai_model``     — required for the real provider.
          * ``ai_api_key``   — optional (local servers ignore auth).
          * ``ai_require_schema`` — toggle JSON-mode validation.

        If ``ai_base_url`` or ``ai_model`` is empty, the configurable
        provider is treated as "not configured" and the factory
        returns the deterministic fallback. Same for any ping failure.
        """
        base_url = str(
            getattr(self._settings, "ai_base_url", "") or ""
        ).strip()
        model = str(
            getattr(self._settings, "ai_model", "") or ""
        ).strip()
        api_key = str(
            getattr(self._settings, "ai_api_key", "") or ""
        ).strip()
        timeout = float(
            getattr(self._settings, "ai_request_timeout_seconds", 60.0) or 60.0
        )
        require_json = bool(
            getattr(self._settings, "ai_require_schema", True)
        )
        if not base_url or not model:
            # Not configured -> fall back silently. Same UX as
            # an Ollama provider whose URL is empty.
            return DeterministicFallbackProvider()
        provider = OpenAICompatibleProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            require_json=require_json,
        )
        if provider.ping():
            return provider
        provider.close()
        return DeterministicFallbackProvider()

    # ---- convenience for callers that want to inspect the name ------- #

    def configured_provider_name(self) -> str:
        """Return the configured provider name (lower-cased).

        Useful for logging and for the verifier. The actual
        runtime provider may be the fallback — check
        ``AssistantResponse.provider_used`` for that.
        """
        return self._provider_name()

    def configured_model(self) -> str:
        """Return the configured model identifier.

        ``"ollama"`` → ``Settings.ollama_model`` (default
        ``"llama3.1"``). ``"openai_compatible"`` →
        ``Settings.ai_model``. Any other provider name → ``""``.
        """
        name = self._provider_name()
        if name == "ollama":
            return str(
                getattr(self._settings, "ollama_model", "llama3.1") or "llama3.1"
            ).strip()
        if name == "openai_compatible":
            return str(
                getattr(self._settings, "ai_model", "") or ""
            ).strip()
        return ""

    def is_available(self) -> bool:
        """Return True iff the configured real provider can be reached.

        The check runs the same path ``build()`` would: a real
        Ollama / OpenAI-compatible ping. A failed ping returns
        False; the factory's ``build()`` will return the
        deterministic fallback in that case.

        This is the signal the
        ``GET /api/v1/chat/provider-status`` endpoint surfaces.
        """
        name = self._provider_name()
        if name == "ollama":
            base_url = str(
                getattr(self._settings, "ollama_base_url", "") or ""
            ).strip()
            if not base_url:
                return False
            try:
                provider = OllamaProvider(
                    base_url=base_url,
                    model=str(
                        getattr(self._settings, "ollama_model", "llama3.1")
                        or "llama3.1"
                    ).strip(),
                    timeout=5.0,
                )
                ok = provider.ping()
                provider.close()
                return ok
            except Exception:
                return False
        if name == "openai_compatible":
            base_url = str(
                getattr(self._settings, "ai_base_url", "") or ""
            ).strip()
            model = str(
                getattr(self._settings, "ai_model", "") or ""
            ).strip()
            if not base_url or not model:
                return False
            try:
                provider = OpenAICompatibleProvider(
                    base_url=base_url,
                    model=model,
                    api_key=str(
                        getattr(self._settings, "ai_api_key", "") or ""
                    ).strip(),
                    timeout=5.0,
                    require_json=bool(
                        getattr(self._settings, "ai_require_schema", True)
                    ),
                )
                ok = provider.ping()
                provider.close()
                return ok
            except Exception:
                return False
        return False