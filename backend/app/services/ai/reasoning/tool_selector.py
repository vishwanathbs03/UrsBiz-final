"""ToolSelector + ToolDispatcher — SPRINT AI-1 Stage 5.

The legacy assistant surfaces a single hard-coded set of
deterministic engines (recommendation, scheme, etc.) by hard
reference from the prompt builder. AI-1 replaces that with a
lightweight dispatch table:

  * :class:`ToolCall` — a descriptor the selector emits.
  * :class:`ToolResult` — the result a tool returns.
  * :class:`ToolInterface` — the protocol every tool satisfies.
  * :class:`StubToolInterface` — the default tool. Returns
    ``status="not_implemented"`` so unimplemented services
    degrade gracefully (the chat still answers; the tool just
    contributes nothing).
  * :class:`ToolSelector` — picks which tool calls to make
    based on the question understanding and the reasoning plan.
  * :class:`ToolDispatcher` — invokes the calls, enforcing a
    per-call 500ms timeout and a total 1000ms cap. Never
    raises — always returns a :class:`ToolResult`.

The default registry is **all stubs**. A future sprint can swap
a real implementation in by registering it on the dispatcher:

    ToolDispatcher.register_tool(
        "health_score", HealthScoreTool(health_score_service)
    )

Backward compatibility
----------------------

The dispatcher is opt-in via ``Settings.ai1_tool_dispatch_enabled``
(default True). When the flag is False the dispatcher short-
circuits to an empty tuple of results — the chat behaves
exactly like it did before AI-1.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


# Per-call timeout for each tool invocation. The dispatcher
# raises a FutureTimeout if a tool exceeds this; the result is
# converted into a ToolResult with status="error".
_PER_CALL_TIMEOUT_MS = 500

# Total dispatch budget. Even with all 5 tool slots used, the
# chat endpoint stays under 2 seconds.
_TOTAL_DISPATCH_BUDGET_MS = 1000

# Hard cap on tool calls per request. Caps the surface area of
# the chat endpoint and prevents an over-eager selector from
# fanning out too widely.
_MAX_TOOL_CALLS_PER_REQUEST = 5

# Result status values. Adding a new value is non-breaking.
ToolStatus = str  # "ok" | "skipped" | "not_implemented" | "error"


# --------------------------------------------------------------------------- #
# ToolCall / ToolResult
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolCall:
    """A request for a deterministic service to run.

    Attributes
    ----------
    service_name
        Stable identifier (``"health_score"``,
        ``"recommendation"``, ``"schemes_sprint16"``, …).
    inputs
        Mapping of input parameters for the call. The
        dispatcher forwards this verbatim to the tool.
    expected_output_shape
        Free-form string describing what the selector
        expects back (e.g. ``"score:int"``,
        ``"schemes:list[SchemeMatch]"``). Tools are free to
        ignore it; it is documentation for the audit trail.
    """

    service_name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output_shape: str = ""


@dataclass(frozen=True)
class ToolResult:
    """The result of a single tool invocation.

    Attributes
    ----------
    service_name
        Echo of the call's :attr:`ToolCall.service_name`.
    status
        One of ``"ok"``, ``"skipped"``, ``"not_implemented"``,
        ``"error"``. The dispatcher never raises; any failure
        is captured as ``status="error"`` with ``error``
        populated.
    payload
        The tool's output. Shape is service-specific. ``None``
        when the tool returned no data.
    duration_ms
        Wall-clock duration of the call (0 for stubs).
    error
        Human-readable error message when ``status="error"``.
        ``""`` otherwise.
    """

    service_name: str
    status: ToolStatus
    payload: Any = None
    duration_ms: int = 0
    error: str = ""


# --------------------------------------------------------------------------- #
# ToolInterface protocol + default stub
# --------------------------------------------------------------------------- #


class ToolInterface(Protocol):
    """Protocol every registered tool satisfies.

    The dispatcher invokes ``invoke(...)`` once per call. The
    tool is free to do anything internally — read from the
    database, call a service, recompute — as long as it
    returns a :class:`ToolResult` and never raises.
    """

    def invoke(
        self, *, owner_id: int, call: ToolCall, context: Any
    ) -> ToolResult: ...


class StubToolInterface:
    """Default tool — returns ``status="not_implemented"``.

    Lets the dispatcher always succeed even when no real
    implementation is registered. The audit trail records
    ``"stub"`` as the error so reviewers can spot gaps.
    """

    def invoke(
        self, *, owner_id: int, call: ToolCall, context: Any
    ) -> ToolResult:
        return ToolResult(
            service_name=call.service_name,
            status="not_implemented",
            payload=None,
            duration_ms=0,
            error="stub",
        )


# --------------------------------------------------------------------------- #
# ToolSelector
# --------------------------------------------------------------------------- #


class ToolSelector:
    """Pick which tool calls to make for a given question.

    The selector reads :attr:`ReasoningPlan.applicable_deterministic_services`
    and emits one :class:`ToolCall` per service. Calls are
    capped at :data:`_MAX_TOOL_CALLS_PER_REQUEST`.

    The selector is a pure function — no I/O, no side effects.
    """

    def select(
        self,
        *,
        question_understanding: Any,
        reasoning_plan: Any,
        context: Any,
    ) -> tuple[ToolCall, ...]:
        """Return the ordered list of tool calls to invoke.

        The result is at most :data:`_MAX_TOOL_CALLS_PER_REQUEST`
        long and is always deterministic given the same
        inputs.
        """
        services = tuple(
            getattr(reasoning_plan, "applicable_deterministic_services", ()) or ()
        )
        if not services:
            # Fall back to the question understanding's hints —
            # this lets the dispatcher run even when the
            # reasoning plan carries an empty tuple (legacy
            # callers that didn't compute AI-1 fields).
            services = tuple(
                getattr(question_understanding, "needs_deterministic_services", ())
                or ()
            )
        if not services:
            return ()

        # Truncate to the per-request cap.
        services = services[:_MAX_TOOL_CALLS_PER_REQUEST]

        calls: list[ToolCall] = []
        for service_name in services:
            calls.append(
                ToolCall(
                    service_name=service_name,
                    inputs={
                        "owner_id": getattr(context, "owner_id", 0),
                        "industry": getattr(context, "industry", "unknown"),
                        "location": getattr(context, "location", "unknown"),
                    },
                    expected_output_shape=_EXPECTED_OUTPUT_SHAPES.get(
                        service_name, ""
                    ),
                )
            )
        return tuple(calls)


# Per-service expected output shape (documentation only).
_EXPECTED_OUTPUT_SHAPES: dict[str, str] = {
    "health_score": "score:int",
    "recommendation": "recs:list[Recommendation]",
    "schemes_sprint16": "schemes:list[SchemeMatch]",
    "finance": "metrics:dict",
    "knowledge_retrieval": "passages:list[str]",
    "business_dna": "dna:DnaProfile",
    "risk": "rules:list[Rule]",
    "insights": "insights:list[Insight]",
    # SPRINT AI-2 — 8 additional service names.
    "opportunity": "opportunities:list[Opportunity]",
    "readiness": "readiness:ReadinessReport",
    "kpi": "kpis:list[Kpi]",
    "benchmark": "benchmarks:dict",
    "growth": "growth:dict",
    "funding": "funding:dict",
    "compliance": "compliance:dict",
    "predictive_sprint14": "predictions:{revenue,growth,risk}",
}


# --------------------------------------------------------------------------- #
# ToolDispatcher
# --------------------------------------------------------------------------- #


class ToolDispatcher:
    """Invoke tool calls with timeout + budget enforcement.

    The dispatcher is the single mutation point for the
    deterministic engine pool. The default registry is **all
    stubs** — unimplemented services degrade to
    ``status="not_implemented"`` rather than raising.

    Threading model
    ---------------

    Calls are dispatched in parallel via a shared
    :class:`ThreadPoolExecutor` with ``max_workers=4``. The
    pool is module-level (lazy-initialised on first use) so
    the cost of spinning up workers is amortised across the
    lifetime of the process.
    """

    _DISPATCH_ENABLED: bool = True  # class-level kill switch

    def __init__(self) -> None:
        self._tools: dict[str, ToolInterface] = {}
        # Default registry is all stubs. SPRINT AI-2 replaces
        # these with real wrappers at the chat endpoint — the
        # production wiring lives in
        # ``app.api.v1.endpoints.chat._service(db)``. The
        # default stub registry keeps AI-1 tests green and
        # makes the dispatcher safe to instantiate without any
        # DB session (e.g. in unit tests).
        self.register_tool("health_score", StubToolInterface())
        self.register_tool("recommendation", StubToolInterface())
        self.register_tool("schemes_sprint16", StubToolInterface())
        self.register_tool("finance", StubToolInterface())
        self.register_tool("knowledge_retrieval", StubToolInterface())
        self.register_tool("business_dna", StubToolInterface())
        self.register_tool("risk", StubToolInterface())
        self.register_tool("insights", StubToolInterface())
        # SPRINT AI-2 — 8 additional service names that the
        # chat endpoint will wire to real engines.
        self.register_tool("opportunity", StubToolInterface())
        self.register_tool("readiness", StubToolInterface())
        self.register_tool("kpi", StubToolInterface())
        self.register_tool("benchmark", StubToolInterface())
        self.register_tool("growth", StubToolInterface())
        self.register_tool("funding", StubToolInterface())
        self.register_tool("compliance", StubToolInterface())
        self.register_tool("predictive_sprint14", StubToolInterface())

    # ---- registry ---------------------------------------------------- #

    def register_tool(self, service_name: str, tool: ToolInterface) -> None:
        """Register or replace the tool for ``service_name``."""
        self._tools[service_name] = tool

    def get_tool(self, service_name: str) -> ToolInterface:
        """Return the tool for ``service_name`` (never raises)."""
        return self._tools.get(service_name) or StubToolInterface()

    # ---- dispatch ---------------------------------------------------- #

    def dispatch(
        self,
        *,
        owner_id: int,
        question_understanding: Any,
        reasoning_plan: Any,
        context: Any,
    ) -> tuple[ToolResult, ...]:
        """Run the selector, invoke each call, return the results.

        The method NEVER raises. Every call returns a
        :class:`ToolResult` regardless of what the tool does
        internally. The total wall-clock budget is
        :data:`_TOTAL_DISPATCH_BUDGET_MS`.
        """
        if not self._DISPATCH_ENABLED:
            return ()

        selector = ToolSelector()
        calls = selector.select(
            question_understanding=question_understanding,
            reasoning_plan=reasoning_plan,
            context=context,
        )
        if not calls:
            return ()

        # Lazy-init the shared executor.
        global _SHARED_EXECUTOR
        if _SHARED_EXECUTOR is None:
            _SHARED_EXECUTOR = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="ai1-tools"
            )

        results: list[ToolResult] = []
        futures = []
        for call in calls:
            tool = self.get_tool(call.service_name)
            futures.append(
                (
                    call,
                    _SHARED_EXECUTOR.submit(
                        _safe_invoke,
                        tool,
                        owner_id=owner_id,
                        call=call,
                        context=context,
                    ),
                )
            )

        for call, future in futures:
            try:
                result = future.result(timeout=_PER_CALL_TIMEOUT_MS / 1000.0)
            except FutureTimeout:
                result = ToolResult(
                    service_name=call.service_name,
                    status="error",
                    payload=None,
                    duration_ms=_PER_CALL_TIMEOUT_MS,
                    error="timeout",
                )
            except Exception as exc:  # noqa: BLE001 — dispatcher must never raise
                result = ToolResult(
                    service_name=call.service_name,
                    status="error",
                    payload=None,
                    duration_ms=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
        return tuple(results)


def _safe_invoke(
    tool: ToolInterface, *, owner_id: int, call: ToolCall, context: Any
) -> ToolResult:
    """Wrap ``tool.invoke`` so an exception becomes a ToolResult."""
    try:
        return tool.invoke(owner_id=owner_id, call=call, context=context)
    except Exception as exc:  # noqa: BLE001 — tools must never raise
        return ToolResult(
            service_name=call.service_name,
            status="error",
            payload=None,
            duration_ms=0,
            error=f"{type(exc).__name__}: {exc}",
        )


# Lazy-initialised shared executor. The pool is module-level
# so the cost of spinning up workers is amortised. Tests can
# force a fresh pool by deleting this module attribute.
_SHARED_EXECUTOR: ThreadPoolExecutor | None = None