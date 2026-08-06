/**
 * Assistant header — page title, last-analysis timestamp,
 * Refresh and Clear Chat actions. Mirrors the Insights
 * header pattern so the visual rhythm is consistent
 * across the app.
 *
 * H7.8C — the "Deterministic · local" pill is replaced by
 * a live provider-status indicator sourced from
 * ``GET /api/v1/chat/provider-status``. The pill renders
 * a green dot + provider name when the configured
 * provider is reachable and a red dot + "rule engine"
 * label when the deterministic fallback is active.
 *
 * The endpoint is auth-gated and exposes no secrets, base
 * URL, or API key — only the provider *name*, model, and
 * an availability flag.
 */

"use client";

import { useEffect, useState } from "react";
import { Building2, CircleAlert, CircleCheck, RefreshCcw, Sparkles, Trash2 } from "lucide-react";
import { DashboardCard } from "@/components/dashboard/DashboardCard";
import { Button } from "@/components/ui/button";
import { ApiError, chatService, type ChatProviderStatus } from "@/services/chat-service";
import { cn } from "@/lib/utils";

interface AssistantHeaderProps {
  lastAnalyzedAt: string | null;
  isFetching: boolean;
  onRefresh: () => void;
  onClear: () => void;
  messageCount: number;
  /** Optional right-aligned controls (e.g. server-history toggle). */
  rightSlot?: React.ReactNode;
}

/**
 * H7.8C — provider-status fetch state machine. The pill
 * surfaces one of four mutually-exclusive states:
 *
 *   loading    — request is in flight or not yet completed
 *   available  — backend returned 200, ``available=true``,
 *                ``fallback_active=false`` → the configured
 *                provider is reachable
 *   fallback   — backend returned 200, ``fallback_active=true``
 *                or ``available=false`` → the deterministic
 *                UrsBiz fallback is currently in use
 *   auth       — backend returned 401 → the user is genuinely
 *                unauthenticated; the Assistant page should
 *                already have been guarded by the auth
 *                middleware, so this state is mostly defensive
 *   error      — any other failure (network, 5xx) → render a
 *                neutral "unavailable" message and never echo
 *                the raw backend payload
 *
 * Raw backend JSON is NEVER rendered. Only the canonical
 * provider name and model identifier are surfaced.
 */
type ProviderState =
  | { kind: "loading" }
  | { kind: "available"; provider: string; model: string }
  | { kind: "fallback" }
  | { kind: "auth" }
  | { kind: "error" };

function toProviderState(
  status: ChatProviderStatus | null,
  loadError: unknown | null,
): ProviderState {
  if (status) {
    if (status.available && !status.fallback_active) {
      return {
        kind: "available",
        provider: status.configured_provider,
        model: status.model,
      };
    }
    return { kind: "fallback" };
  }
  if (loadError) {
    if (loadError instanceof ApiError && loadError.isUnauthenticated) {
      return { kind: "auth" };
    }
    return { kind: "error" };
  }
  return { kind: "loading" };
}

export function AssistantHeader({
  lastAnalyzedAt,
  isFetching,
  onRefresh,
  onClear,
  messageCount,
  rightSlot,
}: AssistantHeaderProps) {
  const [providerStatus, setProviderStatus] = useState<ChatProviderStatus | null>(
    null,
  );
  const [providerError, setProviderError] = useState<unknown | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await chatService.fetchProviderStatus();
        if (!cancelled) {
          setProviderStatus(s);
          setProviderError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setProviderStatus(null);
          setProviderError(err);
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const providerState = toProviderState(providerStatus, providerError);

  return (
    <DashboardCard
      badge="AI Assistant"
      title="AI Business Assistant"
      caption={
        providerState.kind === "available"
          ? `Hybrid AI · ${providerState.provider} (${providerState.model}) — grounded mode is the default.`
          : "Ask anything about your business. Responses are grounded in the Digital Twin, Recommendations, Roadmap, Insights, and Rules."
      }
      trailing={
        <div className="flex items-center gap-2">
          {rightSlot}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={isFetching}
            aria-label={isFetching ? "Refreshing assistant data" : "Refresh assistant data"}
          >
            <RefreshCcw
              className={cn(
                "size-4 transition-transform",
                isFetching && "animate-spin",
              )}
              aria-hidden="true"
            />
            <span className="hidden sm:inline">
              {isFetching ? "Refreshing" : "Refresh"}
            </span>
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClear}
            disabled={messageCount === 0}
            aria-label="Clear chat"
          >
            <Trash2 className="size-4" aria-hidden="true" />
            <span className="hidden sm:inline">Clear Chat</span>
          </Button>
        </div>
      }
    >
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <ProviderStatusPill state={providerState} />
        <span className="inline-flex items-center gap-1.5">
          <Building2 className="size-3.5 text-primary" aria-hidden="true" />
          Last analysis
        </span>
        <span className="font-mono text-foreground">
          {lastAnalyzedAt ? formatTimestamp(lastAnalyzedAt) : "—"}
        </span>
        {messageCount > 0 && (
          <span className="ml-auto text-foreground/70">
            {messageCount} message{messageCount === 1 ? "" : "s"}
          </span>
        )}
      </div>
    </DashboardCard>
  );
}

/**
 * Provider status pill — renders one of the four canonical
 * states defined by ``ProviderState``. Never echoes raw backend
 * JSON, never displays the API key, base URL, or upstream
 * authorization header.
 */
function ProviderStatusPill({ state }: { state: ProviderState }) {
  switch (state.kind) {
    case "loading":
      return (
        <span
          className="inline-flex items-center gap-1.5 text-muted-foreground"
          data-testid="provider-status-pill"
          data-state="loading"
        >
          <Sparkles className="size-3.5 text-primary animate-pulse" aria-hidden="true" />
          Checking provider…
        </span>
      );
    case "available":
      return (
        <span
          className="inline-flex items-center gap-1.5 text-emerald-700"
          data-testid="provider-status-pill"
          data-state="available"
        >
          <CircleCheck className="size-3.5" aria-hidden="true" />
          {state.provider} connected ({state.model})
        </span>
      );
    case "fallback":
      return (
        <span
          className="inline-flex items-center gap-1.5 text-amber-700"
          data-testid="provider-status-pill"
          data-state="fallback"
        >
          <CircleAlert className="size-3.5" aria-hidden="true" />
          Using UrsBiz verified intelligence — fallback active
        </span>
      );
    case "auth":
      return (
        <span
          className="inline-flex items-center gap-1.5 text-muted-foreground"
          data-testid="provider-status-pill"
          data-state="auth"
        >
          <CircleAlert className="size-3.5" aria-hidden="true" />
          Sign in to use AI Assistant
        </span>
      );
    case "error":
      return (
        <span
          className="inline-flex items-center gap-1.5 text-muted-foreground"
          data-testid="provider-status-pill"
          data-state="error"
        >
          <CircleAlert className="size-3.5" aria-hidden="true" />
          AI provider unavailable — UrsBiz fallback ready
        </span>
      );
  }
}

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}
