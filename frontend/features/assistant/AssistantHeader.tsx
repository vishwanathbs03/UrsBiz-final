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
import { chatService, type ChatProviderStatus } from "@/services/chat-service";
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

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await chatService.fetchProviderStatus();
        if (!cancelled) setProviderStatus(s);
      } catch {
        if (!cancelled) setProviderStatus(null);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <DashboardCard
      badge="AI Assistant"
      title="AI Business Assistant"
      caption={
        providerStatus
          ? `Hybrid AI · ${providerStatus.configured_provider} (${providerStatus.model}) — grounded mode is the default.`
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
        <ProviderStatusPill status={providerStatus} />
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
 * Provider status pill — green when the configured provider
 * is reachable, red when the deterministic fallback is
 * active. Renders the provider *name* and model identifier
 * only; never the base URL, API key, or auth header.
 */
function ProviderStatusPill({
  status,
}: {
  status: ChatProviderStatus | null;
}) {
  if (!status) {
    return (
      <span className="inline-flex items-center gap-1.5 text-muted-foreground">
        <Sparkles className="size-3.5 text-primary" aria-hidden="true" />
        Provider status…
      </span>
    );
  }
  if (status.available && !status.fallback_active) {
    return (
      <span
        className="inline-flex items-center gap-1.5 text-emerald-700"
        data-testid="provider-status-pill"
        data-state="available"
      >
        <CircleCheck className="size-3.5" aria-hidden="true" />
        {status.configured_provider} ({status.model}) connected
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1.5 text-amber-700"
      data-testid="provider-status-pill"
      data-state="fallback"
    >
      <CircleAlert className="size-3.5" aria-hidden="true" />
      Provider unavailable — using rule engine
    </span>
  );
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
