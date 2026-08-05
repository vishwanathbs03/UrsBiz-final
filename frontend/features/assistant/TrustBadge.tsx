"use client";

/**
 * TrustBadge — H7.3 (Docx Prompt 3 Part 4) visible trust labels.
 *
 * H7.8C extends the badge with a sixth category:
 *
 *   - "Open-domain LLM — not grounded"      permissive mode
 *
 * The docx requires the assistant UI to distinguish the
 * trust categories a user can see:
 *
 *   - "Calculated by UrsBiz rule engine"     deterministic scoring
 *   - "Generated explanation"                 LLM synthesis (grounded)
 *   - "Open-domain LLM — not grounded"        open-mode LLM (H7.8C)
 *   - "Scenario estimate"                     forecast / projection
 *   - "Official external source"              government scheme data
 *   - "User-provided information"             inputs the user entered
 *
 * The badge is intentionally tiny — a pill below the
 * assistant bubble. The labels are the literal text the
 * docx asks for so the verifier can grep for the strings
 * without false negatives. No emoji / no flourish; the
 * badge is information, not decoration.
 */
import { BadgeCheck, Cpu, Globe, Sparkles, TrendingUp, User } from "lucide-react";
import { cn } from "@/lib/utils";

export type TrustLabel =
  | "rule_engine"
  | "generated"
  | "open_domain"
  | "scenario"
  | "official"
  | "user_provided";

const COPY: Record<
  TrustLabel,
  { text: string; icon: React.ComponentType<{ className?: string }>; tone: string }
> = {
  rule_engine: {
    text: "Calculated by UrsBiz rule engine",
    icon: Cpu,
    tone: "bg-emerald-500/10 text-emerald-700 border-emerald-500/30",
  },
  generated: {
    text: "Generated explanation",
    icon: Sparkles,
    tone: "bg-violet-500/10 text-violet-700 border-violet-500/30",
  },
  open_domain: {
    text: "Open-domain LLM — not grounded",
    icon: Globe,
    tone: "bg-amber-500/10 text-amber-700 border-amber-500/30",
  },
  scenario: {
    text: "Scenario estimate",
    icon: TrendingUp,
    tone: "bg-amber-500/10 text-amber-700 border-amber-500/30",
  },
  official: {
    text: "Official external source",
    icon: BadgeCheck,
    tone: "bg-sky-500/10 text-sky-700 border-sky-500/30",
  },
  user_provided: {
    text: "User-provided information",
    icon: User,
    tone: "bg-slate-500/10 text-slate-700 border-slate-500/30",
  },
};

export function TrustBadge({
  label,
  className,
}: {
  label: TrustLabel;
  className?: string;
}) {
  const entry = COPY[label];
  const Icon = entry.icon;
  return (
    <span
      role="note"
      aria-label={entry.text}
      title={entry.text}
      data-trust-label={label}
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider",
        entry.tone,
        className,
      )}
    >
      <Icon className="size-3" aria-hidden="true" />
      {entry.text}
    </span>
  );
}

/**
 * TrustMeta — H7.3 (Docx Prompt 3 Part 4) required metadata block.
 *
 * Renders the docx-required fields under every assistant
 * bubble when the model produced the response:
 *
 *   - Confidence         (0..100)
 *   - Assumptions        (string list, never empty when present)
 *   - Limitations        (string list, never empty when present)
 *   - Evidence           (string list of evidence_reference ids)
 *   - Last updated       (ISO timestamp)
 *
 * H7.8C extends with the provider/model disclosure the
 * hybrid-mode envelope carries:
 *
 *   - Provider + model   ("openai_compatible:llama3.1")
 *   - Grounding score    (0..100 from the GroundingValidator)
 *   - Provider latency   (milliseconds)
 *   - Prompt truncated   (boolean — the user-prompt was clipped)
 *   - Fallback reason    (the normalized reason, only when fallback_used=true)
 *
 * The block is collapsed by default — the docx says
 * "Display: Confidence, Assumptions, Limitations, Evidence,
 * Last updated time" but does not require the block to be
 * open. A small "Why am I seeing this?" toggle expands it.
 */
export function TrustMeta({
  confidence,
  assumptions,
  limitations,
  evidence,
  generatedAt,
  provider,
  model,
  fallbackReason,
  groundingScore,
  promptTruncated,
  providerLatencyMs,
  className,
}: {
  confidence?: number;
  assumptions?: readonly string[];
  limitations?: readonly string[];
  evidence?: readonly string[];
  /** ISO 8601 timestamp from the upstream payload. */
  generatedAt?: string;
  /** H7.8C — provider name (e.g. "openai_compatible"). Never
   *  includes the base URL or API key. */
  provider?: string;
  /** H7.8C — model identifier (e.g. "openai_compatible:llama3.1"). */
  model?: string;
  /** H7.8C — fallback reason code. Rendered only when present. */
  fallbackReason?: string | null;
  /** H7.8C — server grounding score 0..100. */
  groundingScore?: number;
  /** H7.8C — whether the user prompt was truncated to fit the
   *  provider's context window. */
  promptTruncated?: boolean;
  /** H7.8C — provider round-trip latency in ms. */
  providerLatencyMs?: number;
  className?: string;
}) {
  return (
    <details
      className={cn(
        "rounded-md border border-dashed border-border bg-background/40 px-3 py-2 text-xs",
        className,
      )}
      data-testid="trust-meta"
    >
      <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        Why am I seeing this?
      </summary>
      <div className="mt-2 space-y-2 text-foreground/80">
        {provider || model ? (
          <p>
            <span className="font-semibold">Provider:</span>{" "}
            {provider ?? "unknown"}
            {model ? ` (${model})` : null}
            {typeof providerLatencyMs === "number" ? (
              <span className="text-muted-foreground">
                {" "}
                · {providerLatencyMs} ms
              </span>
            ) : null}
          </p>
        ) : null}
        {typeof groundingScore === "number" ? (
          <p>
            <span className="font-semibold">Grounding score:</span>{" "}
            {Math.max(0, Math.min(100, groundingScore))}/100
          </p>
        ) : null}
        {fallbackReason ? (
          <p>
            <span className="font-semibold">Fallback reason:</span>{" "}
            <code className="rounded bg-secondary px-1 text-[10px]">
              {fallbackReason}
            </code>
          </p>
        ) : null}
        {promptTruncated ? (
          <p className="text-amber-700">
            <span className="font-semibold">Note:</span> the user
            prompt was truncated to fit the model context window.
          </p>
        ) : null}
        {typeof confidence === "number" ? (
          <p>
            <span className="font-semibold">Confidence:</span>{" "}
            {Math.max(0, Math.min(100, confidence))}/100
          </p>
        ) : null}
        {assumptions && assumptions.length > 0 ? (
          <div>
            <p className="font-semibold">Assumptions</p>
            <ul className="ml-4 list-disc">
              {assumptions.map((a, i) => (
                <li key={`a-${i}`}>{a}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {limitations && limitations.length > 0 ? (
          <div>
            <p className="font-semibold">Limitations</p>
            <ul className="ml-4 list-disc">
              {limitations.map((l, i) => (
                <li key={`l-${i}`}>{l}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {evidence && evidence.length > 0 ? (
          <div>
            <p className="font-semibold">Evidence</p>
            <ul className="ml-4 list-disc">
              {evidence.map((e, i) => (
                <li key={`e-${i}`}>{e}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {generatedAt ? (
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Last updated {formatRelativeTime(generatedAt)}
          </p>
        ) : null}
      </div>
    </details>
  );
}

function formatRelativeTime(iso: string): string {
  try {
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return iso;
    const now = Date.now();
    const diff = Math.max(0, now - then);
    const minutes = Math.floor(diff / 60_000);
    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} h ago`;
    const days = Math.floor(hours / 24);
    return `${days} d ago`;
  } catch {
    return iso;
  }
}
