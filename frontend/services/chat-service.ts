"use client";

/**
 * Frontend service for the Sprint 7 Part 3 conversation
 * persistence endpoints.
 *
 *   POST   /api/v1/chat                  createSession
 *   GET    /api/v1/chat                  listSessions
 *   GET    /api/v1/chat/{id}             getSession
 *   DELETE /api/v1/chat/{id}             deleteSession
 *   POST   /api/v1/chat/{id}/message     appendMessage
 *   GET    /api/v1/chat/provider-status  fetchProviderStatus  (H7.8C)
 *
 * Every call goes through the shared `apiClient` so the
 * cookie-based auth and JSON encoding stay in one place.
 */

import { apiClient, ApiError } from "./api-client";

export interface ChatSource {
  topic: string;
  detail: string;
}

export interface ChatGenerationMeta {
  provider: string;
  model: string;
  mode: "grounded" | "open";
  fallback_used: boolean;
  fallback_reason: string | null;
  generation_method: "generative" | "deterministic";
  schema_validated: boolean;
  grounding_validated: boolean;
  server_grounding_score: number;
  evidence_count: number;
  confidence: number | null;
  assumptions: string[];
  limitations: string[];
  evidence_references: string[];
  generated_at: string;
  prompt_truncated: boolean;
  provider_latency_ms: number | null;
  grounded_payload?: Record<string, unknown> | null;
}

export interface ChatProviderStatus {
  configured_provider: string;
  runtime_provider: string;
  model: string;
  available: boolean;
  schema_required: boolean;
  fallback_active: boolean;
  modes: Array<"grounded" | "open">;
  default_mode: "grounded" | "open";
}

export interface ChatMessageOut {
  id: number;
  role: "user" | "assistant";
  kind: string;
  content: string;
  sources: ChatSource[];
  created_at: string;
  /**
   * Per-message trust-label flag.
   *
   *  - `true`  → the assistant turn came from the deterministic
   *              fallback / placeholder provider. The UI MUST
   *              render "Calculated by UrsBiz rule engine".
   *  - `false` → the assistant turn came from a real
   *              OpenAI-compatible / Ollama response. The UI
   *              MAY render "Generated explanation".
   */
  fallback_used: boolean;
  /**
   * H7.8C — the full provenance envelope. Present for
   * every assistant turn (whether generative or fallback).
   * The MessageBubble uses this to render the three-state
   * trust badge (grounded / open / fallback) and the
   * TrustMeta disclosure panel.
   */
  generation?: ChatGenerationMeta | null;
}

export interface ChatSessionSummary {
  id: number;
  title: string;
  summary: string;
  message_count: number;
  last_model: string;
  fallback_used: boolean;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDetail extends ChatSessionSummary {
  messages: ChatMessageOut[];
}

export interface ChatMessageAppendResponse {
  user_message: ChatMessageOut;
  assistant_message: ChatMessageOut;
  session: ChatSessionDetail;
}

export const chatService = {
  async listSessions(): Promise<ChatSessionSummary[]> {
    const payload = await apiClient.get<{ sessions: ChatSessionSummary[]; count: number }>(
      "/api/v1/chat"
    );
    return payload.sessions;
  },

  async getSession(sessionId: number): Promise<ChatSessionDetail> {
    return apiClient.get<ChatSessionDetail>(`/api/v1/chat/${sessionId}`);
  },

  async createSession(title = ""): Promise<ChatSessionDetail> {
    return apiClient.post<ChatSessionDetail>("/api/v1/chat", { title });
  },

  async deleteSession(sessionId: number): Promise<{ deleted: boolean; id: number }> {
    return apiClient.delete<{ deleted: boolean; id: number }>(
      `/api/v1/chat/${sessionId}`
    );
  },

  /**
   * Append a user message to a chat session. Returns the
   * pair (user + assistant) plus the updated session.
   *
   * H7.8C — the ``mode`` flag selects grounded vs open
   * dispatch server-side. The default is ``"grounded"``
   * (the evidence-bounded path).
   */
  async appendMessage(
    sessionId: number,
    content: string,
    opts: { mode?: "grounded" | "open" } = {},
  ): Promise<ChatMessageAppendResponse> {
    return apiClient.post<ChatMessageAppendResponse>(
      `/api/v1/chat/${sessionId}/message`,
      { content, mode: opts.mode ?? "grounded" },
    );
  },

  /**
   * H7.8C — provider status fetch. Returns the configured
   * provider, runtime provider, model, availability, and
   * the configured mode list. The endpoint is auth-gated
   * and never exposes secrets, full URLs, or API keys.
   */
  async fetchProviderStatus(): Promise<ChatProviderStatus> {
    return apiClient.get<ChatProviderStatus>("/api/v1/chat/provider-status");
  },
};

export { ApiError };
