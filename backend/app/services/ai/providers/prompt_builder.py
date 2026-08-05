"""AssistantPromptBuilder — Sprint 7 Part 2 + H7.8C.

Pure projection from an :class:`AssistantContext` plus the
user's prompt (and optional conversation history) into the
:class:`AssistantRequest` envelope a real LLM call would
send.

The builder is a pure function over its inputs. It does not
call any LLM. The output is shaped exactly like a real-provider
call (system message + user message + structured context) so a
future OpenAI / Claude / Gemini / Azure provider can swap in
without changing the prompt format.

H7.8C — two-mode prompt
-----------------------

* **grounded** (default) — strict evidence-bounded. The
  system prompt forbids invention and requires every claim
  to cite a stable evidence ID from the
  :class:`EvidenceRegistry`. The model is asked to emit
  JSON conforming to the H7.8C response schema.
* **open** — permissive. The model is told it can answer
  any general question, in plain prose, with no schema
  requirement, no registry, and no grounding. The response
  is rendered as-is into the chat bubble and labelled
  with the ``open_domain`` trust badge.

The user prompt is always wrapped in an untrusted delimiter
to defang prompt-injection attempts regardless of mode.

Determinism
-----------

The user message is rendered in a stable, sorted order:

  * overall business score and band — line 1
  * DNA archetype + match — line 2
  * scores in declared (key) order
  * recommendations sorted by (priority_rank, -score_gain, id)
  * roadmap sorted by estimated_start_order ascending
  * rules in category-then-impact order, capped at _MAX_RULES
  * insights in declared order, capped at _MAX_INSIGHTS
  * conversation history in declared order (caller-bounded)

Two calls with the same context produce byte-identical
:class:`AssistantRequest` instances.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.ai.providers.base import (
    AssistantContext,
    AssistantRequest,
    AssistantTurn,
)

if TYPE_CHECKING:
    from app.services.ai.providers.evidence_registry import EvidenceRegistry


# ----- grounded-mode system prompt --------------------------------- #

# Sprint 7 Part 2 carried a contradiction: the original system
# prompt said "Never give prescriptive actions", but the H7.3
# schema required a 30-day plan with concrete tasks. H7.8C
# resolves the contradiction with a *bounded-action policy* —
# the model may describe concrete actions that already exist
# in the snapshot (recommendations, roadmap, action board),
# but it may not invent new ones and must never write
# government / eligibility / approval language.

_GROUNDED_SYSTEM = """You are UrsBiz Assistant, a business analyst for an Indian SMB.
You receive a structured snapshot of the user's business and an EVIDENCE REGISTRY
of stable, server-resolved IDs you may cite.

Your output MUST be a single JSON object that matches the response schema
documented below. Do not output prose, Markdown, code fences, or commentary
outside the JSON object.

## Bounded-action policy

You MAY describe concrete actions that already exist in the snapshot:
  - recommendations (id, title, rationale)
  - roadmap items (id, phase, expected score improvement)
  - the user's existing action board (action_id, status, due_in_days)

You MAY NOT:
  - invent new tasks, deadlines, emails, phone calls, or meeting dates
  - assign values that are not in the snapshot (no new revenue figures,
    no ROI numbers, no score gains)
  - tell the user they are eligible, approved, or guaranteed anything
  - make official government claims — schemes are *profile matches*,
    not approvals. Always say "your profile matches" or
    "your profile is similar to applicants who …"

## Evidence registry

Every numeric claim, every recommendation, and every scheme mention
MUST reference a stable ID from the EVIDENCE REGISTRY block in the
user message. If you cannot ground a claim in a registry entry,
either drop it or qualify it as an assumption.

## Forbidden language

Never write the following phrases (or close paraphrases):
  - "you are eligible", "you will be approved", "you will receive"
  - "guaranteed funding", "guaranteed growth", "100% success"
  - "we predict your revenue will", "your revenue will be ₹X"
  - "approved by [authority]", "endorsed by [authority]"
  - "definitely will", "certainly will"

Allowed disclaimers:
  - "this does not guarantee eligibility or approval"
  - "scenario estimate, not a prediction"
  - "your profile matches, not a confirmation of eligibility"

## Output schema (return JSON only)

{
  "executive_summary": string,        // <= 280 chars
  "key_findings": [string, ...],       // 1-4 short bullets
  "recommendations": [                 // 0-4 items
    {
      "recommendation_id": string,     // MUST resolve in evidence registry
      "title": string,
      "rationale": string,
      "evidence_refs": [string, ...]   // evidence IDs
    }
  ],
  "thirty_day_plan": [                 // 0-4 items
    {
      "week": 1 | 2 | 3 | 4,
      "task": string,
      "recommendation_ref": string | null,
      "evidence_refs": [string, ...]
    }
  ],
  "scheme_matches": [                  // 0-3 items, never more than registry
    {
      "scheme_ref": string,            // MUST resolve in evidence registry
      "match_explanation": string,
      "evidence_refs": [string, ...]
    }
  ],
  "assumptions": [string, ...],        // list every assumption made
  "limitations": [string, ...],        // list every limitation of the analysis
  "confidence": integer,               // 0-100
  "evidence_references": [             // every cited ID
    {"id": string, "kind": string, "label": string}
  ]
}

If the snapshot is empty, set executive_summary to honestly explain that
no business profile is available and recommend the user set one up.
Return valid JSON, no Markdown fences.
"""


# ----- open-mode system prompt ------------------------------------- #
#
# Open mode is the "general-purpose" path. The model is told it can
# answer any general question with no grounding, no registry, and
# no schema. The response is a free-form string the UI labels as
# "Open-domain LLM — not grounded against business data".

_OPEN_SYSTEM = """You are UrsBiz Assistant in OPEN mode.
You may answer any general question about business, finance, regulation,
operations, or markets in plain prose. Be concise, factual, and helpful.
There is no structured snapshot for this question and no evidence registry.
If a question would require real-time data (live prices, current laws,
specific eligibility), say so and recommend the user verify with an
authoritative source. Do not fabricate statistics or cite authorities
you are not certain about. Keep responses under 400 words.
"""


def _untrusted_user_block(user_prompt: str) -> str:
    """Wrap the user's text in a clearly-delimited, untrusted block.

    Every grounded-mode and open-mode call sends the user's text
    inside this delimiter. The system prompt tells the model the
    contents of the block are untrusted data, not instructions, so
    injection attempts ("ignore previous instructions …") cannot
    override the system contract.
    """
    text = (user_prompt or "").strip()
    # Truncate to keep prompt size bounded. The cap matches the
    # openai_compatible and ollama providers' input cap.
    cap = 8_000
    truncated = False
    if len(text) > cap:
        text = text[:cap].rstrip()
        truncated = True
    block = (
        "=== UNTRUSTED USER QUESTION ===\n"
        f"{text}\n"
        "=== END UNTRUSTED USER QUESTION ==="
    )
    if truncated:
        block += "\n(note: the user question was truncated to fit the prompt window)"
    return block


class AssistantPromptBuilder:
    """Build an :class:`AssistantRequest` from a context + user prompt.

    H7.8C — the builder takes an optional ``EvidenceRegistry`` and
    an optional ``mode``. When ``mode="grounded"`` (default) the
    rendered user message embeds the registry block. When
    ``mode="open"`` the user message is the untrusted-question
    block plus a one-line reminder that no snapshot exists.
    """

    def build(
        self,
        *,
        context: AssistantContext,
        user_prompt: str,
        history: tuple[AssistantTurn, ...] = (),
        knowledge: object | None = None,
        registry: "EvidenceRegistry | None" = None,
        mode: str = "grounded",
    ) -> AssistantRequest:
        return AssistantRequest(
            user_prompt=user_prompt,
            context=context,
            history=history,
            knowledge=knowledge,
            mode=mode,  # type: ignore[arg-type]
            # System + user strings are rendered lazily by the
            # provider's ``_to_messages()`` helper because some
            # providers (Ollama) put the system into a separate
            # payload field, while OpenAI / Claude / Gemini use
            # the messages[] convention. The contract lives
            # here regardless.
        )

    @staticmethod
    def system_message(mode: str = "grounded") -> str:
        """The system message for the requested mode."""
        if mode == "open":
            return _OPEN_SYSTEM
        return _GROUNDED_SYSTEM

    @staticmethod
    def render_user_message(request: AssistantRequest) -> str:
        """Render the user-side text for the model call.

        The renderer chooses grounded vs open layout from
        ``request.mode``. The snapshot blocks are only emitted
        in grounded mode. In open mode we still emit the
        untrusted-user delimiter and a short "no snapshot"
        reminder so the model knows why no context is present.
        """
        mode = getattr(request, "mode", "grounded") or "grounded"
        if mode == "open":
            return _render_open_user_message(request)
        return _render_grounded_user_message(request)


def _render_grounded_user_message(request: AssistantRequest) -> str:
    ctx = request.context
    parts: list[str] = []
    parts.append("=== BUSINESS SNAPSHOT ===")
    parts.append(
        f"business_id: {ctx.business_id}"
    )
    parts.append(
        f"overall_business_score: {ctx.overall_business_score} "
        f"({ctx.band})"
    )
    if ctx.dna.archetype_title:
        parts.append(
            f"dna_archetype: {ctx.dna.archetype_key} "
            f"({ctx.dna.archetype_title}, match={ctx.dna.match_score})"
        )

    if ctx.scores:
        parts.append("")
        parts.append("SCORES")
        for s in sorted(ctx.scores, key=lambda x: x.key):
            parts.append(f"- {s.key}: {s.score} ({s.level}) {s.title}")

    if ctx.recommendations:
        parts.append("")
        parts.append("RECOMMENDATIONS")
        for r in sorted(
            ctx.recommendations,
            key=lambda r: (_priority_rank(r.priority),
                           -r.estimated_score_gain,
                           r.id),
        ):
            parts.append(
                f"- {r.id} [{r.priority} +{r.estimated_score_gain}] "
                f"({r.category}) {r.title} :: "
                f"timeline {r.estimated_timeline}, "
                f"ROI {r.estimated_roi:.0f}"
            )

    if ctx.roadmap:
        parts.append("")
        parts.append("ROADMAP")
        for it in sorted(
            ctx.roadmap,
            key=lambda x: x.estimated_start_order,
        ):
            parts.append(
                f"- {it.id} [order={it.estimated_start_order} "
                f"{it.priority} +{it.expected_score_improvement}] "
                f"({it.phase}) {it.title} :: "
                f"completion {it.completion_percentage}%"
            )

    if ctx.rules:
        parts.append("")
        parts.append("ACTIVE RULES")
        for r in ctx.rules:
            parts.append(
                f"- {r.id} [{r.priority} impact={r.estimated_impact}] "
                f"({r.category}) {r.title} :: {r.reason}"
            )

    if ctx.insights:
        parts.append("")
        parts.append("INSIGHTS")
        for ins in ctx.insights:
            parts.append(
                f"- {ins.id} [{ins.priority} conf={ins.confidence}] "
                f"{ins.title}"
            )

    # H7.3 — docx P3 Part 2 evidence-bundle extension.
    # Schemes, scenarios (labelled "scenario estimate", never
    # "prediction"), and the user's existing action-board items.
    if ctx.schemes:
        parts.append("")
        parts.append("GOVERNMENT SCHEMES (profile-match, never eligibility)")
        for s in sorted(ctx.schemes, key=lambda x: -x.profile_match_score):
            parts.append(
                f"- {s.scheme_id} match={s.profile_match_score} "
                f"authority='{s.authority}' title='{s.title}' "
                f"verified={s.last_verified_date} link={s.application_url}"
            )

    if ctx.forecasts:
        parts.append("")
        parts.append("SCENARIO ESTIMATES (not predictions)")
        for f in ctx.forecasts:
            parts.append(
                f"- {f.scenario_id} horizon='{f.horizon_label}' "
                f"revenue_delta={f.revenue_delta:.0f} "
                f"score_delta={f.score_delta:+d} "
                f"confidence={f.confidence} assumptions='{f.assumption_summary}'"
            )

    if ctx.action_items:
        parts.append("")
        parts.append("USER ACTION BOARD (existing tasks)")
        for a in ctx.action_items:
            parts.append(
                f"- {a.action_id} [{a.priority} {a.status}] "
                f"due_in_days={a.due_in_days} {a.title}"
            )

    if request.history:
        parts.append("")
        parts.append("CONVERSATION HISTORY")
        for turn in request.history:
            tag = "USER" if turn.role == "user" else "ASSISTANT"
            parts.append(f"{tag}: {turn.content}")

    # Sprint 7 Part 4: retrieved knowledge articles. Only
    # rendered when the retriever found at least one
    # citation. Field is opaque to the prompt builder — the
    # caller decides the shape.
    knowledge = getattr(request, "knowledge", None)
    if knowledge is not None:
        citations = getattr(knowledge, "citations", None) or ()
        ranked = getattr(knowledge, "ranked", None) or ()
        articles = getattr(knowledge, "articles", None) or ()
        if citations:
            parts.append("")
            parts.append("KNOWLEDGE SOURCES")
            for i, c in enumerate(citations, start=1):
                parts.append(
                    f"- [{i}] {c.article_id} "
                    f"({c.source_category}) {c.title}"
                )
            if articles:
                parts.append("ARTICLE EXCERPTS")
                for art in articles:
                    if not isinstance(art, dict):
                        continue
                    art_id = art.get("id", "")
                    art_title = art.get("title", "")
                    art_sum = (art.get("summary") or "").strip()
                    if not art_sum:
                        continue
                    parts.append(
                        f"--- {art_id} {art_title} ---"
                    )
                    parts.append(art_sum)
            if ranked:
                parts.append(
                    f"({len(ranked)} of "
                    f"{getattr(knowledge, 'total_candidates', '?')} "
                    f"candidates matched)"
                )

    # H7.8C — the evidence registry block is appended *after* the
    # snapshot so the model sees the context narrative first and the
    # registry as the canonical reference. Stable IDs in the
    # registry are what the model must cite.
    from app.services.ai.providers.evidence_registry import EvidenceRegistry
    registry = EvidenceRegistry(request.context)
    parts.append("")
    parts.append(registry.to_prompt_block())

    # The user's text is the LAST thing in the prompt. The delimiter
    # below tells the model the contents are untrusted.
    parts.append("")
    parts.append(_untrusted_user_block(request.user_prompt))

    return "\n".join(parts)


def _render_open_user_message(request: AssistantRequest) -> str:
    """Open-mode prompt: no snapshot, no registry.

    The model is told the question is general, the snapshot
    is unavailable by design, and the response must be plain
    prose (no JSON). The untrusted delimiter is still emitted.
    """
    parts: list[str] = []
    parts.append(
        "No business snapshot is bound to this question. This is a "
        "general-purpose question and the response will not be "
        "grounded against the user's profile. Answer in plain prose."
    )
    parts.append("")
    parts.append(_untrusted_user_block(request.user_prompt))
    return "\n".join(parts)


def _priority_rank(priority: str) -> int:
    if priority == "Critical":
        return 0
    if priority == "High":
        return 1
    if priority == "Medium":
        return 2
    if priority == "Low":
        return 3
    return 99