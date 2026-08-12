"""SPRINT AI-18 — Universal AI Evaluation Harness.

Question bank — at least 100 evaluation prompts distributed
across the brief's 16 categories.

The brief is explicit:

  * Use DIFFERENT wording for the same underlying task.
  * Do NOT reuse only known flagship prompts.
  * Cover general knowledge AND business analysis AND
    government schemes AND export AND unknown gaps.

The bank returns plain ``str`` prompts the runner feeds into
the production pipeline. Each prompt is tagged with a
``QuestionCategory`` so the runner can compute coverage
metrics.

Adding categories is non-breaking; removing one IS.
"""
from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Category vocabulary
# --------------------------------------------------------------------------- #


class QuestionCategory(str):
    """Plain-string enum of the brief's 16 categories."""

    GENERAL_KNOWLEDGE = "general_knowledge"
    BUSINESS_FACT = "business_fact"
    BUSINESS_ANALYSIS = "business_analysis"
    CALCULATION = "calculation"
    RECOMMENDATION = "recommendation"
    SCENARIO = "scenario"
    FORECAST = "forecast"
    COMPARISON = "comparison"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    RISK = "risk"
    GOVERNMENT_SCHEME = "government_scheme"
    EXPORT = "export"
    ROADMAP = "roadmap"
    EXTERNAL_INFORMATION = "external_information"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Entry dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuestionEntry:
    """One evaluation prompt.

    Attributes
    ----------
    prompt:
        The literal user message.
    category:
        One of :class:`QuestionCategory` values.
    tags:
        Free-form tags the runner keys off (e.g.
        ``{"calculation", "working_capital"}``). Empty tuple
        when no tag is relevant.
    """

    prompt: str
    category: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "tags": list(self.tags),
        }


# --------------------------------------------------------------------------- #
# Bank
# --------------------------------------------------------------------------- #


# Each tuple element is (prompt, category, [tags...]).
# The bank is intentionally heterogeneous — every entry is
# phrased differently so the runner measures GENERAL behaviour,
# not flagship memorisation.
_BANK: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # ---- GENERAL KNOWLEDGE (≥6) ------------------------------------ #
    (
        "What does EBITDA stand for?",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    (
        "Explain working capital in plain English.",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    (
        "Define gross margin as if I'm starting a business tomorrow.",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    (
        "What's the difference between revenue and profit?",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    (
        "How do I think about cash conversion cycle?",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    (
        "Can you walk me through what a P&L statement shows?",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    (
        "What is ROI and why does it matter?",
        QuestionCategory.GENERAL_KNOWLEDGE, (),
    ),
    # ---- BUSINESS FACT (≥6) ---------------------------------------- #
    (
        "How much revenue are we doing today?",
        QuestionCategory.BUSINESS_FACT, ("revenue",),
    ),
    (
        "What's our current headcount?",
        QuestionCategory.BUSINESS_FACT, ("employees",),
    ),
    (
        "Where is our business located?",
        QuestionCategory.BUSINESS_FACT, ("location",),
    ),
    (
        "Tell me our overall business score.",
        QuestionCategory.BUSINESS_FACT, ("score",),
    ),
    (
        "What's the legal name of our company?",
        QuestionCategory.BUSINESS_FACT, ("legal_name",),
    ),
    (
        "How many workers do we have on the payroll?",
        QuestionCategory.BUSINESS_FACT, ("employees",),
    ),
    (
        "What industry are we in?",
        QuestionCategory.BUSINESS_FACT, ("industry",),
    ),
    # ---- BUSINESS ANALYSIS (≥6) ------------------------------------ #
    (
        "Where is our business strongest?",
        QuestionCategory.BUSINESS_ANALYSIS, ("strength",),
    ),
    (
        "What's holding us back right now?",
        QuestionCategory.BUSINESS_ANALYSIS, ("weakness",),
    ),
    (
        "How is our digital twin trending this quarter?",
        QuestionCategory.BUSINESS_ANALYSIS, ("dna",),
    ),
    (
        "What does the latest business score say about us?",
        QuestionCategory.BUSINESS_ANALYSIS, ("score",),
    ),
    (
        "Which archetype does our business resemble most?",
        QuestionCategory.BUSINESS_ANALYSIS, ("archetype",),
    ),
    (
        "Is our growth profile closer to a startup or established firm?",
        QuestionCategory.BUSINESS_ANALYSIS, ("profile",),
    ),
    (
        "Which part of our operations needs attention first?",
        QuestionCategory.BUSINESS_ANALYSIS, ("diagnosis",),
    ),
    # ---- CALCULATION (≥6) ----------------------------------------- #
    (
        "How much revenue do we need to hit ₹3 Cr?",
        QuestionCategory.CALCULATION, ("gap_math",),
    ),
    (
        "What's our gap to ₹5 crore annual revenue?",
        QuestionCategory.CALCULATION, ("gap_math",),
    ),
    (
        "By how much must we grow to reach our target?",
        QuestionCategory.CALCULATION, ("growth_multiple",),
    ),
    (
        "What is the growth multiple between current and target revenue?",
        QuestionCategory.CALCULATION, ("growth_multiple",),
    ),
    (
        "How much working capital do we need?",
        QuestionCategory.CALCULATION, ("working_capital",),
    ),
    (
        "Estimate our working capital requirement.",
        QuestionCategory.CALCULATION, ("working_capital",),
    ),
    (
        "How many senior engineers can we afford at our current burn?",
        QuestionCategory.CALCULATION, ("headcount_cost",),
    ),
    # ---- RECOMMENDATION (≥6) -------------------------------------- #
    (
        "What should we focus on first this quarter?",
        QuestionCategory.RECOMMENDATION, ("priority",),
    ),
    (
        "Which single move will move the needle most?",
        QuestionCategory.RECOMMENDATION, ("priority",),
    ),
    (
        "What's the most impactful action we can take right now?",
        QuestionCategory.RECOMMENDATION, ("priority",),
    ),
    (
        "If I can only do one thing next week, what should it be?",
        QuestionCategory.RECOMMENDATION, ("priority",),
    ),
    (
        "Recommend the top 3 changes we should make.",
        QuestionCategory.RECOMMENDATION, ("multi",),
    ),
    (
        "Which recommendation from our roadmap is most urgent?",
        QuestionCategory.RECOMMENDATION, ("urgency",),
    ),
    (
        "What do you suggest we tackle in the next 30 days?",
        QuestionCategory.RECOMMENDATION, ("urgency",),
    ),
    # ---- SCENARIO (≥6) -------------------------------------------- #
    (
        "What happens if we grow revenue 20% next year?",
        QuestionCategory.SCENARIO, ("growth",),
    ),
    (
        "Simulate a 15% rise in cotton prices — how do we fare?",
        QuestionCategory.SCENARIO, ("input_shock",),
    ),
    (
        "What if we lose our biggest customer tomorrow?",
        QuestionCategory.SCENARIO, ("concentration",),
    ),
    (
        "How would hiring 10 more people change our runway?",
        QuestionCategory.SCENARIO, ("hiring",),
    ),
    (
        "If we expand to a new state, what changes?",
        QuestionCategory.SCENARIO, ("expansion",),
    ),
    (
        "Walk me through a scenario where our costs rise 10%.",
        QuestionCategory.SCENARIO, ("cost_shock",),
    ),
    (
        "What if a key supplier goes out of business?",
        QuestionCategory.SCENARIO, ("supplier",),
    ),
    # ---- FORECAST (≥6) -------------------------------------------- #
    (
        "What does our revenue forecast say for next year?",
        QuestionCategory.FORECAST, ("revenue",),
    ),
    (
        "Where will we be in 12 months if nothing changes?",
        QuestionCategory.FORECAST, ("baseline",),
    ),
    (
        "Predict our headcount requirement 18 months out.",
        QuestionCategory.FORECAST, ("headcount",),
    ),
    (
        "What is our expected revenue trajectory?",
        QuestionCategory.FORECAST, ("revenue",),
    ),
    (
        "Show me the projected order book.",
        QuestionCategory.FORECAST, ("orders",),
    ),
    (
        "What's our 24-month growth outlook?",
        QuestionCategory.FORECAST, ("long_horizon",),
    ),
    # ---- COMPARISON (≥6) ----------------------------------------- #
    (
        "How do we compare to a typical MSME in Tirupur?",
        QuestionCategory.COMPARISON, ("peer",),
    ),
    (
        "Are we growing faster or slower than industry average?",
        QuestionCategory.COMPARISON, ("peer",),
    ),
    (
        "Compare our margin to peer firms.",
        QuestionCategory.COMPARISON, ("peer",),
    ),
    (
        "How does our employee productivity compare to peers?",
        QuestionCategory.COMPARISON, ("peer",),
    ),
    (
        "What about us vs a startup in the same space?",
        QuestionCategory.COMPARISON, ("peer",),
    ),
    (
        "Stack us up against an established competitor.",
        QuestionCategory.COMPARISON, ("peer",),
    ),
    # ---- FINANCIAL (≥6) ------------------------------------------ #
    (
        "What's our monthly burn rate?",
        QuestionCategory.FINANCIAL, ("burn",),
    ),
    (
        "Estimate our annual expenses.",
        QuestionCategory.FINANCIAL, ("expenses",),
    ),
    (
        "How long is our runway at the current burn?",
        QuestionCategory.FINANCIAL, ("runway",),
    ),
    (
        "What's our gross margin?",
        QuestionCategory.FINANCIAL, ("margin",),
    ),
    (
        "Break down our cost structure.",
        QuestionCategory.FINANCIAL, ("cost",),
    ),
    (
        "How much cash do we have on hand?",
        QuestionCategory.FINANCIAL, ("cash",),
    ),
    # ---- OPERATIONAL (≥6) ---------------------------------------- #
    (
        "Which process is our biggest bottleneck?",
        QuestionCategory.OPERATIONAL, ("bottleneck",),
    ),
    (
        "How efficient is our production line?",
        QuestionCategory.OPERATIONAL, ("efficiency",),
    ),
    (
        "What's our supplier reliability?",
        QuestionCategory.OPERATIONAL, ("supplier",),
    ),
    (
        "Are we over- or under-staffed?",
        QuestionCategory.OPERATIONAL, ("staffing",),
    ),
    (
        "How do we track inventory turnover?",
        QuestionCategory.OPERATIONAL, ("inventory",),
    ),
    (
        "Where do we lose the most time in a typical day?",
        QuestionCategory.OPERATIONAL, ("time_loss",),
    ),
    # ---- RISK (≥6) ----------------------------------------------- #
    (
        "What's our biggest business risk?",
        QuestionCategory.RISK, ("top",),
    ),
    (
        "How exposed are we to currency swings?",
        QuestionCategory.RISK, ("currency",),
    ),
    (
        "What is our supply chain vulnerability?",
        QuestionCategory.RISK, ("supply_chain",),
    ),
    (
        "Are we over-reliant on a single customer?",
        QuestionCategory.RISK, ("concentration",),
    ),
    (
        "Which regulatory changes could hurt us most?",
        QuestionCategory.RISK, ("regulatory",),
    ),
    (
        "How resilient are we to commodity price shocks?",
        QuestionCategory.RISK, ("commodity",),
    ),
    # ---- GOVERNMENT SCHEME (≥6) ---------------------------------- #
    (
        "Are there any government schemes we should apply to?",
        QuestionCategory.GOVERNMENT_SCHEME, ("match",),
    ),
    (
        "What subsidies are available for Tirupur textile MSMEs?",
        QuestionCategory.GOVERNMENT_SCHEME, ("subsidy",),
    ),
    (
        "Which central schemes help manufacturers like us?",
        QuestionCategory.GOVERNMENT_SCHEME, ("central",),
    ),
    (
        "Is there a working capital loan scheme for small businesses?",
        QuestionCategory.GOVERNMENT_SCHEME, ("loan",),
    ),
    (
        "Tell me about PMEGP eligibility for our profile.",
        QuestionCategory.GOVERNMENT_SCHEME, ("pmegp",),
    ),
    (
        "What credit-linked schemes can help with machinery?",
        QuestionCategory.GOVERNMENT_SCHEME, ("credit",),
    ),
    # ---- EXPORT (≥6) --------------------------------------------- #
    (
        "What export markets are best for our products?",
        QuestionCategory.EXPORT, ("markets",),
    ),
    (
        "How do I find international buyers for Tirupur textiles?",
        QuestionCategory.EXPORT, ("buyers",),
    ),
    (
        "What export incentives can we claim?",
        QuestionCategory.EXPORT, ("incentives",),
    ),
    (
        "Walk me through the export documentation process.",
        QuestionCategory.EXPORT, ("compliance",),
    ),
    (
        "Which countries have the strongest demand for our category?",
        QuestionCategory.EXPORT, ("demand",),
    ),
    (
        "How do I hedge against USD/INR swings?",
        QuestionCategory.EXPORT, ("forex",),
    ),
    # ---- ROADMAP (≥6) -------------------------------------------- #
    (
        "What's our 90-day roadmap?",
        QuestionCategory.ROADMAP, ("short_term",),
    ),
    (
        "Show me the 12-month plan.",
        QuestionCategory.ROADMAP, ("mid_term",),
    ),
    (
        "What are the milestones for the next quarter?",
        QuestionCategory.ROADMAP, ("short_term",),
    ),
    (
        "Lay out the 3-year strategic priorities.",
        QuestionCategory.ROADMAP, ("long_term",),
    ),
    (
        "Where will we be by the end of this year?",
        QuestionCategory.ROADMAP, ("year_end",),
    ),
    (
        "How should we sequence the next 6 months?",
        QuestionCategory.ROADMAP, ("sequencing",),
    ),
    # ---- EXTERNAL INFORMATION (≥6) ------------------------------- #
    (
        "What is the current repo rate?",
        QuestionCategory.EXTERNAL_INFORMATION, ("macro",),
    ),
    (
        "What is the latest GST rate for textiles?",
        QuestionCategory.EXTERNAL_INFORMATION, ("tax",),
    ),
    (
        "How is the MSME sector performing nationally?",
        QuestionCategory.EXTERNAL_INFORMATION, ("macro",),
    ),
    (
        "What's the prevailing cotton benchmark?",
        QuestionCategory.EXTERNAL_INFORMATION, ("commodity",),
    ),
    (
        "Explain the Udyam registration process.",
        QuestionCategory.EXTERNAL_INFORMATION, ("registration",),
    ),
    (
        "What does the latest RBI policy say for small business loans?",
        QuestionCategory.EXTERNAL_INFORMATION, ("policy",),
    ),
    # ---- MIXED (≥6) ---------------------------------------------- #
    (
        "What is EBITDA and how does it apply to our margin?",
        QuestionCategory.MIXED, ("general_plus_business",),
    ),
    (
        "What does the repo rate mean for our working capital?",
        QuestionCategory.MIXED, ("external_plus_business",),
    ),
    (
        "What's the GST rate on textiles and how will it impact us?",
        QuestionCategory.MIXED, ("external_plus_business",),
    ),
    (
        "Define working capital and tell me what ours is.",
        QuestionCategory.MIXED, ("general_plus_business",),
    ),
    (
        "What is a PMEGP loan and might we qualify?",
        QuestionCategory.MIXED, ("scheme_plus_business",),
    ),
    (
        "What is the cotton benchmark and how does it affect our cost?",
        QuestionCategory.MIXED, ("commodity_plus_business",),
    ),
    # ---- UNKNOWN (≥6) -------------------------------------------- #
    (
        "What is the meaning of life for an MSME founder?",
        QuestionCategory.UNKNOWN, ("philosophical",),
    ),
    (
        "Who will win the next cricket World Cup?",
        QuestionCategory.UNKNOWN, ("off_topic",),
    ),
    (
        "Tell me a joke about supply chains.",
        QuestionCategory.UNKNOWN, ("off_topic",),
    ),
    (
        "What's the best cuisine in Bengaluru?",
        QuestionCategory.UNKNOWN, ("off_topic",),
    ),
    (
        "Should I learn Python or Java first?",
        QuestionCategory.UNKNOWN, ("off_topic",),
    ),
    (
        "What's the answer to the ultimate question of life, the universe, and everything?",
        QuestionCategory.UNKNOWN, ("off_topic",),
    ),
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def all_questions() -> tuple[QuestionEntry, ...]:
    """Return the full bank as :class:`QuestionEntry` tuples."""
    return tuple(
        QuestionEntry(prompt=p, category=c, tags=t)
        for (p, c, t) in _BANK
    )


def questions_by_category(category: str) -> tuple[QuestionEntry, ...]:
    """Return the entries whose ``category == category``."""
    return tuple(q for q in all_questions() if q.category == category)


def category_coverage() -> dict[str, int]:
    """Return ``{category: count}`` for every category in the bank.

    The runner uses this to compute the ``question_coverage``
    metric (PART 6) — every brief category must be present
    AND non-empty.
    """
    out: dict[str, int] = {}
    for q in all_questions():
        out[q.category] = out.get(q.category, 0) + 1
    return out


def category_vocabulary() -> tuple[str, ...]:
    """Return the canonical 16-category vocabulary in declaration order."""
    return (
        QuestionCategory.GENERAL_KNOWLEDGE,
        QuestionCategory.BUSINESS_FACT,
        QuestionCategory.BUSINESS_ANALYSIS,
        QuestionCategory.CALCULATION,
        QuestionCategory.RECOMMENDATION,
        QuestionCategory.SCENARIO,
        QuestionCategory.FORECAST,
        QuestionCategory.COMPARISON,
        QuestionCategory.FINANCIAL,
        QuestionCategory.OPERATIONAL,
        QuestionCategory.RISK,
        QuestionCategory.GOVERNMENT_SCHEME,
        QuestionCategory.EXPORT,
        QuestionCategory.ROADMAP,
        QuestionCategory.EXTERNAL_INFORMATION,
        QuestionCategory.MIXED,
        QuestionCategory.UNKNOWN,
    )


__all__ = [
    "QuestionCategory",
    "QuestionEntry",
    "all_questions",
    "questions_by_category",
    "category_coverage",
    "category_vocabulary",
]
