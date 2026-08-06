"""ScenarioSimulator — Sprint H8.4 Business Scenario Simulator Engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioSimulationResult:
    """Outcome of a Business Scenario Simulation across 8 dimensions."""

    scenario_title: str
    scenario_type: str  # "hiring", "export_growth", "funding", "commodity_cost", "facility_expansion"
    revenue_impact: str
    cashflow_impact: str
    risks_identified: tuple[str, ...] = field(default_factory=tuple)
    capacity_impact: str = ""
    export_impact: str = ""
    hiring_impact: str = ""
    profitability_impact: str = ""
    timeline_horizon: str = "6-12 months"
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    disclaimer: str = "Illustrative scenario estimate — not a prediction"

    def to_markdown(self) -> str:
        """Format simulation result as clean Markdown card."""
        lines: list[str] = []
        lines.append(f"### SCENARIO SIMULATION: {self.scenario_title.upper()}")
        lines.append(f"*{self.disclaimer}*")
        lines.append("")
        lines.append(f"- **Revenue Impact**: {self.revenue_impact}")
        lines.append(f"- **Cashflow Impact**: {self.cashflow_impact}")
        if self.profitability_impact:
            lines.append(f"- **Profitability & Margins**: {self.profitability_impact}")
        if self.capacity_impact:
            lines.append(f"- **Capacity & Utilization**: {self.capacity_impact}")
        if self.export_impact:
            lines.append(f"- **Export Expansion**: {self.export_impact}")
        if self.hiring_impact:
            lines.append(f"- **Hiring & Payroll Overhead**: {self.hiring_impact}")
        lines.append(f"- **Timeline Horizon**: {self.timeline_horizon}")
        if self.risks_identified:
            lines.append("- **Risks & Vulnerabilities**:")
            for r in self.risks_identified:
                lines.append(f"  * {r}")
        if self.assumptions:
            lines.append("- **Key Scenario Assumptions**:")
            for a in self.assumptions:
                lines.append(f"  * {a}")
        return "\n".join(lines)


class ScenarioSimulator:
    """Deterministic MSME Business Scenario Simulator."""

    def is_simulation_query(self, prompt: str) -> bool:
        """Check if user prompt is asking a 'What if' scenario question."""
        p_low = (prompt or "").lower()
        keywords = ("what if", "scenario", "simulate", "if i hire", "if funding", "if exports", "if prices", "if i open", "buy", "machine", "reach")
        return any(kw in p_low for kw in keywords)

    def simulate(self, prompt: str, context: Any) -> ScenarioSimulationResult:
        """Run deterministic financial & operational simulation based on prompt intent and baseline context."""
        p_low = (prompt or "").lower()
        base_revenue = getattr(context, "annual_revenue_inr", 18000000) or 18000000
        score = getattr(context, "overall_business_score", 68)

        # 0. Machine Capex Scenario
        if "machine" in p_low or "equipment" in p_low:
            return ScenarioSimulationResult(
                scenario_title="Automated Machinery Capex Procurement",
                scenario_type="equipment_capex",
                revenue_impact=f"Estimated top-line expansion of +₹{base_revenue * 0.25 / 100000:.1f} Lakh (+25% Output)",
                cashflow_impact="Initial Capex outlay of ₹12 Lakh with 10-month payback horizon",
                profitability_impact="Net profit margin increases by +3.2% due to lower per-unit labor COGS",
                capacity_impact="Daily manufacturing unit output increases from 1,200 to 1,600 units",
                export_impact="Meets international buyer quality precision specifications",
                hiring_impact="Requires 2 trained CNC machine technicians",
                timeline_horizon="4-6 months procurement, installation & calibration",
                risks_identified=(
                    "Short-term cashflow pressure during 60-day installation phase",
                    "Technician learning curve impact during initial 30 days",
                ),
                assumptions=(
                    "Equipment financed via 70% bank term loan under CGTMSE scheme at 8.5% interest",
                    "Maintenance contract included under 3-year OEM warranty",
                ),
            )

        # 1. Hiring Scenario
        if "hire" in p_low or "employee" in p_low:
            match = re.search(r"(\d+)", p_low)
            count = int(match.group(1)) if match else 10
            payroll_cost = count * 35000 * 12  # ₹35k monthly salary assumption
            rev_gain = base_revenue * 0.18  # 18% revenue capacity boost
            return ScenarioSimulationResult(
                scenario_title=f"Hiring {count} Employees",
                scenario_type="hiring",
                revenue_impact=f"Estimated revenue boost of +₹{rev_gain / 100000:.1f} Lakh (+18% capacity)",
                cashflow_impact=f"Annual payroll expenditure increase of ₹{payroll_cost / 100000:.1f} Lakh",
                profitability_impact="Initial net margin dip of ~2.5% during 60-day onboarding, recovering after month 4",
                capacity_impact=f"Production throughput increases by ~22%",
                export_impact="Enables fulfillment of pending international orders",
                hiring_impact=f"Add {count} staff to operations and quality assurance",
                timeline_horizon="3-6 months onboarding & productivity ramp",
                risks_identified=(
                    f"Payroll overhead increase of ₹{payroll_cost / 1200000:.1f}L/month prior to revenue realization",
                    "Initial drop in worker productivity during training",
                ),
                assumptions=(
                    f"Average monthly salary of ₹35,000 per employee",
                    "Existing factory space accommodates additional headcount",
                ),
            )

        # 2. Export Expansion Scenario
        if "export" in p_low:
            match = re.search(r"(\d+)%", p_low)
            pct = int(match.group(1)) if match else 20
            rev_gain = base_revenue * (pct / 100.0)
            return ScenarioSimulationResult(
                scenario_title=f"Export Increase of {pct}%",
                scenario_type="export_growth",
                revenue_impact=f"Projected revenue addition of +₹{rev_gain / 100000:.1f} Lakh",
                cashflow_impact="Working capital cycle extends by 30-45 days due to international shipping credit terms",
                profitability_impact=f"Gross margin expands by +3.5% due to higher export realization rates",
                capacity_impact="Factory operating capacity rises from 70% to 88%",
                export_impact=f"Export revenue share increases by +{pct}%",
                hiring_impact="Requires 2 export documentation & logistics specialists",
                timeline_horizon="6-12 months export shipping & payment cycle",
                risks_identified=(
                    "Currency exchange rate fluctuation risk",
                    "Working capital stretch due to longer 90-day L/C credit cycles",
                ),
                assumptions=(
                    f"Export orders realize {pct}% higher price realization than domestic sales",
                    "ECGC export credit insurance coverage obtained",
                ),
            )

        # 3. Funding Injection Scenario
        if "fund" in p_low or "lakh" in p_low or "crore" in p_low or "capital" in p_low:
            match = re.search(r"(\d+)", p_low)
            amount = int(match.group(1)) if match else 50
            return ScenarioSimulationResult(
                scenario_title=f"Capital Injection of ₹{amount} Lakh",
                scenario_type="funding",
                revenue_impact=f"Enables ₹{base_revenue * 0.25 / 100000:.1f} Lakh revenue expansion via machinery upgrade",
                cashflow_impact=f"Liquidity buffer increases by ₹{amount} Lakh",
                profitability_impact="Operating margin improves by +4.0% through bulk raw material purchases",
                capacity_impact="Automated machinery increases daily unit output by 35%",
                export_impact="Meets quality audit criteria for European buyers",
                hiring_impact="Add 3 skilled machine operators",
                timeline_horizon="6-9 months machinery procurement & commissioning",
                risks_identified=(
                    "Debt servicing obligation if capital includes term loan component",
                    "Depreciation & interest expense impact on net profit",
                ),
                assumptions=(
                    f"₹{amount} Lakh allocated: 60% capex equipment, 40% working capital",
                    "Interest rate assumption of 8.5% p.a. if funded via collateralized debt",
                ),
            )

        # 4. Commodity Cost Increase Scenario
        if "cotton" in p_low or "price" in p_low or "cost" in p_low or "raw material" in p_low:
            return ScenarioSimulationResult(
                scenario_title="Raw Material Cost Increase (Commodity Volatility)",
                scenario_type="commodity_cost",
                revenue_impact="Flat revenue unless product prices are revised upwards by 5-8%",
                cashflow_impact="Cash outlays for raw material inventory rise by ~12%",
                profitability_impact="Gross profit margin compresses by -4.2% if cost increase cannot be passed to buyers",
                capacity_impact="Production volume maintained but inventory holding cost increases",
                export_impact="Export competitiveness impacted against foreign suppliers",
                hiring_impact="Freeze non-essential hiring to preserve cash",
                timeline_horizon="Immediate (1-3 months inventory cycle)",
                risks_identified=(
                    "Margin squeeze from rigid buyer price contracts",
                    "Working capital shortfall due to higher upfront procurement costs",
                ),
                assumptions=(
                    "Raw material constitutes 55% of total Cost of Goods Sold (COGS)",
                    "Partial cost pass-through of 50% achieved with domestic buyers",
                ),
            )

        # 5. Factory Expansion Scenario (Default)
        return ScenarioSimulationResult(
            scenario_title="Second Factory Facility Expansion",
            scenario_type="facility_expansion",
            revenue_impact=f"Doubles maximum production capacity, enabling +₹{base_revenue * 0.6 / 100000:.1f} Lakh revenue over 18 months",
            cashflow_impact="High capex cash outflow during initial 6 months construction phase",
            profitability_impact="Break-even expected at month 10 following operational commencement",
            capacity_impact="Total manufacturing footprint expands by +100%",
            export_impact="Dedicated export manufacturing line established",
            hiring_impact="Requires 15 additional factory floor workers and 1 plant manager",
            timeline_horizon="12-18 months greenfield/brownfield expansion",
            risks_identified=(
                "Fixed overhead cost burden during initial low-utilization months",
                "Construction & statutory compliance approval delays",
            ),
            assumptions=(
                "Industrial land leased in MSME manufacturing cluster",
                "Phase 1 utilization reaches 50% within 4 months of commissioning",
            ),
        )
