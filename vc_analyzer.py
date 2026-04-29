"""
VC Portfolio Asset Analyzer
Analyzes venture capital portfolio companies and generates investment recommendations
using Claude Opus 4.7 with prompt caching for efficient repeated queries.
"""

import anthropic
import json
from typing import Optional


# ── Demo Portfolio ────────────────────────────────────────────────────────────

DEMO_PORTFOLIO = {
    "fund_name": "Horizon Ventures Fund III",
    "fund_size_usd": 150_000_000,
    "vintage_year": 2021,
    "stage_focus": ["Series A", "Series B"],
    "sector_focus": ["SaaS", "FinTech", "HealthTech", "AI/ML", "CleanTech"],
    "portfolio_companies": [
        {
            "name": "DataStream AI",
            "sector": "AI/ML",
            "stage": "Series B",
            "investment_date": "2022-03",
            "initial_investment_usd": 8_000_000,
            "current_valuation_usd": 120_000_000,
            "entry_valuation_usd": 60_000_000,
            "ownership_pct": 12.5,
            "arr_usd": 18_000_000,
            "arr_growth_yoy_pct": 180,
            "gross_margin_pct": 72,
            "burn_rate_monthly_usd": 900_000,
            "runway_months": 18,
            "headcount": 95,
            "key_metrics": {
                "nrr_pct": 138,
                "cac_payback_months": 14,
                "churn_rate_annual_pct": 6,
            },
            "recent_developments": "Launched enterprise tier; closed 3 Fortune 500 pilots",
            "risks": ["Competitive pressure from hyperscalers", "Key-person dependency on CTO"],
        },
        {
            "name": "MedVault",
            "sector": "HealthTech",
            "stage": "Series A",
            "investment_date": "2022-09",
            "initial_investment_usd": 5_000_000,
            "current_valuation_usd": 28_000_000,
            "entry_valuation_usd": 32_000_000,
            "ownership_pct": 14.2,
            "arr_usd": 3_200_000,
            "arr_growth_yoy_pct": 65,
            "gross_margin_pct": 58,
            "burn_rate_monthly_usd": 650_000,
            "runway_months": 11,
            "headcount": 42,
            "key_metrics": {
                "nrr_pct": 105,
                "cac_payback_months": 28,
                "churn_rate_annual_pct": 15,
            },
            "recent_developments": "FDA clearance pending for core product; new CMO hired",
            "risks": ["Regulatory delays", "Long sales cycles in healthcare", "Short runway"],
        },
        {
            "name": "PayFlow",
            "sector": "FinTech",
            "stage": "Series B",
            "investment_date": "2021-11",
            "initial_investment_usd": 12_000_000,
            "current_valuation_usd": 210_000_000,
            "entry_valuation_usd": 85_000_000,
            "ownership_pct": 9.8,
            "arr_usd": 32_000_000,
            "arr_growth_yoy_pct": 95,
            "gross_margin_pct": 65,
            "burn_rate_monthly_usd": 2_100_000,
            "runway_months": 22,
            "headcount": 180,
            "key_metrics": {
                "nrr_pct": 118,
                "cac_payback_months": 18,
                "churn_rate_annual_pct": 8,
            },
            "recent_developments": "Launched in 3 new international markets; processing $2B+ GMV annually",
            "risks": ["Regulatory compliance across jurisdictions", "Fraud/risk management at scale"],
        },
        {
            "name": "CloudOps Pro",
            "sector": "SaaS",
            "stage": "Series A",
            "investment_date": "2023-02",
            "initial_investment_usd": 6_000_000,
            "current_valuation_usd": 45_000_000,
            "entry_valuation_usd": 40_000_000,
            "ownership_pct": 13.0,
            "arr_usd": 7_500_000,
            "arr_growth_yoy_pct": 120,
            "gross_margin_pct": 78,
            "burn_rate_monthly_usd": 550_000,
            "runway_months": 26,
            "headcount": 55,
            "key_metrics": {
                "nrr_pct": 122,
                "cac_payback_months": 16,
                "churn_rate_annual_pct": 9,
            },
            "recent_developments": "PLG motion gaining traction; 40% of new ARR from self-serve",
            "risks": ["Crowded market with AWS/Azure native tools", "SMB-heavy customer base"],
        },
        {
            "name": "GreenGrid",
            "sector": "CleanTech",
            "stage": "Series A",
            "investment_date": "2022-06",
            "initial_investment_usd": 7_000_000,
            "current_valuation_usd": 38_000_000,
            "entry_valuation_usd": 50_000_000,
            "ownership_pct": 11.5,
            "arr_usd": 4_800_000,
            "arr_growth_yoy_pct": 40,
            "gross_margin_pct": 42,
            "burn_rate_monthly_usd": 800_000,
            "runway_months": 14,
            "headcount": 68,
            "key_metrics": {
                "nrr_pct": 95,
                "cac_payback_months": 36,
                "churn_rate_annual_pct": 20,
            },
            "recent_developments": "Government contract signed worth $8M over 3 years; new hardware product launched",
            "risks": ["Hardware margins compress profitability", "Policy/subsidy dependency", "High churn"],
        },
    ],
}


# ── System Prompt (cached — never changes across queries) ─────────────────────

SYSTEM_PROMPT = """You are an expert venture capital analyst and portfolio manager with 20+ years of experience \
in early and growth-stage investing. Your expertise spans SaaS, FinTech, HealthTech, AI/ML, and CleanTech.

When analyzing portfolio companies you:
1. Apply rigorous quantitative analysis using standard VC metrics: ARR growth, NRR, burn multiple, \
Rule of 40 (growth rate + FCF margin), CAC payback period, gross margin
2. Assess qualitative factors: team, market positioning, competitive moats, regulatory exposure, \
product differentiation
3. Flag urgent risks: runway < 12 months, declining growth, NRR < 100, high churn
4. Identify breakout candidates for follow-on investment with specific valuation guidance
5. Calculate portfolio-level concentration risk and MOIC trajectory
6. Provide actionable, specific recommendations — never generic advice

Formatting:
- Lead with the most critical insight or action required
- Use section headers for multi-part analyses
- Quantify recommendations (e.g., "bridge $2–3M to extend runway to 18+ months")
- Flag urgent items with ⚠️
- Be honest about underperformers — do not sugarcoat"""


# ── Analyzer ──────────────────────────────────────────────────────────────────

class VCPortfolioAnalyzer:
    """Analyzes a VC portfolio using Claude with prompt caching."""

    def __init__(self, portfolio: dict, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.portfolio = portfolio
        self._portfolio_context = self._build_portfolio_context()
        self._cache_stats = {"writes": 0, "reads": 0, "tokens_saved": 0}

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, query: str) -> str:
        """
        Stream an analysis of the portfolio for the given query.
        The system prompt and portfolio data are cached after the first call.
        """
        print(f"\n{'─'*62}")
        print(f"  {query}")
        print(f"{'─'*62}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    # Stable portfolio context — cached across all queries
                    {
                        "type": "text",
                        "text": self._portfolio_context,
                        "cache_control": {"type": "ephemeral"},
                    },
                    # Volatile per-query question — never cached
                    {
                        "type": "text",
                        "text": f"\n\n---\n\nQuestion: {query}",
                    },
                ],
            }
        ]

        full_text = ""
        with self.client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                # System prompt is stable — cache it too
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        ) as stream:
            thinking_shown = False
            for event in stream:
                if not hasattr(event, "type"):
                    continue
                if event.type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and block.type == "thinking" and not thinking_shown:
                        print("[Thinking...]\n", flush=True)
                        thinking_shown = True
                elif event.type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and delta.type == "text_delta":
                        print(delta.text, end="", flush=True)
                        full_text += delta.text

            final = stream.get_final_message()
            self._record_cache(final.usage)

        print()
        return full_text

    def portfolio_snapshot(self) -> dict:
        """Compute quick portfolio-level metrics locally (no API call)."""
        companies = self.portfolio["portfolio_companies"]
        total_invested = sum(c["initial_investment_usd"] for c in companies)
        total_fmv = sum(
            c["current_valuation_usd"] * (c["ownership_pct"] / 100)
            for c in companies
        )
        moic = round(total_fmv / total_invested, 2) if total_invested else 0

        # Burn multiple = net burn / net new ARR (lower is better; <1.5 is good)
        def burn_multiple(c):
            monthly_new_arr = (c["arr_usd"] * c["arr_growth_yoy_pct"] / 100) / 12
            if monthly_new_arr <= 0:
                return None
            return round(c["burn_rate_monthly_usd"] / monthly_new_arr, 1)

        # Rule of 40 = ARR growth % + gross margin %
        def rule_of_40(c):
            return c["arr_growth_yoy_pct"] + c["gross_margin_pct"]

        return {
            "fund": self.portfolio["fund_name"],
            "companies": len(companies),
            "total_invested_usd": total_invested,
            "total_fmv_usd": round(total_fmv),
            "portfolio_moic": moic,
            "at_risk": [c["name"] for c in companies if c.get("runway_months", 99) < 12],
            "breakouts": [
                c["name"]
                for c in companies
                if c["arr_growth_yoy_pct"] >= 100
                and c["current_valuation_usd"] > c["entry_valuation_usd"]
            ],
            "company_metrics": [
                {
                    "name": c["name"],
                    "arr_growth_pct": c["arr_growth_yoy_pct"],
                    "nrr_pct": c["key_metrics"]["nrr_pct"],
                    "runway_months": c["runway_months"],
                    "burn_multiple": burn_multiple(c),
                    "rule_of_40": rule_of_40(c),
                    "moic": round(
                        (c["current_valuation_usd"] * c["ownership_pct"] / 100)
                        / c["initial_investment_usd"],
                        2,
                    ),
                }
                for c in companies
            ],
        }

    def print_cache_stats(self):
        """Print cache performance after running queries."""
        s = self._cache_stats
        total = s["writes"] + s["reads"]
        hit_rate = round(s["reads"] / total * 100) if total else 0
        print(f"\n{'═'*62}")
        print("  Cache Performance")
        print(f"{'─'*62}")
        print(f"  Cache writes : {s['writes']}")
        print(f"  Cache hits   : {s['reads']}")
        print(f"  Hit rate     : {hit_rate}%")
        print(f"  Tokens saved : {s['tokens_saved']:,}")
        cost_saved = s["tokens_saved"] * 0.000005 * 0.9  # ~90% savings vs full price
        print(f"  Est. savings : ~${cost_saved:.3f}")
        print(f"{'═'*62}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_portfolio_context(self) -> str:
        p = self.portfolio
        return (
            f"## Fund Overview\n"
            f"Fund: {p['fund_name']}\n"
            f"Fund Size: ${p['fund_size_usd']:,.0f}\n"
            f"Vintage: {p['vintage_year']}\n"
            f"Stage Focus: {', '.join(p['stage_focus'])}\n"
            f"Sector Focus: {', '.join(p['sector_focus'])}\n"
            f"Portfolio Companies: {len(p['portfolio_companies'])}\n\n"
            f"## Company Data\n"
            f"{json.dumps(p['portfolio_companies'], indent=2)}"
        )

    def _record_cache(self, usage):
        created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        if created:
            self._cache_stats["writes"] += 1
        if read:
            self._cache_stats["reads"] += 1
            self._cache_stats["tokens_saved"] += read


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_snapshot(snap: dict):
    print(f"\n{'═'*62}")
    print(f"  {snap['fund']}")
    print(f"{'─'*62}")
    print(f"  Companies      : {snap['companies']}")
    print(f"  Total Invested : ${snap['total_invested_usd']:>15,.0f}")
    print(f"  Portfolio FMV  : ${snap['total_fmv_usd']:>15,.0f}")
    print(f"  Portfolio MOIC : {snap['portfolio_moic']}x")
    print(f"  At Risk (< 12mo runway): {', '.join(snap['at_risk']) or 'None'}")
    print(f"  Breakout Candidates    : {', '.join(snap['breakouts']) or 'None'}")
    print(f"\n  {'Company':<18} {'ARR Gr%':>7} {'NRR%':>6} {'Runway':>7} {'Burn×':>6} {'R40':>5} {'MOIC':>6}")
    print(f"  {'─'*18} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*5} {'─'*6}")
    for m in snap["company_metrics"]:
        bm = f"{m['burn_multiple']:.1f}×" if m["burn_multiple"] else "  —"
        print(
            f"  {m['name']:<18} {m['arr_growth_pct']:>6}% {m['nrr_pct']:>5}%"
            f" {m['runway_months']:>5}mo {bm:>6} {m['rule_of_40']:>5} {m['moic']:>5.2f}×"
        )
    print(f"{'═'*62}\n")


PRESET_QUERIES = [
    ("Full portfolio health assessment",
     "Provide a complete portfolio health assessment. Rank all companies from strongest to weakest "
     "with specific action items for each. Flag any requiring immediate board-level action."),

    ("Follow-on investment opportunities",
     "Which 1–2 companies should receive follow-on capital in the next 6 months? "
     "Provide recommended check size, target ownership, and the valuation ceiling at which the deal still makes sense. "
     "Also identify which company is most at risk of being a write-off."),

    ("Capital efficiency analysis",
     "Calculate and compare the burn multiple and Rule of 40 score for each company. "
     "Rank them by capital efficiency. Which are best positioned to reach profitability first, "
     "and which are destroying capital relative to growth?"),

    ("Custom question", None),
]


def main():
    print(f"\n{'═'*62}")
    print("  VC Portfolio Asset Analyzer")
    print("  Powered by Claude Opus 4.7 + Prompt Caching")
    print(f"{'═'*62}")

    analyzer = VCPortfolioAnalyzer(DEMO_PORTFOLIO)

    # Show quick local snapshot (no API call)
    snap = analyzer.portfolio_snapshot()
    _print_snapshot(snap)

    # Query loop
    while True:
        print("Select an analysis:")
        for i, (label, _) in enumerate(PRESET_QUERIES, 1):
            print(f"  {i}. {label}")
        print("  0. Exit\n")

        choice = input("Choice: ").strip()
        if choice == "0":
            break

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(PRESET_QUERIES):
                raise ValueError
        except ValueError:
            print("Invalid choice.\n")
            continue

        label, query = PRESET_QUERIES[idx]

        if query is None:
            query = input("Enter your question: ").strip()
            if not query:
                continue

        analyzer.analyze(query)

    analyzer.print_cache_stats()
    print("\nGoodbye.\n")


if __name__ == "__main__":
    main()
