from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from prompts.research_prompts import RESEARCH_SYSTEM, RESEARCH_USER
from state_management.workflow_state import ResearchReport, SystemState, WorkflowState
from tools.logger import get_logger

log = get_logger(__name__)

_ASSET_TYPE_MAP = {
    "BTC": "crypto", "ETH": "crypto", "BNB": "crypto",
    "AAPL": "stock", "MSFT": "stock", "TSLA": "stock",
    "EUR": "forex", "GBP": "forex",
    "GC": "commodity", "CL": "commodity",
}


class ResearchAgent(BaseAgent):
    name = "research_agent"

    async def run(self, state: WorkflowState) -> WorkflowState:
        state.current_state = SystemState.RESEARCHING
        asset_type = _detect_asset_type(state.ticker)

        # Search web for context
        web_skill = self.skill_registry.get("web_research")
        sources: list[dict] = []
        if web_skill:
            query = (
                f"best machine learning methods predict {state.ticker} "
                f"{state.timeframe} price {state.metric_to_optimize}"
            )
            web_result = web_skill.execute({"query": query, "max_results": 5})
            sources = web_result.metadata.get("results", [])

        # Build prompt
        user_prompt = RESEARCH_USER.format(
            asset_type=asset_type,
            ticker=state.ticker,
            timeframe=state.timeframe,
            metric_name=state.metric_to_optimize,
            is_lower_better=True,
            threshold=state.target_threshold,
            tried_methodologies=state.tried_methodologies or "none yet",
        )

        raw = self._invoke([
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])

        report = _parse_report(raw, asset_type, sources)
        state.research_report = report

        if report.recommended_methodologies:
            best = report.recommended_methodologies[0]
            skill_name = best.get("skill_to_use", best.get("name", "xgboost"))
            state.selected_methodology = skill_name
            state.inner_loop_attempt = 0
            state.outer_loop_iteration += 1

        self._log.info(
            "research_complete",
            asset_type=asset_type,
            methodologies=[m.get("name") for m in report.recommended_methodologies],
            outer_iter=state.outer_loop_iteration,
        )
        return state


def _detect_asset_type(ticker: str) -> str:
    ticker_upper = ticker.upper()
    for key, asset_type in _ASSET_TYPE_MAP.items():
        if key in ticker_upper:
            return asset_type
    return "financial_asset"


def _parse_report(raw: str, asset_type: str, sources: list) -> ResearchReport:
    try:
        # Strip any accidental markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        data = json.loads(clean)
        if sources and not data.get("sources"):
            data["sources"] = sources
        return ResearchReport(**data)
    except Exception as exc:
        log.warning("research_parse_failed", error=str(exc), raw_preview=raw[:200])
        return ResearchReport(
            asset_type=asset_type,
            recommended_methodologies=[
                {"name": "xgboost", "rank": 1, "skill_to_use": "xgboost",
                 "rationale": "fallback", "expected_metric_range": {},
                 "tradeoffs": {}, "implementation_hints": {}}
            ],
            feature_engineering_suggestions=[],
            ensemble_strategies=[],
            sources=sources,
        )
