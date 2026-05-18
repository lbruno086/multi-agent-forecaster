from __future__ import annotations

from typing import Any

import numpy as np

from agent_skills.base_skill import BaseSkill, SkillResult
from configs.settings import settings
from tools.logger import get_logger

log = get_logger(__name__)


class WebResearchSkill(BaseSkill):
    name = "web_research"
    description = "Searches the web via Tavily API for financial forecasting methodologies."

    def execute(self, params: dict[str, Any]) -> SkillResult:
        query: str = params["query"]
        max_results: int = params.get("max_results", 5)
        search_depth: str = params.get("search_depth", "advanced")

        results = self._search(query, max_results, search_depth)

        return SkillResult(
            skill_name=self.name,
            predictions=np.array([]),
            fold_scores=[],
            train_score=0.0,
            val_score=0.0,
            model=None,
            metadata={"results": results, "query": query},
        )

    def _search(self, query: str, max_results: int, search_depth: str) -> list[dict]:
        if not settings.tavily_api_key:
            log.warning("tavily_key_missing", msg="Returning empty results")
            return []
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in response.get("results", [])
            ]
        except Exception as exc:
            log.error("tavily_search_failed", error=str(exc))
            return []

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "query": "str — search query",
                "max_results": "int (default 5)",
                "search_depth": "str 'basic' | 'advanced' (default 'advanced')",
            }
        }
