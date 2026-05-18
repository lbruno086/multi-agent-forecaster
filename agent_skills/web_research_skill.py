from __future__ import annotations

from typing import Any

import numpy as np

from agent_skills.base_skill import BaseSkill, SkillResult
from tools.logger import get_logger

log = get_logger(__name__)


class WebResearchSkill(BaseSkill):
    name = "web_research"
    description = "Searches the web via DuckDuckGo (no API key required)."

    def execute(self, params: dict[str, Any]) -> SkillResult:
        query: str = params["query"]
        max_results: int = params.get("max_results", 5)

        results = self._search(query, max_results)

        return SkillResult(
            skill_name=self.name,
            predictions=np.array([]),
            fold_scores=[],
            train_score=0.0,
            val_score=0.0,
            model=None,
            metadata={"results": results, "query": query},
        )

    def _search(self, query: str, max_results: int) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                    "score": 1.0,
                }
                for r in (raw or [])
            ]
        except Exception as exc:
            log.warning("web_search_failed", error=str(exc))
            return []

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "query": "str — search query",
                "max_results": "int (default 5)",
            }
        }
