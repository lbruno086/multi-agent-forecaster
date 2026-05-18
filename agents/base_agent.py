from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI

from agent_skills.skill_registry import SkillRegistry
from configs.settings import settings
from state_management.workflow_state import WorkflowState
from tools.logger import get_logger

log = get_logger(__name__)


def build_llm(temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm.model,
        temperature=temperature if temperature is not None else settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


class BaseAgent(ABC):
    name: str

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.llm: ChatOpenAI = llm or build_llm()
        self.skill_registry: SkillRegistry = skill_registry or SkillRegistry()
        self._log = get_logger(self.name)

    @abstractmethod
    async def run(self, state: WorkflowState) -> WorkflowState: ...

    def _invoke(self, messages: list[dict[str, str]]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        response = self.llm.invoke(lc_messages)
        return response.content.strip()
