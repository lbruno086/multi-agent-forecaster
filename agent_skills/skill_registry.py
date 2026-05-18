from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

from agent_skills.base_skill import BaseSkill
from tools.logger import get_logger

log = get_logger(__name__)

_SKILLS_DIR = Path(__file__).parent


class SkillRegistry:
    """Auto-discovers and caches BaseSkill subclasses from agent_skills/."""

    def __init__(self) -> None:
        self._registry: dict[str, BaseSkill] = {}
        self._scan()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _scan(self) -> None:
        for path in sorted(_SKILLS_DIR.glob("*_skill.py")):
            module_name = f"agent_skills.{path.stem}"
            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseSkill)
                        and obj is not BaseSkill
                        and hasattr(obj, "name")
                    ):
                        instance = obj()
                        self._registry[instance.name] = instance
                        log.info("skill_registered", skill=instance.name, version=instance.version)
            except Exception as exc:
                log.warning("skill_scan_failed", file=path.name, error=str(exc))

    def reload(self) -> None:
        """Re-scan skills directory. Call after Code Generation Agent creates a new skill."""
        self._registry.clear()
        self._scan()

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> BaseSkill | None:
        return self._registry.get(name)

    def get_or_create(self, name: str, fallback: Callable[[], BaseSkill] | None = None) -> BaseSkill:
        skill = self.get(name)
        if skill is not None:
            return skill
        if fallback is not None:
            skill = fallback()
            self._registry[skill.name] = skill
            log.info("skill_created_via_fallback", skill=skill.name)
            return skill
        raise KeyError(
            f"Skill '{name}' not found. Available: {self.list_available()}"
        )

    def register(self, skill: BaseSkill) -> None:
        self._registry[skill.name] = skill
        log.info("skill_manually_registered", skill=skill.name)

    def list_available(self) -> list[str]:
        return sorted(self._registry)
