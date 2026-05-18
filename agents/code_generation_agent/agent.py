from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from agent_skills.base_skill import BaseSkill
from agents.base_agent import BaseAgent
from prompts.codegen_prompts import CODEGEN_SYSTEM, CODEGEN_USER
from state_management.workflow_state import SystemState, WorkflowState

_SKILLS_DIR = Path(__file__).parent.parent.parent / "agent_skills"


class CodeGenerationAgent(BaseAgent):
    name = "code_generation_agent"

    async def run(self, state: WorkflowState) -> WorkflowState:
        state.current_state = SystemState.GENERATING_CODE
        methodology = state.selected_methodology

        # Check registry first — never generate what already exists
        existing = self.skill_registry.get(methodology)
        if existing:
            self._log.info("skill_already_exists", skill=methodology)
            state.generated_skill_name = methodology
            return state

        # Generate new skill via LLM
        research = state.research_report
        hints = {}
        if research:
            for m in research.recommended_methodologies:
                if m.get("skill_to_use") == methodology or m.get("name") == methodology:
                    hints = m.get("implementation_hints", {})
                    break

        skill_name = methodology.lower().replace(" ", "_").replace("-", "_")
        user_prompt = CODEGEN_USER.format(
            methodology_name=methodology,
            skill_name=skill_name,
            description=f"Auto-generated skill for {methodology}",
            implementation_hints=hints,
        )

        raw_code = self._invoke([
            {"role": "system", "content": CODEGEN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])

        clean_code = _strip_fences(raw_code)

        if not _validate_code(clean_code):
            state.error = f"Generated code for '{methodology}' failed syntax validation."
            self._log.error("codegen_syntax_invalid", methodology=methodology)
            return state

        skill_file = _SKILLS_DIR / f"{skill_name}_skill.py"
        skill_file.write_text(clean_code, encoding="utf-8")
        self._log.info("skill_file_written", path=str(skill_file))

        self.skill_registry.reload()

        if self.skill_registry.get(skill_name):
            state.generated_skill_name = skill_name
            state.selected_methodology = skill_name
            self._log.info("skill_registered_after_generation", skill=skill_name)
        else:
            state.error = (
                f"Skill file written for '{skill_name}' but failed to register. "
                "Check generated code for import errors."
            )

        return state


def _strip_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _validate_code(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
