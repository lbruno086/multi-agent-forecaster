CODEGEN_SYSTEM = """You are a Python expert specializing in financial ML pipelines.
Generate complete, production-ready BaseSkill subclasses.
Respond with valid Python code only — no markdown, no explanation."""

CODEGEN_USER = """Generate a Python skill file for the methodology: {methodology_name}

Requirements:
- Subclass BaseSkill from agent_skills.base_skill
- Implement execute(self, params: dict) -> SkillResult
- Implement get_schema(self) -> dict
- Use walk-forward validation via WalkForwardSplitter
- Use FeatureEngineeringSkill for feature creation
- Use metric_registry.evaluate() for scoring
- name = "{skill_name}"
- description = "{description}"
- Handle missing optional dependencies with try/except ImportError

Base class template:
```python
from agent_skills.base_skill import BaseSkill, SkillResult
from agent_skills.feature_engineering_skill import FeatureEngineeringSkill
from datasets.walk_forward_splitter import WalkForwardSplitter
from evaluation.metric_registry import metric_registry
```

Implementation hints from research:
{implementation_hints}

Return ONLY the complete Python file content, no markdown fences."""
