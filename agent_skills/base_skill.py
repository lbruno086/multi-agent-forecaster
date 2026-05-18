from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class SkillResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    skill_name: str
    predictions: np.ndarray
    fold_scores: list[float]
    train_score: float
    val_score: float
    model: Any
    model_path: str | None = None
    params_used: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class BaseSkill(ABC):
    name: str
    description: str
    version: str = "1.0.0"

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> SkillResult: ...

    @abstractmethod
    def get_schema(self) -> dict[str, Any]: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"
