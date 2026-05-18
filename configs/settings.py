from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "configs" / "default.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_yaml = _load_yaml()


class LLMConfig(BaseModel):
    provider: str = _yaml["llm"]["provider"]
    model: str = _yaml["llm"]["model"]
    temperature: float = _yaml["llm"]["temperature"]
    max_tokens: int = _yaml["llm"]["max_tokens"]


class DataConfig(BaseModel):
    source: str = _yaml["data"]["source"]
    cache_dir: Path = _ROOT / _yaml["data"]["cache_dir"].lstrip("./")


class MLflowConfig(BaseModel):
    tracking_uri: str = str(_ROOT / _yaml["mlflow"]["tracking_uri"].lstrip("./"))
    experiment_name: str = _yaml["mlflow"]["experiment_name"]


class SystemConfig(BaseModel):
    max_inner_attempts: int = _yaml["system"]["max_inner_attempts"]
    max_outer_iterations: int = _yaml["system"]["max_outer_iterations"]
    state_backend: Literal["memory", "redis"] = _yaml["system"]["state_backend"]
    log_level: str = _yaml["system"]["log_level"]


class RedisConfig(BaseModel):
    host: str = _yaml["redis"]["host"]
    port: int = _yaml["redis"]["port"]
    db: int = _yaml["redis"]["db"]
    url: str = ""


class OptunaConfig(BaseModel):
    n_trials: int = _yaml["optuna"]["n_trials"]
    direction: str = _yaml["optuna"]["direction"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    redis_url: str = Field(default="", alias="REDIS_URL")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)

    @field_validator("deepseek_api_key")
    @classmethod
    def api_key_required(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Add it to your .env file.\n"
                "See .env.example for reference."
            )
        return v

    @model_validator(mode="after")
    def sync_overrides(self) -> "Settings":
        if self.mlflow_tracking_uri:
            self.mlflow.tracking_uri = self.mlflow_tracking_uri
        if self.redis_url:
            self.redis.url = self.redis_url
        return self


settings = Settings()
