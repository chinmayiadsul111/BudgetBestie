"""
Central config loader.

Every other module imports `get_config()` instead of reading config.yaml or
os.environ directly. This keeps config access in one testable place and means
switching environments (local -> staging -> prod) never requires touching
business logic.
"""
from __future__ import annotations

import os
import functools
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when required config or env values are missing or invalid."""


class Config:
    """
    Thin wrapper around the parsed YAML dict with dotted-path access,
    e.g. cfg.get("llm.provider").
    """

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw

    def get(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self._raw
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_path: str) -> Any:
        sentinel = object()
        value = self.get(dotted_path, sentinel)
        if value is sentinel:
            raise ConfigError(f"Missing required config key: '{dotted_path}'")
        return value

    def as_dict(self) -> dict[str, Any]:
        return self._raw


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Cached singleton loader. Loading is cheap but we don't want every agent
    re-parsing YAML on every call inside a hot LangGraph loop.
    """
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    if not CONFIG_PATH.exists():
        raise ConfigError(f"config.yaml not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    return Config(raw)


def env(key: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable, with a clear error if it's required and missing."""
    value = os.environ.get(key, default)
    if required and not value:
        raise ConfigError(
            f"Environment variable '{key}' is required but not set. "
            f"Check your .env file against config/.env.example."
        )
    return value


def reset_config_cache() -> None:
    """Useful in tests when swapping config files between test cases."""
    get_config.cache_clear()
