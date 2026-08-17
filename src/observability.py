"""
LangSmith observability wiring.

LangSmith traces every LLM call, tool call, and graph transition automatically
once the right env vars are set — this module just validates that the config
and environment agree with each other and fails with a clear message rather
than silently tracing nothing.
"""
from __future__ import annotations

import os

from src.config_loader import get_config, env
from src.logging_setup import get_logger

logger = get_logger(__name__)


def init_observability() -> bool:
    """
    Returns True if LangSmith tracing is active, False if intentionally disabled.
    Never raises — a missing API key should degrade to "no tracing", not crash the app.
    """
    cfg = get_config()
    enabled = cfg.get("observability.langsmith.enabled", False)

    if not enabled:
        logger.info("langsmith_disabled_by_config")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    api_key = env("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning(
            "langsmith_enabled_but_api_key_missing",
            extra={"hint": "set LANGCHAIN_API_KEY in .env — falling back to no tracing"},
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    project_name = cfg.get("observability.langsmith.project_name", "budgetbestie")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    logger.info("langsmith_enabled", extra={"project": project_name})
    return True
