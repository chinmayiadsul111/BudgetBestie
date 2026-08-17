"""
LLM provider factory.

Every agent asks this module for a chat model and never imports
langchain_ollama / langchain_openai directly. That means switching the whole
pipeline from local Ollama to Azure OpenAI (or plain OpenAI) is a ONE-LINE
change in config.yaml (llm.provider), not a code change.
"""
from __future__ import annotations

from typing import Any

from src.config_loader import get_config, env
from src.logging_setup import get_logger

logger = get_logger(__name__)


class UnsupportedProviderError(ValueError):
    pass


def get_chat_model(temperature: float | None = None) -> Any:
    """
    Returns a LangChain-compatible chat model instance based on
    config.yaml -> llm.provider.
    """
    cfg = get_config()
    provider = cfg.get("llm.provider", "ollama")
    temp = temperature if temperature is not None else cfg.get("llm.temperature", 0.2)
    max_tokens = cfg.get("llm.max_tokens", 1024)
    timeout = cfg.get("llm.request_timeout_seconds", 60)

    logger.info("llm_provider_selected", extra={"provider": provider, "temperature": temp})

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=cfg.get("llm.ollama.base_url", "http://localhost:11434"),
            model=cfg.get("llm.ollama.model", "llama3.1:8b"),
            temperature=temp,
            num_predict=max_tokens,
        )

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=env("AZURE_OPENAI_ENDPOINT", required=True),
            api_key=env("AZURE_OPENAI_API_KEY", required=True),
            azure_deployment=env(
                "AZURE_OPENAI_DEPLOYMENT", default=cfg.get("llm.azure_openai.deployment")
            ),
            api_version=cfg.get("llm.azure_openai.api_version", "2024-08-01-preview"),
            temperature=temp,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=env("OPENAI_API_KEY", required=True),
            model=cfg.get("llm.openai.model", "gpt-4o-mini"),
            temperature=temp,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    raise UnsupportedProviderError(
        f"Unknown llm.provider '{provider}'. Expected one of: ollama, azure_openai, openai."
    )
