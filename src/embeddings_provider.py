"""
Embeddings provider factory — mirrors llm_provider.py's switch pattern.
"""
from __future__ import annotations

from typing import Any

from src.config_loader import get_config, env
from src.device_utils import get_device
from src.logging_setup import get_logger

logger = get_logger(__name__)


def get_embeddings_model() -> Any:
    cfg = get_config()
    provider = cfg.get("embeddings.provider", "ollama")
    device = get_device()

    logger.info("embeddings_provider_selected", extra={"provider": provider, "device": device})

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=cfg.get("embeddings.ollama.base_url", "http://localhost:11434"),
            model=cfg.get("embeddings.ollama.model", "nomic-embed-text"),
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=env("OPENAI_API_KEY", required=True),
            model=cfg.get("embeddings.openai.model", "text-embedding-3-small"),
        )

    raise ValueError(f"Unknown embeddings.provider '{provider}'. Expected: ollama, openai.")
