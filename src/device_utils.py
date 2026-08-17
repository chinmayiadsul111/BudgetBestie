"""
Device selection: decides whether embedding/local-inference workloads run on
GPU or CPU, honoring config overrides, and never crashing the app if CUDA is
misconfigured (common on shared/dev machines).
"""
from __future__ import annotations

from src.config_loader import get_config
from src.logging_setup import get_logger

logger = get_logger(__name__)


def resolve_device() -> str:
    """
    Returns "cuda" or "cpu".

    Resolution order:
      1. config.compute.device == "cpu" or "cuda" -> forced, no detection.
      2. config.compute.device == "auto" -> probe torch.cuda.is_available().
      3. Any probing error -> fall back to CPU if configured to do so,
         otherwise re-raise (fail loudly rather than silently degrade perf
         in an environment that expected GPU guarantees, e.g. a training box).
    """
    cfg = get_config()
    requested = cfg.get("compute.device", "auto")
    log_selection = cfg.get("compute.log_device_selection", True)
    fallback_ok = cfg.get("compute.fallback_to_cpu_on_error", True)

    if requested in ("cpu", "cuda"):
        if log_selection:
            logger.info("device_selection_forced", extra={"device": requested})
        return requested

    if requested != "auto":
        logger.warning(
            "device_selection_invalid_value_defaulting_to_auto",
            extra={"configured_value": requested},
        )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        if log_selection:
            detail = {"device": device, "cuda_available": cuda_available}
            if cuda_available:
                detail["gpu_name"] = torch.cuda.get_device_name(0)
                detail["gpu_count"] = torch.cuda.device_count()
            logger.info("device_selection_auto_detected", extra=detail)
        return device
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any torch/driver issue
        logger.warning(
            "device_auto_detection_failed",
            extra={"error": str(exc), "falling_back_to_cpu": fallback_ok},
        )
        if fallback_ok:
            return "cpu"
        raise


_DEVICE_CACHE: str | None = None


def get_device() -> str:
    """Cached accessor — detection runs once per process."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        _DEVICE_CACHE = resolve_device()
    return _DEVICE_CACHE
