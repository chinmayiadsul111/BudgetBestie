"""
Single entrypoint for the whole project. No CLI flags — everything that would
normally be a flag lives in config/config.yaml instead.

    python run.py

starts the FastAPI server. Interactive terminal demo (no server) is in
demo.py — see COMMANDS.md for both.
"""
import uvicorn

from src.config_loader import get_config

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "src.api:app",
        host=cfg.get("server.host", "0.0.0.0"),
        port=cfg.get("server.port", 8000),
        reload=cfg.get("server.reload", False),
    )
