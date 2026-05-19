from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logger(name: str = "cancer_latent_space", level: int = logging.INFO) -> logging.Logger:
    """Create a compact stdout logger suitable for R system2() capture."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | [PY:%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def require_file(path: Path, label: str | None = None) -> None:
    """Raise a clear error when a required input is missing."""
    if not Path(path).exists():
        name = label or "Required file"
        raise FileNotFoundError(f"{name} not found: {path}")
