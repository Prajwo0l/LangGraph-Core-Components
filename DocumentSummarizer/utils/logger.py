"""
utils/logger.py — Centralized logging setup.
"""

import logging
import sys
from pathlib import Path

from config import Config


def get_logger(name: str, cfg: Config | None = None) -> logging.Logger:
    """
    Return a logger with the given *name*.
    On first call the root handler is configured; subsequent calls just
    return the named child logger.
    """
    cfg = cfg or Config()

    root = logging.getLogger()
    if root.handlers:                   # already configured
        return logging.getLogger(name)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(cfg.log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    return logging.getLogger(name)
