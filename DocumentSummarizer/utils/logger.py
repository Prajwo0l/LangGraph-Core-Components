"""
utils/logger.py — Centralized logging setup.

FIX: The old implementation checked `if root.handlers` to skip re-configuration.
This silently matched Python's built-in `lastResort` handler, locking in INFO
level permanently and ignoring any --log-level flag passed by the user.

New design:
  - Track whether WE have configured logging via a module-level flag.
  - Always reconfigure if a new cfg is explicitly passed in.
  - get_logger(name, cfg) with a real cfg → reconfigures handlers to new level.
  - get_logger(name) with no cfg → returns existing logger without overwriting.
"""

import logging
import sys
from pathlib import Path

_CONFIGURED = False          # True once we have set up our own handlers
_CURRENT_LEVEL = "INFO"      # Track what level we configured


def get_logger(name: str, cfg=None) -> logging.Logger:
    """
    Return a named logger, configuring root handlers if needed.

    Parameters
    ----------
    name : Module name — pass __name__.
    cfg  : Config instance.  If provided AND logging has not been configured
           yet (or the level has changed), root handlers are set up / updated.
           If None, the existing configuration is left untouched.
    """
    global _CONFIGURED, _CURRENT_LEVEL

    if cfg is not None:
        desired_level = cfg.log_level.upper()
        # Reconfigure whenever: first time, OR the requested level has changed.
        if not _CONFIGURED or desired_level != _CURRENT_LEVEL:
            _setup_handlers(cfg)
            _CONFIGURED = True
            _CURRENT_LEVEL = desired_level

    return logging.getLogger(name)


def _setup_handlers(cfg) -> None:
    """
    Remove all existing root handlers (including Python's lastResort),
    then attach a fresh console + file handler at the configured level.
    """
    root = logging.getLogger()

    # Remove every existing handler — including the default lastResort handler
    # that Python attaches before any logging calls are made.
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(cfg.log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)
