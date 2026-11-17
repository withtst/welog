"""Local rotating logs for GUI launches, including pythonw without a console."""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path


def default_log_dir() -> Path:
    project = Path(__file__).resolve().parents[2]
    return project / "logs" if (project / "pyproject.toml").exists() else Path.home() / "WeLog" / "logs"


def configure_logging(directory: Path | None = None, *, install_hooks: bool = True) -> Path:
    directory = (directory if directory is not None else default_log_dir()).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "welog.log"
    logger = logging.getLogger("welog")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_welog_handler", False):
            logger.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler._welog_handler = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    if install_hooks:
        def exception_hook(kind, value, traceback):
            logger.critical("uncaught_exception", exc_info=(kind, value, traceback))
            if sys.stderr is not None:
                sys.__excepthook__(kind, value, traceback)

        def thread_hook(args):
            logger.critical("uncaught_thread_exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

        sys.excepthook = exception_hook
        threading.excepthook = thread_hook
    logger.info("logging_initialized python=%s log=%s", sys.version.split()[0], path)
    return path
