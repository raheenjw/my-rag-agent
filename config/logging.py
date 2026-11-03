"""Application logging configuration — console + logs/ directory."""

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: int = logging.INFO) -> Path:
    """Configure console and file logging. Returns logs directory path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # All app logs
    app_log = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
    app_log.setFormatter(formatter)
    root.addHandler(app_log)

    # Retrieval pipeline logs (also written to dedicated file)
    retrieval_logger = logging.getLogger("services.retrieval")
    retrieval_logger.setLevel(level)
    retrieval_file = logging.FileHandler(LOGS_DIR / "retrieval.log", encoding="utf-8")
    retrieval_file.setFormatter(formatter)
    retrieval_logger.addHandler(retrieval_file)

    return LOGS_DIR
