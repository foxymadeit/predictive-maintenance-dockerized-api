import logging
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for Python logging.

    Converts log records into a JSON-serializable dictionary including 
    timestamp, log level, logger name, and the message.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record as a JSON string.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            A JSON-encoded string containing the log details and any 
            available extra information.
        """
        log: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # extra info if available
        if hasattr(record, "extra"):
            log.update(record.extra)

        return json.dumps(log, ensure_ascii=False)


def setup_json_logger(
    name: str = "app",
    *,
    log_dir: Path = Path("logs"),
    log_file: str = "api.log",
) -> logging.Logger:
    """Configures and returns a logger with JSON formatting.

    Sets up a logger that outputs to both the console (stdout) and a specified 
    log file. It ensures the log directory exists and disables propagation 
    to prevent duplicate logs.

    Args:
        name (str): Name of the logger instance. Defaults to "app".
        log_dir (Path): Directory where the log file will be stored. 
            Defaults to Path("logs").
        log_file (str): Name of the log file. Defaults to "api.log".

    Returns:
        A configured logging.Logger instance ready for JSON-structured logging.

    Example:
        >>> logger = setup_json_logger("api_service")
        >>> logger.info("Model prediction started")
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = JsonFormatter()

    # stdout handler 
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # file handler
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger