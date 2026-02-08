import logging
import os
import sys
import threading
from typing import Optional

import torch.distributed as dist
import tqdm

_LOCK = threading.Lock()
_STATE: dict[str, tuple[int, Optional[str]]] = {}


def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _global_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank())


def _logger_key(name: Optional[str]) -> str:
    return "<root>" if name is None else name


def _compute_output_file(output: str) -> str:
    if os.path.splitext(output)[-1] in (".txt", ".log"):
        return output
    return os.path.join(output, "logs", "log.txt")


def _remove_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_ijepath_owned", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that writes through tqdm to avoid progress-bar corruption."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure root/named logger with tqdm-safe stdout and optional file logging."""
    logging.captureWarnings(capture_warnings)
    logger = logging.getLogger(name)
    logger_key = _logger_key(name)
    state = (int(level), output)

    with _LOCK:
        if _STATE.get(logger_key) == state:
            return logger

        _remove_owned_handlers(logger)
        logger.setLevel(level)
        if name is not None:
            logger.propagate = False

        fmt = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y%m%d %H:%M:%S")

        if _is_main_process():
            stdout_handler = TqdmLoggingHandler(stream=sys.stdout)
            stdout_handler.setLevel(level)
            stdout_handler.setFormatter(formatter)
            stdout_handler._ijepath_owned = True  # type: ignore[attr-defined]
            logger.addHandler(stdout_handler)

        if output:
            filename = _compute_output_file(output)
            if not _is_main_process():
                filename = f"{filename}.rank{_global_rank()}"
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)

            file_handler = logging.FileHandler(filename, mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler._ijepath_owned = True  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

        _STATE[logger_key] = state

    return logger
