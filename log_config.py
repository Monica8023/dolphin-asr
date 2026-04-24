import logging
import logging.handlers
import os
import queue

from config import nacos_config as cfg


_LOG_QUEUE: queue.Queue | None = None
_QUEUE_LISTENER: logging.handlers.QueueListener | None = None


def setup_logging() -> None:
    global _LOG_QUEUE, _QUEUE_LISTENER
    log_path = cfg.get("log_path", "./log")
    log_level = cfg.get("log_level", "INFO")
    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, "dolphin-asr.log")
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in list(root.handlers):
        if isinstance(handler, (logging.handlers.RotatingFileHandler, logging.StreamHandler)) and not isinstance(handler, logging.handlers.QueueHandler):
            root.removeHandler(handler)

    if _LOG_QUEUE is None:
        _LOG_QUEUE = queue.Queue(-1)
    if _QUEUE_LISTENER is None:
        _QUEUE_LISTENER = logging.handlers.QueueListener(_LOG_QUEUE, file_handler, console_handler)
        _QUEUE_LISTENER.start()

    if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
        root.addHandler(logging.handlers.QueueHandler(_LOG_QUEUE))

    for noisy_logger in ("funasr", "modelscope", "modelscope.pipelines"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
