import logging
import logging.handlers
import os

from config import nacos_config as cfg


def setup_logging() -> None:
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

    # 避免重复添加（热重载场景）
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        root.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(console_handler)

    # 屏蔽第三方库（FunASR / ModelScope）的内部日志，只保留 WARNING 及以上
    for noisy_logger in ("funasr", "modelscope", "modelscope.pipelines"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
