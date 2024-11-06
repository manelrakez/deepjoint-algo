import sys

from pathlib import Path

from loguru import logger


LOGURU_DEFAULT_KWARGS = {"enqueue": True, "diagnose": True, "backtrace": False}


def set_logger(output_dir: Path | None = None):
    logger.remove()
    logger.add(sys.stderr, **LOGURU_DEFAULT_KWARGS)
    if output_dir:
        logger.add(output_dir / "output.log", **LOGURU_DEFAULT_KWARGS)
