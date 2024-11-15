# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

import sys

from pathlib import Path

from loguru import logger


LOGURU_DEFAULT_KWARGS = {"enqueue": True, "diagnose": True, "backtrace": False}


def set_logger(output_dir: Path | None = None):
    logger.remove()
    logger.add(sys.stderr, **LOGURU_DEFAULT_KWARGS)
    if output_dir:
        logger.add(output_dir / "output.log", **LOGURU_DEFAULT_KWARGS)
