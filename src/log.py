import logging
import sys
from datetime import datetime
from pathlib import Path

from src.settings import settings

LOG_ROOT = Path("logs")
LOG_ROOT.mkdir(exist_ok=True)


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()

    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(settings.LOG_LEVEL)
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )

    exec_name = Path(sys.argv[0]).stem
    log_file = (
        LOG_ROOT / f"{exec_name}_{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root.addHandler(fh)

    return root
