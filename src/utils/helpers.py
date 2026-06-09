# filename: src/utils/helpers.py
# purpose:  Shared utilities — JSON serialisation, artifact persistence
# version:  1.0

# stdlib
import json
import logging
from pathlib import Path

# third-party
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalar types that json.dumps rejects by default."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    logger.info("Saved JSON: %s", path)
