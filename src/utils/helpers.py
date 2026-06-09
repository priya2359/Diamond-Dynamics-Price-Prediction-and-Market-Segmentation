# filename: src/utils/helpers.py
# purpose:  Shared utilities — JSON serialisation, artifact persistence, figure saving
# version:  1.1

# stdlib
import json
import logging
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
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


def save_figure(fig: plt.Figure, name: str, figures_dir: Path, dpi: int = 150) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
