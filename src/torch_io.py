"""torch.load wrapper for tensor checkpoints (state dicts) — avoids pickle FutureWarning when supported."""

from pathlib import Path
from typing import Any, Union

import torch


def load_state_dict_checkpoint(path: Union[str, Path], map_location: Any = None):
    """Load a .pt file saved via torch.save(model.state_dict(), ...)."""
    path = Path(path)
    kwargs = {"map_location": map_location}
    try:
        return torch.load(path, **kwargs, weights_only=True)
    except TypeError:
        return torch.load(path, **kwargs)
