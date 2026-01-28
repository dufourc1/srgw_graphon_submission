from typing import Optional

import torch

DEFAULT_DTYPE = torch.float32


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed_torch(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
