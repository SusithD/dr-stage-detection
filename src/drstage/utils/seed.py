"""Random seed initialization for reproducibility."""

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set seeds for random, numpy, and torch for reproducibility.

    Args:
        seed: Seed value.
        deterministic: If True, enable deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def worker_init_fn(worker_id: int) -> None:
    """Worker initialization for DataLoader multiprocessing.

    Args:
        worker_id: Worker ID from DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
