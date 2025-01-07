from typing import Sequence

import torch


def one_hot_fast(shape: Sequence[int],
                 indices: torch.Tensor,
                 one_hot_dim: int = -1,
                 dtype: torch.dtype = torch.int64,
                 device: torch.device = "cpu") -> torch.Tensor:
    """
    Create one hot encoding for the indices
    """
    one_hot = torch.zeros(
        shape,
        dtype=dtype,
        device=device,
        )
    one_hot.scatter_(one_hot_dim, indices.unsqueeze(one_hot_dim), 1)
    return one_hot
