from typing import Sequence

import torch


def create_anchor_points_and_stride_tensors(grid_sizes: torch.Tensor, strides: Sequence[int], offset: float = 0.5):
    """
    grid_sizes: (n, 2) -> (y, x)
    anchor_points: (n, 2) -> xy, with x rising first
    strides_mask: (n) -> stride for each anchor point
    """
    anchor_points_list = []
    strides_all = []
    for (h, w), s in zip(grid_sizes, strides):
        anchor_points = torch.stack(torch.meshgrid(torch.arange(w, dtype=torch.float),
                                                   torch.arange(h, dtype=torch.float), indexing="xy"), 2).view(-1, 2)
        anchor_points += offset
        anchor_points_list.append(anchor_points)
        strides_all.append(torch.full(((w*h.item()) ,), s))
    anchor_points = torch.cat(anchor_points_list, 0)
    strides_mask = torch.cat(strides_all, dim=0)
    return anchor_points, strides_mask
