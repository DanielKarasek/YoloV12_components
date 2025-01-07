from typing import Literal, Optional

import torch
from torch import nn
from torch.nn.functional import cross_entropy


class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction


    def forward(self,
                pred_distribution: torch.Tensor,
                target: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Distribution Focal Loss
        params:
            pred: (b,  reg_max)
            target: (b,  1)
        """
        left_target_bin = target.long()
        right_target_bin = left_target_bin + 1
        left_weight = right_target_bin - target
        right_weight = target - left_target_bin
        loss = (cross_entropy(pred_distribution, left_target_bin.view(-1), reduction="none") * left_weight
              + cross_entropy(pred_distribution, right_target_bin.view(-1), reduction="none") * right_weight)
        if weights is not None:
            loss = (loss.view(-1, 4) * weights.unsqueeze(-1)).flatten()
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss =  loss.sum()

        return loss
