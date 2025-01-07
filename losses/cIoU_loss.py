from typing import Literal, Optional

import torch
from torch import nn

from YoloV12.bbox_utilities.bbox_iou import bbox_ciou_1to1


class IoULoss(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred_bboxes: torch.Tensor,
                target_bboxes: torch.Tensor,
                weights: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the IoU Loss
        params:
            pred_bboxes: (b, 4)
            target_bboxes: (b, 4)
            weights: (b) Optional
        """
        IoU = bbox_ciou_1to1(pred_bboxes, target_bboxes, box_format="xyxy").squeeze()
        IoU_loss = 1 - IoU if weights is None else (1 - IoU) * weights
        if self.reduction == "mean":
            IoU_loss = IoU_loss.mean()
        elif self.reduction == "sum":
            IoU_loss = IoU_loss.sum()
        return IoU_loss
