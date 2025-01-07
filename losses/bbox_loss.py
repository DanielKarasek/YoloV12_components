from typing import Literal

import torch
from torch import nn

from YoloV12.bbox_utilities.bbox_conversion import xyxy2offsets
from YoloV12.losses.cIoU_loss import IoULoss
from YoloV12.losses.distribution_focal_loss import DistributionFocalLoss


class BboxLoss(nn.Module):
    # Combines IoU loss and DFL
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "sum"):
        super(BboxLoss, self).__init__()
        self.dfl = DistributionFocalLoss(reduction=reduction)
        self.iou_loss = IoULoss(reduction=reduction)
        self.reduction = reduction

    def forward(self,
                pred_distribution: torch.Tensor,
                pred_bboxes: torch.Tensor,
                target_bboxes: torch.Tensor,
                anchor_points: torch.Tensor,
                target_score_weights: torch.Tensor,):
        """
        Forward pass of the Bbox Loss
        params:
            pred_distribution: (n, 4*bin_count)
            pred_bboxes: (n, 4)  # xyxy
            target_bboxes: (n, 4) # xyxy
            anchor_points: (n, 2),
            target_score_weights: (n)
        """
        IoU_loss = self.iou_loss(pred_bboxes, target_bboxes, target_score_weights)

        dist_bins = pred_distribution.shape[-1] // 4
        ltrb_target_bboxes = xyxy2offsets(target_bboxes, anchor_points).clamp_(0, dist_bins - 1 - 1e-6)
        dfl_loss = self.dfl(pred_distribution.view(-1, dist_bins), ltrb_target_bboxes.view(-1), target_score_weights)

        return  IoU_loss, dfl_loss
