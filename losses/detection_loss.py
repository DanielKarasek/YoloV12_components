from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from YoloV12.bbox_utilities.bbox_conversion import cxcywh2xyxy, distribution2offsets, offset2xyxy, xywh2xyxy
from YoloV12.losses.bbox_loss import BboxLoss
from YoloV12.assigners.tal import TaskAlignedAssignerLayer
from YoloV12.yolo_utils import create_anchor_points_and_stride_tensors



class DetectionLoss(nn.Module):

    def __init__(self,
                 bbox_bin_cnt: int,
                 cls_cnt: int,
                 strides: Sequence[int],
                 bbox_scale: float,
                 cls_scale: float,
                 dfl_scale: float,
                 ):
        super(DetectionLoss, self).__init__()

        self._bin_cnt = bbox_bin_cnt
        self._cls_cnt = cls_cnt
        self._output_size = self._bin_cnt * 4 + self._cls_cnt

        self._strides = torch.tensor(strides)
        self._tal = TaskAlignedAssignerLayer(cls_cnt=cls_cnt)
        self._classification_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._bbox_loss = BboxLoss(reduction="sum")

        self._bbox_scale = bbox_scale
        self._cls_scale = cls_scale
        self._dfl_scale = dfl_scale


    def linearize_targets(self, gt_concatenate: torch.Tensor, batch_size: int ) -> (
            Tuple[torch.Tensor, torch.Tensor]):
        """
        params:
            gt_concatenate: (total_boxes, 1{=img_idx}+4{=bbox}+1{=cls})
            b_size: int
        returns:
            targets: (b, total_boxes, 4{=bbox}+1{=cls}) bbox==XYXY
        """
        total_boxes, len_data = gt_concatenate.shape
        if total_boxes == 0:
            targets = torch.zeros(batch_size, 0, len_data-1)
            gt_mask = torch.zeros(batch_size, 0)
        else:
            img_idx, counts = torch.unique(gt_concatenate[:, 0], return_counts=True)
            targets = torch.zeros(batch_size, counts.max(), len_data-1, device=gt_concatenate.device)
            gt_mask = torch.zeros(batch_size, counts.max(), device=gt_concatenate.device)
            counts_sum = 0
            for img_idx, counts in zip(img_idx, counts):
                img_idx = img_idx.long().item()
                targets[img_idx, :counts] = gt_concatenate[counts_sum:counts_sum+counts, 1:]
                gt_mask[img_idx, :counts] = 1
                counts_sum += counts
        return targets, gt_mask

    @staticmethod
    def set_bboxes2xyxy_img_coordinates(bboxes: torch.Tensor, img_size: torch.Tensor) -> torch.Tensor:
        """
        params:
            bboxes: (b, max_bboxes, 4) # XYWH
            img_sizes: (b, 2)
        returns:
            bboxes: (b, max_bboxes, 4) # XYXY
        """
        return cxcywh2xyxy(bboxes).mul_(img_size.repeat(2).view(1,1,4))

    @staticmethod
    def extract_positive(pred_distribution: torch.Tensor, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor,
                         target_scores_onehot: torch.Tensor, anchor_points: torch.Tensor, fg_mask: torch.Tensor) -> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        fg_mask = fg_mask.bool()
        pred_bboxes = pred_bboxes[fg_mask]
        pred_distribution = pred_distribution[fg_mask]
        target_bboxes = target_bboxes[fg_mask]
        anchor_points = anchor_points.view(1, -1, 2).expand(3, -1, 2)
        anchor_points = anchor_points[fg_mask]
        weights = target_scores_onehot[fg_mask].sum(-1)
        return pred_distribution, pred_bboxes, target_bboxes, anchor_points, weights

    def get_linearized_split_predictions(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        params:
            predictions: layer_cnt*[(b, W_l, H_l, 4*bin_count + cls_cnt)]
        returns:
            pred_dist: (b, K, 4*bin_cnt)
            pred_cls: (b, K, cls_cnt)
        """
        bs = len(predictions[0])
        predictions_linearized = torch.cat([layer_predictions.view(bs, -1, self._output_size)
                                            for layer_predictions in predictions], dim=1)
        pred_dist_raw, pred_scores_raw = predictions_linearized.split([self._bin_cnt * 4, self._cls_cnt], dim=-1)
        return pred_dist_raw, pred_scores_raw

    def get_grid_sizes_and_image_size(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        params:
            predictions: layer_cnt*[(b, H_l, W_l, 4*bin_count + cls_cnt)]
        returns:
            grid_sizes: (layer_cnt, 2) # yx
            img_size: (b, 2) # yx
        """
        grid_sizes = torch.cat([torch.tensor(layer.shape[1:3]) for layer in predictions], dim=0).view(-1, 2)
        img_size = (grid_sizes * self._strides.repeat_interleave(2).view(-1, 2))[0]
        return grid_sizes, img_size

    def preprocess_targets(self, targets: Dict[str, torch.Tensor], bs: int, img_size: torch.Tensor) -> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        params:
            targets: Dict{img_idx: (n), gt_bboxes: (n, 4),  gt_labels: (n)}
            bs: int
        returns:
            gt_bboxes: (bs, max_single_image_bboxes, 4) # XYXY in img_coordinates
            gt_cls: (bs, max_single_image_bboxes, 1)
            gt_mask: (bs, max_single_image_bboxes)
        """
        targets_linearized, gt_mask = self.linearize_targets(
            torch.cat([targets["img_idx"].unsqueeze(-1), targets["gt_bboxes"], targets["gt_labels"].unsqueeze(-1)], dim=1), bs)
        gt_bboxes, gt_cls = targets_linearized.split([4, 1], dim=-1)
        gt_bboxes = self.set_bboxes2xyxy_img_coordinates(gt_bboxes, img_size)
        return gt_bboxes, gt_cls, gt_mask

    def preprocess_predictions(self,
                               pred_dist_raw: torch.Tensor, pred_scores_raw: torch.Tensor,
                               anchor_points: torch.Tensor, bs: int) -> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        params:
            pred_dist_raw: layer_cnt*[(b, K, 4*bin_count)]
            pred_scores_raw: layer_cnt*[(b, K, cls_cnt)]
            anchor_points: (K, 2)
        returns:
            pred_dist: (b, K, 4*bin_cnt)
            pred_bboxes: (b, K, 4) # XYXY in grid_coordinates
            pred_scores: (b, K, cls_cnt)
        """
        pred_dist = pred_dist_raw.view(bs, -1, 4, self._bin_cnt).softmax(dim=-1).view(bs, -1, 4*self._bin_cnt)
        pred_bboxes_g = distribution2offsets(pred_dist, torch.arange(self._bin_cnt))
        pred_bboxes_g = offset2xyxy(pred_bboxes_g, anchor_points)
        pred_scores = pred_scores_raw.sigmoid()
        return pred_dist, pred_bboxes_g, pred_scores

    # todo: Finish testing all of the internal functionality
    # todo: test - Try Forward pass and compare with Ultralytics so it works approximately the same
    #  (values don't have to be necessarly the same)
    # todo: Check Ultralytics if the scaling is same except for scale up
    # todo: Add scaling factor - if TAL doesn't find enough quality bboxes, the error isn't too small
    def forward(self, predictions: List[torch.Tensor], targets: Dict[str, torch.Tensor])-> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        params:
            predictions: layer_cnt*[(b, H_l, W_l, 4*bin_count + cls_cnt)]
            targets: Dict{img_idx: (n), gt_bboxes: (n, 4),  gt_labels: (n)}
        returns:
            loss: (1) batch scaled sum of cls_loss, box_loss, dfl_loss
                      losses are normalized so the magnitude coming from each individual image i
                      s approximately the smae
        """
        cIoU_loss, cls_loss, dfl_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        bs = len(predictions[0])
        grid_sizes, img_size = self.get_grid_sizes_and_image_size(predictions)


        gt_bboxes_img, gt_cls, gt_mask = self.preprocess_targets(targets, bs, img_size)

        anchor_points_g, strides_mask = create_anchor_points_and_stride_tensors(grid_sizes, self._strides, offset=0.5)

        pred_dist_raw, pred_scores_raw = self.get_linearized_split_predictions(predictions)
        pred_dist, pred_bboxes_g, pred_scores = self.preprocess_predictions(pred_dist_raw,
                                                                            pred_scores_raw,
                                                                            anchor_points_g, bs)

        pred_bboxes_img = (pred_bboxes_g * strides_mask[None, :, None]).type(gt_bboxes_img.dtype)
        anchor_points_img = (anchor_points_g * strides_mask[:, None]).type(gt_bboxes_img.dtype)

        # TAL
        target_scores_onehot, target_bboxes, _, _, fg_mask = self._tal(anchor_points_img,
                                                                       gt_bboxes_img,
                                                                       gt_cls,
                                                                       pred_bboxes_img,
                                                                       pred_scores,
                                                                       gt_mask)
        # This keeps loss magnitude same across images (images with more detections don't take over)
        img_idx_weights = torch.max(target_scores_onehot.view(bs, -1).sum(-1), torch.ones(bs,
                                                                                         dtype=target_scores_onehot.dtype,
                                                                                         device=target_scores_onehot.device))

        cls_loss_unscaled = self._classification_loss(pred_scores_raw,
                                                      target_scores_onehot).view(bs, -1).sum(-1)
        # TODO: test if this isn't just contraproductive since this loss isn't based upon number of detections
        #       but rather just the grid size
        # cls_loss = (cls_loss_unscaled / img_idx_weights).sum()
        cls_loss = cls_loss_unscaled.sum()
        # get bbox_losses
        if fg_mask.sum():
            target_bboxes /= strides_mask.unsqueeze(-1)
            # extract positives
            positive_examples = self.extract_positive(pred_dist, pred_bboxes_g, target_bboxes,
                                                      target_scores_onehot, anchor_points_g, fg_mask)
            pred_dist, pred_bboxes_g, target_bboxes_g, anchor_points_g, target_scores_weights = positive_examples
            def create_img_weight_tensor(fg_mask: torch.Tensor, img_idx_weights: torch.Tensor) -> torch.Tensor:
                positive_per_image = fg_mask.sum(-1)
                weight_tensor = torch.repeat_interleave(img_idx_weights, positive_per_image)
                return weight_tensor

            weight_tensor = create_img_weight_tensor(fg_mask, img_idx_weights)
            target_scores_weights /= weight_tensor
            cIoU_loss, dfl_loss = self._bbox_loss(pred_dist.view(-1, 4*self._bin_cnt),
                                                  pred_bboxes_g.view(-1, 4),
                                                  target_bboxes_g.view(-1, 4),
                                                  anchor_points_g,
                                                  target_scores_weights)

        # Scale_losses -> Perhaps lambda for scaling upwards
        cls_loss *= self._cls_scale
        cIoU_loss *= self._bbox_scale
        dfl_loss *= self._dfl_scale

        return cls_loss + cIoU_loss + dfl_loss, cls_loss.detach(), cIoU_loss.detach(), dfl_loss.detach()
