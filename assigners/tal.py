from typing import Tuple

import torch
from torch import nn

from YoloV12.bbox_utilities.bbox_iou import bbox_ciou_1to1
from YoloV12.torch_utils import one_hot_fast


# TODO: Efficient types for the masks (avoid casting as much as possible)
# TODO: Benchmark the code (speed + memory utilization)
# TODO: check both CUDA and CPU implementations
# TODO: how to extend to multilabel classification (split up bboxes and then combine so I can assign single bbox?)
# TODO: extend so Bbox can be predicted by anchor point can predict bbox with little negative offset (how to work
#       with the distribution?)
class TaskAlignedAssignerLayer(nn.Module):
    """
    This is implementation of the Task Aligned Assigner Layer. It is based on paper https://arxiv.org/pdf/2108.07755.pdf.
    This code is optimized for GPU through usage of masks.

    General Pseudocode:
    1. Get anchor points which lie inside any GT_bounding box (use mask to denote which points are inside which bbox)
    2. Calculate alignment metrics for each anchor point and GT bounding box pair
    3. Select top k alignment metrics for each GT bounding box
    4. Filter the top k selection to ensure that each anchor point is assigned to at most one GT bounding box
    5. Create target matrices for the model based on the filtered selection
    """
    def __init__(self,
                 classification_pow: float = 1.,
                 iou_pow: float = 6.,
                 top_k: int = 15,
                 cls_cnt: int = 80):
        super(TaskAlignedAssignerLayer, self).__init__()
        self.classification_pow = classification_pow
        self.iou_pow = iou_pow
        self.top_k = top_k
        self.cls_cnt = cls_cnt

    @torch.no_grad()
    def forward(self,
                points: torch.Tensor, gt_bboxes: torch.Tensor,
                gt_labels: torch.Tensor, pred_bboxes: torch.Tensor,
                pred_scores: torch.Tensor, gt_mask: torch.Tensor) -> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Forward pass of the Task Aligned Assigner Layer
        dimension descriptions:
            K = grid_size_w * grid_size_h
            nc = number of classes
            batch_max_boxes = max number of bboxes in single image in a batch (e.g. if there are 3 gt bboxes in one image,
                                                                               and 2 in another, then batch_max_boxes = 3)
        params:
            points: (K, 2) # FLOAT
            bboxes_gt: (batch, batch_max_boxes, 4) # float XYXY
            bboxes_pred: (batch, K, 4) # float XYXY
            pred_scores: (batch, K, nc) # float [0, 1]
            gt_labels: (batch, batch_max_boxes, 1) # int (ideally long)
            gt_mask: (batch batch_max_boxes) # bool

        returns:
            target_labels_onehot: (batch, K, class_cnt), # long
            target_scores: (batch, K) # long
            target_bboxes: (batch, K, 4) # float XYXY
            assignment_mask: (batch, batch_max_boxes, K) # int8
            overlaps: (batch, batch_max_boxes, K) # float # TODO: Will I ever use this?
            fg_mask: (batch, batch_max_boxes) # int8
        """
        pos_mask = self.get_points_in_bbox_mask(points, gt_bboxes)
        pos_mask *= gt_mask.unsqueeze(-1).expand_as(pos_mask)
        alignment_metrics, overlaps = self.get_alignment_metrics(gt_bboxes, gt_labels, pred_bboxes, pred_scores,
                                                                 pos_mask)

        assignment_mask = self.get_top_k(alignment_metrics)
        assignment_mask, fg_mask, target_bbox_indices = self.filter_multi_box_assignment(assignment_mask, overlaps)

        target_labels_onehot, target_labels, target_bboxes = self.create_target_matrices(gt_labels,
                                                                                         gt_bboxes,
                                                                                         fg_mask,
                                                                                         target_bbox_indices)

        alignment_metrics *= assignment_mask
        # WARNING: Changes alignment_metrics in place
        target_scores_onehot = self.normalize_scores(overlaps, alignment_metrics, target_labels_onehot)

        return target_scores_onehot, target_bboxes, assignment_mask, overlaps, fg_mask

    def create_target_matrices(self,
                               gt_labels: torch.Tensor,
                               gt_bboxes: torch.Tensor,
                               fg_mask: torch.Tensor,
                               target_bbox_indices: torch.tensor)-> (torch.Tensor, torch.Tensor):
        """
        Create the target matrices
        params:
            gt_labels: (batch, batch_max_boxes, 1) # int (ideally long)
            gt_bboxes: (batch, batch_max_boxes, 4) # float XYXY
            fg_mask: (batch, batch_max_boxes) # int8
            target_bbox_indices: (batch, K) # long

        returns:
            one_hot_target_labels: (batch, K, class_cnt), # int8
            target_labels: (batch, K) # long
            target_bboxes: (batch, K, 4) # float XYXY
        """
        b, K = target_bbox_indices.shape
        gt_labels = gt_labels.squeeze().long()
        target_labels = gt_labels.gather(1, target_bbox_indices)
        target_bboxes = gt_bboxes.gather(1, target_bbox_indices.unsqueeze(-1).expand(-1, -1, 4))
        target_labels_one_hot = one_hot_fast((b, K, self.cls_cnt),
                                             target_labels,
                                             2,
                                             dtype=gt_bboxes.dtype,
                                             device=target_labels.device, )
        fg_mask_expanded = fg_mask.unsqueeze(2).expand(-1, -1, self.cls_cnt)
        target_labels_one_hot.mul_(fg_mask_expanded)
        return target_labels_one_hot, target_labels, target_bboxes


    def get_alignment_metrics(self,
                              gt_bboxes: torch.Tensor,
                              gt_labels: torch.Tensor, pred_bboxes: torch.Tensor,
                              pred_scores: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Get the alignment metrics for each anchor point and GT bounding box pair
        params:
            bboxes_gt: (batch, batch_max_boxes, 4) # Float XYXY
            bboxes_pred: (batch, K, 4) # Float XYXY
            label_gt: (batch, batch_max_boxes, 1) # int (long ideally)
            scores_pred: (batch, K, nc) # Float [0, 1]
            mask: (batch, batch_max_boxes, K) # Float

        returns:
            alignment_metrics: (batch, batch_max_boxes, K), # Float
            overlaps: (batch, batch_max_boxes, K) # Float

        """
        batch_size, batch_max_boxes, k = mask.shape
        pred_scores_idx = gt_labels.long().expand(-1, -1, k) # batch, batch_max_boxes, K, 1
        mask = mask.bool()
        pred_scores = pred_scores.unsqueeze(1).expand(-1, batch_max_boxes, -1, -1) # batch, batch_max_boxes, k, nc
        target_scores = torch.gather(pred_scores, 3, pred_scores_idx[..., None]).squeeze()
        target_scores *= mask
        target_scores = target_scores.float()

        gt_bboxes = gt_bboxes.unsqueeze(2).expand(-1, -1, k, -1)[mask]
        pred_bboxes = pred_bboxes.unsqueeze(1).expand(-1, batch_max_boxes, -1, -1)[mask]

        overlaps = torch.zeros((batch_size, batch_max_boxes, k), device=gt_bboxes.device)
        # Only calculate IoU for the masked boxes?

        overlaps[mask] = bbox_ciou_1to1(gt_bboxes, pred_bboxes, box_format="xyxy").squeeze(-1).clamp_(0)
        overlaps *= mask.float()
        alignment_metrics = target_scores ** self.classification_pow * overlaps ** self.iou_pow
        return alignment_metrics, overlaps

    def get_top_k(self, alignment_metrics: torch.Tensor):
        """
        Parameters:
            alignment_metrics: (batch, batch_max_boxes, K) # Float
        returns:
            assignment_mask: (batch, batch_max_boxes, K) # int8
        """
        top_k_metrics, top_k_indices = torch.topk(alignment_metrics, self.top_k, dim=-1, largest=True)
        top_k_mask = (top_k_metrics > 1e-9)
        top_k_indices.masked_fill_(~top_k_mask, 0)


        assignment_mask = torch.zeros_like(alignment_metrics, dtype=torch.int8)
        top_k_mask = top_k_mask.to(assignment_mask.dtype)
        for k in torch.arange(0, self.top_k):
            assignment_mask.scatter_add_(-1, top_k_indices[:, :, k:k+1], top_k_mask[:, :, k:k+1])
        assignment_mask.masked_fill_(assignment_mask>1, 0)
        return assignment_mask.to(torch.int8)


    @staticmethod
    def get_points_in_bbox_mask(points: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Check if the points are inside the bbox
        params:
            points: (K, 2) # FLOAT
            bboxes: (batch, max_boxes, 4) # FLOAT

        returns:
            mask_positive_samples: (batch, max_boxes, K) # FLOAT
        """
        chunks = bboxes.chunk(2, dim=-1)
        min_point, max_point = chunks[0].unsqueeze(2), chunks[1].unsqueeze(2)
        point_diff = torch.cat((min_point-points[None, None], points[None, None]-max_point),dim=-1)
        return point_diff.amax(-1).lt_(1e-6)

    @staticmethod
    def normalize_scores(masked_overlaps: torch.Tensor,
                         masked_alignment_metrics: torch.Tensor,
                         target_scores: torch.Tensor) -> torch.Tensor:
        """
        Per instance normalization of the target scores to max IoU among all the anchor points for given instance

        params:
            masked_overlaps: (batch, batch_max_boxes, K) # Float
            masked_alignment_metrics: (batch, batch_max_boxes, K) # Float
            target_scores: (batch, K, cls_num) # Float

        returns:
            normalized_target_scores: (batch, K, cls_num) # Float
        """
        masked_alignment_metrics /= masked_alignment_metrics.amax(dim=-1, keepdim=True) + 1e-9
        masked_alignment_metrics *= masked_overlaps.amax(dim=-1, keepdim=True)
        normalization_factor = masked_alignment_metrics.amax(dim=1)
        target_scores *= normalization_factor.unsqueeze(-1)
        return target_scores

    @staticmethod
    def filter_multi_box_assignment(assignment_mask: torch.Tensor, overlaps: torch.Tensor):
        """
        Filter the assignment mask to ensure that each box is assigned to at most one gt box
        assignment_mask: (batch, batch_max_boxes, K) # int8
        params:
            overlaps: (batch, batch_max_boxes, K) # Float

        returns:
            assignment_mask: (batch, batch_max_boxes, K) # int8
            fg_mask: (batch, batch_max_boxes) # int8
            target_bbox_indices: (batch, K) # long
        """
        fg_mask = assignment_mask.sum(dim=1, dtype=torch.int8)
        if torch.max(fg_mask) > 1:
            max_indices = overlaps.argmax(dim=1)

            max_mask = torch.zeros_like(assignment_mask, dtype=torch.bool)
            max_mask.scatter_(1, max_indices.unsqueeze(1), 1)
            assignment_mask *= max_mask
            fg_mask = assignment_mask.sum(dim=1, dtype=torch.int8)
        target_bbox_indices = assignment_mask.argmax(dim=1)
        return assignment_mask, fg_mask, target_bbox_indices

def setup_cls_pred_scores(pred_scores: torch.Tensor) -> torch.Tensor:
    for i in range(len(pred_scores)):
        pred_scores[i] = (torch.arange(5).float().repeat(5).view(5, 5) +
                          torch.arange(5).float().view(-1, 1) / 10.).view(25, 1).expand_as(pred_scores[i]) / 10.
    return pred_scores

def setup_true_bboxes(bboxes_pred: torch.Tensor, bboxes_gt: torch.Tensor, anchor_points: int) -> torch.Tensor:
    def get_single_idx(row: int, col: int, col_len: int) -> int:
        return row * col_len + col

    def get_inside_indices(bbox: torch.Tensor, anchor_points: int, col_len: int) -> torch.Tensor:
        return torch.tensor(
            [get_single_idx(row, col, col_len) for row in range(anchor_points) for col in range(anchor_points) if
             (bbox[0] <= row < bbox[2] and bbox[1] <= col < bbox[3])])

    def get_all_inside_indices(bboxes: torch.Tensor, anchor_points: int, col_len: int) -> torch.Tensor:
        return torch.cat([get_inside_indices(bbox, anchor_points, col_len) for bbox in bboxes]).flatten().unique()

    gt_bboxes_len = bboxes_gt[0].shape[0]
    for i in range(len(bboxes_pred)):
        all_pos_indices = get_all_inside_indices(bboxes_gt[i], anchor_points, anchor_points)
        torch.manual_seed(10)
        for j, pos_indices in enumerate(all_pos_indices):
            bboxes_pred[i, pos_indices] = bboxes_gt[i, j  % gt_bboxes_len] + torch.rand(4, device="cuda") * 0.5
    return bboxes_pred

def setup_data_tal():
    bs = 2
    max_boxes = 3
    anchor_points = 5
    # bboxes = torch.zeros([bs, max_boxes, 4])
    # bboxes[0, 0, :] = torch.tensor([0, 0, 1, 1])
    # chunks = bboxes.chunk(2, dim=-1)

    # Test points_in_bbox
    points = torch.stack(
        torch.meshgrid(torch.arange(anchor_points, device="cuda"), torch.arange(anchor_points, device="cuda")),
        dim=2).view(-1, 2)
    bboxes_gt = torch.zeros([bs, max_boxes, 4], device="cuda")
    bboxes_gt[0, 0, :] = torch.tensor([0, 0, 2, 2])
    bboxes_gt[0, 1, :] = torch.tensor([1, 1, 3, 4])
    bboxes_gt[0, 2, :] = torch.tensor([0, 0, 4, 4])
    bboxes_gt[1, 0, :] = torch.tensor([1, 2.7, 4, 4])
    bboxes_gt[1, 1, :] = torch.tensor([2, 2, 4, 3.5])
    bboxes_gt[1, 2, :] = torch.tensor([1.7, 1, 3, 2.7])
    gt_mask = torch.zeros([bs, max_boxes], device="cuda")
    gt_mask[0] = torch.tensor([1, 0, 0])
    gt_mask[1] = torch.tensor([1, 1, 1])
    # mask = points_in_bbox(points, bboxes_gt)

    # Test get_alignment_metrics
    bboxes_pred = torch.zeros([bs, anchor_points ** 2, 4], device="cuda")
    pred_scores = torch.zeros([bs, anchor_points ** 2, 5], device="cuda")


    pred_scores = setup_cls_pred_scores(pred_scores)
    bboxes_pred = setup_true_bboxes(bboxes_pred, bboxes_gt, anchor_points)

    gt_labels = torch.zeros([bs, max_boxes, 1], device="cuda")
    gt_labels[0] = torch.tensor([[0], [1], [2]])
    gt_labels[1] = torch.tensor([[0], [1], [2]])
    return points, bboxes_gt, gt_labels, bboxes_pred, pred_scores, gt_mask



def main():
    # Todo: add test_tal which asserts the values
    points, bboxes_gt, gt_labels, bboxes_pred, pred_scores, gt_mask = setup_data_tal()
    tal = TaskAlignedAssignerLayer(cls_cnt=3)
    target_scores, target_bboxes, assignment_mask, overlaps, fg_mask = tal.forward(points, bboxes_gt, gt_labels,
                                                                                   bboxes_pred, pred_scores, gt_mask)
    print(target_scores, target_bboxes, assignment_mask, overlaps, fg_mask)






if __name__ == "__main__":
    # test_dfl()
    # test_iou()
    # test_bbox_loss()
    test_data = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [1, 0, 1, 1, 1]])
    ...
