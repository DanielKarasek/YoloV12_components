import math
from typing import Literal

import torch
import torchvision

from YoloV12.bbox_utilities.bbox_conversion import boxes_to_corners, BoxFormat


def _bbox_iou_many2many(bboxes1: torch.Tensor,
                        bboxes2: torch.Tensor,
                        box_format: BoxFormat = "cxcywh",
                        iou_type: Literal['iou', 'giou', 'diou', 'ciou'] = 'iou',
                        eps: float = 1e-7) -> torch.Tensor:
    bboxes1 = boxes_to_corners(bboxes1, box_format)
    bboxes2 = boxes_to_corners(bboxes2, box_format)
    if iou_type == "iou":
        return torchvision.ops.box_iou(bboxes1, bboxes2)
    elif iou_type == "giou":
        return torchvision.ops.generalized_box_iou(bboxes1, bboxes2)
    elif iou_type == "diou":
        return torchvision.ops.distance_box_iou(bboxes1, bboxes2, eps)
    elif iou_type == "ciou":
        return torchvision.ops.complete_box_iou(bboxes1, bboxes2, eps)


def bbox_giou_many2many(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                        box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_many2many(bboxes1, bboxes2, box_format, 'giou', eps)


def bbox_diou_many2many(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                        box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_many2many(bboxes1, bboxes2, box_format, 'diou', eps)


def bbox_iou_many2many(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                       box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_many2many(bboxes1, bboxes2, box_format, 'ciou', eps)


def bbox_ciou_many2many(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                        box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_many2many(bboxes1, bboxes2, box_format, 'ciou', eps)


def _bbox_iou_1to1(bboxes1: torch.Tensor,
                   bboxes2: torch.Tensor,
                   box_format: BoxFormat = "cxcywh",
                   iou_type: Literal['iou', 'giou', 'diou', 'ciou'] = 'iou',
                   eps: float = 1e-7) -> torch.Tensor:
    bboxes1 = boxes_to_corners(bboxes1, box_format)
    bboxes2 = boxes_to_corners(bboxes2, box_format)
    x1, y1, x2, y2 = torch.chunk(bboxes1, 4, dim=-1)
    x1_, y1_, x2_, y2_ = torch.chunk(bboxes2, 4, dim=-1)
    w1, h1, w2, h2 = x2 - x1, y2 - y1, x2_ - x1_, y2_ - y1_
    inter = ((torch.min(x2, x2_) - torch.max(x1, x1_)).clamp(0) *
             (torch.min(y2, y2_) - torch.max(y1, y1_)).clamp(0))
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if iou_type == 'iou':
        return iou
    cw = torch.max(x2, x2_) - torch.min(x1, x1_)
    ch = torch.max(y2, y2_) - torch.min(y1, y1_)
    min_rectangle_area = cw * ch + eps
    if iou_type == 'giou':
        giou = iou - (min_rectangle_area - union) / min_rectangle_area
        return giou
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((x2 + x1 - x2_ - x1_) ** 2 + (y2 + y1 - y2_ - y1_) ** 2) / 4
    if iou_type == 'diou':
        diou = iou - rho2 / c2
        return diou
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou


def bbox_giou_1to1(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                   box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_1to1(bboxes1, bboxes2, box_format, 'giou', eps)


def bbox_diou_1to1(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                   box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_1to1(bboxes1, bboxes2, box_format, 'diou', eps)


def bbox_iou_1to1(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                  box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_1to1(bboxes1, bboxes2, box_format, 'iou', eps)


def bbox_ciou_1to1(bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                   box_format: BoxFormat = "cxcywh", eps: float = 1e-7) -> torch.Tensor:
    return _bbox_iou_1to1(bboxes1, bboxes2, box_format, 'ciou', eps)
