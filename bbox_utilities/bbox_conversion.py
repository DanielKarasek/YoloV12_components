from typing import Literal

import torch

BoxFormat = Literal ["cxcywh", "xywh", "xyxy"]


class InvalidBoxFormat(Exception):
    pass


def xyxy2xyxy(bboxes: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    if in_place:
        raise NotImplementedError("In place conversion not implemented")
    boxes_xyxy = bboxes[..., 0:4].clone()
    return boxes_xyxy


def xywh2xyxy(bboxes: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    if in_place:
        raise NotImplementedError("In place conversion not implemented")
    bboxes_shape = bboxes.shape
    box1_x1 = bboxes[..., 0:1]
    box1_y1 = bboxes[..., 1:2]
    box1_x2 = bboxes[..., 0:1] + bboxes[..., 2:3]
    box1_y2 = bboxes[..., 1:2] + bboxes[..., 3:4]
    return torch.stack([box1_x1, box1_y1, box1_x2, box1_y2], dim=-1).view(*bboxes_shape[:-1], 4)

def xyxy2xywh(bboxes: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    if in_place:
        raise NotImplementedError("In place conversion not implemented")
    bboxes_shape = bboxes.shape
    box1_w = bboxes[..., 2:3] - bboxes[..., 0:1]
    box1_h = bboxes[..., 3:4] - bboxes[..., 1:2]
    box1_x = bboxes[..., 0:1]
    box1_y = bboxes[..., 1:2]
    return torch.stack([box1_x, box1_y, box1_w, box1_h], dim=-1).view(*bboxes_shape[:-1], 4)

def cxcywh2xyxy(bboxes: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    if in_place:
        raise NotImplementedError("In place conversion not implemented")
    bboxes_shape = bboxes.shape
    box1_x1 = bboxes[..., 0:1] - bboxes[..., 2:3] / 2
    box1_y1 = bboxes[..., 1:2] - bboxes[..., 3:4] / 2
    box1_x2 = bboxes[..., 0:1] + bboxes[..., 2:3] / 2
    box1_y2 = bboxes[..., 1:2] + bboxes[..., 3:4] / 2
    return torch.stack([box1_x1, box1_y1, box1_x2, box1_y2], dim=-1).view(*bboxes_shape[:-1], 4)


def xyxy2cxcywh(bboxes: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    if in_place:
        raise NotImplementedError("In place conversion not implemented")
    bboxes_shape = bboxes.shape
    box1_w = bboxes[..., 2:3] - bboxes[..., 0:1]
    box1_h = bboxes[..., 3:4] - bboxes[..., 1:2]
    box1_x = (bboxes[..., 0:1] + bboxes[..., 2:3]) / 2
    box1_y = (bboxes[..., 1:2] + bboxes[..., 3:4]) / 2

    return torch.stack([box1_x, box1_y, box1_w, box1_h], dim=-1).view(*bboxes_shape[:-1], 4)


def offset2xyxy(bbox_offsets: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """
    Convert the distribution to xyxy format
    params:
        bbox_offsets: (..., N, 4)
        anchor_points: (N, 2)
    returns:
        xyxy_bbox: (..., N, 4)
    """
    xyxy_bbox = torch.empty_like(bbox_offsets)
    min_offset, max_offset = bbox_offsets.chunk(2, dim=-1)
    xyxy_bbox[..., :2] = anchor_points - min_offset
    xyxy_bbox[..., 2:] = anchor_points + max_offset
    return xyxy_bbox


def xyxy2offsets(xyxy_bbox: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """
    Convert the bbox to offsets
    params:
        xyxy_bbox: (..., N, 4)
        anchor_points: (N, 2)
    returns:
        bbox_offsets: (..., N, 4)
    """
    bbox_offsets = torch.empty_like(xyxy_bbox)
    bbox_offsets[..., :2] = anchor_points - xyxy_bbox[..., :2]
    bbox_offsets[..., 2:] = xyxy_bbox[..., 2:] - anchor_points
    return bbox_offsets


def offsets2distribution(offsets: torch.Tensor, bin_weights: torch.Tensor) -> torch.Tensor:
    # TODO: Each distribution should probably be just combination of 2 bins (left and right)
    ...


def distribution2offsets(distribution_bboxes: torch.Tensor,
                         distribution_bin_weights: torch.Tensor) -> torch.Tensor:
    """
    Convert the distribution bboxes to offsets
    params:
        distribution_bboxes: (..., 4*bin_count)
        distribution_bin_weights: (bin_count)
    """
    bin_count = distribution_bin_weights.shape[-1]
    distribution_bboxes = distribution_bboxes.view(*distribution_bboxes.shape[:-1], 4, bin_count)
    # bboxes => (..., 4, bin_count) distribution_bin_weights => (4, bin_count)
    distribution_bin_weights = distribution_bin_weights.unsqueeze(0).expand(4, bin_count)
    matching_shape = [1] * (distribution_bboxes.ndim - 2) + [4, bin_count]
    distribution_bin_weights = distribution_bin_weights.view(*matching_shape)
    distribution_offsets = (distribution_bboxes * distribution_bin_weights).sum(dim=-1)
    return distribution_offsets


def boxes_to_corners(bboxes: torch.Tensor,
                     box_format: BoxFormat = "cxcywh") -> (
        torch.Tensor):
    """
    Converts bounding boxes to 4 tensors each representing one corner coordinate.
     e.g. first tensor contains all x1 corners

    Bboxes in format: [Batch_size, 4], where 4 represents [x, y, w, h] (midpoint) or [x1, y1, x2, y2] (corners).
    Out format: 4 tensors of [Batch_size] size
    # doctests
    >>> bboxes = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
    >>> boxes_to_corners(bboxes, "cxcywh")
    tensor([[0.5000, 0.5000, 1.5000, 1.5000],
            [1.0000, 1.0000, 3.0000, 3.0000]])
    >>> bboxes = torch.tensor([[1, 1, 2, 2], [2, 2, 4, 4]])
    >>> boxes_to_corners(bboxes, "xyxy")
    tensor([[1, 1, 2, 2],
            [2, 2, 4, 4]])
    """
    if box_format not in ["cxcywh", "xywh", "xyxy"]:
        raise InvalidBoxFormat("Box format should be midpoint or corners")
    if box_format == "cxcywh":
        bboxes = cxcywh2xyxy(bboxes)
    elif box_format == "xywh":
        bboxes = xywh2xyxy(bboxes)

    return bboxes
