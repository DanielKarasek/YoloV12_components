from typing import Literal

import torch

BoxFormat = Literal ["cxcywh", "xywh", "xyxy", "offset"]


class InvalidBoxFormat(Exception):
    pass


def xyxy2xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    return bboxes


def xyxy2xywh(bboxes: torch.Tensor) -> torch.Tensor:
    bboxes_new = torch.empty_like(bboxes)
    bboxes_new[..., :] = bboxes
    bboxes_new[..., 2:] -= bboxes[..., :2]
    return bboxes_new

def xyxy2cxcywh(bboxes: torch.Tensor) -> torch.Tensor:
    bboxes_new = torch.empty_like(bboxes)
    bboxes_new[..., 0:2] = (bboxes[..., 0:2] + bboxes[..., 2:4]) / 2
    bboxes_new[..., 2:4] = bboxes[..., 2:4] - bboxes[..., 0:2]
    return bboxes_new

def xyxy2offset(xyxy_bbox: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """
    Anchor points (n, 2) are used as anchors from which the ltrb offsets should be calculated
    """
    bbox_offsets = torch.empty_like(xyxy_bbox)
    bbox_offsets[..., :2] = anchor_points - xyxy_bbox[..., :2]
    bbox_offsets[..., 2:] = xyxy_bbox[..., 2:] - anchor_points
    return bbox_offsets


def xywh2xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    bboxes_new = torch.empty_like(bboxes)
    bboxes_new[..., 0:2] = bboxes[..., 0:2]
    bboxes_new[..., 2:4] = bboxes[..., 0:2] + bboxes[..., 2:4]
    return bboxes_new

def xywh2cxcywh(bboxes: torch.Tensor) -> torch.Tensor:
    bboxes_new = torch.empty_like(bboxes)
    bboxes_new[..., 0:2] = bboxes[..., 0:2] - bboxes[..., 2:4] / 2
    bboxes_new[..., 2:4] = bboxes[..., 2:4]
    return bboxes_new

def xywh2offset(xywh_bbox: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """
    Anchor points (n, 2) are used as anchors from which the ltrb offsets should be calculated
    """
    xyxy = xywh2xyxy(xywh_bbox)
    bbox_offsets = xyxy2offset(xyxy, anchor_points)
    return bbox_offsets


def cxcywh2xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    new_bboxes = torch.empty_like(bboxes)
    new_bboxes[..., 0:2] = bboxes[..., 0:2] - bboxes[..., 2:4] / 2
    new_bboxes[..., 2:4] = bboxes[..., 0:2] + bboxes[..., 2:4] / 2
    return new_bboxes


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

    import doctest