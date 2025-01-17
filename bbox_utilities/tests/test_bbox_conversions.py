import pytest
import torch


@pytest.fixture
def xyxy_bboxes():
    return torch.tensor([[2.,2.,4.,4.],
                         [3.,3.,8.,8.],
                         [1.,1.,13.,10.]])

@pytest.fixture
def basic_anchor_points():
    return torch.tensor([[3,3], [7,7], [9,9]])


# TODO: expand these
def test_xyxy_offset(xyxy_bboxes, basic_anchor_points):

    from YoloV12.bbox_utilities.bbox_conversion import xyxy2offset
    from YoloV12.bbox_utilities.bbox_conversion import offset2xyxy

    offsets = xyxy2offset(xyxy_bboxes, basic_anchor_points)
    xyxy_bboxes_2 = offset2xyxy(offsets, basic_anchor_points)
    assert torch.allclose(xyxy_bboxes, xyxy_bboxes_2)

def test_xyxy_xywh_conversions(xyxy_bboxes):

    from YoloV12.bbox_utilities.bbox_conversion import xyxy2xywh
    from YoloV12.bbox_utilities.bbox_conversion import xywh2xyxy

    xywh_bboxes = xyxy2xywh(xyxy_bboxes)
    xyxy_bboxes_2 = xywh2xyxy(xywh_bboxes)
    assert torch.allclose(xyxy_bboxes, xyxy_bboxes_2)

def test_xyxy_cxcy_wh_conversions(xyxy_bboxes):
    from YoloV12.bbox_utilities.bbox_conversion import xyxy2cxcywh
    from YoloV12.bbox_utilities.bbox_conversion import cxcywh2xyxy

    cxcywh_bboxes = xyxy2cxcywh(xyxy_bboxes)
    xyxy_bboxes_2 = cxcywh2xyxy(cxcywh_bboxes)
    assert torch.allclose(xyxy_bboxes, xyxy_bboxes_2)

if __name__ == "__main__":
    ...