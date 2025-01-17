import pytest
import torch

@pytest.fixture
def setup_bbox_loss_test_data():
    pred_bboxes = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]])
    pred_distribution = torch.tensor([[0.1, 0.9, 0.8, 0.6, 0.5, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8],
                                      [0.5, 0.5, 0.9, 0.6, 0.4, 0.1, 0.6, 0.5, 0.2, 0.9, 0.2, 0.8],
                                      [0.9, 0.1, 0.8, 0.9, 0.2, 0.1, 0.9, 0.2, 0.5, 0.6, 0.4, 0.8]])

    target_bboxes = torch.tensor([[-0.1, -0.2, 0.6, 0.7], [0.9, 1.2, 1.5, 3.5], [4.3, 4.7, 5.2, 5.3]])
    anchor_points = torch.tensor([[0, 0], [1, 2], [2, 2]])
    weights = torch.tensor([1, 0.5, 0.3])
    return pred_distribution, pred_bboxes, target_bboxes, anchor_points, weights

def test_dfl(setup_bbox_loss_test_data):
    from YoloV12.losses.distribution_focal_loss import DistributionFocalLoss
    from YoloV12.bbox_utilities.bbox_conversion import xyxy2offset
    dfl = DistributionFocalLoss("sum")

    pred_distribution, pred_bboxes, target_bboxes, anchor_points, weights = setup_bbox_loss_test_data

    dist_bins = pred_distribution.shape[-1] // 4
    ltrb_target_bboxes = xyxy2offset(target_bboxes, anchor_points).clamp_(0, dist_bins - 1 - 1e-6)

    loss = dfl(pred_distribution.view(-1, dist_bins), ltrb_target_bboxes.view(-1), weights)
    assert torch.isclose(loss, torch.tensor(8.4642))

def test_iou(setup_bbox_loss_test_data):
    # USES cIoU loss
    from YoloV12.losses.cIoU_loss import IoULoss
    iou = IoULoss("sum")
    _, pred_bboxes, target_bboxes, _, weights = setup_bbox_loss_test_data
    loss = iou(pred_bboxes, target_bboxes, weights)
    assert torch.isclose(loss, torch.tensor(1.6213734))

def test_bbox_loss(setup_bbox_loss_test_data):
    from YoloV12.losses.bbox_loss import BboxLoss
    bbox_loss = BboxLoss("sum")
    pred_distribution, pred_bboxes, target_bboxes, anchor_points, target_score_weights = setup_bbox_loss_test_data
    IoU_loss, dfl_loss  = bbox_loss(pred_distribution, pred_bboxes, target_bboxes, anchor_points, target_score_weights)
    assert torch.isclose(IoU_loss, torch.tensor(1.6213734)) and torch.isclose(dfl_loss, torch.tensor(8.4642))

