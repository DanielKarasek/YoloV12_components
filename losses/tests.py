import unittest

import torch

from YoloV12.bbox_utilities.bbox_conversion import xywh2xyxy
from YoloV12.losses.detection_loss import DetectionLoss
import pytest

from YoloV12.yolo_utils import create_anchor_points_and_stride_tensors

import pytest
import torch

@pytest.fixture
def detection_config(request):
    """Config fixture that can switch between 'small' and 'medium' based on a marker or parameter."""
    param = getattr(request, "param", "small")  # Default to "small" if not specified
    if param == "small":
        return {
            "name": "small",
            "bin_count": 2,
            "cls_cnt": 1,
            "strides": [1, 3],
            "W_list": [3, 1],
            "H_list": [3, 1],
            "img_size": (3, 3)
        }
    elif param == "medium":
        return {
            "name": "medium",
            "bin_count": 5,
            "cls_cnt": 3,
            "strides": [1, 2, 4],
            "W_list": [16, 8, 4],
            "H_list": [24, 12, 6],
            "img_size": (16, 24)
        }


@pytest.fixture
def targets(detection_config):
    cfg = detection_config

    targets = {}
    targets[0] = {
        "img_idx": torch.tensor([0, 0, 0]),
        "gt_bboxes": torch.tensor([
            [0.05, 0.15, 0.10, 0.30],
            [0.25, 0.35, 0.40, 0.60],
            [0.50, 0.55, 0.30, 0.25]
            ]),
        }

    targets[1] = {
        "img_idx": torch.tensor([1]),
        "gt_bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "gt_labels": torch.tensor([1])
        }

    targets[2] = {
        "img_idx": torch.empty((0,)),
        "gt_bboxes": torch.empty((0, 4)),
        "gt_labels": torch.empty((0,), dtype=torch.long)
        }
    if cfg["name"] == "small":
        # Image 0: three bounding boxes
        targets[0]["gt_labels"] = torch.tensor([0, 1, 0])
    elif cfg["name"] == "medium":
        targets[0]["gt_labels"] = torch.tensor([0, 1, 2])
    return targets


@pytest.fixture
def predictions(config):
    """Create predictions based on the provided config."""
    bin_cnt = config["bin_count"]
    cls_cnt = config["cls_cnt"]
    W_list = config["W_list"]
    H_list = config["H_list"]
    batch_size = 2  # Example batch size

    predictions = []
    for i in range(len(W_list)):
        W_i, H_i = W_list[i], H_list[i]
        pred_shape = (batch_size, W_i, H_i, 4 * bin_cnt + cls_cnt)
        predictions.append(torch.zeros(pred_shape))
    return predictions

@pytest.fixture
def anchors_and_strides(detection_config):
    cfg = detection_config
    # TODO: todo
    # anchor_points = torch.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
    # strides = torch.tensor(cfg["strides"])
    return anchor_points, strides


@pytest.fixture
def detection_loss_obj(detection_config):
    """
    Returns a DetectionLoss instance based on the 'small' configuration.
    Swap to detection_config_medium for bigger tests or parametrize if needed.
    """
    from YoloV12.losses.detection_loss import DetectionLoss

    cfg = detection_config
    return DetectionLoss(
        bbox_bin_cnt=cfg["bin_count"],
        cls_cnt=cfg["cls_cnt"],
        strides=cfg["strides"],
        bbox_scale=cfg["iou_weight"],
        dfl_scale=cfg["dfl_weight"],
        cls_scale=cfg["cls_weight"],
    )





@pytest.fixture
def grid_sizes_and_image_size():
    grid_sizes = torch.tensor([[16, 24], [8, 12], [4, 6]])
    img_size = torch.tensor([16, 24])
    return grid_sizes, img_size

# Split into multiple fixtures
@pytest.fixture
def setup_detection_test_data():
    # ------------------------------------------------------------------------------
    # 1) Define shapes and hyperparameters
    # ------------------------------------------------------------------------------
    batch_size = 3
    layer_cnt = 3  # e.g. YOLO typically predicts at multiple scales
    bin_count = 5  # example
    cls_cnt = 3  # example: number of classes
    W_list = [16, 8, 4]  # output widths at each layer
    H_list = [24, 12, 6]  # output heights at each layer
    strides = [1, 2, 4]  # example: strides at each layer

    # ------------------------------------------------------------------------------
    # 2) Create "targets"
    #    keys = {0, 1, 2} for our 3 images
    # ------------------------------------------------------------------------------
    targets = {}

    # Image 0: three bounding boxes
    targets[0] = {
        "img_idx": torch.tensor([0, 0, 0]),  # just the image index
        "gt_bboxes": torch.tensor([
            [0.05, 0.15, 0.10, 0.30],
            [0.25, 0.35, 0.40, 0.60],
            [0.50, 0.55, 0.30, 0.25]
            ]),
        "gt_labels": torch.tensor([0, 1, 2])  # example class indices
        }

    # Image 1: one bounding box (e.g. [x1, y1, x2, y2] in normalized coords)
    targets[1] = {
        "img_idx": torch.tensor([1]),  # just the image index
        "gt_bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "gt_labels": torch.tensor([1])  # single label
        }

    # Image 2: zero bounding boxes
    targets[2] = {
        "img_idx": torch.empty((0,)),  # just the image index
        "gt_bboxes": torch.empty((0, 4)),  # shape = (0,4)
        "gt_labels": torch.empty((0,), dtype=torch.long)  # no labels
        }

    # ------------------------------------------------------------------------------
    # 3) Create "predictions"
    #    We'll create a list of tensors, one per layer:
    #    predictions[i] shape = (batch_size, W_i, H_i, 4*bin_count + cls_cnt)
    # ------------------------------------------------------------------------------

    predictions = []
    for i in range(layer_cnt):
        W_i = W_list[i]
        H_i = H_list[i]
        predictions.append(
            torch.zeros(batch_size, W_i, H_i, 4 * bin_count + cls_cnt)
            )
    # ------------------------------------------------------------------------------
    # Stack targets into single dictionary
    # ------------------------------------------------------------------------------
    targets = {
        "img_idx": torch.cat([targets[i]["img_idx"] for i in range(batch_size)]),
        "gt_bboxes": torch.cat([targets[i]["gt_bboxes"] for i in range(batch_size)]),
        "gt_labels": torch.cat([targets[i]["gt_labels"] for i in range(batch_size)])
        }

    # ------------------------------------------------------------------------------
    # Setup the DetectionLoss object
    # ------------------------------------------------------------------------------

    detection_loss = DetectionLoss(bin_count,
                                   cls_cnt,
                                   strides,
                                   1,
                                   2,
                                   3)
    return predictions, targets, detection_loss


@pytest.fixture
def targets_processed():
    gt_bboxes = torch.tensor([[[0.8000, 3.6000, 2.4000, 10.8000],
                               [4.0000, 8.4000, 10.4000, 22.8000],
                               [8.0000, 13.2000, 12.8000, 19.2000]],

                              [[1.6000, 4.8000, 6.4000, 14.4000],
                               [0.0000, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.0000, 0.0000, 0.0000]],

                              [[0.0000, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.0000, 0.0000, 0.0000]]])
    gt_cls = torch.tensor([[[0.],
                            [1.],
                            [2.]],

                           [[1.],
                            [0.],
                            [0.]],

                           [[0.],
                            [0.],
                            [0.]]])
    gt_mask = torch.tensor([[1., 1., 1.],
                            [1., 0., 0.],
                            [0., 0., 0.]])
    return gt_bboxes, gt_cls, gt_mask


def test_get_grid_sizes_and_image_size(setup_detection_test_data, grid_sizes_and_image_size):
    predictions, targets, detection_loss = setup_detection_test_data
    grid_sizes, img_size = detection_loss.get_grid_sizes_and_image_size(predictions)
    gt_grid_sizes, gt_img_size = grid_sizes_and_image_size
    assert grid_sizes.shape == (3, 2) and "There should be w,h for each of the 3 layers"
    assert img_size.shape == (2,) and "There should be w,h for the original image size"
    assert ((grid_sizes == gt_grid_sizes).all())
    assert ((img_size == gt_img_size).all())


def test_target_preprocess(setup_detection_test_data, grid_sizes_and_image_size, targets_processed):
    predictions, targets, detection_loss = setup_detection_test_data
    bs = len(predictions[0])
    grid_sizes, img_size = grid_sizes_and_image_size
    bboxes, cls, mask = detection_loss.preprocess_targets(targets, bs, img_size)
    gt_bboxes, gt_cls, gt_mask = targets_processed
    assert torch.allclose(gt_bboxes, bboxes)
    assert torch.allclose(gt_cls, cls)
    assert torch.allclose(gt_mask, mask)


@pytest.fixture
def strides_and_offset():
    strides = torch.tensor([1, 2, 4])
    offset = 0.5
    return strides, offset


# TODO: This tests should be in some other place
@pytest.fixture
def anchor_points_and_strides_setup():
    grid_sizes = torch.tensor([[2, 2], [2, 2], [1, 1]])
    strides = torch.tensor([1, 2, 4])
    offset = 0.5
    anchor_points = torch.tensor([[0.5000, 0.5000], [0.5000, 1.5000], [1.5000, 0.5000], [1.5000, 1.5000],
                                  [0.5000, 0.5000], [0.5000, 1.5000], [1.5000, 0.5000], [1.5000, 1.5000],
                                  [0.5000, 0.5000]])
    stride_tensor = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 4])
    return grid_sizes, strides, offset, anchor_points, stride_tensor


def test_anchor_points_and_strides(anchor_points_and_strides_setup):
    grid_sizes, strides, offset, gt_anchor_points, gt_stride_tensor = anchor_points_and_strides_setup
    anchor_points, stride_tensor = create_anchor_points_and_stride_tensors(grid_sizes, strides, offset)
    assert torch.allclose(anchor_points, gt_anchor_points)
    assert torch.allclose(stride_tensor, gt_stride_tensor)


@pytest.fixture
def get_data_small():
    batch_size = 2
    layer_cnt = 2  # e.g. YOLO typically predicts at multiple scales
    bin_count = 2  # example
    cls_cnt = 1  # example: number of classes
    W_list = [3, 1]  # output widths at each layer
    H_list = [3, 1]  # output heights at each layer
    strides = [1, 3]  # example: strides at each layer

    # ------------------------------------------------------------------------------
    # 2) Create "targets"
    #    keys = {0, 1, 2} for our 3 images
    # ------------------------------------------------------------------------------
    targets = {}

    # Image 0: three bounding boxes
    targets[0] = {
        "img_idx": torch.tensor([0, 0, 0]),  # just the image index
        "gt_bboxes": torch.tensor([
            [0.05, 0.15, 0.10, 0.30],
            [0.25, 0.35, 0.40, 0.60],
            [0.50, 0.55, 0.30, 0.25]
            ]),
        "gt_labels": torch.tensor([0, 1, 0])  # example class indices
        }

    # Image 1: one bounding box (e.g. [x1, y1, x2, y2] in normalized coords)
    targets[1] = {
        "img_idx": torch.tensor([1]),  # just the image index
        "gt_bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "gt_labels": torch.tensor([1])  # single label
        }

    # Image 2: zero bounding boxes
    targets[2] = {
        "img_idx": torch.empty((0,)),  # just the image index
        "gt_bboxes": torch.empty((0, 4)),  # shape = (0,4)
        "gt_labels": torch.empty((0,), dtype=torch.long)  # no labels
        }

    # ------------------------------------------------------------------------------
    # 3) Create "predictions"
    #    We'll create a list of tensors, one per layer:
    #    predictions[i] shape = (batch_size, W_i, H_i, 4*bin_count + cls_cnt)
    # ------------------------------------------------------------------------------

    predictions = []
    for i in range(layer_cnt):
        W_i = W_list[i]
        H_i = H_list[i]
        predictions.append(
            torch.zeros(batch_size, W_i, H_i, 4 * bin_count + cls_cnt)
            )
    # ------------------------------------------------------------------------------
    # Stack targets into single dictionary
    # ------------------------------------------------------------------------------
    targets = {
        "img_idx": torch.cat([targets[i]["img_idx"] for i in range(batch_size)]),
        "gt_bboxes": torch.cat([targets[i]["gt_bboxes"] for i in range(batch_size)]),
        "gt_labels": torch.cat([targets[i]["gt_labels"] for i in range(batch_size)])
        }

    # ------------------------------------------------------------------------------
    # Setup the DetectionLoss object
    # ------------------------------------------------------------------------------

    detection_loss = DetectionLoss(bin_count,
                                   cls_cnt,
                                   strides,
                                   1,
                                   2,
                                   3)
    return predictions, targets, detection_loss


@pytest.fixture
def small_slightly_altered_data(get_data_small):
    predictions, targets, detection_loss = get_data_small
    new_distribution = torch.tensor([1] * 8)
    predictions[0][0, 0, 0, 0:8] = new_distribution
    predictions[0][0, 0, 0, 8] = 1
    return predictions, targets, detection_loss


@pytest.fixture
def small_slightly_altered_data_with_linearized_predictions(get_data_small):
    predictions, targets, detection_loss = get_data_small
    new_distribution = torch.tensor([1, -1] * 4)
    predictions[0][0, 0, 0, 0:8] = new_distribution
    predictions[0][0, 0, 0, 8] = 1
    pred_dist_raw, pred_scores_raw = detection_loss.get_linearized_split_predictions(predictions)
    return pred_dist_raw, pred_scores_raw, targets, detection_loss

@pytest.fixture
def small_anchor_points():
    anchor_points, strides = create_anchor_points_and_stride_tensors(torch.tensor([[3, 3], [1, 1]]), torch.tensor([1, 3]), 0.5)
    return anchor_points, strides

def test_prediction_linearization(small_slightly_altered_data_with_linearized_predictions):
    pred_dist_raw, pred_scores_raw, targets, detection_loss = small_slightly_altered_data_with_linearized_predictions
    assert pred_dist_raw[0, 0, :].allclose(torch.tensor([1., -1] * 4))
    assert pred_dist_raw[:, 1:, :].allclose(torch.tensor([0.] * 8))
    assert pred_scores_raw[0, 0, 0].allclose(torch.tensor([1.]))
    assert pred_scores_raw[:, 1:, 0].allclose(torch.tensor([0.]))


def test_predictions_preprocess(small_slightly_altered_data_with_linearized_predictions,
                                small_anchor_points):
    pred_dist_raw, pred_scores_raw, targets, detection_loss = small_slightly_altered_data_with_linearized_predictions
    anchor_points_g, _ = small_anchor_points
    bs = pred_dist_raw.shape[0]

    pred_dist, pred_bboxes_g, pred_scores = detection_loss.preprocess_predictions(pred_dist_raw,
                                                                                  pred_scores_raw,
                                                                                  anchor_points_g, bs)

    assert pred_dist[0, 0, :].allclose(torch.tensor([0.8808, 0.1192] * 4), atol=1e-5)
    assert pred_dist[0, 1:, :].allclose(torch.tensor([0.5, 0.5] * 4))
    assert pred_bboxes_g[0, 0, :].allclose(torch.tensor([0.3808, 0.3808, 0.6192, 0.6192]), atol=1e-5)
    assert pred_bboxes_g[0, 1:, :].allclose(anchor_points_g[1:].repeat((1, 2)) + torch.tensor([-0.5, -0.5, 0.5, 0.5]))
    assert pred_scores[0, 0, 0].allclose(torch.tensor([0.7311]), atol=1e-4)
    assert pred_scores[0, 1:, 0].allclose(torch.tensor([0.5]))



if __name__ == "__main__":
    unittest.main()
