from copy import deepcopy

from torch import nn

from ultralytics.utils.ops import xywh2xyxy
from YoloV12.losses.detection_loss import DetectionLoss

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
            "batch_size": 3,
            "bin_count": 2,
            "cls_cnt": 1,
            "strides": [1, 3],
            "W_list": [3, 1],
            "H_list": [3, 1],
            "grid_sizes": torch.tensor([[3, 3], [1, 1]]),
            "img_size": (3, 3),
            "iou_weight": 1,
            "dfl_weight": 2,
            "cls_weight": 3
            }
    elif param == "medium":
        return {
            "name": "medium",
            "batch_size": 3,
            "bin_count": 5,
            "cls_cnt": 3,
            "strides": [1, 2, 4],
            "W_list": [16, 8, 4],
            "H_list": [24, 12, 6],
            "grid_sizes": torch.tensor([[16, 24], [8, 12], [4, 6]]),
            "img_size": (16, 24),
            "iou_weight": 1,
            "dfl_weight": 2,
            "cls_weight": 3
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
    targets = {
        "img_idx": torch.cat([targets[i]["img_idx"] for i in range(cfg["batch_size"])]),
        "gt_bboxes": torch.cat([targets[i]["gt_bboxes"] for i in range(cfg["batch_size"])]),
        "gt_labels": torch.cat([targets[i]["gt_labels"] for i in range(cfg["batch_size"])])
        }
    return targets


@pytest.fixture
def predictions(detection_config, request):
    """Create predictions based on the provided config."""
    manual_seed = getattr(request, "manual_seed", 32)
    pred_distribution = getattr(request, "pred_distribution", "zeros")
    bin_cnt = detection_config["bin_count"]
    cls_cnt = detection_config["cls_cnt"]
    w_list = detection_config["W_list"]
    h_list = detection_config["H_list"]
    batch_size = detection_config["batch_size"]

    predictions = []
    for i in range(len(w_list)):
        w_i, h_i = w_list[i], h_list[i]
        pred_shape = (batch_size, w_i, h_i, 4 * bin_cnt + cls_cnt)
        if pred_distribution == "zeros":
            predictions.append(torch.zeros(pred_shape))
        elif pred_distribution == "normal_random":
            predictions.append(torch.rand(pred_shape))
        else:
            raise ValueError(f"Unknown pred_distribution: {pred_distribution}")
    return predictions


@pytest.fixture
def grid_sizes_and_image_size(detection_config):
    cfg = detection_config
    grid_sizes = cfg["grid_sizes"].clone()
    img_size = torch.tensor(cfg["img_size"])
    return grid_sizes, img_size


@pytest.fixture
def anchors_and_strides(grid_sizes_and_image_size, detection_config):
    grid_sizes, _ = grid_sizes_and_image_size
    anchors_points, strides = create_anchor_points_and_stride_tensors(grid_sizes,
                                                                      torch.tensor(detection_config["strides"]),
                                                                      0.5)
    return anchors_points, strides


@pytest.fixture
def detection_loss_obj(detection_config):
    """
    Returns a DetectionLoss instance based on the 'small' configuration.
    Swap to detection_config_medium for bigger tests or parametrize if needed.
    """

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
def targets_processed():
    gt_bboxes = torch.tensor([[[0.0000, 0.0000, 1.6000, 7.2000],
                               [0.8000, 1.2000, 7.2000, 15.6000],
                               [5.6000, 10.2000, 10.4000, 16.2000]],

                              [[-0.8000, 0.0000, 4.0000, 9.6000],
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


# TODO: This tests should be in some other place
@pytest.fixture
def anchor_points_and_strides_raw():
    grid_sizes = torch.tensor([[2, 2], [2, 2], [1, 1]])
    strides = torch.tensor([1, 2, 4])
    offset = 0.5
    anchor_points = torch.tensor([[0.5000, 0.5000], [1.5000, 0.5000], [0.5000, 1.5000], [1.5000, 1.5000],
                                  [0.5000, 0.5000], [1.5000, 0.5000], [0.5000, 1.5000], [1.5000, 1.5000],
                                  [0.5000, 0.5000]])
    stride_tensor = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 4])
    return grid_sizes, strides, offset, anchor_points, stride_tensor


@pytest.fixture
def small_slightly_altered_data(predictions, targets, detection_loss_obj):
    new_distribution = torch.tensor([1, -1] * 4)
    predictions[0][0, 0, 0, 0:8] = new_distribution
    predictions[0][0, 0, 0, 8] = 1
    return predictions, targets, detection_loss_obj


@pytest.fixture
def small_slightly_altered_data_with_linearized_predictions(small_slightly_altered_data, detection_config):
    cfg = detection_config
    predictions, targets, detection_loss_obj = small_slightly_altered_data
    predictions_linearized = torch.cat([layer_predictions.view(cfg["batch_size"], -1,
                                                               cfg["bin_count"] * 4 + cfg["cls_cnt"])
                                        for layer_predictions in predictions], dim=1)
    pred_dist_raw, pred_scores_raw = predictions_linearized.split([cfg["bin_count"] * 4, cfg["cls_cnt"]],
                                                                  dim=-1)

    return pred_dist_raw, pred_scores_raw, targets, detection_loss_obj


@pytest.mark.parametrize("detection_config", ["small", "medium"], indirect=True)
@pytest.mark.parametrize("predictions", ["zeros"], indirect=True)
def test_get_grid_sizes_and_image_size(predictions, targets, detection_loss_obj, grid_sizes_and_image_size):
    grid_sizes, img_size = detection_loss_obj.get_grid_sizes_and_image_size(predictions)
    gt_grid_sizes, gt_img_size = grid_sizes_and_image_size
    assert grid_sizes.shape == gt_grid_sizes.shape and "There should be w,h for each of the 3 layers"
    assert img_size.shape == gt_img_size.shape and "There should be w,h for the original image size"
    assert ((grid_sizes == gt_grid_sizes).all())
    assert ((img_size == gt_img_size).all())


@pytest.mark.parametrize("detection_config", ["medium"], indirect=True)
@pytest.mark.parametrize("predictions", ["zeros"], indirect=True)
def test_target_preprocess(predictions, targets, detection_loss_obj, grid_sizes_and_image_size, targets_processed):
    bs = len(predictions[0])
    grid_sizes, img_size = grid_sizes_and_image_size
    bboxes, cls, mask = detection_loss_obj.preprocess_targets(targets, bs, img_size)
    gt_bboxes, gt_cls, gt_mask = targets_processed
    assert torch.allclose(gt_bboxes, bboxes)
    assert torch.allclose(gt_cls, cls)
    assert torch.allclose(gt_mask, mask)


def test_anchor_points_and_strides(anchor_points_and_strides_raw):
    grid_sizes, strides, offset, gt_anchor_points, gt_stride_tensor = anchor_points_and_strides_raw
    anchor_points, stride_tensor = create_anchor_points_and_stride_tensors(grid_sizes, strides, offset)
    assert torch.allclose(anchor_points, gt_anchor_points)
    assert torch.allclose(stride_tensor, gt_stride_tensor)


@pytest.mark.parametrize("detection_config", ["small"], indirect=True)
@pytest.mark.parametrize("predictions", ["zeros"], indirect=True)
def test_prediction_linearization(small_slightly_altered_data):
    predictions, targets, detection_loss_obj = small_slightly_altered_data
    pred_dist_raw, pred_scores_raw = detection_loss_obj.get_linearized_split_predictions(predictions)
    assert pred_dist_raw[0, 0, :].allclose(torch.tensor([1., -1] * 4))
    assert pred_dist_raw[:, 1:, :].allclose(torch.tensor([0.] * 8))
    assert pred_scores_raw[0, 0, 0].allclose(torch.tensor([1.]))
    assert pred_scores_raw[:, 1:, 0].allclose(torch.tensor([0.]))


@pytest.mark.parametrize("detection_config", ["small"], indirect=True)
@pytest.mark.parametrize("predictions", ["zeros"], indirect=True)
def test_predictions_preprocess(small_slightly_altered_data_with_linearized_predictions,
                                anchors_and_strides):
    pred_dist_raw, pred_scores_raw, targets, detection_loss = small_slightly_altered_data_with_linearized_predictions
    anchor_points_g, _ = anchors_and_strides
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

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[1:3] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        from ultralytics.utils.tal import TORCH_1_10
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, nc, bin_cnt, strides, bbox_gain, dfl_gain, cls_gain, device,
                 tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = {"box": bbox_gain, "cls": cls_gain, "dfl": dfl_gain}
        self.stride = strides  # model strides
        self.nc = nc  # number of classes
        self.no = nc + bin_cnt * 4
        self.reg_max = bin_cnt
        self.device = device

        self.use_dfl = bin_cnt > 1

        from ultralytics.utils.tal import TaskAlignedAssigner
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        from ultralytics.utils.loss import BboxLoss
        self.bbox_loss = BboxLoss(bin_cnt).to(device)
        self.proj = torch.arange(bin_cnt, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        from ultralytics.utils.tal import dist2bbox
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
            )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[1:3], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[0, 1, 0, 1]])
        gt_labels, gt_bboxes, = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            )
        # Upsacling if target_scores_sum < 1
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # Normalizing so that maximum value is 1 (forces renormalization across whole)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
                )

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["cls"]  # cls gain
        loss[2] *= self.hyp["dfl"]  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

@pytest.fixture
def medium_random_seed_32_detection_loss():
    total, cls, cIoU, dfl = torch.tensor([9459.2070, 9432.3486, 1.1073517, 25.7510])
    return total, cls, cIoU, dfl

@pytest.fixture
def v8_detection_loss(detection_config):
    return v8DetectionLoss(nc=detection_config["cls_cnt"],
                           bin_cnt=detection_config["bin_count"],
                           strides=detection_config["strides"],
                           bbox_gain=detection_config["iou_weight"],
                           dfl_gain=detection_config["dfl_weight"],
                           cls_gain=detection_config["cls_weight"],
                           device="cpu")


@pytest.mark.parametrize("detection_config", ["medium"], indirect=True)
@pytest.mark.parametrize("predictions", ["normal_random"], indirect=True)
def test_forward_integration(predictions, targets, detection_loss_obj, v8_detection_loss, medium_random_seed_32_detection_loss):
    gt_total, gt_cls, gt_cIoU, gt_dfl = medium_random_seed_32_detection_loss

    targets_v8 = deepcopy(targets)
    predictions_v8 = deepcopy(predictions)
    targets_v8["bboxes"] = targets_v8["gt_bboxes"]
    targets_v8["cls"] = targets_v8["gt_labels"]
    targets_v8["batch_idx"] = targets_v8["img_idx"]

    loss = detection_loss_obj(predictions, targets)
    v8_loss = v8_detection_loss(predictions_v8, targets_v8)
    assert loss[0].allclose(gt_total)
    assert loss[1].allclose(gt_cls)
    assert loss[2].allclose(gt_cIoU)
    assert loss[3].allclose(gt_dfl)


#
# def test_forward()

if __name__ == "__main__":
    ...
