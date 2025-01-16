import torch

from YoloV12.modules.v12_0_head import DetectionHeadV12_0


def test_detection_head_v12_0():
    num_classes, dist_bin_cnt, ch = 5, 10, (32, 64, 128)
    head = DetectionHeadV12_0(num_classes, dist_bin_cnt, ch)
    x = [torch.rand(1, 32, 64, 64), torch.rand(1, 64, 32, 32), torch.rand(1, 128, 16, 16)]
    y = head(x)
    assert len(y) == 3
    assert y[0].shape == (1, 64, 64, num_classes + dist_bin_cnt * 4)
    assert y[1].shape == (1, 32, 32, num_classes + dist_bin_cnt * 4)
    assert y[2].shape == (1, 16, 16, num_classes + dist_bin_cnt * 4)
