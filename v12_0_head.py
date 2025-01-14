from torch import nn


class DetectionHeadV12_0(nn.Module):
    def __init__(self, num_classes: int, distribution_bin_cnt: int):
        super(DetectionHeadV12_0, self).__init__()
        self._num_classes = num_classes
        self._distribution_bin_cnt = distribution_bin_cnt

        self.

    def forward(self, x):
