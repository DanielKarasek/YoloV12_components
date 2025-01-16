from typing import Sequence

import torch
from functorch import einops
from torch import nn

from YoloV12.modules.general_modules import Conv, DWConv


class DetectionHeadV12_0(nn.Module):

    def __init__(self,
                 num_classes: int,
                 distribution_bin_cnt: int,
                 ch: Sequence[int] = ()):
        super(DetectionHeadV12_0, self).__init__()
        self._num_classes = num_classes
        self._distribution_bin_cnt = distribution_bin_cnt
        distrib_ch = max((16, ch[0] // 4, self._distribution_bin_cnt * 4))
        nc_ch = max(ch[0], self._num_classes)
        dist_branch = (nn.Sequential(Conv(x, distrib_ch, 3, 1),
                                     Conv(distrib_ch, distrib_ch, 3, 1),
                                     nn.Conv2d(distrib_ch, 4 * self._distribution_bin_cnt, 1)) for x in ch)
        self.dist_branch = nn.ModuleList(dist_branch)
        cls_branch = (nn.Sequential(DWConv(x, nc_ch, 3, 1),
                                    DWConv(nc_ch, nc_ch, 3, 1),
                                    nn.Conv2d(nc_ch, self._num_classes, 1)) for x in ch)

        self.cls_branch = nn.ModuleList(cls_branch)

    #
    def forward(self, x: Sequence[torch.Tensor]):
        results = []
        for i in range(len(x)):
            results.append(torch.cat([self.dist_branch[i](x[i]), self.cls_branch[i](x[i])], dim=1))
        for i in range(len(results)):
            results[i] = einops.rearrange(results[i], "b c h w -> b h w c")
        return results


