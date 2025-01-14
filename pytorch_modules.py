import math

from torch import nn

class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size, stride, padding, groups: int = 1, bias=False, activation_fn=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = activation_fn

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act else x

class DWConv(nn.Module):
    # todo: annotate the arguments
    # todo: activation_fn interface
    def __init__(self, in_ch: int, out_ch: int, k_size, stride, padding, bias=False, activation_fn=None):
        super(DWConv, self).__init__()
        group_cnt = math.gcd(in_ch, out_ch)
        self.conv = Conv(in_ch, out_ch, k_size, stride, padding, groups=group_cnt, bias=bias)


    def forward(self, x):
        return self.conv(x)


class DWSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size, stride, padding, bias=False, activation_fn=None):
        super(DWSeparableConv, self).__init__()
        dw_conv = DWConv(in_ch, in_ch, k_size, stride, padding, bias=False, activation_fn=activation_fn)
        pointwise_conv = Conv(in_ch, out_ch, 1, 1, 0, bias=bias, activation_fn=activation_fn)

        self.conv = nn.Sequential(dw_conv, pointwise_conv)

    def forward(self, x):
        return self.conv(x)