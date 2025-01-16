import torch
from torch import nn

from YoloV12.modules.general_modules import Conv, DWConv, DWSeparableConv


def test_conv():
    x = torch.randn(1, 3, 224, 224)
    conv = Conv(3, 16, 3, 1, 1, activation_fn=nn.ReLU())
    y = conv(x)
    assert torch.all(y>=0)
    assert y.shape == (1, 16, 224, 224)


def test_dw_conv():
    x = torch.randn(1, 3, 224, 224)
    dw_conv = DWConv(3, 6, 3, 1, 1)
    y = dw_conv(x)
    assert dw_conv.conv.conv.groups == 3
    assert y.shape == (1, 6, 224, 224)


def test_dw_separable_conv():
    x = torch.randn(1, 3, 224, 224)
    dw_sep_conv = DWSeparableConv(3, 6, 3, 1, 1)
    y = dw_sep_conv(x)
    assert y.shape == (1, 6, 224, 224)
