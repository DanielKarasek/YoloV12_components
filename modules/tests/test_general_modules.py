import torch
from torch import nn

from YoloV12.modules.general_modules import (Attention, BasicConv, CspC3Block, CspC3BlockWithPSA, DWConv,
    DWSeparableConv, GElanWithCspC3Block, GElanWithPSA, GElanWithResidualBlock, PSA, ResidualBlock)


def test_conv():
    x = torch.randn(1, 3, 224, 224)
    conv = BasicConv(3, 16, 3, 1, 1, act_fn=nn.ReLU())
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


def test_residual_block():
    x = torch.randn(1, 32, 224, 224)
    block = ResidualBlock(32, 2, 0.5, True)
    y = block(x)
    assert y.shape == (1, 32, 224, 224)


def test_csp_c3_block():
    x = torch.randn(1, 32, 224, 224)
    block = CspC3Block(32, 64, 2, True)
    y = block(x)
    assert y.shape == (1, 64, 224, 224)


def test_gelan_with_residual_block():
    x = torch.randn(1, 32, 224, 224)
    block = GElanWithResidualBlock(32, 64, 2, True)
    y = block(x)
    assert y.shape == (1, 64, 224, 224)


def test_gelan_with_csp_c3_block():
    x = torch.randn(1, 32, 224, 224)
    block = GElanWithCspC3Block(32, 64, 2, True)
    y = block(x)
    assert y.shape == (1, 64, 224, 224)


def test_attention():
    x = torch.randn(1, 32, 16, 16)
    attn = Attention(32, 1, 0.5)
    y = attn(x)
    assert y.shape == (1, 32, 16, 16)


def test_psa():
    x = torch.randn(1, 32, 16, 16)
    block = PSA(32, num_heads=4, attention_scale=0.5, shortcut=True)
    y = block(x)
    assert y.shape == (1, 32, 16, 16)

def test_csp_c3_block_with_psa():
    x = torch.randn(1, 32, 16, 16)
    block = CspC3BlockWithPSA(32, 64, 2, True)
    y = block(x)
    assert y.shape == (1, 64, 16, 16)

def test_gelan_with_psa():
    x = torch.randn(1, 32, 16, 16)
    block = GElanWithPSA(32, 64, 2, True)
    y = block(x)
    assert y.shape == (1, 64, 16, 16)

if __name__ == "__main__":
    ...
