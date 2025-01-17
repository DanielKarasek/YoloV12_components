import math
from typing import Annotated, Optional, Sequence, TypeVar, Union

import einops
import torch
from torch import nn

# TODO: make blocks "inner blocks" a parameter (or something that isn't a damn fking inheritance)
# TODO: Integrate DefaultActFN into infra
# TODO: try DWSepConv in deeper layers

class Shape:
    shape: Sequence[int]

    def __init__(self, *shape):
        self.shape = shape


T = TypeVar("T")

# Create a “meta” alias for a list of T:
NN2DParam = Union[T, Annotated[Sequence[T], Shape(2, )]]
ActFn = Optional[nn.Module]


class DefaultActFn(nn.Module):
    act_fn: nn.Module = nn.SiLU(inplace=True)

    def __init__(self):
        super(DefaultActFn, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(x)


def make_divisible(x: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(x + divisor / 2) // divisor * divisor)
    new_value = round(new_value)
    return new_value


def keep_dims_pad(k_size: NN2DParam):
    return k_size // 2 if isinstance(k_size, int) else [x // 2 for x in k_size]


class BasicConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: NN2DParam[int] = 1, stride: NN2DParam[int] = 1,
                 padding: Optional[NN2DParam[int]] = None, groups: int = 1, bias: bool = False,
                 act_fn: nn.Module = DefaultActFn(), **kwargs):
        super(BasicConv, self).__init__()
        if padding is None:
            padding = keep_dims_pad(k)
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, groups=groups, bias=bias, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_fn

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act else x


class DWConv(nn.Module):
    # todo: activation_fn interface
    def __init__(self, in_ch: int, out_ch: int, k: NN2DParam[int], stride: NN2DParam[int],
                 padding: Optional[NN2DParam[int]] = None, bias: bool = False,
                 act_fn: nn.Module = DefaultActFn()):
        super(DWConv, self).__init__()
        group_cnt = math.gcd(in_ch, out_ch)
        self.conv = BasicConv(in_ch, out_ch, k, stride, padding, groups=group_cnt, bias=bias, act_fn=act_fn)

    def forward(self, x):
        return self.conv(x)


class DWSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: NN2DParam[int], stride: NN2DParam[int],
                 padding: Optional[NN2DParam[int]] = None, bias: bool = False,
                 act_fn: nn.Module = DefaultActFn()):
        super(DWSeparableConv, self).__init__()
        dw_conv = DWConv(in_ch, in_ch, k, stride, padding, bias=False, act_fn=act_fn)
        pointwise_conv = BasicConv(in_ch, out_ch, 1, 1, 0, bias=bias, act_fn=act_fn)

        self.conv = nn.Sequential(dw_conv, pointwise_conv)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels: int, num_repeats: int = 1,
                 hidden_ch_scale: float = 0.5, skip: bool = True,
                 groups: int = 1):
        super(ResidualBlock, self).__init__()

        self.num_repeats = num_repeats

        hidden_ch = make_divisible(int(hidden_ch_scale * channels), 4)
        assert hidden_ch % groups == 0, "hidden_ch must be divisible by groups"

        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            sequence = nn.Sequential(
                BasicConv(channels, hidden_ch, k=1),
                BasicConv(hidden_ch, channels, k=3, groups=groups)
                )
            self.layers.append(sequence)

        self.skip = skip

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.skip else layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attention_scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.key_dim = int(self.head_dim * attention_scale)
        self.scale = self.head_dim ** -0.5
        kw_dim = self.key_dim * num_heads * 2
        v_dim = self.head_dim * num_heads
        qkv_dim = v_dim + kw_dim
        self.qkw = BasicConv(dim, qkv_dim, 1, act_fn=None)
        self.attn = nn.Softmax(dim=-1)
        self.proj = BasicConv(dim, dim, 1, act_fn=None)
        self.pe = BasicConv(dim, dim, 3, groups=dim, act_fn=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkw(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=-2)

        q = einops.rearrange(k, "b n_heads k_dim N-> b n_heads N k_dim")
        qk_scaled = q @ k * self.scale
        attn = self.attn(qk_scaled)
        attn = einops.rearrange(attn, "b n_heads N1 one_hot_N -> b n_heads one_hot_N N1")
        # v == (B, num_heads, head_dim, N)
        x = (v @ attn).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

class PSA(nn.Module):
    def __init__(self, in_ch: int, num_heads: int, attention_scale: float = 0.5, shortcut: bool = True):
        super().__init__()
        self.attention = Attention(in_ch, num_heads, attention_scale)
        self.ffn = nn.Sequential(BasicConv(in_ch, in_ch * 2, 1), BasicConv(in_ch * 2, in_ch, 1, act_fn=None))
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x) + x if self.shortcut else self.attention(x)
        x = self.ffn(x) + x if self.shortcut else self.ffn(x)
        return x


class CspC3Block(nn.Module):
    """Basic CSP-like block architecture with 3 convolutions"""

    def __init__(self, in_ch: int, out_ch: int, n: int, shortcut: bool = True,
                 hidden_ch_scale: float = 0.5, groups: int = 1):
        super().__init__()
        hidden_ch = int(out_ch * hidden_ch_scale)
        self.cv_preprocess = BasicConv(in_ch, hidden_ch * 2, 1, 1)
        self.cv_out = BasicConv(2 * hidden_ch, out_ch, 1, 1)
        self.m = nn.Sequential(*[ResidualBlock(hidden_ch, hidden_ch, 1, shortcut, groups) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, deep = self.cv_preprocess(x).chunk(2, dim=1)
        y = [skip, self.m(deep)]
        y = torch.cat(y, dim=1)
        return self.cv_out(y)



class CspC3BlockWithPSA(nn.Module):
    """Basic CSP-like block architecture with 3 convolutions and PSA bottleneck"""

    def __init__(self, in_ch: int, out_ch: int, n: int, shortcut: bool = True,
                 hidden_ch_scale: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * hidden_ch_scale)
        self.cv_preprocess = BasicConv(in_ch, hidden_ch * 2, 1, 1)
        self.cv_out = BasicConv(2 * hidden_ch, out_ch, 1, 1)
        self.m = nn.Sequential(
            *[PSA(hidden_ch, num_heads=max(hidden_ch // 64, 1), attention_scale=0.5, shortcut=shortcut) for _ in
              range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, deep = self.cv_preprocess(x).chunk(2, dim=1)
        y = [skip, self.m(deep)]
        y = torch.cat(y, dim=1)
        return self.cv_out(y)


class GElanWithPSA(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int, shortcut: bool = True,
                 hidden_ch_scale: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * hidden_ch_scale)
        self.cv_preprocess = BasicConv(in_ch, hidden_ch * 2, 1, 1)
        self.cv_out = BasicConv((2 + n) * hidden_ch, out_ch, 1, 1)
        self.m = nn.ModuleList(
            PSA(hidden_ch, num_heads=max(hidden_ch // 64, 1), attention_scale=0.5, shortcut=shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, deep = self.cv_preprocess(x).chunk(2, dim=1)
        y = [skip, deep]
        for layer in self.m:
            y.append(layer(y[-1]))
        y = torch.cat(y, dim=1)
        return self.cv_out(y)


class GElanWithResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int, shortcut: bool = True,
                 hidden_ch_scale: float = 0.5, groups: int = 1):
        super().__init__()
        hidden_ch = int(out_ch * hidden_ch_scale)
        self.cv_preprocess = BasicConv(in_ch, hidden_ch * 2, 1, 1)
        self.cv_out = BasicConv((2 + n) * hidden_ch, out_ch, 1, 1)
        self.m = nn.ModuleList(ResidualBlock(hidden_ch, hidden_ch, 1., shortcut, groups) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, deep = self.cv_preprocess(x).chunk(2, dim=1)
        y = [skip, deep]
        for layer in self.m:
            y.append(layer(y[-1]))
        y = torch.cat(y, dim=1)
        return self.cv_out(y)


# TODO: CSPC3Block shouldn't have N==2 as default
# TODO: Double scale down - Do we want to parametrize this or make it default or what
class GElanWithCspC3Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int, shortcut: bool = True,
                 hidden_ch_scale: float = 0.5, groups: int = 1):
        super().__init__()
        hidden_ch = int(out_ch * hidden_ch_scale)
        self.cv_preprocess = BasicConv(in_ch, hidden_ch * 2, 1, 1)
        self.cv_out = BasicConv((2 + n) * hidden_ch, out_ch, 1, 1)
        self.m = nn.ModuleList(CspC3Block(hidden_ch, hidden_ch, 2, shortcut, groups=groups) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, deep = self.cv_preprocess(x).chunk(2, dim=1)
        y = [skip, deep]
        for layer in self.m:
            y.append(layer(y[-1]))
        y = torch.cat(y, dim=1)
        return self.cv_out(y)


if __name__ == "__main__":
    ...
