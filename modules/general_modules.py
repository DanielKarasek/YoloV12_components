import math
from typing import Annotated, Optional, Sequence, Tuple, TypeVar, Union

import attrs
from torch import nn


class Shape:
    shape: Sequence[int]

    def __init__(self, *shape):
        self.shape = shape

T = TypeVar("T")

# Create a “meta” alias for a list of T:
NN2DParam = Union[T, Annotated[Sequence[T], Shape(2, )]]

# TODO: maybe add defaults for K and Stride
class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size: NN2DParam[int], stride: NN2DParam[int],
                 padding: Optional[NN2DParam[int]]=None, groups: int = 1, bias: bool=False,
                 act_fn: nn.Module=nn.SiLU(inplace=True)):
        super(Conv, self).__init__()
        if padding is None:
            padding = keep_dims_pad(k_size)
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_fn

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act else x

class DWConv(nn.Module):
    # todo: annotate the arguments
    # todo: activation_fn interface
    def __init__(self, in_ch: int, out_ch: int, k_size: NN2DParam[int], stride: NN2DParam[int],
                 padding: Optional[NN2DParam[int]]=None, bias: bool=False, act_fn: nn.Module=nn.SiLU(inplace=True)):
        super(DWConv, self).__init__()
        group_cnt = math.gcd(in_ch, out_ch)
        self.conv = Conv(in_ch, out_ch, k_size, stride, padding, groups=group_cnt, bias=bias, act_fn=act_fn)


    def forward(self, x):
        return self.conv(x)


class DWSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size: NN2DParam[int], stride: NN2DParam[int],
                 padding: Optional[NN2DParam[int]]=None, bias: bool=False, act_fn: nn.Module=nn.SiLU(inplace=True)):
        super(DWSeparableConv, self).__init__()
        dw_conv = DWConv(in_ch, in_ch, k_size, stride, padding, bias=False, act_fn=act_fn)
        pointwise_conv = Conv(in_ch, out_ch, 1, 1, 0, bias=bias, act_fn=act_fn)

        self.conv = nn.Sequential(dw_conv, pointwise_conv)

    def forward(self, x):
        return self.conv(x)


def keep_dims_pad(k_size: NN2DParam):
    return k_size // 2 if isinstance(k_size, int) else [x // 2 for x in k_size]


if __name__ == "__main__":
    ...