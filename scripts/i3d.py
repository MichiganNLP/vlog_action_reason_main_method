from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_3_t, _size_6_t
from torch.nn.modules.utils import _triple


def compute_same_padding_size(kernel_size: int, stride: int, size: int) -> int:
    return max(kernel_size - (stride - -size % stride), 0)


def compute_same_padding(x: torch.Tensor, kernel_size: _size_3_t, stride: _size_3_t) -> torch.Tensor:
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)

    t, h, w = x.shape[2:]

    pad_t = compute_same_padding_size(kernel_size[0], stride[0], t)
    pad_h = compute_same_padding_size(kernel_size[1], stride[1], h)
    pad_w = compute_same_padding_size(kernel_size[2], stride[2], w)

    pad_t_f = pad_t // 2
    pad_t_b = pad_t - pad_t_f
    pad_h_f = pad_h // 2
    pad_h_b = pad_h - pad_h_f
    pad_w_f = pad_w // 2
    pad_w_b = pad_w - pad_w_f

    return F.pad(x, (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b))


class MaxPool3dSamePadding(nn.MaxPool3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(compute_same_padding(x, self.kernel_size, self.stride))


class Unit3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t = 1, stride: _size_3_t = 1,
                 activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.relu, use_batch_norm: bool = True,
                 bias: bool = False, eps: float = 1e-3, bn_momentum: float = 0.01) -> None:
        super().__init__()

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.activation_fn = activation_fn

        # We always want padding to be 0 here (the default).
        # We will dynamically pad based on input size in forward function.
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,  # noqa
                                stride=stride, bias=bias)  # noqa
        # (noqa needed - see https://youtrack.jetbrains.com/issue/PY-48669)

        self.bn = nn.BatchNorm3d(out_channels, eps=eps, momentum=bn_momentum) if use_batch_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = compute_same_padding(x, self.kernel_size, self.stride)
        x = self.conv3d(x)
        if self.bn:
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: _size_6_t, eps: float = 1e-3) -> None:
        super().__init__()
        self.b0 = Unit3D(in_channels, out_channels[0], kernel_size=1, eps=eps)
        self.b1a = Unit3D(in_channels, out_channels[1], kernel_size=1, eps=eps)
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_size=3, eps=eps)
        self.b2a = Unit3D(in_channels, out_channels[3], kernel_size=1, eps=eps)
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_size=3, eps=eps)
        self.b3a = MaxPool3dSamePadding(kernel_size=3, stride=1)
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_size=1, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class I3D(nn.Module):
    """Inception-v1 I3D architecture.

    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/abs/1705.07750.

    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        https://arxiv.org/abs/1409.4842.
    """

    def __init__(self, num_classes: int = 400, spatial_squeeze: bool = True, in_channels: int = 3,
                 dropout_keep_prob: float = 0.5, eps: float = 1e-3) -> None:
        super().__init__()
        self.spatial_squeeze = spatial_squeeze

        self.Conv3d_1a_7x7 = Unit3D(in_channels, 64, kernel_size=7, stride=2, eps=eps)
        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Conv3d_2b_1x1 = Unit3D(64, 64, kernel_size=1, eps=eps)
        self.Conv3d_2c_3x3 = Unit3D(64, 192, kernel_size=3, eps=eps)
        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Mixed_3b = InceptionModule(192, (64, 96, 128, 16, 32, 32), eps=eps)
        self.Mixed_3c = InceptionModule(256, (128, 128, 192, 32, 96, 64), eps=eps)
        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=3, stride=2)
        self.Mixed_4b = InceptionModule(128 + 192 + 96 + 64, (192, 96, 208, 16, 48, 64), eps=eps)
        self.Mixed_4c = InceptionModule(192 + 208 + 48 + 64, (160, 112, 224, 24, 64, 64), eps=eps)
        self.Mixed_4d = InceptionModule(160 + 224 + 64 + 64, (128, 128, 256, 24, 64, 64), eps=eps)
        self.Mixed_4e = InceptionModule(128 + 256 + 64 + 64, (112, 144, 288, 32, 64, 64), eps=eps)
        self.Mixed_4f = InceptionModule(112 + 288 + 64 + 64, (256, 160, 320, 32, 128, 128), eps=eps)
        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=2, stride=2)
        self.Mixed_5b = InceptionModule(256 + 320 + 128 + 128, (256, 160, 320, 32, 128, 128), eps=eps)
        self.Mixed_5c = InceptionModule(256 + 320 + 128 + 128, (384, 192, 384, 48, 128, 128), eps=eps)
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(384 + 384 + 128 + 128, num_classes, kernel_size=1, activation_fn=None,
                             use_batch_norm=False, bias=True, eps=eps)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:  # noqa
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.avg_pool(x)

        if return_logits:
            x = self.logits(self.dropout(x))

        # It's `(batch, time, classes)` which is what we want to work with.
        return x.squeeze(-1).squeeze(-1) if self.spatial_squeeze else x
