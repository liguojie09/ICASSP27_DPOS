"""Dual Polarity-Orbit Stem (DPOS) for native 8-bit grayscale images."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DPOSStem(nn.Module):
    """Combine signed canonical and local full-wave convolution responses.

    The original output-channel budget is split 3:1 between the two branches.
    Both convolutions are bias-free. Normalization, activation, and the rest of
    the backbone belong after this module and are not modified here.

    Args:
        out_channels: Total output channels; must be a positive multiple of four.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        padding_mode: Padding mode supported by ``torch.nn.Conv2d``.
    """

    def __init__(
        self,
        out_channels: int = 32,
        kernel_size: int | tuple[int, int] = 7,
        stride: int | tuple[int, int] = 2,
        padding: int | tuple[int, int] | str = 3,
        *,
        dilation: int | tuple[int, int] = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        if out_channels < 4 or out_channels % 4:
            raise ValueError("out_channels must be a positive multiple of four.")
        signed_channels = 3 * out_channels // 4
        options = dict(
            in_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=False,
        )
        self.signed = nn.Conv2d(out_channels=signed_channels, **options)
        self.magnitude = nn.Conv2d(
            out_channels=out_channels - signed_channels, **options
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return stem features for floating-point input ``[N, 1, H, W]``.

        Inputs represent native 8-bit pixels divided by 255, with values in
        [0, 1]. Integer centering and the sign reduction use float32, matching
        the paper's native-pixel construction at 224 x 224 resolution.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected grayscale input with shape [N, 1, H, W].")
        if not x.is_floating_point():
            raise TypeError("Convert native pixels to floating point and divide by 255.")

        # Recover the native pixel grid before centering to preserve the orbit.
        q = torch.round(255.0 * x.float())
        r = 2.0 * q - 255.0
        u = r / 255.0

        # The first centered pixel is nonzero and resolves an exactly zero sum.
        score = r.sum(dim=(-2, -1), keepdim=True)
        sigma = torch.where(score == 0, r[..., :1, :1].sign(), score.sign())
        canonical = sigma * u

        signed = self.signed(canonical.to(dtype=self.signed.weight.dtype))
        magnitude = self.magnitude(u.to(dtype=self.magnitude.weight.dtype)).abs()
        return torch.cat((signed, magnitude), dim=1)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d) -> DPOSStem:
        """Copy an existing bias-free grayscale or RGB input convolution.

        RGB weights are summed along the input-channel axis. The first three
        quarters of output filters initialize the signed branch; the remainder
        initialize the magnitude branch, retaining their original order.
        This method uses only the supplied in-memory weights; it loads no files.
        """
        if conv.in_channels not in (1, 3) or conv.groups != 1:
            raise ValueError("Expected an ungrouped grayscale or RGB convolution.")
        if conv.bias is not None:
            raise ValueError("The source convolution must be bias-free.")

        stem = cls(
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            padding_mode=conv.padding_mode,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        split = stem.signed.out_channels
        with torch.no_grad():
            grayscale = conv.weight.sum(dim=1, keepdim=True)
            stem.signed.weight.copy_(grayscale[:split])
            stem.magnitude.weight.copy_(grayscale[split:])
        stem.requires_grad_(conv.weight.requires_grad)
        return stem.train(conv.training)
