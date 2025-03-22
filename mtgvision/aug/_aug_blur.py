#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


__all__ = [
    "BlurGaussian",
    "BlurJpegCompression",
    "BlurDownscale",
]

from dataclasses import dataclass
from typing import Optional

import jax
import jax.random as jrandom
import dm_pix as pix
from jax import lax
from jax.tree_util import register_dataclass

from mtgvision.aug._base import (
    AugItems,
    Augment,
    AugPrngHint,
    JnpFloat32,
    jax_static_field,
)
from mtgvision.aug._util_args import ArgIntHint, sample_int
from mtgvision.aug._util_jax import ResizeMethod
from mtgvision.aug._util_jpeg import rgb_img_jpeg_compression_jax


# ========================================================================= #
# functions                                                                 #
# ========================================================================= #


def _rgb_downscale_upscale(
    src: JnpFloat32,
    scale: float,  # static
    method: ResizeMethod,  # static
    antialias: bool = True,  # static
) -> JnpFloat32:
    method = ResizeMethod(method).value
    # checks
    if scale >= 1:
        return src
    # downscale
    img = jax.image.resize(
        src,
        (int(src.shape[0] * scale), int(src.shape[1] * scale), src.shape[2]),
        method=method,
        antialias=antialias,
    )
    # upscale
    return jax.image.resize(
        img,
        (src.shape[0], src.shape[1], src.shape[2]),
        method=method,
        antialias=antialias,
    )


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


@register_dataclass
@dataclass(frozen=True)
class BlurGaussian(Augment):
    """
    Apply a box blur to the image.

    # ~3 sigma on each side of the kernel's center covers ~99.7% of the
    # probability mass. There is some fiddling for smaller values. Source:
    # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    kernel_size = ((sigma - 0.8) / 0.3 + 1) / 0.5 + 1

    sigma=0.5 --> kernel_size=1
    sigma=1.0 --> kernel_size=4.333
    sigma=2.0 --> kernel_size=11
    sigma=3.0 --> kernel_size=17.667
    """

    p: float = jax_static_field(default=0.5)
    kernel_size: int = jax_static_field(default=7)
    sigma: ArgIntHint = jax_static_field(default=(0, 2))  # min 0
    aug_mask: bool = jax_static_field(default=False)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask:
            sigma = sample_int(key, self.sigma)

            def true_branch(x):
                # image
                im = x.image
                if x.has_image:
                    im = pix.gaussian_blur(x.image, sigma, self.kernel_size)
                # mask
                mask = x.mask
                if x.has_mask and self.aug_mask:
                    mask = pix.gaussian_blur(x.mask, sigma, self.kernel_size)
                # done
                return x.override(image=im, mask=mask)

            return lax.cond(
                sigma > 0,
                true_branch,
                lambda x: x,
                x,
            )

        return x


# @register_dataclass
# @dataclass(frozen=True)
# class BlurMedian(Augment):
#     """
#     Apply a median blur to the image.
#     """
#     p: float = jax_static_field(default=0.5)
#
#     def __init__(
#         self,
#         radius: ArgIntHint = (0, 3),
#         aug_mask: bool = False,
#         p: float = 0.5,
#     ):
#         super().__init__(p=p)
#         self._radius = ArgIntRange.from_arg(radius, min_val=0)
#         self._aug_mask = aug_mask
#
#     def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
#         if x.has_image or x.has_mask:
#             r = self._radius.sample(key)
#             if r > 0:
#                 # image
#                 im = x.image
#                 if x.has_image:
#                     im = cv2.medianBlur(x.image, r)
#                 # mask
#                 mask = x.mask
#                 if x.has_mask and self._aug_mask:
#                     mask = cv2.medianBlur(x.mask, r)
#                 # done
#                 return x.override(image=im, mask=mask)
#         return x


@register_dataclass
@dataclass(frozen=True)
class BlurJpegCompression(Augment):
    """
    Apply jpeg compression to the image.
    """

    p: float = jax_static_field(default=0.5)
    quality: int = jax_static_field(default=(10, 100))  # min 1, max 100

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            quality = sample_int(key, self.quality)
            im = rgb_img_jpeg_compression_jax(
                src=x.image,  # don't need to copy
                quality=quality,
            )
            return x.override(image=im)
        return x


@register_dataclass
@dataclass(frozen=True)
class BlurDownscale(Augment):
    """
    Downscale the image and then upscale it back to the original size.
    """

    p: float = jax_static_field(default=0.5)
    scale_levels: tuple[float, ...] = jax_static_field(default=(1 / 8, 1 / 4, 1 / 2, 1))
    scale_p: Optional[tuple[float, ...]] = jax_static_field(default=None)
    aug_mask: bool = jax_static_field(default=False)
    inter: ResizeMethod = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: ResizeMethod = jax_static_field(default=ResizeMethod.LINEAR)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask:

            def true_branch(x, scale):
                # image
                im = x.image
                if x.has_image:
                    im = _rgb_downscale_upscale(
                        src=x.image,  # don't need to copy
                        scale=scale,
                        method=self.inter,
                    )
                # mask
                mask = x.mask
                if x.has_mask and self.aug_mask and self.inter_mask is not None:
                    mask = _rgb_downscale_upscale(
                        src=x.mask,  # don't need to copy
                        scale=scale,
                        method=self.inter_mask,
                    )
                # done
                return x.override(image=im, mask=mask)

            return lax.switch(
                jrandom.choice(key, len(self.scale_levels), (), p=self.scale_p),
                branches=[
                    lambda x: true_branch(x, scale) for scale in self.scale_levels
                ],
                operand=x,
            )

        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    def _main():
        BlurGaussian().quick_test()
        BlurJpegCompression().quick_test()
        BlurDownscale().quick_test()

    _main()
