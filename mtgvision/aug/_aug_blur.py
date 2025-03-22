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

import jax
import dm_pix as pix

from mtgvision.aug._base import AugItems, Augment, AugPrngHint, JnpFloat32
from mtgvision.aug._util_args import (
    ArgFloatHint,
    ArgFloatRange,
    ArgIntHint,
    ArgIntRange,
)
from mtgvision.aug._util_jax import ResizeMethod


# ========================================================================= #
# functions                                                                 #
# ========================================================================= #


def _rgb_img_jpeg_compression(
    src: JnpFloat32,
    quality: int,
) -> JnpFloat32:
    # # to uint
    # img = np.clip(src * 255, 0, 255).astype(np.uint8)
    # # encode, decode
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # _, encimg = cv2.imencode(".jpg", img, encode_param)
    # img_transformed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    # # to float
    # return img_transformed.astype(np.float32) / 255

    raise NotImplementedError


def _rgb_downscale_upscale(
    src: JnpFloat32,
    scale: float,
    method: ResizeMethod,
    antialias: bool = True,
) -> JnpFloat32:
    method = ResizeMethod(method).value
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


def _rgb_gaussian_blur_sigma(
    src: JnpFloat32,
    sigma: int,
    kernel_scale: float = 1.0,
) -> JnpFloat32:
    # ~3 sigma on each side of the kernel's center covers ~99.7% of the
    # probability mass. There is some fiddling for smaller values. Source:
    # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    kernel_size = ((sigma - 0.8) / 0.3 + 1) / 0.5 + 1
    # apply gaussian blur
    return pix.gaussian_blur(src, sigma, kernel_size * kernel_scale)


def _rgb_gaussian_blur_kernel(
    src: JnpFloat32,
    kernel_size: int,
    sigma_scale: float = 1.0,
) -> JnpFloat32:
    sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    # apply gaussian blur
    return pix.gaussian_blur(src, sigma * sigma_scale, kernel_size)


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


class BlurGaussian(Augment):
    """
    Apply a box blur to the image.
    """

    def __init__(
        self,
        radius: ArgIntHint = (0, 3),
        aug_mask: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._radius = ArgIntRange.from_arg(radius, min_val=0)
        self._aug_mask = aug_mask

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask:
            r = self._radius.sample(prng)
            if r > 0:
                # image
                im = x.image
                if x.has_image:
                    im = _rgb_gaussian_blur_kernel(src=x.image, kernel_size=r)
                # mask
                mask = x.mask
                if x.has_mask and self._aug_mask:
                    mask = _rgb_gaussian_blur_kernel(src=x.mask, kernel_size=r)
                # done
                return x.override(image=im, mask=mask)
        return x


# class BlurMedian(Augment):
#     """
#     Apply a median blur to the image.
#     """
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
#     def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
#         if x.has_image or x.has_mask:
#             r = self._radius.sample(prng)
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


class BlurJpegCompression(Augment):
    """
    Apply jpeg compression to the image.
    """

    def __init__(self, quality: ArgIntHint = (10, 100), p: float = 0.5):
        super().__init__(p=p)
        self._quality = ArgIntRange.from_arg(quality, min_val=0, max_val=100)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_jpeg_compression(
                src=x.image,  # don't need to copy
                quality=self._quality.sample(prng),
            )
            return x.override(image=im)
        return x


class BlurDownscale(Augment):
    """
    Downscale the image and then upscale it back to the original size.
    """

    def __init__(
        self,
        scale: ArgFloatHint = (1 / 8, 1),
        aug_mask: bool = False,
        inter: ResizeMethod = ResizeMethod.LINEAR,
        inter_mask: ResizeMethod = ResizeMethod.LINEAR,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._scale = ArgFloatRange.from_arg(scale, min_val=0, max_val=1)
        self._aug_mask = aug_mask
        self._inter = ResizeMethod(inter)
        self._inter_mask = ResizeMethod(inter_mask)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask:
            scale = self._scale.sample(prng)
            if scale < 1:
                # image
                im = x.image
                if x.has_image:
                    im = _rgb_downscale_upscale(
                        src=x.image,  # don't need to copy
                        scale=scale,
                        method=self._inter,
                    )
                # mask
                mask = x.mask
                if x.has_mask and self._aug_mask and self._inter_mask is not None:
                    mask = _rgb_downscale_upscale(
                        src=x.mask,  # don't need to copy
                        scale=scale,
                        method=self._inter_mask,
                    )
                # done
                return x.override(image=im, mask=mask)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
