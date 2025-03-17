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
    "BlurBox",
    "BlurMedian",
    "BlurJpegCompression",
    "BlurDownscale",
]

import cv2
import numpy as np

from mtgvision.aug._base import AugItems, Augment, AugPrngHint, NpFloat32
from mtgvision.aug._util_args import (
    ArgFloatHint,
    ArgFloatRange,
    ArgIntHint,
    ArgIntRange,
)

# ========================================================================= #
# functions                                                                 #
# ========================================================================= #


def _rgb_img_jpeg_compression(
    src: NpFloat32,
    quality: int,
) -> NpFloat32:
    # to uint
    img = np.clip(src * 255, 0, 255).astype(np.uint8)
    # encode, decode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    img_transformed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    # to float
    return img_transformed.astype(np.float32) / 255


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


class BlurBox(Augment):
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
                    im = cv2.blur(x.image, (r, r))
                # mask
                mask = x.mask
                if x.has_mask and self._aug_mask:
                    mask = cv2.blur(x.mask, (r, r))
                # done
                return x.override(image=im, mask=mask)
        return x


class BlurMedian(Augment):
    """
    Apply a median blur to the image.
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
                    im = cv2.medianBlur(x.image, r)
                # mask
                mask = x.mask
                if x.has_mask and self._aug_mask:
                    mask = cv2.medianBlur(x.mask, r)
                # done
                return x.override(image=im, mask=mask)
        return x


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
        interpolation: int = cv2.INTER_LINEAR,  # cv2.
        interpolation_mask: int = cv2.INTER_LINEAR,  # cv2.
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._scale = ArgFloatRange.from_arg(scale, min_val=0, max_val=1)
        self._aug_mask = aug_mask

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask:
            scale = self._scale.sample(prng)
            if scale < 1:
                # image
                im = x.image
                if x.has_image:
                    im = cv2.resize(
                        x.image,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                # mask
                mask = x.mask
                if x.has_mask and self._aug_mask:
                    mask = cv2.resize(
                        x.mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
                    )
                # done
                return x.override(image=im, mask=mask)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
