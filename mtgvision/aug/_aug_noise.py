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
    "NoiseMultiplicativeGaussian",
    "NoiseAdditiveGaussian",
    "NoisePoison",
    "NoiseSaltPepper",
    "RandomErasing",
]


import numpy as np
from mtgvision.aug._base import AugItems, Augment, AugPrngHint, NpFloat32
from mtgvision.aug._util_args import ArgFloatHint, ArgFloatRange


# ========================================================================= #
# IN-PLACE NOISE                                                            #
# ========================================================================= #


def _rgb_img_inplace_noise_multiplicative_speckle(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    strength: float = 0.1,
    channelwise: bool = True,
) -> NpFloat32:
    if channelwise:
        src[...] *= prng.normal(1, strength, src.shape)
    else:
        src[...] *= prng.normal(1, strength, (*src.shape[:2], 1))
    return src


def _rgb_img_inplace_noise_additive_gaussian(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    strength: float = 0.1,
    channelwise: bool = True,
) -> NpFloat32:
    if channelwise:
        src[...] += prng.normal(0, strength, src.shape)
    else:
        src[...] += prng.normal(0, strength, (*src.shape[:2], 1))
    return src


def _rgb_img_inplace_noise_poison(
    prng: AugPrngHint, *, src: NpFloat32, strength: float = 0.1, eps: float = 0
) -> NpFloat32:
    src[...] = prng.poisson(src * strength) / (strength + eps)
    return src


def _inplace_noise_salt_pepper(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    strength: float = 0.1,
    channelwise: bool = False,
) -> NpFloat32:
    if channelwise:
        num_noise = int(strength * src.size)
        salt_or_pepper = prng.integers(0, 2, num_noise)
        ix = prng.integers(0, src.shape[0], num_noise)
        iy = prng.integers(0, src.shape[1], num_noise)
        ic = prng.integers(0, src.shape[2], num_noise)
        src[ix, iy, ic] = salt_or_pepper
    else:
        num_noise = int(strength * np.prod(src.shape[:2]))
        salt_or_pepper = prng.integers(0, 2, num_noise)
        ix = prng.integers(0, src.shape[0], num_noise)
        iy = prng.integers(0, src.shape[1], num_noise)
        src[ix, iy, :] = salt_or_pepper[:, None]
    return src


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


class NoiseMultiplicativeGaussian(Augment):
    """
    Add multiplicative gaussian noise to the image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(
        self,
        strength: ArgFloatHint = (0, 0.1),
        channelwise: bool = True,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        self._strength = ArgFloatRange.from_arg(strength, min_val=0)
        self._channelwise = channelwise
        self._inplace = inplace

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_noise_multiplicative_speckle(
                prng,
                src=x.image if self._inplace else x.image.copy(),
                strength=self._strength.sample(prng),
                channelwise=self._channelwise,
            )
            return x.override(image=im)
        return x


class NoiseAdditiveGaussian(Augment):
    """
    Add additive gaussian noise to the image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(
        self,
        strength: ArgFloatHint = (0, 0.1),
        channelwise: bool = True,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        self._strength = ArgFloatRange.from_arg(strength, min_val=0)
        self._channelwise = channelwise
        self._inplace = inplace

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_noise_additive_gaussian(
                prng,
                src=x.image if self._inplace else x.image.copy(),
                strength=self._strength.sample(prng),
                channelwise=self._channelwise,
            )
            return x.override(image=im)
        return x


class NoisePoison(Augment):
    """
    Add poisson noise to the image.
    """

    def __init__(
        self,
        strength: ArgFloatHint = (0, 0.1),
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        self._strength = ArgFloatRange.from_arg(strength, min_val=0)
        self._inplace = inplace
        # checks
        if self._strength.low == 0:
            raise ValueError(f"{self.__class__.__name__} cannot have a strength of 0")

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_noise_poison(
                prng,
                src=x.image if self._inplace else x.image.copy(),
                strength=self._strength.sample(prng),
            )
            return x.override(image=im)
        return x


class NoiseSaltPepper(Augment):
    """
    Add salt and pepper noise to the image.
    """

    def __init__(
        self,
        strength: ArgFloatHint = (0, 0.2),
        channelwise: bool = False,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        self._strength = ArgFloatRange.from_arg(strength, min_val=0, max_val=1)
        self._channelwise = channelwise
        self._inplace = inplace

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _inplace_noise_salt_pepper(
                prng,
                src=x.image if self._inplace else x.image.copy(),
                strength=self._strength.sample(prng),
                channelwise=self._channelwise,
            )
            return x.override(image=im)
        return x


class RandomErasing(Augment):
    """
    Randomly erase a rectangular region in the image, leaving mask and points unchanged.
    """

    def __init__(
        self,
        scale: ArgFloatHint = (0.2, 0.7),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._scale = ArgFloatRange.from_arg(scale, min_val=0, max_val=1)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            img = x.image.copy()
            h, w = img.shape[:2]
            area = h * w
            target_area = self._scale.sample(prng) * area
            aspect_ratio = prng.uniform(0.3, 1 / 0.3)
            rh = int((target_area * aspect_ratio) ** 0.5)
            rw = int((target_area / aspect_ratio) ** 0.5)
            if rh < h and rw < w:
                x0 = prng.integers(0, w - rw)
                y0 = prng.integers(0, h - rh)
                fill_value = prng.uniform(0, 1, size=(rh, rw, img.shape[2]))
                img[y0 : y0 + rh, x0 : x0 + rw, :] = fill_value
            return x.override(image=img)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
