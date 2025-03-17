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

import math
from typing import Literal

import numpy as np
from mtgvision.aug._base import AugItems, Augment, AugPrngHint, NpFloat32
from mtgvision.aug._util_args import (
    ArgFloatHint,
    ArgFloatRange,
    ArgIntRange,
    ArgStrLiterals,
    ArgStrLiteralsHint,
)


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
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    peak: float = 1.0,
    eps: float = 0,
) -> NpFloat32:
    src[...] = prng.poisson(src * 255 * peak) / (peak * 255 + eps)
    return src


def _rgb_inplace_noise_salt_pepper(
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


def _rgb_inplace_random_erasing(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    scale_min_max: tuple[float, float] = (0.2, 0.7),  # [0, 1]
    aspect_min_max: tuple[float, float] = (1, 3),  # [1, inf]
    color: Literal[
        "random", "uniform_random", "zero", "one", "mean"
    ] = "uniform_random",
    inside: bool = True,
) -> NpFloat32:
    h, w = src.shape[:2]
    # scale
    scale = prng.uniform(*scale_min_max)
    target_area = scale * (h * w)
    # aspect
    aspect_ratio = prng.uniform(*aspect_min_max)
    if prng.random() < 0.5:
        aspect_ratio = 1 / aspect_ratio
    block_w = int((target_area / aspect_ratio) ** 0.5)
    block_h = int((target_area * aspect_ratio) ** 0.5)
    # get coords
    if inside:
        cx = prng.integers(block_w // 2, w - block_w // 2)
        cy = prng.integers(block_h // 2, h - block_h // 2)
    else:
        cx = prng.integers(0 - block_w // 2, w + block_w // 2)
        cy = prng.integers(0 - block_h // 2, h + block_h // 2)
    # clamp to valid ranges
    block_x0 = max(0, cx - block_w // 2)
    block_y0 = max(0, cy - block_h // 2)
    block_x1 = min(w, cx + block_w // 2)
    block_y1 = min(h, cy + block_h // 2)
    # color
    if color == "random":
        c = prng.uniform(0, 1, size=(block_h, block_w, src.shape[2]))
    elif color == "uniform_random":
        c = prng.uniform(0, 1, size=(1, 1, src.shape[2]))
    elif color in ("zero", "black"):
        c = 0
    elif color in ("one", "white"):
        c = 1
    elif color == "mean":
        c = src[block_y0:block_y1, block_x0:block_x1].mean()
    else:
        raise ValueError(f"invalid color: {color}")
    # fill
    src[block_y0:block_y1, block_x0:block_x1, ...] = c
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

    Sample peak values in the log domain.
    * peak=0 is full noise
    * peak=0.1 is some noise
    * peak=inf is no noise
    """

    def __init__(
        self,
        peak: ArgFloatHint = (0.01, 0.2),
        logprob: bool = True,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        self._peak = ArgFloatRange.from_arg(peak, min_val=0)
        self._inplace = inplace
        # log prob
        self._logprob = logprob
        if logprob:
            self._peak = ArgFloatRange(
                low=math.log(self._peak.low),
                high=math.log(self._peak.high),
            )
        # checks
        if self._peak.low == 0:
            raise ValueError(f"{self.__class__.__name__} cannot have a strength of 0")

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            peak = self._peak.sample(prng)
            if self._logprob:
                peak = math.exp(peak)
            # scale values from [0 (no aug), 1 (full aug)] to [inf (no aug), 0 (full aug)]
            # peak=max(1e-5, 1 / (self._strength.sample(prng) + 1e-5) - 1),
            im = _rgb_img_inplace_noise_poison(
                prng,
                src=x.image if self._inplace else x.image.copy(),
                peak=max(1e-7, peak),
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
            im = _rgb_inplace_noise_salt_pepper(
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
        aspect: ArgFloatHint = (1, 3),
        color: ArgStrLiteralsHint = "uniform_random",
        n: ArgIntRange = 1,
        inside: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._n = ArgIntRange.from_arg(n, min_val=0)
        self._scale = ArgFloatRange.from_arg(scale, min_val=0, max_val=1)
        self._aspect = ArgFloatRange.from_arg(aspect, min_val=1)
        self._color = ArgStrLiterals.from_arg(
            color, allowed_vals=("random", "uniform_random", "black", "white", "mean")
        )
        self._inside = inside

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            n = self._n.sample(prng)
            if n > 0:
                im = x.image.copy()
                for _ in range(n):
                    im = _rgb_inplace_random_erasing(
                        prng,
                        src=im,
                        scale_min_max=(self._scale.low, self._scale.high),
                        aspect_min_max=(self._aspect.low, self._aspect.high),
                        color=self._color.sample(prng),
                        inside=self._inside,
                    )
                return x.override(image=im)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
