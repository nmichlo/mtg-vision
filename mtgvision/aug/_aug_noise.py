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
]


import numpy as np
from mtgvision.aug._base import AugItems, Augment, AugPrngHint


# ========================================================================= #
# IN-PLACE NOISE                                                            #
# ========================================================================= #


NpFloat32 = np.ndarray[np.float32]


def _rgb_img_inplace_noise_multiplicative_speckle(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    strength: float = 0.1,
    channelwise: bool = True,
) -> None:
    if channelwise:
        src[...] *= prng.normal(1, strength, src.shape)
    else:
        src[...] *= prng.normal(1, strength, (*src.shape[:2], 1))


def _rgb_img_inplace_noise_additive_gaussian(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    strength: float = 0.1,
    channelwise: bool = True,
) -> None:
    if channelwise:
        src[...] += prng.normal(0, strength, src.shape)
    else:
        src[...] += prng.normal(0, strength, (*src.shape[:2], 1))


def _rgb_img_inplace_noise_poison(
    prng: AugPrngHint, *, src: NpFloat32, strength: float = 0.1
) -> None:
    src[...] = prng.poisson(src * strength) / strength


def _inplace_noise_salt_pepper(
    prng: AugPrngHint,
    *,
    src: NpFloat32,
    strength: float = 0.1,
    channelwise: bool = False,
) -> None:
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


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


class NoiseMultiplicativeGaussian(Augment):
    def __init__(self, strength: float = 0.1, channelwise: bool = True, p: float = 0.5):
        super().__init__(p=p)
        self._strength = strength
        self._channelwise = channelwise

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_noise_multiplicative_speckle(
                prng,
                src=x.image,
                strength=self._strength,
                channelwise=self._channelwise,
            )
            return x.override(image=im)
        return x


class NoiseAdditiveGaussian(Augment):
    def __init__(self, strength: float = 0.1, channelwise: bool = True, p: float = 0.5):
        super().__init__(p=p)
        self._strength = strength
        self._channelwise = channelwise

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_noise_additive_gaussian(
                prng,
                src=x.image,
                strength=self._strength,
                channelwise=self._channelwise,
            )
            return x.override(image=im)
        return x


class NoisePoison(Augment):
    def __init__(self, strength: float = 0.1, p: float = 0.5):
        super().__init__(p=p)
        self._strength = strength

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_noise_poison(
                prng, src=x.image, strength=self._strength
            )
            return x.override(image=im)
        return x


class NoiseSaltPepper(Augment):
    def __init__(
        self, strength: float = 0.1, channelwise: bool = False, p: float = 0.5
    ):
        super().__init__(p=p)
        self._strength = strength
        self._channelwise = channelwise

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _inplace_noise_salt_pepper(
                prng,
                src=x.image,
                strength=self._strength,
                channelwise=self._channelwise,
            )
            return x.override(image=im)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
