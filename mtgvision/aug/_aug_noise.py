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

from dataclasses import dataclass

import jax.numpy as jnp
from typing import Literal
import jax.random as jrandom
import numpy as np
from jax import lax

from mtgvision.aug._base import AugItems, Augment, AugPrngHint, JnpFloat32
from mtgvision.aug._util_args import (
    ArgFloatHint,
    ArgIntHint,
    ArgStrLiteralsHint,
    sample_float,
    sample_int,
    sample_str,
)

from jax.tree_util import register_dataclass
from mtgvision.aug._base import jax_static_field


# ========================================================================= #
# IN-PLACE NOISE                                                            #
# ========================================================================= #


def _rgb_img_noise_multiplicative_speckle(
    key: AugPrngHint,
    *,
    src: JnpFloat32,
    strength: float = 0.1,
    channelwise: bool = True,
) -> JnpFloat32:
    rand = lax.cond(
        channelwise,
        lambda: jrandom.normal(key, src.shape),
        lambda: jrandom.normal(key, (*src.shape[:2], 1)).repeat(3, -1),
    )
    return src * (rand * strength + 1)


def _rgb_img_noise_additive_gaussian(
    key: AugPrngHint,
    *,
    src: JnpFloat32,
    strength: float = 0.1,
    channelwise: bool = True,
) -> JnpFloat32:
    rand = lax.cond(
        channelwise,
        lambda: jrandom.normal(key, src.shape),
        lambda: jrandom.normal(key, (*src.shape[:2], 1)).repeat(3, -1),
    )
    return src + (rand * strength)


def _rgb_img_noise_poison(
    key: AugPrngHint,
    *,
    src: JnpFloat32,
    peak: float = 1.0,
    eps: float = 0,
) -> JnpFloat32:
    p = 255 * peak
    return jrandom.poisson(key, src * p) / (p + eps)


def _rgb_inplace_noise_salt_pepper(
    key: AugPrngHint,
    *,
    src: JnpFloat32,
    strength: float = 0.1,
    channelwise: bool = False,
) -> JnpFloat32:
    rand = lax.cond(
        channelwise,
        lambda: jrandom.uniform(key, src.shape),
        lambda: jrandom.uniform(key, (*src.shape[:2], 1)).repeat(3, -1),
    )
    # set salt and pepper
    src = jnp.where(rand < (strength), 0, src)
    src = jnp.where(rand < (strength / 2), 1, src)
    return src


def _rgb_inplace_random_erasing(
    key: AugPrngHint,
    *,
    src: JnpFloat32,
    scale_min_max: tuple[float, float] = (0.2, 0.7),  # [0, 1]
    aspect_min_max: tuple[float, float] = (1, 3),  # [1, inf]
    color: Literal[
        "random", "uniform_random", "zero", "one", "mean"
    ] = "uniform_random",
    inside: bool = True,
) -> JnpFloat32:
    h, w = src.shape[:2]
    # scale
    scale = jrandom.uniform(key, (), jnp.float32, *scale_min_max)
    target_area = scale * (h * w)
    # aspect
    aspect_ratio = jrandom.uniform(key, (), jnp.float32, *aspect_min_max)
    aspect_ratio = lax.cond(
        jrandom.uniform(key) < 0.5,
        lambda: 1 / aspect_ratio,
        lambda: aspect_ratio,
    )

    block_w = ((target_area / aspect_ratio) ** 0.5).astype(jnp.int32)
    block_h = ((target_area * aspect_ratio) ** 0.5).astype(jnp.int32)
    # get coords
    (mx, Mx), (my, My) = lax.cond(
        inside,
        lambda: ((block_w // 2, w - block_w // 2), (block_h // 2, h - block_h // 2)),
        lambda: (
            (0 - block_w // 2, w + block_w // 2),
            (0 - block_h // 2, h + block_h // 2),
        ),
    )
    cx = jrandom.randint(key, (), mx, Mx)
    cy = jrandom.randint(key, (), my, My)

    # clamp to valid ranges
    block_x0 = jnp.maximum(0, cx - block_w // 2)
    block_y0 = jnp.maximum(0, cy - block_h // 2)
    block_x1 = jnp.minimum(w, cx + block_w // 2)
    block_y1 = jnp.minimum(h, cy + block_h // 2)

    index = {
        "uniform_random": 0,
        "zero": 1,
        "black": 1,
        "one": 2,
        "white": 2,
    }

    # color
    c = lax.switch(
        index[color],
        [
            lambda: jrandom.uniform(key, (src.shape[-1],)),
            lambda: jnp.zeros((src.shape[-1],)),
            lambda: jnp.ones((src.shape[-1],)),
        ],
    )

    # bool mask over fill region
    masky = jnp.where(np.arange(h) >= block_y0, np.arange(h) < block_y1, False)
    maskx = jnp.where(np.arange(w) >= block_x0, np.arange(w) < block_x1, False)
    mask = masky[:, None] & maskx[None, :]

    # fill the region
    src = jnp.where(mask[:, :, None], c[None, None, :], src)
    return src


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


@register_dataclass
@dataclass(frozen=True)
class NoiseMultiplicativeGaussian(Augment):
    """
    Add multiplicative gaussian noise to the image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    p: float = jax_static_field(default=0.5)
    strength: ArgFloatHint = jax_static_field(default=(0, 0.1))  # min 0
    channelwise: bool = jax_static_field(default=True)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_noise_multiplicative_speckle(
                key,
                src=x.image,
                strength=sample_float(key, self.strength),
                channelwise=self.channelwise,
            )
            return x.override(image=im)
        return x


@register_dataclass
@dataclass(frozen=True)
class NoiseAdditiveGaussian(Augment):
    """
    Add additive gaussian noise to the image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    p: float = jax_static_field(default=0.5)
    strength: ArgFloatHint = jax_static_field(default=(0, 0.1))  # min 0
    channelwise: bool = jax_static_field(default=True)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_noise_additive_gaussian(
                key,
                src=x.image,
                strength=sample_float(key, self.strength),
                channelwise=self.channelwise,
            )
            return x.override(image=im)
        return x


@register_dataclass
@dataclass(frozen=True)
class NoisePoison(Augment):
    """
    Add poisson noise to the image.

    Sample peak values in the log domain.
    * peak=0 is full noise
    * peak=0.1 is some noise
    * peak=inf is no noise
    """

    p: float = jax_static_field(default=0.5)
    peak: ArgFloatHint = jax_static_field(default=(0.01, 0.2))  # min 0
    logprob: bool = jax_static_field(default=True)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            peak = sample_float(key, self.peak)
            if self.logprob:
                peak = jnp.exp(peak)
            # scale values from [0 (no aug), 1 (full aug)] to [inf (no aug), 0 (full aug)]
            # peak=max(1e-5, 1 / (self._strength.sample(prng) + 1e-5) - 1),
            im = _rgb_img_noise_poison(
                key,
                src=x.image,
                peak=jnp.maximum(1e-7, peak),
            )
            return x.override(image=im)
        return x


@register_dataclass
@dataclass(frozen=True)
class NoiseSaltPepper(Augment):
    """
    Add salt and pepper noise to the image.
    """

    p: float = jax_static_field(default=0.5)
    strength: ArgFloatHint = jax_static_field(default=(0, 0.2))  # min 0, max 1
    channelwise: bool = jax_static_field(default=False)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_inplace_noise_salt_pepper(
                key,
                src=x.image.copy(),
                strength=sample_float(key, self.strength),
                channelwise=self.channelwise,
            )
            return x.override(image=im)
        return x


@register_dataclass
@dataclass(frozen=True)
class RandomErasing(Augment):
    """
    Randomly erase a rectangular region in the image, leaving mask and points unchanged.
    """

    p: float = jax_static_field(default=0.5)
    scale: ArgFloatHint = jax_static_field(default=(0.2, 0.7))  # min 0, max 1
    n: ArgIntHint = jax_static_field(default=1)  # min 0
    aspect: ArgFloatHint = jax_static_field(default=(1, 3))  # min 1
    color: ArgStrLiteralsHint = jax_static_field(default="uniform_random")
    inside: bool = jax_static_field(default=False)

    @staticmethod
    def _norm_range(min_max):
        if isinstance(min_max, (tuple, list)):
            m, M = min_max
            return (m, M)
        return (min_max, min_max)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            n = sample_int(key, self.n)

            def apply(im):
                return _rgb_inplace_random_erasing(
                    key,
                    src=im,
                    scale_min_max=self._norm_range(self.scale),
                    aspect_min_max=self._norm_range(self.aspect),
                    color=sample_str(key, self.color),
                    inside=self.inside,
                )

            im = lax.fori_loop(0, n, lambda i, im: apply(im), x.image.copy())
            return x.override(image=im)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    def _main():
        for Aug in [
            NoiseMultiplicativeGaussian,
            NoiseAdditiveGaussian,
            NoisePoison,
            NoiseSaltPepper,
            RandomErasing,
        ]:
            Aug().quick_test()

    _main()
