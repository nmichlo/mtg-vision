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
    "ColorGamma",
    "ColorBrightness",
    "ColorContrast",
    "ColorExposure",
    "ColorSaturation",
    "ColorHue",
    "ColorTint",
    "ColorInvert",
    "ColorGrayscale",
    "ImgClip",
    "ColorFadeWhite",
    "ColorFadeBlack",
]

import jax.numpy as jnp
import jax.random as jrandom
import dm_pix as pix

from mtgvision.aug import AugImgHint
from mtgvision.aug._base import AugmentItems, AugPrngHint, JnpFloat32
from mtgvision.aug._util_args import ArgFloatHint, sample_float
from dataclasses import dataclass

from jax.tree_util import register_dataclass

from mtgvision.aug._base import jax_static_field


# ========================================================================= #
# IN-PLACE NOISE                                                            #
# ========================================================================= #


def _rgb_img_gamma(src: JnpFloat32, gamma: float = 1.0) -> JnpFloat32:
    return jnp.power(src, gamma)


def _rgb_img_brightness(src: JnpFloat32, brightness: float = 0.0) -> JnpFloat32:
    return src + brightness


def _rgb_img_contrast(src: JnpFloat32, contrast: float = 1.0) -> JnpFloat32:
    # effectively LERP between image and 0.5
    return contrast * src + (1 - contrast) * 0.5


def _rgb_img_exposure(src: JnpFloat32, exposure: float = 0.0) -> JnpFloat32:
    return src * 2**exposure


def _rgb_img_saturation(src: JnpFloat32, saturation: float = 1.0) -> JnpFloat32:
    # taken from kornia.enhance.adjust_saturation_with_gray_subtraction
    grey = pix.rgb_to_grayscale(src, keep_dims=True)
    # blend the image with the grayscaled image
    return (1 - saturation) * grey + saturation * src


def _rgb_img_hue(src: JnpFloat32, hue: float = 0.0) -> JnpFloat32:
    return pix.adjust_hue(jnp.clip(src, 0, 1), hue)


def _rgb_img_tint(
    src: JnpFloat32,
    tint_rgb: tuple[float, float, float] = (0, 0, 0),
) -> JnpFloat32:
    return src + jnp.asarray(tint_rgb, dtype=jnp.float32)[None, None, :]


def _rgb_img_fade_white(src: JnpFloat32, ratio: float = 0.33) -> JnpFloat32:
    return (1 - ratio) * src + (ratio * 1)


def _rgb_img_fade_black(src: JnpFloat32, ratio: float = 0.5) -> JnpFloat32:
    return (1 - ratio) * src  # + (ratio*0)


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


def _ran(key, m, M, shape=()):
    return jrandom.uniform(key, shape, minval=m, maxval=M)


@register_dataclass
@dataclass(frozen=True)
class ColorGamma(AugmentItems):
    p: float = jax_static_field(default=0.5)
    gamma: ArgFloatHint = jax_static_field(default=(0, 0.5))  # min 0

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.gamma)
        return _rgb_img_gamma(src=image, gamma=val)


@register_dataclass
@dataclass(frozen=True)
class ColorBrightness(AugmentItems):
    """
    Adjust the brightness of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    p: float = jax_static_field(default=0.5)
    brightness: ArgFloatHint = jax_static_field(default=(-0.5, 0.5))  # min -1, max 1

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.brightness)
        return _rgb_img_brightness(src=image, brightness=val)


@register_dataclass
@dataclass(frozen=True)
class ColorContrast(AugmentItems):
    """
    Adjust the contrast of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    p: float = jax_static_field(default=0.5)
    contrast: ArgFloatHint = jax_static_field(default=(0.5, 2))  # min 0

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.contrast)
        return _rgb_img_contrast(src=image, contrast=val)


@register_dataclass
@dataclass(frozen=True)
class ColorExposure(AugmentItems):
    """
    Adjust the exposure of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    p: float = jax_static_field(default=0.5)
    exposure: ArgFloatHint = jax_static_field(default=(0.5, 2))

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.exposure)
        return _rgb_img_exposure(src=image, exposure=val)


@register_dataclass
@dataclass(frozen=True)
class ColorSaturation(AugmentItems):
    """
    Adjust the saturation of an image.
    """

    p: float = jax_static_field(default=0.5)
    saturation: ArgFloatHint = jax_static_field(default=(0, 2))  # min 0

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.saturation)
        return _rgb_img_saturation(src=image, saturation=val)


@register_dataclass
@dataclass(frozen=True)
class ColorHue(AugmentItems):
    """
    Adjust the hue of an image.
    """

    p: float = jax_static_field(default=0.5)
    hue: ArgFloatHint = jax_static_field(default=(-0.1, 0.1))  # min -1, max 1

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.hue)
        return _rgb_img_hue(src=image, hue=val)


@register_dataclass
@dataclass(frozen=True)
class ColorTint(AugmentItems):
    """
    Adjust the tint of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    p: float = jax_static_field(default=0.5)
    tint: ArgFloatHint = jax_static_field(default=(-0.3, 0.3))  # min -1, max 1

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_tint(
            src=image,
            tint_rgb=(
                sample_float(key, self.tint),
                sample_float(key, self.tint),
                sample_float(key, self.tint),
            ),
        )


@register_dataclass
@dataclass(frozen=True)
class ColorInvert(AugmentItems):
    """
    Invert the colors of an image.
    """

    p: float = jax_static_field(default=0.5)

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return 1 - image


@register_dataclass
@dataclass(frozen=True)
class ColorGrayscale(AugmentItems):
    """
    Convert an image to grayscale.
    """

    p: float = jax_static_field(default=0.5)

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        im = jnp.mean(image, axis=-1, keepdims=True)
        im = jnp.repeat(im, 3, axis=-1)
        return im


@register_dataclass
@dataclass(frozen=True)
class ImgClip(AugmentItems):
    """
    Clip the values of an image to the range [0, 1].
    """

    p: float = jax_static_field(default=1.0)

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return jnp.clip(image, 0, 1)

    def _augment_mask(self, key: AugPrngHint, mask: AugImgHint) -> AugImgHint:
        return jnp.clip(mask, 0, 1)


@register_dataclass
@dataclass(frozen=True)
class ColorFadeWhite(AugmentItems):
    """
    Fade the colors of an image to white.

    0 is no fade, 1 is full fade.
    """

    p: float = jax_static_field(default=0.5)
    ratio: float = jax_static_field(default=(0, 0.5))  # min 0, max 1

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.ratio)
        return _rgb_img_fade_white(src=image, ratio=val)


@register_dataclass
@dataclass(frozen=True)
class ColorFadeBlack(AugmentItems):
    """
    Fade the colors of an image to black.

    0 is no fade, 1 is full fade.
    """

    p: float = jax_static_field(default=0.5)
    ratio: float = jax_static_field(default=(0, 0.5))  # min 0, max 1

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        val = sample_float(key, self.ratio)
        return _rgb_img_fade_black(src=image, ratio=val)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    def _main():
        for Aug in [
            ColorGamma,
            ColorBrightness,
            ColorContrast,
            ColorExposure,
            ColorSaturation,
            ColorHue,
            ColorTint,
            ColorInvert,
            ColorGrayscale,
            ImgClip,
            ColorFadeWhite,
            ColorFadeBlack,
        ]:
            Aug().quick_test()

    _main()
