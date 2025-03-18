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

import jax
import jax.numpy as jnp
import jax.random as jrandom
from tqdm import tqdm
import dm_pix as pix

from mtgvision.aug import AugImgHint
from mtgvision.aug._base import AugmentItems, AugPrngHint, NpFloat32
from mtgvision.aug._util_args import ArgFloatHint, ArgFloatRange


# ========================================================================= #
# IN-PLACE NOISE                                                            #
# ========================================================================= #


def _rgb_img_gamma(src: NpFloat32, gamma: float = 1.0) -> NpFloat32:
    return jnp.power(src, gamma)


def _rgb_img_brightness(src: NpFloat32, brightness: float = 0.0) -> NpFloat32:
    return src + brightness


def _rgb_img_contrast(src: NpFloat32, contrast: float = 1.0) -> NpFloat32:
    # effectively LERP between image and 0.5
    return contrast * src + (1 - contrast) * 0.5


def _rgb_img_exposure(src: NpFloat32, exposure: float = 0.0) -> NpFloat32:
    return src * 2**exposure


def _rgb_img_saturation(src: NpFloat32, saturation: float = 1.0) -> NpFloat32:
    # taken from kornia.enhance.adjust_saturation_with_gray_subtraction
    grey = pix.rgb_to_grayscale(src, keep_dims=True)
    # blend the image with the grayscaled image
    return (1 - saturation) * grey + saturation * src


def _rgb_img_hue(src: NpFloat32, hue: float = 0.0) -> NpFloat32:
    return pix.adjust_hue(jnp.clip(src, 0, 1), hue)


def _rgb_img_tint(
    src: NpFloat32,
    tint_rgb: tuple[float, float, float] = (0, 0, 0),
) -> NpFloat32:
    return src + jnp.asarray(tint_rgb, dtype=jnp.float32)[None, None, :]


def _rgb_img_fade_white(src: NpFloat32, ratio: float = 0.33) -> NpFloat32:
    return (1 - ratio) * src + (ratio * 1)


def _rgb_img_fade_black(src: NpFloat32, ratio: float = 0.5) -> NpFloat32:
    return (1 - ratio) * src  # + (ratio*0)


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


class ColorGamma(AugmentItems):
    def __init__(self, gamma: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._gamma = ArgFloatRange.from_arg(gamma, min_val=0)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_gamma(src=image, gamma=self._gamma.sample(prng))


class ColorBrightness(AugmentItems):
    """
    Adjust the brightness of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, brightness: ArgFloatHint = (-0.5, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._brightness = ArgFloatRange.from_arg(brightness, min_val=-1, max_val=1)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_brightness(src=image, brightness=self._brightness.sample(prng))


class ColorContrast(AugmentItems):
    """
    Adjust the contrast of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, contrast: ArgFloatHint = (0.5, 2), p: float = 0.5):
        super().__init__(p=p)
        self._contrast = ArgFloatRange.from_arg(contrast, min_val=0)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_contrast(src=image, contrast=self._contrast.sample(prng))


class ColorExposure(AugmentItems):
    """
    Adjust the exposure of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, exposure: ArgFloatHint = (0.5, 2), p: float = 0.5):
        super().__init__(p=p)
        self._exposure = ArgFloatRange.from_arg(exposure)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_exposure(src=image, exposure=self._exposure.sample(prng))


class ColorSaturation(AugmentItems):
    """
    Adjust the saturation of an image.

    TODO: relies on torch and kornia
    """

    def __init__(self, saturation: ArgFloatHint = (0, 2), p: float = 0.5):
        super().__init__(p=p)
        self._saturation = ArgFloatRange.from_arg(saturation, min_val=0)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_saturation(src=image, saturation=self._saturation.sample(prng))


class ColorHue(AugmentItems):
    """
    Adjust the hue of an image.

    TODO: relies on torch and kornia
    """

    def __init__(self, hue: ArgFloatHint = (-0.1, 0.1), p: float = 0.5):
        super().__init__(p=p)
        self._hue = ArgFloatRange.from_arg(hue, min_val=-1, max_val=1)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_hue(src=image, hue=self._hue.sample(prng))


class ColorTint(AugmentItems):
    """
    Adjust the tint of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, tint: ArgFloatHint = (-0.3, 0.3), p: float = 0.5):
        super().__init__(p=p)
        self._tint = ArgFloatRange.from_arg(tint, min_val=-1, max_val=1)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_tint(
            src=image,
            tint_rgb=(
                self._tint.sample(prng),
                self._tint.sample(prng),
                self._tint.sample(prng),
            ),
        )


class ColorInvert(AugmentItems):
    """
    Invert the colors of an image.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return 1 - image


class ColorGrayscale(AugmentItems):
    """
    Convert an image to grayscale.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        im = jnp.mean(image, axis=-1, keepdims=True)
        im = jnp.repeat(im, 3, axis=-1)
        return im


class ImgClip(AugmentItems):
    """
    Clip the values of an image to the range [0, 1].
    """

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return jnp.clip(image, 0, 1)


class ColorFadeWhite(AugmentItems):
    """
    Fade the colors of an image to white.

    0 is no fade, 1 is full fade.
    """

    def __init__(self, ratio: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._ratio = ArgFloatRange.from_arg(ratio, min_val=0, max_val=1)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_fade_white(src=image, ratio=self._ratio.sample(prng))


class ColorFadeBlack(AugmentItems):
    """
    Fade the colors of an image to black.

    0 is no fade, 1 is full fade.
    """

    def __init__(self, ratio: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._ratio = ArgFloatRange.from_arg(ratio, min_val=0, max_val=1)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return _rgb_img_fade_black(src=image, ratio=self._ratio.sample(prng))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":
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
        aug = Aug()
        key = jrandom.key(32)
        src = jrandom.uniform(key, (224, 224, 3), jnp.float32)
        aug = jax.jit(aug.__call__)
        for i in tqdm(range(15000), desc=f"{Aug.__name__}"):
            aug(image=src)
