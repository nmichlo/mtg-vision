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
]


import numpy as np
from mtgvision.aug._base import AugItems, Augment, AugPrngHint, NpFloat32
from mtgvision.aug._util_args import ArgFloatHint, ArgFloatRange


# ========================================================================= #
# IN-PLACE NOISE                                                            #
# ========================================================================= #


def _rgb_img_inplace_gamma(src: NpFloat32, gamma: float = 1.0) -> NpFloat32:
    src[...] = np.power(src, gamma)
    return src


def _rgb_img_inplace_brightness(src: NpFloat32, brightness: float = 0.0) -> NpFloat32:
    src[...] += brightness
    return src


def _rgb_img_inplace_contrast(src: NpFloat32, contrast: float = 1.0) -> NpFloat32:
    src[...] = (src - 0.5) * contrast + 0.5
    return src


def _rgb_img_inplace_exposure(src: NpFloat32, exposure: float = 0.0) -> NpFloat32:
    src[...] = 1 - np.exp(-src * exposure)
    return src


def _rgb_img_inplace_saturation(src: NpFloat32, saturation: float = 1.0) -> NpFloat32:
    # TODO: relies on torch and kornia
    import kornia.enhance as ke
    import torch

    src[...] = (
        ke.adjust_saturation_with_gray_subtraction(torch.as_tensor(src), saturation)
        .numpy()
        .astype(np.float32)
    )
    return src


def _rgb_img_inplace_hue(src: NpFloat32, hue: float = 0.0) -> NpFloat32:
    # TODO: relies on torch and kornia
    import kornia.enhance as ke
    import torch

    src[...] = ke.adjust_hue(torch.as_tensor(src), hue).numpy().astype(np.float32)
    return src


def _rgb_img_inplace_tint(
    src: NpFloat32, tint_rgb: tuple[float, float, float] = (0, 0, 0)
) -> NpFloat32:
    src[...] = src + np.array(tint_rgb)[None, None, :]
    return src


def _clip_image(src: NpFloat32) -> NpFloat32:
    return np.clip(src, 0, 1)


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


class ColorGamma(Augment):
    def __init__(self, gamma: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._gamma = ArgFloatRange.from_arg(gamma)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_gamma(
                src=x.image.copy(),
                gamma=self._gamma.sample(prng),
            )
            return x.override(image=im)
        return x


class ColorBrightness(Augment):
    """
    Adjust the brightness of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, brightness: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._brightness = ArgFloatRange.from_arg(brightness, min_val=-1, max_val=1)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_brightness(
                src=x.image.copy(),
                brightness=self._brightness.sample(prng),
            )
            return x.override(image=im)
        return x


class ColorContrast(Augment):
    """
    Adjust the contrast of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, contrast: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._contrast = ArgFloatRange.from_arg(contrast, min_val=0)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_contrast(
                src=x.image.copy(),
                contrast=self._contrast.sample(prng),
            )
            return x.override(image=im)
        return x


class ColorExposure(Augment):
    """
    Adjust the exposure of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, exposure: ArgFloatHint = (0, 0.5), p: float = 0.5):
        super().__init__(p=p)
        self._exposure = ArgFloatRange.from_arg(exposure, min_val=-1, max_val=1)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_exposure(
                src=x.image.copy(),
                exposure=self._exposure.sample(prng),
            )
            return x.override(image=im)
        return x


class ColorSaturation(Augment):
    """
    Adjust the saturation of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.

    TODO: relies on torch and kornia
    """

    def __init__(self, saturation: ArgFloatHint = (0, 0.2), p: float = 0.5):
        super().__init__(p=p)
        self._saturation = ArgFloatRange.from_arg(saturation, min_val=0)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_saturation(
                src=x.image.copy(),
                saturation=self._saturation.sample(prng),
            )
            return x.override(image=im)
        return x


class ColorHue(Augment):
    """
    Adjust the hue of an image.

    TODO: relies on torch and kornia
    """

    def __init__(self, hue: ArgFloatHint = (0, 0.1), p: float = 0.5):
        super().__init__(p=p)
        self._hue = ArgFloatRange.from_arg(hue, min_val=-1, max_val=1)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_hue(
                src=x.image.copy(),
                hue=self._hue.sample(prng),
            )
            return x.override(image=im)
        return x


class ColorTint(Augment):
    """
    Adjust the tint of an image.

    *NB* Can result in image values outside the range [0, 1], clip images after using this.
    """

    def __init__(self, tint: ArgFloatHint = (0, 0.2), p: float = 0.5):
        super().__init__(p=p)
        self._tint = ArgFloatRange.from_arg(tint, min_val=-1, max_val=1)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _rgb_img_inplace_tint(
                src=x.image.copy(),
                tint_rgb=(
                    self._tint.sample(prng),
                    self._tint.sample(prng),
                    self._tint.sample(prng),
                ),
            )
            return x.override(image=im)
        return x


class ColorInvert(Augment):
    """
    Invert the colors of an image.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = 1 - x.image
            return x.override(image=im)
        return x


class ColorGrayscale(Augment):
    """
    Convert an image to grayscale.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = np.mean(x.image, axis=-1, keepdims=True)
            im = np.repeat(im, 3, axis=-1)
            return x.override(image=im)
        return x


class ImgClip(Augment):
    """
    Clip the values of an image to the range [0, 1].
    """

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image:
            im = _clip_image(x.image.copy())
            return x.override(image=im)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
