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
    "NpFloat32",
    # hints
    "AugImgHint",
    "AugMaskHint",
    "AugPointsHint",
    "AugPrngHint",
    # results
    "AugItems",
    # base
    "Augment",
    "AugmentItems",
]

import abc

import jax
import jax.random as jrandom
import numpy.random as np_random
from jax import lax

from typing_extensions import final, NamedTuple


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


NpFloat32 = jax.Array

AugImgHint = NpFloat32
AugMaskHint = NpFloat32
AugPointsHint = NpFloat32
AugPrngHint = jax.Array


_MISSING = object()


class AugItems(NamedTuple):
    """
    Named tuple for holding augment items that are passed between augments and returned.
    """

    image: AugImgHint | None = None
    mask: AugMaskHint | None = None
    points: AugPointsHint | None = None

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_points(self) -> bool:
        return self.points is not None

    def override(
        self,
        image: AugImgHint | None = _MISSING,
        mask: AugMaskHint | None = _MISSING,
        points: AugPointsHint | None = _MISSING,
    ) -> "AugItems":
        # defaults
        if image is _MISSING:
            image = self.image
        if mask is _MISSING:
            mask = self.mask
        if points is _MISSING:
            points = self.points
        # handle xor, not allowed to change from None to set or set to None
        if (self.image is None) != (image is None):
            raise ValueError("Cannot change image from None to set or set to None")
        if (self.mask is None) != (mask is None):
            raise ValueError("Cannot change mask from None to set or set to None")
        if (self.points is None) != (points is None):
            raise ValueError("Cannot change points from None to set or set to None")
        # TODO check for NaN?
        # override
        return AugItems(image=image, mask=mask, points=points)

    def get_bounds_hw(self) -> tuple[int, int]:
        if self.has_image and self.has_mask:
            if self.image.shape[:2] != self.mask.shape[:2]:
                raise ValueError(
                    f"Image and mask do not have the same dimensions, got: {self.image.shape[:2]} != {self.mask.shape[:2]}"
                )
            return self.image.shape[:2]
        elif self.has_image:
            return self.image.shape[:2]
        elif self.has_mask:
            return self.mask.shape[:2]
        else:
            raise ValueError("Cannot get image dimensions without image or mask")


# ========================================================================= #
# Base                                                                      #
# ========================================================================= #


class Augment(abc.ABC):
    """
    Based augment supporting:
    - images (float32, (H, W, 3) channel)
    - masks (float32, (H, W, 1) channel)
    - keypoints (float32, (N, 2) channel)
    """

    def __init__(self, p: float = 1.0):
        self._p = p

    @final
    def aug_image(self, image: AugImgHint) -> AugImgHint:
        return self.__call__(image=image).image

    @final
    def aug_mask(self, mask: AugMaskHint) -> AugMaskHint:
        return self.__call__(mask=mask).mask

    @final
    def aug_points(self, points: AugPointsHint) -> AugPointsHint:
        return self.__call__(points=points).points

    @final
    def __call__(
        self,
        *,
        image: AugImgHint | None = None,
        mask: AugMaskHint | None = None,
        points: AugPointsHint | None = None,
        seed: int | AugPrngHint | None = None,
    ) -> AugItems:
        # check inputs
        if all(x is None for x in [image, mask, points]):
            raise ValueError("At least one of image, mask, or points must be provided")
        # make items
        items = AugItems(image=image, mask=mask, points=points)
        # make prng
        if seed is None:
            prng = jrandom.key(np_random.randint(0, 2**32 - 1))
        elif isinstance(seed, int):
            prng = jrandom.key(seed)
        elif isinstance(seed, jax.Array):
            prng = seed  # key
        else:
            raise ValueError(
                f"Invalid seed type: {type(seed)}, must be int or RandomState"
            )
        # augment
        results = self._call(prng, items)
        # check results correspond
        for a, b, name in zip(items, results, ["image", "mask", "points"]):
            if (a is None) and (b is not None):
                raise ValueError(
                    f"Augment {self.__class__.__name__} returned a {name} but it was not provided as input."
                )
            elif (a is not None) and (b is None):
                raise ValueError(
                    f"Augment {self.__class__.__name__} did not return a {name} but it was provided as input."
                )
        # done!
        return results

    @final
    def _call(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        """
        If augments refer to each other, then this is the method to call NOT __call__.
        """
        # split the prng so that each augment gets a different seed
        # this is important for chaining augments together so that if an augment
        # is swapped out, the seed state of later augments does not change.
        return lax.cond(
            jrandom.uniform(prng) < self._p,
            lambda: self._apply(prng, x),
            lambda: x,
        )

    @abc.abstractmethod
    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        """
        Used to apply the augment to the input items OR chain augments together by
        referring to `self._call`

        Don't pass prng into child methods, rather split the PRNG for each child.
        Each new level of function calls should split the PRNG before going down a
        layer.
        - Always split at the START of _apply.
        - Always split before varying numbers of prng calls.
        """
        raise NotImplementedError


# ========================================================================= #
# Conditional Augments                                                      #
# ========================================================================= #


class AugmentItems(Augment):
    """
    Conditional augment that applies an augment if there is a value present
    """

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        # augment image
        image = x.image
        if x.has_image:
            image = self._augment_image(prng, x.image)
        # augment mask
        mask = x.mask
        if x.has_mask:
            mask = self._augment_mask(prng, x.mask)
        # augment points
        points = x.points
        if x.has_points:
            hw = x.get_bounds_hw()
            points = self._augment_points(prng, x.points)
        # done
        return x.override(image=image, mask=mask, points=points)

    def _augment_image(self, prng: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return image

    def _augment_mask(self, prng: AugPrngHint, mask: AugMaskHint) -> AugMaskHint:
        return mask

    def _augment_points(
        self, prng: AugPrngHint, points: AugPointsHint, hw: tuple[int, int]
    ) -> AugPointsHint:
        return points


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
