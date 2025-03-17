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
]

import abc

import cv2
import numpy as np
import numpy.random as npr

from typing_extensions import final, Literal, NamedTuple


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


NpFloat32 = np.ndarray[np.float32]

AugImgHint = NpFloat32
AugMaskHint = NpFloat32
AugPointsHint = NpFloat32
AugPrngHint = npr.Generator


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
        # checks
        if image is not None and np.any(np.isnan(image)):
            raise ValueError("Image contains NaN values")
        if mask is not None and np.any(np.isnan(mask)):
            raise ValueError("Mask contains NaN values")
        if points is not None and np.any(np.isnan(points)):
            raise ValueError("Points contains NaN values")
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

    def applied_rotation_matrix(
        self,
        M: np.ndarray,
        *,
        mode: Literal["affine", "perspective"],
        inter: int = cv2.INTER_LINEAR,
        inter_mask: int = cv2.INTER_NEAREST,
    ) -> "AugItems":
        if mode == "affine":
            warp = cv2.warpAffine
            invert_transform = cv2.invertAffineTransform
            transform = cv2.transform
        else:
            warp = cv2.warpPerspective
            invert_transform = np.linalg.inv
            transform = cv2.perspectiveTransform
        # Get dimensions
        h, w = self.get_bounds_hw()
        # Warp image and mask
        im = warp(self.image, M, (w, h), flags=inter) if self.has_image else None
        mask = warp(self.mask, M, (w, h), flags=inter_mask) if self.has_mask else None
        # Transform points
        points = None
        if self.has_points:
            M_inv = invert_transform(M)
            points = transform(self.points[None, :, :], M_inv)[0]
        # done
        return self.override(image=im, mask=mask, points=points)


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
        if seed is None or isinstance(seed, int):
            prng = npr.default_rng(seed)
        elif isinstance(seed, npr.Generator):
            prng = seed
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
        if prng.uniform() < self._p:
            # split the prng so that each augment gets a different seed
            # this is important for chaining augments together so that if an augment
            # is swapped out, the seed state of later augments does not change.
            return self._apply(self._child_prng(prng), x)
        else:
            return x

    @abc.abstractmethod
    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        """
        Used to apply the augment to the input items OR chain augments together by
        referring to `self._call`

        Don't pass prng into child methods, rather split the PRNG for each child.
        Each new level of function calls should split the PRNG before going down a
        layer.

        e.g. _call splits the PRNG and calls _apply
        e.g. _apply splits the PRNG and passes this to _call, then uses the original PRNG for its own operations.
        """
        raise NotImplementedError

    @classmethod
    @final
    def _child_prng(cls, prng: AugPrngHint) -> AugPrngHint:
        """
        Like jax.random.split, but for numpy.random.RandomState. This is used when
        chaining augmentations together to ensure that each augmentation gets a
        different random seed. This is useful because it means swapping out augments
        does not change the seed state of later augments.
        """
        return npr.default_rng(prng.integers(2**32))

    @classmethod
    @final
    def _n_child_prng(cls, prng: AugPrngHint, n: int) -> list[AugPrngHint]:
        return [cls._child_prng(prng) for _ in range(n)]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
