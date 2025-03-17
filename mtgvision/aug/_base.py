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

import numpy as np
import numpy.random as npr

from typing_extensions import final, NamedTuple


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


AugImgHint = np.ndarray[np.float32]
AugMaskHint = np.ndarray[np.float32]
AugPointsHint = np.ndarray[np.float32]
AugPrngHint = npr.Generator


_MISSING = object()


class AugItems(NamedTuple):
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
        # handle xor, not allowed to change from None to set or set to None
        if (image is _MISSING) != (self.image is None):
            raise ValueError("Cannot change image from None to set or set to None")
        if (mask is _MISSING) != (self.mask is None):
            raise ValueError("Cannot change mask from None to set or set to None")
        if (points is _MISSING) != (self.points is None):
            raise ValueError("Cannot change points from None to set or set to None")
        # override
        return AugItems(
            image=self.image if (image is _MISSING) else image,
            mask=self.mask if (mask is _MISSING) else mask,
            points=self.points if (points is _MISSING) else points,
        )


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
