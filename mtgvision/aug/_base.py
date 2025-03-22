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
    "JnpFloat32",
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
from dataclasses import dataclass, field
from typing import Callable, Optional, Type, TypeVar

import jax
import jax.random as jrandom
import numpy.random as np_random
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_dataclass
from tqdm import tqdm
from typing_extensions import final


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


JnpFloat32 = jax.Array

AugImgHint = JnpFloat32
AugMaskHint = JnpFloat32
AugPointsHint = JnpFloat32
AugPrngHint = jax.Array


T = TypeVar("T")


# dataclass field that is STATIC for jax
def jax_static_field(**kwargs) -> Type[T]:
    return field(**kwargs, metadata=dict(static=True))


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


_MISSING = object()


@register_dataclass
@dataclass(frozen=True)
class AugItems:
    """
    Named tuple for holding augment items that are passed between augments and returned.
    """

    image: Optional[AugImgHint] = None
    mask: Optional[AugMaskHint] = None
    points: Optional[AugPointsHint] = None

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


@register_dataclass
@dataclass(frozen=True)
class Augment(abc.ABC):
    """
    Based augment supporting:
    - images (float32, (H, W, 3) channel)
    - masks (float32, (H, W, 1) channel)
    - keypoints (float32, (N, 2) channel)

    All subclasses should be wrapped with:
    ```
    @register_dataclass
    @dataclass(frozen=True)
    class Subclass(Augment):
        ...
    ```
    """

    p: float = jax_static_field(default=1.0)

    def jitted_call(self) -> "Callable[[AugPrngHint, AugItems], AugItems]":
        return jax.jit(self.__call__)

    def easy_apply(
        self,
        *,
        image: AugImgHint | None = None,
        mask: AugMaskHint | None = None,
        points: AugPointsHint | None = None,
        seed: int | AugPrngHint | None = None,
        jit: bool = True,
    ) -> AugItems:
        # check inputs
        if all(x is None for x in [image, mask, points]):
            raise ValueError("At least one of image, mask, or points must be provided")
        # make items
        items = AugItems(image=image, mask=mask, points=points)
        # make prng key
        if seed is None:
            key = jrandom.key(np_random.randint(0, 2**32 - 1))
        elif isinstance(seed, int):
            key = jrandom.key(seed)
        elif isinstance(seed, jax.Array):
            key = seed  # key
        else:
            raise ValueError(
                f"Invalid seed type: {type(seed)}, must be int or RandomState"
            )
        # augment
        results = self.__call__(key, items)
        # check results correspond
        for name in ["image", "mask", "points"]:
            a = getattr(items, name)
            b = getattr(results, name)
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

    # DON'T CALL THIS DIRECTLY DOWNSTREAM
    def __call__(self, key: AugPrngHint, x: AugItems) -> AugItems:
        """
        If augments refer to each other, then this is the method to call NOT _apply.
        """
        # split the key so that each augment gets a different prng seed
        # this is important for chaining augments together so that if an augment
        # is swapped out, the seed state of later augments does not change.
        key, subkey = jrandom.split(key)
        return lax.cond(
            jrandom.uniform(key) < self.p,
            self._apply,
            lambda key, x: x,
            subkey,
            x,
        )

    @abc.abstractmethod
    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        """
        Used to apply the augment to the input items OR chain augments together by
        referring to `self._call`

        Don't pass key into child methods, rather split the PRNG for each child.
        Each new level of function calls should split the PRNG before going down a
        layer.
        - Always split at the START of _apply.
        - Always split before varying numbers of prng calls.
        """
        raise NotImplementedError

    def quick_test(
        self,
        image_size=(192, 128, 3),
        mask_size=(192, 128, 1),
        points_size=(10, 2),
        n: Optional[int] = 1000,
        jit: bool = True,
    ):
        key = jrandom.key(42)
        img = jrandom.uniform(key, image_size, jnp.float32)
        mask = jrandom.uniform(key, mask_size, jnp.float32)
        points = jrandom.uniform(key, points_size, jnp.float32)
        items = AugItems(image=img, mask=mask, points=points)
        call = self.jitted_call() if jit else self
        if n is not None:
            for _ in tqdm(range(n), desc=f"{self.__class__.__name__}"):
                call(key, items)
        else:
            call(key, items)


# ========================================================================= #
# Conditional Augments                                                      #
# ========================================================================= #


@register_dataclass
@dataclass(frozen=True)
class AugmentItems(Augment, abc.ABC):
    """
    Conditional augment that applies an augment if there is a value present
    """

    @final
    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        # augment image
        image = x.image
        if x.has_image:
            image = self._augment_image(key, x.image)
        # augment mask
        mask = x.mask
        if x.has_mask:
            mask = self._augment_mask(key, x.mask)
        # augment points
        points = x.points
        if x.has_points:
            hw = x.get_bounds_hw()
            points = self._augment_points(key, x.points, hw)
        # done
        return x.override(image=image, mask=mask, points=points)

    def _augment_image(self, key: AugPrngHint, image: AugImgHint) -> AugImgHint:
        return image

    def _augment_mask(self, key: AugPrngHint, mask: AugMaskHint) -> AugMaskHint:
        return mask

    def _augment_points(
        self, key: AugPrngHint, points: AugPointsHint, hw: tuple[int, int]
    ) -> AugPointsHint:
        return points


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
