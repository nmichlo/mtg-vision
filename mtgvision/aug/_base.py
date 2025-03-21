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
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

import jax
import jax.random as jrandom
import numpy.random as np_random
import jax.numpy as jnp
from jax import lax
from jax._src.tree_util import register_dataclass
from tqdm import tqdm

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
# Helper                                                                    #
# ========================================================================= #


# NOTE: by convention for jax, the first arg is usually the one that is differentiated.
# - (params, start_params) -> float
LossModel = Callable[[NamedTuple, NamedTuple, NamedTuple], float]

# - (params, pts) -> transformed_pts
ParamModel = Callable[[NamedTuple, NamedTuple, "xnp.ndarray"], "xnp.ndarray"]

# - (pts) -> transformed_pts   |   not intended for differentiation
FullModel = Callable[["xnp.ndarray"], "xnp.ndarray"]


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


T = TypeVar("T")


def jax_static_field(**kwargs) -> Type[T]:
    return field(**kwargs, metadata=dict(static=True))


@register_dataclass
@dataclass
class Items:
    image: jax.Array = None
    mask: jax.Array = None
    points: jax.Array = None


@register_dataclass
@dataclass
class ParamsRanSeq:
    augments: list[Any] = jax_static_field(default_factory=list)
    n_min: Optional[int] = jax_static_field(default=None)
    n_max: Optional[int] = jax_static_field(default=None)
    shuffle: bool = jax_static_field(default=False)

    def apply(self, key, items: Items) -> Items:
        # 0. get bounds, default is always number of augments
        m = len(self.augments) if self.n_min is None else self.n_min
        M = len(self.augments) if self.n_max is None else self.n_max
        # 1. get random order of ALL augments
        #    jax struggles with indexing based on a traced array. So we need to use
        #    switch to select the correct augment based on the random index
        order = jrandom.choice(key, len(self.augments), (M,), replace=False)
        branches = [lambda k, itms: aug.apply(k, itms) for aug in self.augments]
        branch = lambda i, itms: jax.lax.switch(
            order[i], branches, key, itms
        )  # is order correctly re-done each time?
        # 2. choose number of augments to apply, and apply them
        n = jrandom.randint(key, (), m, M)
        # 3. loop and apply
        return lax.fori_loop(0, n, branch, items)


@register_dataclass
@dataclass
class ParamsBrightness:
    brightness_min: float = jax_static_field(default=-0.1)
    brightness_max: float = jax_static_field(default=0.1)

    def apply(self, key, items: Items) -> Items:
        brightness = jrandom.uniform(
            key,
            minval=self.brightness_min,
            maxval=self.brightness_max,
        )
        items.image += brightness
        return items


@register_dataclass
@dataclass
class ParamsExposure:
    exposure_min: float = jax_static_field(default=-0.1)
    exposure_max: float = jax_static_field(default=0.1)

    def apply(self, key, items: Items) -> Items:
        brightness = jrandom.uniform(
            key,
            minval=self.exposure_min,
            maxval=self.exposure_max,
        )
        items.image *= brightness
        return items


if __name__ == "__main__":
    key = jrandom.key(42)
    img = jrandom.uniform(key, (192, 128, 3), jnp.float32)

    aug = ParamsRanSeq(augments=[ParamsBrightness(), ParamsExposure()])

    items = Items(image=img)
    aug.apply = jax.jit(aug.apply)

    for i in tqdm(range(100000)):
        items_new = aug.apply(key, items)


#
# class AbstractModelMaker(abc.ABC):
#     # <private: model>
#     # - annotated with ClassVar to make sure pydantic doesn't use it
#     # - base config classes should NOT share the same Params type, this is so we can link the param type to the config uniquely
#     # * static variables should always be hashable, eg. int, str, tuple, etc. (not list)
#     Params: ClassVar[T]
#
#     # these should point to static functions that can be JIT compiled with jax and thus
#     # cached! dynamic dispatch should be handled by examining the params.
#     AUGMENT_FN: ClassVar[ParamModel]
#     RANDOM_
#
#
#     MODEL_BACKWARD: ClassVar[ParamModel]
#     MODEL_LOSS: ClassVar[LossModel] = staticmethod(noop_loss)
#
#     def tree_flatten_with_keys(self):
#         return (((GetAttrKey('x'), self.x), (GetAttrKey('y'), self.y)), None)
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
