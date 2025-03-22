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
    "OneOf",
    "AllOf",
    "SomeOf",
    "SampleOf",
]

import abc
from dataclasses import dataclass

import jax
import jax.random as jrandom
from jax import lax
from jax.tree_util import register_dataclass

from mtgvision.aug._base import AugItems, Augment, AugPrngHint, jax_static_field


# ========================================================================= #
# Base Chain Augments                                                       #
# ========================================================================= #


@dataclass(frozen=True)
class _Chain(Augment, abc.ABC):
    p: float = jax_static_field(default=1.0)
    augments: tuple[Augment, ...] = jax_static_field(default_factory=list)


@dataclass(frozen=True)
class _Shuffle(_Chain, abc.ABC):
    p: float = jax_static_field(default=1.0)
    n_min: int | None = jax_static_field(default=None)
    n_max: int | None = jax_static_field(default=None)
    replace: bool = jax_static_field(default=False)

    def _apply(self, key: AugPrngHint, items: AugItems) -> AugItems:
        """
        Sample the augments to apply, with or without replacement.
        """
        # 0. get bounds, default is always number of augments
        m = len(self.augments) if self.n_min is None else self.n_min
        M = len(self.augments) if self.n_max is None else self.n_max
        # 1. choose number of augments to apply, and apply them
        n = jrandom.randint(key, (), m, M)
        # 2. get random order of ALL augments, TODO: should be (n,)
        order = jrandom.choice(key, len(self.augments), (M,), replace=self.replace)
        # 3. loop and apply
        #    jax struggles with indexing based on a traced array. So we need to use
        #    switch to select the correct augment based on the random index
        return lax.fori_loop(
            0,
            n,
            lambda i, itms: jax.lax.switch(order[i], self.augments, key, itms),
            items,
        )


# ========================================================================= #
# Chain Augments                                                            #
# ========================================================================= #


# one of the augments
@register_dataclass
@dataclass(frozen=True)
class OneOf(_Chain):
    p: float = jax_static_field(default=1.0)

    def _apply(self, key: AugPrngHint, items: AugItems) -> AugItems:
        # DOES NOT SUPPORT len(augments) == 0
        i = jrandom.randint(key, (), 0, len(self.augments))
        return lax.switch(i, self.augments, key, items)


# all augments
@register_dataclass
@dataclass(frozen=True)
class AllOf(_Chain):
    p: float = jax_static_field(default=1.0)

    def _apply(self, key: AugPrngHint, items: AugItems) -> AugItems:
        for aug in self.augments:
            items = aug(key, items)
        return items


# sample without replacement
@register_dataclass
@dataclass(frozen=True)
class SomeOf(_Shuffle):
    # DOES NOT SUPPORT len(augments) == 0
    p: float = jax_static_field(default=1.0)
    replace: bool = jax_static_field(default=False)


# sample with replacement
@register_dataclass
@dataclass(frozen=True)
class SampleOf(_Shuffle):
    # DOES NOT SUPPORT len(augments) == 0
    p: float = jax_static_field(default=1.0)
    replace: bool = jax_static_field(default=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    def _main():
        from mtgvision.aug import NoOp

        # test
        from mtgvision.aug._aug_chain import OneOf
        from mtgvision.aug._aug_chain import AllOf
        from mtgvision.aug._aug_chain import SomeOf
        from mtgvision.aug._aug_chain import SampleOf

        # augs
        for n in [16, 4, 1, 0]:
            print()
            print(n)
            print()
            augs = [NoOp() for _ in range(n)]
            # test
            AllOf(augments=augs).quick_test()
            if n != 0:
                OneOf(augments=augs).quick_test()
                SomeOf(augments=augs).quick_test()
                SampleOf(augments=augs).quick_test()

    _main()
