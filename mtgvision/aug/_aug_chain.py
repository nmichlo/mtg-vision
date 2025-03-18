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

import jax.random as jrandom

from mtgvision.aug._base import AugItems, Augment, AugPrngHint
from mtgvision.aug._util_args import ArgIntHint, ArgIntRange


# ========================================================================= #
# Base Chain Augments                                                       #
# ========================================================================= #


class _Chain(Augment):
    def __init__(
        self,
        *augments: Augment,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self._augments = list(augments or ())

    def _get_augments(self, prng: AugPrngHint):
        return self._augments

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        augments = self._get_augments(prng)
        for augment in augments:
            _, split = jrandom.split(prng)
            x = augment._call(split, x)
        return x


class _Shuffle(_Chain):
    def __init__(
        self,
        *augments: Augment,
        p: float = 1.0,
        n: ArgIntHint | None = None,
        replace: bool = False,
    ):
        super().__init__(*augments, p=p)
        self._n = ArgIntRange.from_arg(
            n,
            min_val=0,
            max_val=len(augments),
            clip_max=True,
            default_val=len(augments),
        )
        self._replace = replace

    def _get_augments(self, prng: AugPrngHint):
        """
        Sample the augments to apply, with or without replacement.
        """
        k = self._n.sample(prng)
        idxs = jrandom.choice(prng, len(self._augments), (k,), replace=self._replace)
        return [self._augments[i] for i in idxs]


# ========================================================================= #
# Chain Augments                                                            #
# ========================================================================= #


# one of the augments
class OneOf(_Chain):
    def _get_augments(self, prng: AugPrngHint):
        idx = jrandom.choice(prng, len(self._augments))
        return [self._augments[idx]]


# all augments
class AllOf(_Chain):
    def _get_augments(self, prng: AugPrngHint):
        return self._augments


# sample without replacement
class SomeOf(_Shuffle):
    def __init__(
        self,
        *augments: Augment,
        p: float = 1.0,
        n: int | tuple[int, int] | None = None,
    ):
        super().__init__(*augments, p=p, n=n, replace=False)


# sample with replacement
class SampleOf(_Shuffle):
    def __init__(
        self,
        *augments: Augment,
        p: float = 1.0,
        n: int | tuple[int, int] | None = None,
    ):
        super().__init__(*augments, p=p, n=n, replace=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
