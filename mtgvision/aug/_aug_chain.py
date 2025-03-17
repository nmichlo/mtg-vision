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

import warnings
from mtgvision.aug._base import AugItems, Augment, AugPrngHint


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
        if not augments:
            raise ValueError("At least one augment must be provided")
        self._augments = augments

    def _get_augments(self, prng: AugPrngHint):
        return self._augments

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        for augment in self._get_augments(prng):
            x = augment._call(self._child_prng(prng), x)
        return x


class _Shuffle(_Chain):
    def __init__(
        self,
        *augments: Augment,
        p: float = 1.0,
        n: int | tuple[int, int] | None = None,
        replace: bool = False,
    ):
        super().__init__(*augments, p=p)
        # CHECK N: int
        if n is None:
            n = len(augments)
        if isinstance(n, int):
            n = (n, n)
        if len(n) != 2:
            raise ValueError("n must be an int or a tuple of two ints")
        # CHECK N: (LOW,HIGH)
        low, high = n
        if low < 1:
            raise ValueError(f"(low, high) must be greater than 0, got: {n}")
        if low > high:
            raise ValueError(f"(low, high) must be in increasing order, got: {n}")
        if high > len(augments):
            warnings.warn(
                "high is greater than the number of augments, reducing to the number of augments"
            )
            high = len(augments)
        if low > len(augments):
            warnings.warn(
                "low is greater than the number of augments, reducing to the number of augments"
            )
            low = len(augments)
        # SET N
        self._n_low = low
        self._n_high = high
        self._replace = replace

    def _get_augments(self, prng: AugPrngHint):
        """
        Sample the augments to apply, with or without replacement.
        """
        k = prng.integers(self._n_low, self._n_high + 1)
        return prng.choice(self._augments, size=k, replace=self._replace)


# ========================================================================= #
# Chain Augments                                                            #
# ========================================================================= #


# one of the augments
class OneOf(_Chain):
    def _get_augments(self, prng: AugPrngHint):
        return [prng.choice(self._augments)]


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
