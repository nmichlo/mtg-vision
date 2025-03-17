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
    "ArgIntRange",
    "ArgFloatRange",
    "ArgIntHint",
    "ArgFloatHint",
]

from typing import NamedTuple

import numpy as np

from mtgvision.aug import AugPrngHint


# ========================================================================= #
# Hints                                                                     #
# ========================================================================= #


ArgIntHint = int | tuple[int, int]
ArgFloatHint = float | tuple[float, float]


# ========================================================================= #
# Ranges                                                                    #
# ========================================================================= #


class ArgIntRange(NamedTuple):
    """
    Represents a range of integer values that can be sampled from.
    """

    low: int
    high: int

    @classmethod
    def from_arg(
        cls,
        value: ArgIntHint,
        *,
        min_val: int | None = None,
        max_val: int | None = None,
        default_val: int | None = None,
        allow_equal: bool = True,
    ) -> "ArgIntRange":
        # get values
        if isinstance(value, tuple):
            low, high = value
        elif isinstance(value, int):
            low = high = value
        elif value is None:
            if default_val is None:
                raise RuntimeError("Int value cannot be None, if default value is None")
            low = high = default_val
        else:
            raise TypeError(f"Invalid type for int value: {type(value)}")
        # check values
        if min_val is not None and max_val is not None:
            if min_val > max_val:
                raise ValueError(
                    f"min_val: {min_val} must be less than or equal to max_val: {max_val}"
                )
        if min_val is not None and low < min_val:
            raise ValueError(
                f"low int value: {low} must be greater than or equal to {min_val}"
            )
        if max_val is not None and high > max_val:
            raise ValueError(
                f"high int value: {high} must be less than or equal to {max_val}"
            )
        if low > high:
            raise ValueError(
                f"low int value: {low} must be less than or equal to high int value: {high}"
            )
        if not allow_equal and low == high:
            raise ValueError(
                f"low int value: {low} must not be equal to high int value: {high}"
            )
        return cls(low=low, high=high)

    def sample(self, prng: AugPrngHint, size=None):
        return prng.integers(self.low, self.high + 1, size=size)  # excludes high


class ArgFloatRange(NamedTuple):
    """
    Represents a range of float values that can be sampled from.
    """

    low: float
    high: float

    @classmethod
    def from_arg(
        cls,
        value: ArgFloatHint,
        *,
        min_val: float | None = None,
        max_val: float | None = None,
        default_val: float | None = None,
        allow_equal: bool = True,
    ) -> "ArgFloatRange":
        # get values
        if isinstance(value, tuple):
            low, high = value
        elif isinstance(value, (float, int)):
            low = high = value
        elif value is None:
            if default_val is None:
                raise RuntimeError(
                    "Float value cannot be None, if default value is None"
                )
            low = high = default_val
        else:
            raise TypeError(f"Invalid type for float value: {type(value)}")
        # check values
        if min_val is not None and max_val is not None:
            if min_val > max_val:
                raise ValueError(
                    f"min_val: {min_val} must be less than or equal to max_val: {max_val}"
                )
        if min_val is not None and low < min_val:
            raise ValueError(
                f"low float value: {low} must be greater than or equal to {min_val}"
            )
        if max_val is not None and high > max_val:
            raise ValueError(
                f"high float value: {high} must be less than or equal to {max_val}"
            )
        if low > high:
            raise ValueError(
                f"low float value: {low} must be less than or equal to high float value: {high}"
            )
        if not allow_equal and low == high:
            raise ValueError(
                f"low float value: {low} must not be equal to high float value: {high}"
            )
        return cls(low=low, high=high)

    def sample(self, prng: AugPrngHint, size=None):
        val = prng.uniform(self.low, self.high, size=size)
        if isinstance(val, np.ndarray):
            val = val.astype(np.float32)
        return val


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
