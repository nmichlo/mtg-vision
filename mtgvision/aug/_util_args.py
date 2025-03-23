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
    "ArgIntHint",
    "ArgFloatHint",
    "ArgStrLiteralsHint",
]

import jax.numpy as jnp
import jax.random as jrandom

from mtgvision.aug import AugPrngHint


# ========================================================================= #
# Hints                                                                     #
# ========================================================================= #

ArgIntHint = int | tuple[int, int]
ArgFloatHint = float | tuple[float, float]
ArgStrLiteralsHint = str | tuple[str, ...]


def _get_default(value, default):
    if value is None:
        m = M = default
    else:
        m, M = value
        if m is None:
            m = default
        if M is None:
            M = default
    return m, M


def sample_float_defaults(
    key: AugPrngHint, value: ArgFloatHint, default: float, shape=()
) -> float:
    value = _get_default(value, default)
    return sample_float(key, value, shape)


def sample_float(key: AugPrngHint, value: ArgFloatHint, shape=()) -> float:
    if isinstance(value, (tuple, list)):
        m, M = value
        return jrandom.uniform(key, shape, jnp.float32, minval=m, maxval=M)
    else:
        return jnp.full(shape, value)


def sample_int_defaults(
    key: AugPrngHint, value: ArgIntHint, default: int, shape=()
) -> int:
    value = _get_default(value, default)
    return sample_int(key, value, shape)


def sample_int(key: AugPrngHint, value: ArgIntHint, shape=()) -> int:
    if isinstance(value, (tuple, list)):
        m, M = value
        return jrandom.randint(key, shape, m, M + 1, jnp.int32)
    else:
        return jnp.full(shape, value)


def sample_str(key: AugPrngHint, value: ArgStrLiteralsHint) -> str:
    if isinstance(value, (tuple, list)):
        return jrandom.choice(key, value)
    return value


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
