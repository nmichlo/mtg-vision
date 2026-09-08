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


from __future__ import annotations

import abc
import random
import warnings
from abc import ABC
from collections.abc import Callable

# a step in an augmentation pipeline; `None` is a no-op
type Transform[T] = Callable[[T], T] | None
# the concrete sequence types `Applicator.__init__` unpacks as a single argument
# -- kept concrete (not `Sequence[Transform[T]]`) so `isinstance` fully narrows it
type _Transforms[T] = list[Transform[T]] | set[Transform[T]] | tuple[Transform[T], ...]


def seed_all(seed: int) -> None:
    # random
    random.seed(seed)
    # numpy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        warnings.warn("numpy not found, skipping seed")
    # torch
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        warnings.warn("torch not found, skipping seed")


# ============================================================================ #
# Random Application of Functions                                              #
# ============================================================================ #


class Applicator[T](ABC):
    def __init__(self, *callables: Transform[T] | _Transforms[T]) -> None:
        # a single list/set/tuple argument is unpacked; anything else is varargs
        if len(callables) == 1 and isinstance(callables[0], (list, set, tuple)):
            items: list[Transform[T]] = list(callables[0])
        else:
            items = []
            for c in callables:
                assert not isinstance(c, (list, set, tuple))
                items.append(c)
        if len(items) < 1:
            raise RuntimeError("There must be a callable")
        self.callables: list[Transform[T]] = items

    def __call__(self, x: T) -> T:
        return self._apply(x)

    @staticmethod
    def _call(c: Transform[T], x: T) -> T:
        if c is None:
            return x
        elif callable(c):
            return c(x)
        else:
            raise RuntimeError(f"Unsupported Callable Type: {type(c)}")

    @abc.abstractmethod
    def _apply(self, x: T) -> T:
        pass


class ApplyOrdered[T](Applicator[T]):
    def _apply(self, x: T) -> T:
        for c in self.callables:
            x = Applicator._call(c, x)
        return x


class ApplyShuffled[T](Applicator[T]):
    def __init__(self, *callables: Transform[T] | _Transforms[T]) -> None:
        super().__init__(*callables)
        self.indices: list[int] = list(range(len(self.callables)))

    def _apply(self, x: T) -> T:
        random.shuffle(self.indices)
        for i in self.indices:
            x = Applicator._call(self.callables[i], x)
        return x


class ApplyChoice[T](Applicator[T]):
    def _apply(self, x: T) -> T:
        return Applicator._call(random.choice(self.callables), x)
