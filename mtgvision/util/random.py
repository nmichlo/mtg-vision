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


import abc
import warnings
from abc import ABC
import random
from functools import partial


def seed_all(seed: int):
    # random
    random.seed(seed)
    # numpy
    try:
        import np

        np.random.seed(seed)
    except (ImportError, ModuleNotFoundError):
        warnings.warn("numpy not found, skipping seed")

    # torch
    try:
        import torch

        torch.manual_seed(seed)
    except (ImportError, ModuleNotFoundError):
        warnings.warn("torch not found, skipping seed")


# ============================================================================ #
# Random Application of Functions                                              #
# ============================================================================ #


class Applicator(ABC):
    def __init__(self, *callables, p: float = 1.0):
        self.p = p
        self.callables = callables
        if len(self.callables) == 1 and type(self.callables[0]) in [list, set, tuple]:
            self.callables = self.callables[0]
        if len(self.callables) < 1:
            raise RuntimeError("There must be a callable")

    def __call__(self, x):
        if random.random() < self.p:
            return self._apply(x)
        else:
            return x

    @staticmethod
    def _call(c, x):
        if c is None:
            return x
        elif callable(c):
            return c(x)
        else:
            raise RuntimeError("Unsupported Callable Type: {}".format(type(c)))

    @abc.abstractmethod
    def _apply(self, x):
        raise NotImplementedError


class NoOp(Applicator):
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def _apply(self, x):
        return x


class ApplyFn(Applicator):
    def __init__(self, callable, *args, p: float = 1.0, **kwargs):
        callable = partial(callable, *args, **kwargs)
        super().__init__(callable, p=p)

    def _apply(self, x):
        return Applicator._call(self.callables[0], x)


class ApplyChance(Applicator):
    def __init__(self, *callables, p: float = 0.5):
        super().__init__(callables, p=p)
        self.p = p

    def _apply(self, x):
        for c in self.callables:
            x = Applicator._call(c, x)
        return x


class ApplySequence(Applicator):
    def _apply(self, x):
        for c in self.callables:
            x = Applicator._call(c, x)
        return x


class ApplyShuffled(Applicator):
    def __init__(self, *callables, p: float = 1.0):
        super().__init__(callables, p=p)
        self.indices = list(range(len(self.callables)))

    def _apply(self, x):
        indices = list(self.indices)
        random.shuffle(indices)
        for i in indices:
            x = Applicator._call(self.callables[i], x)
        return x


class ApplyOne(Applicator):
    def _apply(self, x):
        return Applicator._call(random.choice(self.callables), x)
