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
    def __init__(self, *callables):
        self.callables = callables
        if len(self.callables) == 1 and type(self.callables[0]) in [list, set, tuple]:
            self.callables = self.callables[0]
        if len(self.callables) < 1:
            raise RuntimeError("There must be a callable")

    def __call__(self, x):
        return self._apply(x)

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
        pass


class ApplyOrdered(Applicator):
    def _apply(self, x):
        for c in self.callables:
            x = Applicator._call(c, x)
        return x


class ApplyShuffled(Applicator):
    def __init__(self, *callables):
        super().__init__(callables)
        self.indices = list(range(len(self.callables)))

    def _apply(self, x):
        random.shuffle(self.indices)
        for i in self.indices:
            x = Applicator._call(self.callables[i], x)
        return x


class ApplyChoice(Applicator):
    def _apply(self, x):
        return Applicator._call(random.choice(self.callables), x)
