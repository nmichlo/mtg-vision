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


import os
import random
from pathlib import Path
from typing import Callable, Generic, TypeVar

from mtgvision.util.files import init_dir
import mtgvision.util.parallel as upar

# ========================================================================= #
# Lazies                                                                    #
# TODO: remove, this is kind of silly                                       #
# ========================================================================= #


K = TypeVar("K")
V = TypeVar("V")


class ILazy(Generic[V]):
    def get(self, persist: bool = False, force: bool = False) -> V:
        raise NotImplementedError()


class LazyValue(Generic[K, V], ILazy[V]):
    def __init__(self, resource: K, gen: Callable[[K], V]):
        self._resource = resource
        self._gen = gen
        self._obj = None

    @property
    def resource(self) -> K:
        return self._resource

    def get(self, persist: bool = False, force: bool = False) -> V:
        if force or self._obj is None:
            obj = self._gen(self._resource)
            if persist:
                self._obj = obj
            return obj
        else:
            return self._obj


class LazyFile(Generic[K, V], ILazy[V]):
    def __init__(
        self,
        resource: K,
        local: Path,
        gen_save: Callable[[K, Path], None],
        load: Callable[[Path], V],
    ):
        self._resource = resource
        self._local = local
        self._load = load
        self._gen_save = gen_save
        self._obj = None

    @property
    def resource(self) -> K:
        return self._resource

    def get(self, persist: bool = False, force: bool = False) -> V:
        if force or self._obj is None:
            if force or not os.path.exists(self._local):
                init_dir(self._local, is_file=True)
                self._gen_save(self._resource, self._local)
            obj = self._load(self._local)
            if persist:
                self._obj = obj
            return obj
        else:
            return self._obj


class LazyList(Generic[V]):
    def __init__(self, lazies: list[ILazy[V]], persist: bool = False):
        self._lazies = lazies
        self._persist = persist

    def __len__(self):
        return len(self._lazies)

    def __getitem__(self, item):
        return self.get(item)

    def get(self, idx: int | slice, threads: int = 1):
        dat = self._lazies[idx]
        if isinstance(idx, slice):
            return [
                x for x in upar.run_threaded(lambda d: d.get(), dat, threads=threads)
            ]
        else:
            return dat.get(persist=self._persist)

    def ran(self) -> V:
        return random.choice(self)


class PregenList(Generic[V]):
    def __init__(self, items: list[V]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, item) -> V:
        return self._items[item]

    def get(self, idx: int | slice) -> V:
        return self._items[idx]

    def ran(self) -> V:
        return random.choice(self)
