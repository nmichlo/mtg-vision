#  MIT License
#
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import random
from .. import util
from .files import init_dir

# ========================================================================= #
# Lazies                                                                    #
# TODO: remove, this is kind of silly                                       #
# ========================================================================= #


class Lazy(object):

    def __init__(self, resource, gen):
        self._resource = resource
        self._gen = gen
        self._obj = None

    def get(self, persist=False, force=False):
        if force or self._obj is None:
            obj = self._gen(self._resource)
            if persist:
                self._obj = obj
            return obj
        else:
            return self._obj


class LazyFile(object):
    def __init__(self, resource, local, gen_save, load):
        self._resource = resource
        self._local = local
        self._load = load
        self._gen_save = gen_save
        self._obj = None

    def get(self, persist=False, force=False):
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


class LazyList(object):
    def __init__(self, lazies, persist=False):
        self._lazies = lazies
        self._persist = persist

    def __len__(self):
        return len(self._lazies)

    def __getitem__(self, item):
        return self.get(item)

    def get(self, item, threads=1):
        dat = self._lazies[item]
        if type(item) == slice:
            return [x for x in util.run_threaded(lambda d: d.get(), dat, threads=threads)]
        else:
            return dat.get(persist=self._persist)

    def ran(self):
        return random.choice(self)


class PregenList(object):
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, item):
        return self._items[item]

    def get(self, item):
        return self._items[item]

    def ran(self):
        return random.choice(self)

