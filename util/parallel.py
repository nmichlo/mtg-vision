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

import multiprocessing as mp
from multiprocessing.pool import ThreadPool, Pool
from threading import Thread

# ========================================================================= #
# PARALLEL                                                                  #
# ========================================================================= #


def _starfunc(x: tuple):
    """Map a tuple: (func, *args) to the function: func(*args)"""
    return x[0](*x[1] if type(x[1]) == tuple else x[1])


_POOL = None
_DISABLE_PARALLEL = False


def run_parallel(func: callable, task_iter: iter) -> iter:
    if type(task_iter) == int:
        task_iter = range(task_iter)
    tasks = ((func, task) for task in task_iter)

    global _DISABLE_PARALLEL
    if _DISABLE_PARALLEL:
        return (_starfunc(task) for task in tasks)
    else:
        global _POOL
        if _POOL is None:
            _POOL = Pool(mp.cpu_count())
        return _POOL.imap_unordered(_starfunc, tasks)


def run_threaded(func: callable, task_iter: iter, threads=mp.cpu_count()):
    if type(task_iter) == int:
        task_iter = range(task_iter)

    if threads < 2:
        return (func(task) for task in task_iter)
    else:
        with ThreadPool(processes=threads) as pool:
            tasks = ((func, task) for task in task_iter)
            return pool.imap_unordered(_starfunc, tasks)


class ThreadLoop:
    def __init__(self, update=None):
        self.stopped = False
        if callable(update):
            self.update = update

    def start(self):
        Thread(target=self.loop, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def loop(self):
        while True:
            if self.stopped:
                return
            self.update()

    def update(self):
        raise NotImplementedError('Implement Me')