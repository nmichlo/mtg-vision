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

import time


# ============================================================================ #
# Global Random Singleton                                                      #
# ============================================================================ #


class GLOBAL_TIMER:

    _timers = {}
    _init_time = time.time()

    @staticmethod
    def time():
        return time.time() - GLOBAL_TIMER._init_time

    @staticmethod
    def start(name):
        GLOBAL_TIMER._timers[name] = time.time()

    @staticmethod
    def stop(name):
        old = GLOBAL_TIMER._timers[name]
        del GLOBAL_TIMER._timers[name]
        return time.time() - old

    @staticmethod
    def finish(name):
        print("{}: {}".format(name, GLOBAL_TIMER.stop(name)))
