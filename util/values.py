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

import numpy as np

# ============================================================================ #
# Helper Functions                                                             #
# ============================================================================ #


def asrt(x, msg='Assertion Failed'):
    assert x, msg
    return x

def asrt_in(x, lst):
    asrt(x in lst, '{}" must be in: {}'.format(x, lst))
    return x

def asrt_f16(x):
    asrt(type(x) == np.ndarray and x.dtype == np.float16)
    return x

def asrt_f32(x):
    asrt(type(x) == np.ndarray and x.dtype == np.float32)
    return x

def asrt_f64(x):
    asrt(type(x) == np.ndarray and x.dtype == np.float64)
    return x

def asrt_f128(x):
    asrt(type(x) == np.ndarray and x.dtype == np.float128)
    return x

def asrt_float(x):
    asrt(type(x) == np.ndarray and x.dtype in [np.float16, np.float32, np.float64, np.float128])
    return x



def default(x, default):
    return x if (x is not None) else (default() if callable(default) else default)

def defined_dict(**args):
    return { k: v for k, v in args.items() if v is not None }


# ============================================================================ #
# Singleton Interface                                                         #
# ============================================================================ #


class ISingleton(object):
    def __init__(self):
        raise Exception('This class is a Singleton!')