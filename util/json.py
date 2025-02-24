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
import jsonpickle
from .files import init_dir
import json


# ============================================================================ #
# Json File Cache                                                              #
# ============================================================================ #

CACHE = True

# # TODO, make this easier to use... and more reliable
class JsonCache(object):
    def __init__(self, cache_file, refresh=not CACHE):
        self.path = cache_file
        self.data = None
        self.refresh = refresh
        self.save = True

    def __enter__(self) -> dict:
        init_dir(self.path, is_file=True)
        if self.refresh or not os.path.isfile(self.path):
            self.data = {}
            self.refresh = False
        else:
            try:
                with open(self.path, 'r') as file_stream:
                    self.data = json.load(file_stream)
            except Exception as e:
                self.data = {}
                print("WARNING: Error loading cache: {} ({})".format(self.path, e))
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        init_dir(self.path, is_file=True)
        if self.save:
            with open(self.path, 'w') as file_stream:
                json.dump(self.data, file_stream)
        else:
            print('Skipping Save Cache: {}'.format(self.path))
        self.data = None

    # @staticmethod
    # def __call__(self, cache_file):
    #     def wrap(fn):
    #         def do(*args, **kwargs):
    #             with JsonCache(cache_file) as data:
    #                 if 'data' not in data:
    #                     data['data'] = fn(*args, **kwargs)
    #                 return data['data']
    #         return do
    #     return wrap


# ============================================================================ #
# Json Pickle                                                                  #
# ============================================================================ #


# class JsonPickler:
#
#     @staticmethod
#     def loads(string):
#         return jsonpickle.decode(string)
#
#     @staticmethod
#     def load(file_name):
#         with open(file_name, 'r') as file:
#             string = file.read()
#         return JsonPickler.loads(string)
#
#     @staticmethod
#     def dumps(obj):
#         jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
#         return jsonpickle.encode(obj)
#
#     @staticmethod
#     def dump(obj, file_name):
#         init_dir(file_name, is_file=True)
#         with open(file_name, 'w') as file:
#             string = JsonPickler.dumps(obj)
#             file.write(string)









# import pickle
# import time
# import inspect
# import base64
# import hashlib
#
# debug = False
#
# def log(s):
#     if debug:
#         print(s)
#
# caches = dict()
# updated_caches = []
#
# def get_cache(fname):
#     if fname in caches:
#         return caches[fname]
#     try:
#         with open(fname, "rb") as f:
#             c = pickle.load(f)
#     except:
#         c = dict()
#     caches[fname] = c
#     return c
#
# def write_to_cache(fname, obj):
#     updated_caches.append(fname)
#     caches[fname] = obj
#
# def cleanup():
#     for fname in updated_caches:
#         with open(fname, "wb") as f:
#             pickle.dump(caches[fname], f)
#
# def get_fn_hash(f):
#     return base64.b64encode(hashlib.sha1(inspect.getsource(f).encode("utf-8")).digest())
#
# NONE = 0
# ARGS = 1
# KWARGS = 2
#
# def cache(fname="cache/cache.pkl", timeout=-1, key=ARGS | KWARGS):
#
#     def impl(fn):
#         load_t = time.time()
#         c = get_cache(fname)
#         log("loaded cache in {:.2f}s".format(time.time() - load_t))
#
#         def d(*args, **kwargs):
#             log("checking cache on {}".format(fn.__name__))
#             if key == ARGS | KWARGS:
#                 k = pickle.dumps((fn.__name__, args, kwargs))
#             if key == ARGS:
#                 k = pickle.dumps((fn.__name__, args))
#             if key == KWARGS:
#                 k = pickle.dumps((fn.__name__, kwargs))
#             if key == NONE:
#                 k = pickle.dumps((fn.__name__))
#             if k in c:
#                 h, t, to, res = c[k]
#                 if get_fn_hash(fn) == h and (to < 0 or (time.time() - t) < to):
#                     log("cache hit.")
#                     return res
#             log("cache miss.")
#             res = fn(*args, **kwargs)
#             c[k] = (get_fn_hash(fn), time.time(), timeout, res)
#             save_t = time.time()
#             write_to_cache(fname, c)
#             log("saved cache in {:.2f}s".format(time.time() - save_t))
#             return res
#
#         return d
#
#     return impl