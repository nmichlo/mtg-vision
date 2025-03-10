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
import mtgvision.util.files as ufls
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
        ufls.init_dir(self.path, is_file=True)
        if self.refresh or not os.path.isfile(self.path):
            self.data = {}
            self.refresh = False
        else:
            try:
                with open(self.path, "r") as file_stream:
                    self.data = json.load(file_stream)
            except Exception as e:
                self.data = {}
                print("WARNING: Error loading cache: {} ({})".format(self.path, e))
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        ufls.init_dir(self.path, is_file=True)
        if self.save:
            with open(self.path, "w") as file_stream:
                json.dump(self.data, file_stream)
        else:
            print("Skipping Save Cache: {}".format(self.path))
        self.data = None
