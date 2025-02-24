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
from .values import default


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def get_image_paths(folder, extensions=None, prefixed=False) -> list:
    extensions = default(extensions, lambda: ['.jpg', '.png', '.jpeg'])
    images = []

    folder = os.path.join(folder, '')

    for root, dirs, files in os.walk(folder):
        # strip roots
        if not prefixed:
            if root.startswith(folder):
                root = root[len(folder):]
            else:
                raise Exception('This should never happen')
        # append files
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                # TODO: this may not always work, must be a better way.
                images.append(os.path.join(root, file))
    return images


def init_dir(*paths, is_file=False) -> str:
    path = os.path.join(*paths)
    dirs = path if not is_file else os.path.dirname(path)
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    return path


# ========================================================================= #
# PATH BUILDER                                                              #
# ========================================================================= #


class Folder(object):
    def __init__(self, *paths, init=False):
        self._root = os.path.join(*paths)
        if init:
            init_dir(self._root)

    @property
    def path(self):
        return self._root

    def cd(self, *paths, init=False):
        return Folder(self._root, *paths, init=init)

    def to(self, *paths, init=False, is_file=False):
        path = os.path.join(self._root, *paths)
        if init:
            return init_dir(path, is_file=is_file)
        return path

    def __str__(self):
        return self._root

    def __repr__(self):
        return str(self)