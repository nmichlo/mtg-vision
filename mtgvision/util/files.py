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
from os import PathLike


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def get_image_paths(
    folder: str | PathLike, extensions: list[str] | None = None, prefixed: bool = False
) -> list:
    if not extensions:
        extensions = [".jpg", ".png", ".jpeg"]
    images = []
    folder = os.path.join(str(folder), "")  # ensure trailing slash
    for root, dirs, files in os.walk(folder):
        # strip roots
        if not prefixed:
            if root.startswith(folder):
                root = root[len(folder) :]
            else:
                raise Exception("This should never happen")
        # append files
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                # TODO: this may not always work, must be a better way.
                images.append(os.path.join(root, file))
    return images


def init_dir(*parts: str, is_file: bool = False) -> str:
    path: str = os.path.join(*parts)
    dirs = path if not is_file else os.path.dirname(path)
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    return path
