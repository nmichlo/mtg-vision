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


import functools
from math import ceil
from os import PathLike
from typing import TypeVar

import cv2
import jax.image
import jax.numpy as jnp
import numpy as np
from PIL import Image


# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #


T = TypeVar("T")


def ensure_float32(fn: T) -> T:
    """
    Decorator to ensure that a function returns a numpy array of type np.float32.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if not isinstance(result, jnp.ndarray):
            raise Exception(
                f"Function {fn.__name__} did not return a numpy array, got: {type(result)}"
            )
        if result.dtype != jnp.float32:
            raise Exception(
                f"Function {fn.__name__} did not return a numpy array of type {jnp.float32}, got: {result.dtype}"
            )
        return result

    return wrapper


def asrt_float(x):
    assert isinstance(x, (jnp.ndarray, np.ndarray)) and x.dtype in [
        jnp.float16,
        jnp.float32,
        jnp.float64,
    ], f"Expected any np.float*, got {x.dtype}"
    return x


# ========================================================================= #
# OpenCV Wrapper - IO                                                       #
# ========================================================================= #


def imwrite(path: str | PathLike, img: jnp.ndarray):
    """
    Save an image to disk, support both float images in range [0, 1]
    and int or uint images in range [0, 255]
    """
    if img.dtype in [jnp.float16, jnp.float32, jnp.float64]:
        img = (img * 255).astype(jnp.uint8)
    cv2.imwrite(str(path), img)


def imread_float32(path: str | PathLike) -> jnp.ndarray:
    """
    Read an image from disk, and convert it to a float32 image in range [0, 1].
    """
    img = cv2.imread(str(path))
    if img is None:
        raise Exception("Image not found: {}".format(str(path)))
    return img_float32(img)


def _imshow(image, window_name="image", scale=1):
    """
    Display an image in a window temporarily, this should be used
    with additional wait logic to keep the window open.
    """
    if scale != 1 and scale is not None:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, image)
    cv2.moveWindow(window_name, 100, 100)


def imshow_loop(image, window_name="image", scale=1):
    """
    Display an image in a window, the window will stay open until the user
    presses the escape key or closes the window.
    """
    _imshow(image, window_name, scale)
    while True:
        k = cv2.waitKey(100)
        if (k == 27) or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
            cv2.destroyAllWindows()
            break


# ============================================================================ #
# Basic Operations                                                             #
# ============================================================================ #


def img_clip(img: jnp.ndarray) -> jnp.ndarray:
    """
    Clip an image to the correct range, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if img.dtype in [jnp.float16, jnp.float32, jnp.float64]:
        im = jnp.clip(img, 0, 1)
    elif img.dtype in [jnp.uint8, jnp.int32, jnp.int64, jnp.uint64]:
        im = jnp.clip(img, 0, 255)
    else:
        raise Exception(f"Unsupported Type: {img.dtype}")
    assert im.dtype == img.dtype
    return im


def img_uint8(img: jnp.ndarray | Image.Image) -> jnp.ndarray:
    """
    Convert an image to uint8, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if isinstance(img, Image.Image):
        return jnp.asarray(img, dtype=jnp.uint8)
    elif isinstance(img, (jnp.ndarray, np.ndarray)):
        if img.dtype in [jnp.uint8]:
            return img
        elif img.dtype in [jnp.float16, jnp.float32, jnp.float64]:
            return (img_clip(img) * 255.0).astype(jnp.uint8)
        elif img.dtype in [jnp.int32]:
            return img_clip(img)
        else:
            raise Exception(f"Unsupported Numpy Type: {img.dtype}")
    else:
        raise Exception(f"Unsupported Type: {type(img)}")


def img_float32(img: jnp.ndarray | Image.Image) -> jnp.ndarray:
    """
    Convert an image to float32, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if isinstance(img, Image.Image):
        return jnp.asarray(img, dtype=jnp.float32) / 255.0
    elif isinstance(img, (jnp.ndarray, np.ndarray)):
        if img.dtype in [jnp.float32]:
            return img_clip(img)
        elif img.dtype in [jnp.float16, jnp.float64]:
            return img_clip(img.astype(jnp.float32))
        elif img.dtype in [jnp.uint8, jnp.int32]:
            return img_clip(img.astype(jnp.float32) / 255.0)
        else:
            raise Exception(f"Unsupported Numpy Type: {img.dtype}")
    else:
        raise Exception(f"Unsupported Type: {type(img)}")


# ============================================================================ #
# Merge Operations                                                             #
# ============================================================================ #


@ensure_float32
def rgba_over_rgb(
    fg_rgba: jnp.ndarray,
    bg_rgb: jnp.ndarray,
):
    """
    Merge a foreground RGBA image with a background RGB image, supports float
    images in range [0, 1] and int or uint images in range [0, 255]. Both must
    be the same type and size.
    """
    assert fg_rgba.ndim == 3 and fg_rgba.shape[2] == 4
    assert bg_rgb.ndim == 3 and bg_rgb.shape[2] == 3
    return rgb_mask_over_rgb(
        fg_rgb=fg_rgba[:, :, :3], fg_mask=fg_rgba[:, :, 3], bg_rgb=bg_rgb
    )


def rgb_mask_over_rgb(
    fg_rgb: jnp.ndarray,
    fg_mask: jnp.ndarray,
    bg_rgb: jnp.ndarray,
):
    """
    Merge a foreground RGB image with a mask image over a background RGB image,
    supports float images in range [0, 1] and int or uint images in range [0, 255].
    All images must be the same type and size.
    """
    assert fg_rgb.ndim == 3 and fg_rgb.shape[2] == 3
    assert fg_mask.ndim == 2
    assert bg_rgb.ndim == 3 and bg_rgb.shape[2] == 3
    fg = fg_rgb * fg_mask[..., None]
    bg = bg_rgb * (1 - fg_mask[..., None])
    return img_clip(bg + fg)


# ============================================================================ #
# Basic Transforms                                                             #
# - ALL SIZES ARE: >>> (H, W) <<< NOT: (W, H)                                  #
# ============================================================================ #


@ensure_float32
def resize(
    img: jnp.ndarray, size_hw: tuple[int, int], shrink: bool = True
) -> jnp.ndarray:
    """Resize an image to a new size."""
    h, w = size_hw
    # OpenCV is W*H not H*W
    return jax.image.resize(
        img,
        shape=(w, h, *img.shape[2:]),
        method="cubic" if not shrink else "linear",
    )


@ensure_float32
def remove_border_resized(
    img: jnp.ndarray, border_width: int, size_hw: tuple[int, int] = None
) -> jnp.ndarray:
    """Remove a border of specified size (crop) from an image and resize it."""
    (ih, iw) = img.shape[:2]
    crop = img[border_width : ih - border_width, border_width : iw - border_width, :]
    if size_hw is not None:
        crop = resize(crop, size_hw)
    return crop


@ensure_float32
def crop_to_size(
    img: jnp.ndarray, size_hw: tuple[int, int], pad: bool = False
) -> jnp.ndarray:
    """
    Crop an image to a new size, if pad is True then the image is padded
    if the new size is smaller than the original. Otherwise, the image is
    cropped to the center.
    """
    assert len(img.shape) == 3
    (ih, iw), (sh, sw) = (img.shape[:2], size_hw)
    if ih == sh and iw == sw:
        ret = img
    else:
        # calc size
        rh, rw = (ih / sh, iw / sw)
        r = min(rh, rw) if (not pad) else max(rh, rw)
        rh, rw = int(ih / r), int(iw / r)
        # resize
        resized = resize(img, (rh, rw))
        if pad:
            zeros = jnp.zeros((sh, sw, img.shape[2]), dtype=img.dtype)
            y0, x0 = (sh - rh) // 2, (sw - rw) // 2
            zeros[y0 : y0 + rh, x0 : x0 + rw, :] = resized
            ret = zeros
        else:
            y0, x0 = (rh - sh) // 2, (rw - sw) // 2
            ret = resized[y0 : y0 + sh, x0 : x0 + sw, :]
    return ret


@ensure_float32
def rotate_bounded(img: jnp.ndarray, deg: float) -> jnp.ndarray:
    """
    Rotate an image by a specified number of degrees, the image is bounded
    to the original size, and the image is resized to fit the
    bounding box.
    """
    (h, w) = img.shape[:2]
    (cy, cx) = (h // 2, w // 2)
    # rotation matrix
    M = cv2.getRotationMatrix2D(center=(cx, cy), angle=deg, scale=1.0)  # anticlockwise
    cos, sin = jnp.abs(M[0, 0]), jnp.abs(M[0, 1])
    # new bounds
    nw, nh = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    # adjust the rotation matrix
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    # transform
    return cv2.warpAffine(img, M, (nw, nh))


# ========================================================================= #
# DRAW                                                                      #
# ========================================================================= #


@ensure_float32
def round_rect_mask(
    size_hw: tuple[int, int], radius: int | None = None, radius_ratio: float = 0.045
) -> jnp.ndarray:
    """
    Create a mask with a rounded edges.
    """
    if radius is None:
        radius = int(ceil(max(size_hw) * radius_ratio))
    img = jnp.ones(size_hw[:2], dtype=jnp.float32)
    # corner piece
    corner = np.zeros((radius, radius), dtype=jnp.float32)
    cv2.circle(corner, (0, 0), radius, 1, cv2.FILLED)
    # fill corners
    y1, x1 = size_hw[:2]
    img.at[y1 - radius :, x1 - radius :].set(jnp.rot90(corner, 0))  # br
    img.at[:radius, x1 - radius :].set(jnp.rot90(corner, 1))  # tr
    img.at[:radius, :radius].set(jnp.rot90(corner, 2))  # tl
    img.at[y1 - radius :, :radius].set(jnp.rot90(corner, 3))  # bl
    return img
