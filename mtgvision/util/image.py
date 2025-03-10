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


import base64
import functools
from math import ceil
from os import PathLike
from typing import Literal, TypeVar

import cv2
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
        if not isinstance(result, np.ndarray):
            raise Exception(
                f"Function {fn.__name__} did not return a numpy array, got: {type(result)}"
            )
        if result.dtype != np.float32:
            raise Exception(
                f"Function {fn.__name__} did not return a numpy array of type {np.float32}, got: {result.dtype}"
            )
        return result

    return wrapper


def asrt_float(x):
    assert isinstance(x, np.ndarray) and x.dtype in [
        np.float16,
        np.float32,
        np.float64,
    ], f"Expected any np.float*, got {x.dtype}"
    return x


# ========================================================================= #
# OpenCV Wrapper - IO                                                       #
# ========================================================================= #


def imwrite(path: str | PathLike, img: np.ndarray):
    """
    Save an image to disk, support both float images in range [0, 1]
    and int or uint images in range [0, 255]
    """
    if img.dtype in [np.float16, np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def imread_float(path: str | PathLike) -> np.ndarray[np.float32]:
    """
    Read an image from disk, and convert it to a float32 image in range [0, 1].
    """
    img = cv2.imread(str(path))
    if img is None:
        raise Exception("Image not found: {}".format(str(path)))
    return img_float32(img)


def imshow(image, window_name="image", scale=1):
    """
    Display an image in a window temporarily, this should be used
    with additional wait logic to keep the window open.
    """
    if scale != 1 and scale is not None:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, image)


def imshow_loop(image, window_name="image", scale=1):
    """
    Display an image in a window, the window will stay open until the user
    presses the escape key or closes the window.
    """
    imshow(image, window_name, scale)
    while True:
        k = cv2.waitKey(100)
        if (k == 27) or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
            cv2.destroyAllWindows()
            break


@ensure_float32
def safe_imread(path):
    """
    Read an image from disk, if the image is not found, return a blank image.
    """
    try:
        return imread_float(path)
    except Exception:
        return np.zeros((1, 1, 3), np.float32)


# ============================================================================ #
# Types                                                                        #
# ============================================================================ #


def image2base64(
    img: np.ndarray | Image.Image,
    encode: Literal["jpg", "png"] = "png",
) -> bytes:
    """
    Convert an image to a base64 encoded string, supports any kind of input image.
    """
    img = img_uint8(img)
    _, encoding = cv2.imencode(".{}".format(encode), img)
    return base64.b64encode(encoding)


# ============================================================================ #
# Basic Operations                                                             #
# ============================================================================ #


def img_clip(img: np.ndarray) -> np.ndarray:
    """
    Clip an image to the correct range, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if img.dtype in [np.float16, np.float32, np.float64]:
        return np.clip(img, 0, 1)
    elif img.dtype in [np.uint8, np.int32]:
        return np.clip(img, 0, 255)
    else:
        raise Exception(f"Unsupported Type: {img.dtype}")


def img_uint8(img: np.ndarray | Image.Image) -> np.ndarray[np.uint8]:
    """
    Convert an image to uint8, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if isinstance(img, Image.Image):
        return np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if img.dtype in [np.uint8]:
            return img
        elif img.dtype in [np.float16, np.float32, np.float64]:
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        elif img.dtype in [np.int32]:
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise Exception(f"Unsupported Numpy Type: {img.dtype}")
    else:
        raise Exception(f"Unsupported Type: {type(img)}")


def img_float32(img: np.ndarray | Image.Image) -> np.ndarray[np.float32]:
    """
    Convert an image to float32, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if isinstance(img, Image.Image):
        return img_float32(np.asarray(img, dtype=np.float32))
    elif isinstance(img, np.ndarray):
        if img.dtype in [np.float32]:
            return img
        elif img.dtype in [np.float16, np.float64]:
            return np.clip(img.astype(np.float32), 0, 1)
        elif img.dtype in [np.uint8, np.int32]:
            return np.divide(np.clip(img, 0, 255), 255.0, dtype=np.float32)
        else:
            raise Exception(f"Unsupported Numpy Type: {img.dtype}")
    else:
        raise Exception(f"Unsupported Type: {type(img)}")


# ============================================================================ #
# Merge Operations                                                             #
# ============================================================================ #


@ensure_float32
def rgba_over_rgb(
    fg_rgba: np.ndarray[np.float32 | np.uint8],
    bg_rgb: np.ndarray[np.float32 | np.uint8],
):
    """
    Merge a foreground RGBA image with a background RGB image, supports float
    images in range [0, 1] and int or uint images in range [0, 255]. Both must
    be the same type and size.
    """
    alpha = fg_rgba[:, :, 3]
    fg = cv2.merge(
        (fg_rgba[:, :, 0] * alpha, fg_rgba[:, :, 1] * alpha, fg_rgba[:, :, 2] * alpha)
    )
    alpha = 1 - alpha
    bg = cv2.merge(
        (bg_rgb[:, :, 0] * alpha, bg_rgb[:, :, 1] * alpha, bg_rgb[:, :, 2] * alpha)
    )
    ret = img_clip(bg + fg)
    return ret


# ============================================================================ #
# Basic Transforms                                                             #
# - ALL SIZES ARE: >>> (H, W) <<< NOT: (W, H)                                  #
# ============================================================================ #


@ensure_float32
def flip_vert(img: np.ndarray) -> np.ndarray:
    """Flip an image vertically."""
    return cv2.flip(img, 0)


@ensure_float32
def flip_horr(img: np.ndarray) -> np.ndarray:
    """Flip an image horizontally."""
    return cv2.flip(img, 1)


@ensure_float32
def flip(img: np.ndarray, horr: bool = True, vert: bool = True) -> np.ndarray:
    """Flip an image either horizontally or vertically."""
    if vert:
        img = flip_vert(img)
    if horr:
        img = flip_horr(img)
    return img


@ensure_float32
def resize(
    img: np.ndarray, size_hw: tuple[int, int], shrink: bool = True
) -> np.ndarray:
    """Resize an image to a new size."""
    h, w = size_hw
    # OpenCV is W*H not H*W
    return cv2.resize(
        img,
        (h, w),
        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC,
    )


@ensure_float32
def remove_border_resized(
    img: np.ndarray, border_width: int, size_hw: tuple[int, int] = None
) -> np.ndarray:
    """Remove a border of specified size (crop) from an image and resize it."""
    (ih, iw) = img.shape[:2]
    crop = img[border_width : ih - border_width, border_width : iw - border_width, :]
    if size_hw is not None:
        crop = resize(crop, size_hw)
    return crop


@ensure_float32
def crop_to_size(
    img: np.ndarray, size_hw: tuple[int, int], pad: bool = False
) -> np.ndarray:
    """
    Crop an image to a new size, if pad is True then the image is padded
    if the new size is smaller than the original. Otherwise, the image is
    cropped to the center.
    """
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
            zeros = np.zeros((sh, sw, img.shape[2]), dtype=img.dtype)
            y0, x0 = (sh - rh) // 2, (sw - rw) // 2
            zeros[y0 : y0 + rh, x0 : x0 + rw, :] = resized
            ret = zeros
        else:
            y0, x0 = (rh - sh) // 2, (rw - sw) // 2
            ret = resized[y0 : y0 + sh, x0 : x0 + sw, :]
    return ret


@ensure_float32
def rotate_bounded(img: np.ndarray, deg: float) -> np.ndarray:
    """
    Rotate an image by a specified number of degrees, the image is bounded
    to the original size, and the image is resized to fit the
    bounding box.
    """
    (h, w) = img.shape[:2]
    (cy, cx) = (h // 2, w // 2)
    # rotation matrix
    M = cv2.getRotationMatrix2D(center=(cx, cy), angle=deg, scale=1.0)  # anticlockwise
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
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
) -> np.ndarray:
    """
    Create a mask with a rounded edges.
    """
    if radius is None:
        radius = int(ceil(max(size_hw) * radius_ratio))
    img = np.ones(size_hw[:2], dtype=np.float32)
    # corner piece
    corner = np.zeros((radius, radius), dtype=np.float32)
    cv2.circle(corner, (0, 0), radius, 1, cv2.FILLED)
    # fill corners
    y1, x1 = size_hw[:2]
    img[y1 - radius :, x1 - radius :] = np.rot90(corner, 0)  # br
    img[:radius, x1 - radius :] = np.rot90(corner, 1)  # tr
    img[:radius, :radius] = np.rot90(corner, 2)  # tl
    img[y1 - radius :, :radius] = np.rot90(corner, 3)  # bl
    return img


# ========================================================================= #
# NOISE                                                                     #
# TODO: this can be replaced with Albumentations or imgaug                  #
# ========================================================================= #


@ensure_float32
def noise_speckle(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """Add speckle noise to an image."""
    out = np.copy(asrt_float(img))
    gauss = np.random.randn(*(img.shape[0], img.shape[1], 3))
    gauss = gauss.reshape((img.shape[0], img.shape[1], 3))
    out[:, :, :3] = img[:, :, :3] * (1 + gauss * strength)
    ret = img_clip(out)
    return ret


@ensure_float32
def noise_gaussian(img: np.ndarray, mean: float = 0, var: float = 0.5) -> np.ndarray:
    """Add gaussian noise to an image."""
    out = np.copy(asrt_float(img))
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], 3))
    gauss = gauss.reshape((img.shape[0], img.shape[1], 3))
    out[:, :, :3] = img[:, :, :3] + gauss
    ret = img_clip(out)
    return ret


@ensure_float32
def noise_salt_pepper(
    img: np.ndarray, strength: float = 0.1, svp: float = 0.5
) -> np.ndarray:
    """Add salt and pepper noise to an image."""
    out = np.copy(asrt_float(img))
    if img.shape[2] > 3:
        alpha = img[:, :, 3]
    # Salt mode
    num_salt = np.ceil(strength * img.size * svp)
    cx, cy, cs = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[cx, cy, cs] = 1
    # Pepper mode
    num_pepper = np.ceil(strength * img.size * (1 - svp))
    cx, cy, cs = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[cx, cy, cs] = 0
    if img.shape[2] > 3:
        out[:, :, 3] = alpha
    ret = img_clip(out)
    return ret


@ensure_float32
def noise_poisson(
    img: np.ndarray, peak: float = 0.1, amount: float = 0.25
) -> np.ndarray:
    """Add poisson noise to an image."""
    img = img_clip(asrt_float(img))
    noise = np.zeros_like(img)
    noise[:, :, :3] = np.random.poisson(img[:, :, :3] * peak) / peak
    ret = img_clip((1 - amount) * img + amount * noise)
    return ret
