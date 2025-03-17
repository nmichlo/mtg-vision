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
import random
from math import ceil
from os import PathLike
from typing import TypeVar

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


def imread_float32(path: str | PathLike) -> np.ndarray[np.float32]:
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


def img_clip(img: np.ndarray) -> np.ndarray:
    """
    Clip an image to the correct range, supports float images in range [0, 1]
    and int or uint images in range [0, 255].
    """
    if img.dtype in [np.float16, np.float32, np.float64]:
        im = np.clip(img, 0, 1)
    elif img.dtype in [np.uint8, np.int32, np.int64, np.uint64]:
        im = np.clip(img, 0, 255)
    else:
        raise Exception(f"Unsupported Type: {img.dtype}")
    assert im.dtype == img.dtype
    return im


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
            return np.multiply(img_clip(img), 255.0, dtype=np.uint8)
        elif img.dtype in [np.int32]:
            return img_clip(img)
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
        return np.divide(img, 255.0, dtype=np.float32)
    elif isinstance(img, np.ndarray):
        if img.dtype in [np.float32]:
            return img_clip(img)
        elif img.dtype in [np.float16, np.float64]:
            return img_clip(img.astype(np.float32))
        elif img.dtype in [np.uint8, np.int32]:
            return img_clip(np.divide(img, 255.0, dtype=np.float32))
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
    assert fg_rgba.ndim == 3 and fg_rgba.shape[2] == 4
    assert bg_rgb.ndim == 3 and bg_rgb.shape[2] == 3
    return rgb_mask_over_rgb(
        fg_rgb=fg_rgba[:, :, :3], fg_mask=fg_rgba[:, :, 3], bg_rgb=bg_rgb
    )


def rgb_mask_over_rgb(
    fg_rgb: np.ndarray[np.float32 | np.uint8],
    fg_mask: np.ndarray[np.float32 | np.uint8],
    bg_rgb: np.ndarray[np.float32 | np.uint8],
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
        (w, h),
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


@ensure_float32
def rotate_cover(img: np.ndarray, deg: float) -> np.ndarray:
    """
    Rotate an image by a specified number of degrees, the image is expanded to cover
    the original size.
    """
    (h, w) = img.shape[:2]
    (cy, cx) = (h // 2, w // 2)
    # rotation matrix
    M = cv2.getRotationMatrix2D(center=(cx, cy), angle=deg, scale=1.0)
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
# RANDOM TRANSFORMS                                                         #
# ========================================================================= #


class Mutate:
    """
    A collection of image random image transformations.
    # TODO: replace with random transforms from e.g. albumentations
    """

    @staticmethod
    def flip(img, horr=True, vert=True):
        return flip(
            img,
            horr=horr and (random.random() >= 0.5),
            vert=vert and (random.random() >= 0.5),
        )

    @staticmethod
    def rotate_bounded(img, deg_min=0, deg_max=360):
        return rotate_bounded(img, deg_min + np.random.random() * (deg_max - deg_min))

    @staticmethod
    def upsidedown(img):
        return np.rot90(img, k=2)

    @staticmethod
    def warp(img, warp_ratio=0.15, warp_ratio_min=-0.05):
        # [top left, top right, bottom left, bottom right]
        (h, w) = (img.shape[0] - 1, img.shape[1] - 1)
        src_pts = np.asarray([(0, 0), (0, w), (h, 0), (h, w)], dtype=np.float32)
        ran = warp_ratio_min + np.random.rand(4, 2) * (
            abs(warp_ratio - warp_ratio_min) * 0.5
        )
        dst_pts = (
            ran * np.asarray([(h, w), (h, -w), (-h, w), (-h, -w)], dtype=np.float32)
            + src_pts
        )
        dst_pts = np.asarray(dst_pts, dtype=np.float32)
        # transform matrix
        transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warp
        return cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]))

    @staticmethod
    def warp_inv(img, warp_ratio=0.5, warp_ratio_min=0.25):
        return Mutate.warp(img, warp_ratio=-warp_ratio, warp_ratio_min=-warp_ratio_min)

    @staticmethod
    def noise(img, amount=0.75):
        noise_type = random.choice(["speckle", "gaussian", "pepper", "poisson"])
        if noise_type == "speckle":
            noisy = noise_speckle(img, strength=0.3)
        elif noise_type == "gaussian":
            noisy = noise_gaussian(img, mean=0, var=0.05)
        elif noise_type == "pepper":
            noisy = noise_salt_pepper(img, strength=0.1, svp=0.5)
        elif noise_type == "poisson":
            noisy = noise_poisson(img, peak=0.8, amount=0.5)
        else:
            raise Exception("Invalid Choice")
        ratio = np.random.random() * amount
        img[:, :, :3] = (ratio) * noisy[:, :, :3] + (1 - ratio) * img[:, :, :3]
        return img

    @staticmethod
    def blur(img, n_max=3):
        n = np.random.randint(0, (n_max - 1) // 2 + 1) * 2 + 1
        return cv2.GaussianBlur(img, (n, n), 0)

    @staticmethod
    def tint(img, amount=0.15):
        for i in range(3):
            r = 1 + amount * (2 * np.random.random() - 1)
            img[:, :, i] = img_clip(r * img[:, :, i])
        return img

    @staticmethod
    def fade_white(img, amount=0.33):
        ratio = np.random.random() * amount
        img[:, :, :3] = (ratio) * 1 + (1 - ratio) * img[:, :, :3]
        return img

    @staticmethod
    def fade_black(img, amount=0.5):
        ratio = np.random.random() * amount
        img[:, :, :3] = (ratio) * 0 + (1 - ratio) * img[:, :, :3]
        return img


# ========================================================================= #
# CUSTOM TRANSFORMS                                                         #
# ========================================================================= #


# ========================================================================= #
# VARS & UTILITIES                                                          #
# ========================================================================= #

T = TypeVar("T")


def ensure_float32_dict(fn: T) -> T:
    """Decorator to ensure transform functions return dicts with float32 images and masks."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if not isinstance(result, dict):
            raise Exception(
                f"Function {fn.__name__} did not return a dict, got: {type(result)}"
            )
        if "image" not in result or "mask" not in result:
            raise Exception(
                f"Function {fn.__name__} did not return a dict with 'image' and 'mask'"
            )
        if (
            not isinstance(result["image"], np.ndarray)
            or result["image"].dtype != np.float32
        ):
            raise Exception(
                f"Function {fn.__name__} did not return 'image' as np.float32"
            )
        if (
            not isinstance(result["mask"], np.ndarray)
            or result["mask"].dtype != np.float32
        ):
            raise Exception(
                f"Function {fn.__name__} did not return 'mask' as np.float32"
            )
        return result

    return wrapper


# ========================================================================= #
# TRANSFORM FUNCTIONS                                                       #
# ========================================================================= #


@ensure_float32_dict
def my_flip_horr(data: dict) -> dict:
    """Flip image and mask horizontally."""
    img = flip_horr(data["image"])
    mask = flip_horr(data["mask"])
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_flip_vert(data: dict) -> dict:
    """Flip image and mask vertically."""
    img = flip_vert(data["image"])
    mask = flip_vert(data["mask"])
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_rotate_bounded(data: dict, deg: float) -> dict:
    """Rotate image and mask by a specified degree with bounding."""
    img = rotate_bounded(data["image"], deg)
    mask = rotate_bounded(data["mask"], deg)
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_upsidedown(data: dict) -> dict:
    """Rotate image and mask 180 degrees."""
    img = np.rot90(data["image"], k=2)
    mask = np.rot90(data["mask"], k=2)
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_shift_scale_rotate(
    data: dict,
    shift_limit=(-0.0625, 0.0625),
    scale_limit=(-0.2, 0.0),
    rotate_limit=(-5, 5),
) -> dict:
    """Apply shift, scale, and rotation to image and mask."""
    img = data["image"].copy()
    mask = data["mask"].copy()
    h, w = img.shape[:2]
    shift_x = np.random.uniform(*shift_limit) * w
    shift_y = np.random.uniform(*shift_limit) * h
    scale = 1 + np.random.uniform(*scale_limit)
    angle = np.random.uniform(*rotate_limit)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += [shift_x, shift_y]
    img_transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    mask_transformed = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_CUBIC)
    return {"image": img_transformed, "mask": mask_transformed}


@ensure_float32_dict
def my_perspective(data: dict, scale=(0, 0.075)) -> dict:
    """Apply perspective transform to image and mask."""
    img = data["image"].copy()
    mask = data["mask"].copy()
    h, w = img.shape[:2]
    src_pts = np.array(
        [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], dtype=np.float32
    )
    distortion = np.random.uniform(*scale)
    offsets = distortion * np.array(
        [[h, w], [-h, w], [h, -w], [-h, -w]], dtype=np.float32
    )
    dst_pts = src_pts + offsets
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_transformed = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    mask_transformed = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_CUBIC)
    return {"image": img_transformed, "mask": mask_transformed}


@ensure_float32_dict
def my_erasing(data: dict, scale=(0.2, 0.7)) -> dict:
    """Randomly erase a rectangular region in the image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    h, w = img.shape[:2]
    area = h * w
    target_area = random.uniform(*scale) * area
    aspect_ratio = random.uniform(0.3, 1 / 0.3)
    rh = int((target_area * aspect_ratio) ** 0.5)
    rw = int((target_area / aspect_ratio) ** 0.5)
    if rh < h and rw < w:
        x = random.randint(0, w - rw)
        y = random.randint(0, h - rh)
        fill_value = np.random.uniform(0, 1, size=(rh, rw, img.shape[2]))
        img[y : y + rh, x : x + rw] = fill_value
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_tint(data: dict, amount=0.15) -> dict:
    """Apply random tint to image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    for i in range(3):
        r = 1 + amount * (2 * np.random.random() - 1)
        img[:, :, i] = img_clip(r * img[:, :, i])
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_blur(data: dict, n_max=7) -> dict:
    """Apply blur to image; mask unchanged."""
    img = Mutate.blur(data["image"], n_max=n_max)  # Assuming uimg.blur exists
    mask = data["mask"]
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_image_compression(data: dict, quality_range=(98, 100)) -> dict:
    """Apply JPEG compression to image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", (img * 255).astype(np.uint8), encode_param)
    img_transformed = cv2.imdecode(encimg, cv2.IMREAD_COLOR).astype(np.float32) / 255
    return {"image": img_transformed, "mask": mask}


@ensure_float32_dict
def my_downscale(data: dict, scale_range=(0.2, 0.9)) -> dict:
    """Downscale and upscale image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    scale = random.uniform(*scale_range)
    h, w = img.shape[:2]
    small_h, small_w = int(h * scale), int(w * scale)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    img_transformed = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return {"image": img_transformed, "mask": mask}


@ensure_float32_dict
def my_noise(data: dict, var=0.05) -> dict:
    """Add Gaussian noise to image; mask unchanged."""
    img = noise_gaussian(data["image"], mean=0, var=var)
    mask = data["mask"]
    return {"image": img, "mask": mask}


@ensure_float32_dict
def my_gamma(data: dict, gamma_limit=(80, 120)) -> dict:
    """Adjust gamma of image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    gamma = random.uniform(gamma_limit[0], gamma_limit[1]) / 100
    img_transformed = np.clip(img**gamma, 0, 1).astype(np.float32)
    return {"image": img_transformed, "mask": mask}
