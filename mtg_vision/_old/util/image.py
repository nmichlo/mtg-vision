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

import base64
from math import ceil
import cv2
import numpy as np
from mtg_vision._old.util import asrt_float

# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #

# global type used by default for images and networks
# TODO: update more things to use this
FLOAT_TYPE = np.float32

# ========================================================================= #
# OpenCV Wrapper - IO                                                       #
# ========================================================================= #


def imwrite(path, img):
    if img.dtype in [np.float16, np.float32, np.float64, np.float128]:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def imread(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception('Image not found: {}'.format(path))
    return fxx(img)

def imshow(image, window_name='image', scale=1):
    if scale != 1 and scale is not None:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, image)

def imshow_loop(image, window_name='image', scale=1):
    imshow(image, window_name, scale)
    while True:
        k = cv2.waitKey(100)
        if (k == 27) or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
            cv2.destroyAllWindows()
            break

def safe_imread(path):
    try:
        return imread(path)
    except Exception as e:
        return np.zeros((1,1,3), FLOAT_TYPE)

# ============================================================================ #
# Types                                                                        #
# ============================================================================ #

def fxx(*items, float_type=FLOAT_TYPE):
    if len(items) == 0:
        raise Exception('Too few args')
    elif len(items) == 1:
        item = items[0]
        if type(item) == np.ndarray:
            if item.dtype in [np.float16, np.float32, np.float64, np.float128]:
                return item.astype(float_type)
            elif item.dtype in [np.uint8, np.int32]:
                return np.divide(item, 255.0, dtype=float_type) # NORMALISE
            else:
                raise Exception('Unsupported Numpy Type: {}'.format(item.dtype))
        elif type(item) in [tuple, list]:
            return np.array(item, dtype=float_type)
        else:
            raise Exception('Unsupported Type: {}'.format(type(item)))
    else:
        return np.array(items, dtype=float_type)

def image2base64(img, type='png'):
    if img.dtype in [np.float16, np.float32, np.float64, np.float128]:
        img = img * 255
    return base64.b64encode(cv2.imencode('.{}'.format(type), img.astype(np.uint8))[1])

# ============================================================================ #
# Basic Operations                                                             #
# ============================================================================ #


def clip(img: np.ndarray):
    if img.dtype in [np.float16, np.float32, np.float64, np.float128]:
        return np.clip(img, 0, 1)
    elif img.dtype in [np.uint8, np.int32]:
        return np.clip(img, 0, 255)
    else:
        raise Exception('Unsupported Type: {}'.format(img.dtype))


def img_uint8(img):
    if img.dtype in [np.uint8]:
        return img
    elif img.dtype in [np.float16, np.float32, np.float64, np.float128]:
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    elif img.dtype in [np.int32]:
        return np.clip(img, 0, 255).astype(np.uint8)
    else:
        raise Exception('Unsupported Numpy Type: {}'.format(img.dtype))

def img_float32(img):
    if img.dtype in [np.float32]:
        return img
    elif img.dtype in [np.float16, np.float64, np.float128]:
        return np.clip(img.astype(np.float32), 0, 1)
    elif img.dtype in [np.uint8, np.int32]:
        return np.divide(np.clip(img, 0, 255), 255.0, dtype=np.float32)
    else:
        raise Exception('Unsupported Numpy Type: {}'.format(img.dtype))



# ============================================================================ #
# Merge Operations                                                             #
# ============================================================================ #

def rgba_over_rgb(fg_rgba, bg_rgb):
    alpha = fg_rgba[:, :, 3]
    fg = cv2.merge((fg_rgba[:, :, 0] * alpha, fg_rgba[:, :, 1] * alpha, fg_rgba[:, :, 2] * alpha))
    alpha = 1 - alpha
    bg = cv2.merge((bg_rgb[:, :, 0] * alpha, bg_rgb[:, :, 1] * alpha, bg_rgb[:, :, 2] * alpha))
    ret = clip(bg + fg)
    assert ret.dtype == FLOAT_TYPE
    return ret

# ============================================================================ #
# Basic Transforms                                                             #
# - ALL SIZES ARE: >>> (H, W) <<< NOT: (W, H)                                  #
# ============================================================================ #

def flip_vert(img):
    ret = cv2.flip(img, 0)
    assert ret.dtype == FLOAT_TYPE
    return ret


def flip_horr(img):
    ret = cv2.flip(img, 1)
    assert ret.dtype == FLOAT_TYPE
    return ret

def flip(img, horr=True, vert=True):
    if vert:
        img = flip_vert(img)
    if horr:
        img = flip_horr(img)
    ret = img
    assert ret.dtype == FLOAT_TYPE
    return ret

def resize(img, size, shrink=True):
    # OpenCV is W*H not H*W
    ret = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    assert ret.dtype == FLOAT_TYPE
    return ret

def remove_border_resized(img, border_width, size=None):
    (ih, iw) = img.shape[:2]
    crop = img[border_width:ih-border_width, border_width:iw-border_width, :]
    if size is not None:
        crop = resize(crop, size)
    ret = crop
    assert ret.dtype == FLOAT_TYPE
    return ret

def crop_to_size(img, size, pad=False):
    (ih, iw), (sh, sw) = (img.shape[:2], size)
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
            zeros[y0:y0+rh,x0:x0+rw,:] = resized
            ret = zeros
        else:
            y0, x0 = (rh - sh) // 2, (rw - sw) // 2
            ret = resized[y0:y0+sh,x0:x0+sw,:]
    assert ret.dtype == img.dtype
    return ret

def rotate_bounded(img, deg):
    (h, w) = img.shape[:2]
    (cy, cx) = (h//2, w//2)
    # rotation matrix
    M = cv2.getRotationMatrix2D(center=(cx, cy), angle=deg, scale=1.0)  # anticlockwise
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    # new bounds
    nw, nh = int((h*sin) + (w*cos)), int((h*cos) + (w*sin))
    # adjust the rotation matrix
    M[0, 2] += (nw/2)-cx
    M[1, 2] += (nh/2)-cy
    # transform
    ret = cv2.warpAffine(img, M, (nw, nh))
    assert ret.dtype == img.dtype
    return ret

# ========================================================================= #
# DRAW                                                                      #
# ========================================================================= #

def round_rect_mask(size, radius=None, radius_ratio=0.045, dtype=FLOAT_TYPE):
    if radius is None:
        radius = ceil(max(size)*radius_ratio)
    img = np.ones(size[:2], dtype=dtype)
    # corner piece
    corner = np.zeros((radius, radius), dtype=dtype)
    cv2.circle(corner, (0, 0), radius, 1, cv2.FILLED)
    # fill corners
    y1, x1 = size[:2]
    img[y1-radius:, x1-radius:] = np.rot90(corner, 0)  # br
    img[:radius, x1-radius:]    = np.rot90(corner, 1)  # tr
    img[:radius, :radius]       = np.rot90(corner, 2)  # tl
    img[y1-radius:, :radius]    = np.rot90(corner, 3)  # bl
    # return
    assert img.dtype == dtype
    return img

# ========================================================================= #
# NOISE                                                                     #
# ========================================================================= #

def noise_speckle(img, strength=0.1):
    out = np.copy(asrt_float(img))
    gauss = np.random.randn(*(img.shape[0], img.shape[1], 3))
    gauss = gauss.reshape((img.shape[0], img.shape[1], 3))
    out[:,:,:3] = img[:,:,:3] * (1 + gauss * strength)
    ret = clip(out)
    assert ret.dtype == FLOAT_TYPE
    return ret

def noise_gaussian(img, mean=0, var=0.5):
    out = np.copy(asrt_float(img))
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], 3))
    gauss = gauss.reshape((img.shape[0], img.shape[1], 3))
    out[:,:,:3] = img[:,:,:3] + gauss
    ret = clip(out)
    assert ret.dtype == FLOAT_TYPE
    return ret

def noise_salt_pepper(img, strength=0.1, svp=0.5):
    out = np.copy(asrt_float(img))
    if img.shape[2] > 3:
        alpha = img[:, :, 3]
    # Salt mode
    num_salt = np.ceil(strength * img.size * svp)
    cx, cy, cs = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
    out[cx, cy, cs] = 1
    # Pepper mode
    num_pepper = np.ceil(strength * img.size * (1-svp))
    cx, cy, cs = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
    out[cx, cy, cs] = 0
    if img.shape[2] > 3:
        out[:, :, 3] = alpha
    ret = clip(out)
    assert ret.dtype == FLOAT_TYPE
    return ret

def noise_poisson(img, peak=0.1, amount=0.25):
    img = clip(asrt_float(img))
    noise = np.zeros_like(img)
    noise[:,:,:3] = np.random.poisson(img[:,:,:3]*peak) / peak
    ret = clip((1-amount)*img + amount*noise)
    assert ret.dtype == FLOAT_TYPE
    return ret

