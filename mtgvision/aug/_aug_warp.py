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


__all__ = [
    "FlipHorizontal",
    "FlipVertical",
    "RotateBounded",
    "Rotate180",
    "PerspectiveWarp",
    "ShiftScaleRotate",
]


import cv2
import numpy as np

from mtgvision.aug import AugItems, Augment, AugPrngHint
from mtgvision.aug._util_args import ArgFloatHint, ArgFloatRange


# ========================================================================= #
# WRAP                                                                      #
# ========================================================================= #


class FlipHorizontal(Augment):
    """
    Flip the image, mask, and points horizontally.
    """

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            _, w = x.get_bounds_hw()
            # Flip image
            im = np.fliplr(x.image) if x.has_image else None
            # Flip mask
            mask = np.fliplr(x.mask) if x.has_mask else None
            # Flip points (x -> W - 1 - x)
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points[:, 0] = w - 1 - points[:, 0]
            return x.override(image=im, mask=mask, points=points)
        return x


class FlipVertical(Augment):
    """
    Flip the image, mask, and points vertically.
    """

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            h, _ = x.get_bounds_hw()
            # Flip image
            im = np.flipud(x.image) if x.has_image else None
            # Flip mask
            mask = np.flipud(x.mask) if x.has_mask else None
            # Flip points (y -> H - 1 - y)
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points[:, 1] = h - 1 - points[:, 1]
            return x.override(image=im, mask=mask, points=points)
        return x


class RotateBounded(Augment):
    """
    Rotate the image, mask, and points within a specified degree range, adjusting the output size to bound the rotated content.
    """

    def __init__(
        self,
        deg: ArgFloatHint = (0, 360),
        inter: int = cv2.INTER_LINEAR,
        inter_mask: int = cv2.INTER_NEAREST,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._deg = ArgFloatRange.from_arg(deg, min_val=-360, max_val=360)
        self._inter = inter
        self._inter_mask = inter_mask

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            deg = self._deg.sample(prng)
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Compute center
            cy, cx = h / 2, w / 2
            # Compute rotation matrix
            M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            # New bounding dimensions
            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))
            # Adjust translation
            M[0, 2] += (nw / 2) - cx
            M[1, 2] += (nh / 2) - cy
            # Warp
            return x.applied_rotation_matrix(
                M, mode="affine", inter=self._inter, inter_mask=self._inter_mask
            )
        return x


class Rotate180(Augment):
    """
    Rotate the image, mask, and points by 180 degrees.
    """

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Rotate image and mask
            im = np.rot90(x.image, k=2) if x.has_image else None
            mask = np.rot90(x.mask, k=2) if x.has_mask else None
            # Transform points (x, y) -> (W - 1 - x, H - 1 - y)
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points[:, 0] = w - 1 - points[:, 0]
                points[:, 1] = h - 1 - points[:, 1]
            return x.override(image=im, mask=mask, points=points)
        return x


class PerspectiveWarp(Augment):
    """
    Apply a random perspective warp to the image, mask, and points by distorting the four corners.
    """

    def __init__(
        self,
        corner_jitter_ratio: ArgFloatHint = (-0.1, 0.1),
        corner_jitter_offset: ArgFloatHint = 0,
        inter: int = cv2.INTER_LINEAR,
        inter_mask: int = cv2.INTER_NEAREST,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._corner_jitter_ratio = ArgFloatRange.from_arg(corner_jitter_ratio)
        self._corner_jitter_offset = ArgFloatRange.from_arg(corner_jitter_offset)
        self._inter = inter
        self._inter_mask = inter_mask

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Random offsets
            ran = self._corner_jitter_ratio.sample(prng, size=(4, 2))
            offsets = ran * np.array(
                [(h, w), (h, -w), (-h, w), (-h, -w)], dtype=np.float32
            )
            offsets += self._corner_jitter_offset.sample(prng, size=(4, 2))
            # Define source points (corners)
            src_pts = np.array(
                [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], dtype=np.float32
            )
            dst_pts = src_pts + offsets
            # Compute perspective transform
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # Warp
            return x.applied_rotation_matrix(
                M, mode="perspective", inter=self._inter, inter_mask=self._inter_mask
            )
        return x


class ShiftScaleRotate(Augment):
    """
    Apply random shift, scale, and rotation to the image, mask, and points using an affine transform.
    """

    def __init__(
        self,
        shift_ratio: ArgFloatHint = (-0.0625, 0.0625),
        scale_ratio: ArgFloatHint = (-0.2, 0.0),
        rotate_limit: ArgFloatHint = (-5, 5),
        inter: int = cv2.INTER_LINEAR,
        inter_mask: int = cv2.INTER_NEAREST,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self._shift_limit = ArgFloatRange.from_arg(shift_ratio, min_val=-1, max_val=1)
        self._scale_limit = ArgFloatRange.from_arg(scale_ratio)
        self._rotate_limit = ArgFloatRange.from_arg(
            rotate_limit, min_val=-360, max_val=360
        )
        self._inter = inter
        self._inter_mask = inter_mask

    def _apply(self, prng: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Sample parameters
            shift_x = self._shift_limit.sample(prng) * w
            shift_y = self._shift_limit.sample(prng) * h
            scale = 1 + self._scale_limit.sample(prng)
            angle = self._rotate_limit.sample(prng)
            # Compute affine matrix
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[:, 2] += [shift_x, shift_y]
            # Warp
            return x.applied_rotation_matrix(
                M, mode="affine", inter=self._inter, inter_mask=self._inter_mask
            )
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
