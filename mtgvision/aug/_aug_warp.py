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

from dataclasses import dataclass

import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_dataclass

from mtgvision.aug import AugItems, Augment, AugPrngHint
from mtgvision.aug._base import jax_static_field
from mtgvision.aug._util_args import ArgFloatHint, sample_float
from mtgvision.aug._util_jax import ResizeMethod


def _get_rotation_matrix_2d(center, angle, scale):
    """
    Same as: cv2.getRotationMatrix2D((cx, cy), deg, 1.0), but for jax
    """
    cx, cy = center
    angle = jnp.deg2rad(angle)
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    M = jnp.array([[cos, -sin, cx], [sin, cos, cy], [0, 0, 1]], dtype=jnp.float32)
    return lax.cond(
        scale == 1.0,
        lambda: M,
        lambda: jnp.dot(
            jnp.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=jnp.float32), M
        ),
    )


# MAYBE WRONG?
# def _get_perspective_transform(src_pts, dst_pts):
#     """
#     Same as: cv2.getPerspectiveTransform(src_pts, dst_pts), but for jax
#     """
#     A = jnp.vstack([src_pts.T, jnp.ones((1, 4), dtype=jnp.float32)])
#     B = jnp.vstack([dst_pts.T, jnp.ones((1, 4), dtype=jnp.float32)])
#     M, _, _, _ = jnp.linalg.lstsq(A.T, B.T, rcond=None)
#     return M.T


def applied_matrix(
    M: jnp.ndarray,
    points: jnp.ndarray,
):
    """
    Apply a matrix transformation to a set of points.
    """
    # add extra dim
    points = jnp.hstack([points, jnp.ones((points.shape[0], 1))])
    # apply
    points = jnp.dot(M, points.T)
    # extract
    return points.T[:, :2]


# MAYBE WRONG?
#     def applied_rotation_matrix(
#         self,
#         M: jnp.ndarray,
#         *,
#         mode: Literal["affine", "perspective"],
#         inter: int = cv2.INTER_LINEAR,
#         inter_mask: int = cv2.INTER_NEAREST,
#     ) -> "AugItems":
#         if mode == "affine":
#             warp = cv2.warpAffine
#             invert_transform = cv2.invertAffineTransform
#             transform = cv2.transform
#         else:
#             warp = cv2.warpPerspective
#             invert_transform = jnp.linalg.inv
#             transform = cv2.perspectiveTransform
#         # Get dimensions
#         h, w = self.get_bounds_hw()
#         # Warp image and mask
#         im = warp(self.image, M, (w, h), flags=inter) if self.has_image else None
#         mask = warp(self.mask, M, (w, h), flags=inter_mask) if self.has_mask else None
#         # Transform points
#         points = None
#         if self.has_points:
#             M_inv = invert_transform(M)
#             points = transform(self.points[None, :, :], M_inv)[0]
#         # done
#         return self.override(image=im, mask=mask, points=points)


# WRONG
# def applied_rotation_matrix(
#     x: AugItems,
#     M: jnp.ndarray,
#     *,
#     mode: str,
#     inter: ResizeMethod,
#     inter_mask: ResizeMethod,
# ) -> AugItems:
#     if mode == "affine":
#         warp = jnp.array
#         invert_transform = jnp.linalg.inv
#         transform = jnp.matmul
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
#     # Get dimensions
#     h, w = x.get_bounds_hw()
#     # Warp image and mask
#     im = warp(M, x.image, (w, h), flags=inter) if x.has_image else None
#     mask = warp(M, x.mask, (w, h), flags=inter_mask) if x.has_mask else None
#     # Transform points
#     points = None
#     if x.has_points:
#         M_inv = invert_transform(M)
#         points = transform(M_inv, x.points.T).T
#     # done
#     return x.override(image=im, mask=mask, points=points)


# ========================================================================= #
# WRAP                                                                      #
# ========================================================================= #


@register_dataclass
@dataclass(frozen=True)
class FlipHorizontal(Augment):
    """
    Flip the image, mask, and points horizontally.
    """

    p: float = jax_static_field(default=0.5)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            _, w = x.get_bounds_hw()
            # Flip image
            im = jnp.fliplr(x.image) if x.has_image else None
            # Flip mask
            mask = jnp.fliplr(x.mask) if x.has_mask else None
            # Flip points (x -> W - 1 - x)
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points.at[:, 0].set(w - 1 - points[:, 0])
            return x.override(image=im, mask=mask, points=points)
        return x


@register_dataclass
@dataclass(frozen=True)
class FlipVertical(Augment):
    """
    Flip the image, mask, and points vertically.
    """

    p: float = jax_static_field(default=0.5)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            h, _ = x.get_bounds_hw()
            # Flip image
            im = jnp.flipud(x.image) if x.has_image else None
            # Flip mask
            mask = jnp.flipud(x.mask) if x.has_mask else None
            # Flip points (y -> H - 1 - y)
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points.at[:, 1].set(h - 1 - points[:, 1])
            return x.override(image=im, mask=mask, points=points)
        return x


@register_dataclass
@dataclass(frozen=True)
class RotateBounded(Augment):
    """
    Rotate the image, mask, and points within a specified degree range, adjusting the output size to bound the rotated content.
    """

    p: float = jax_static_field(default=0.5)
    deg: ArgFloatHint = jax_static_field(default=(0, 360))  # min -360, max 360
    inter: int = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: int = jax_static_field(default=ResizeMethod.NEAREST)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            deg = sample_float(key, self.deg)
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Compute center
            cy, cx = h / 2, w / 2
            # Compute rotation matrix
            M = _get_rotation_matrix_2d((cx, cy), deg, 1.0)
            cos, sin = jnp.abs(M[0, 0]), jnp.abs(M[0, 1])
            # New bounding dimensions
            nw = ((h * sin) + (w * cos)).astype(int)
            nh = ((h * cos) + (w * sin)).astype(int)
            # Adjust translation
            M.at[0, 2].set(M[0, 2] + (nw / 2) - cx)
            M.at[1, 2].set(M[1, 2] + (nh / 2) - cy)
            # Warp
            return applied_rotation_matrix(
                x, M, mode="affine", inter=self.inter, inter_mask=self.inter_mask
            )
        return x


@register_dataclass
@dataclass(frozen=True)
class Rotate180(Augment):
    """
    Rotate the image, mask, and points by 180 degrees.
    """

    p: float = jax_static_field(default=0.5)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Rotate image and mask
            im = jnp.rot90(x.image, k=2) if x.has_image else None
            mask = jnp.rot90(x.mask, k=2) if x.has_mask else None
            # Transform points (x, y) -> (W - 1 - x, H - 1 - y)
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points[:, 0] = w - 1 - points[:, 0]
                points[:, 1] = h - 1 - points[:, 1]
            return x.override(image=im, mask=mask, points=points)
        return x


@register_dataclass
@dataclass(frozen=True)
class PerspectiveWarp(Augment):
    """
    Apply a random perspective warp to the image, mask, and points by distorting the four corners.
    """

    p: float = jax_static_field(default=0.5)
    corner_jitter_ratio: ArgFloatHint = jax_static_field(default=(-0.1, 0.1))
    corner_jitter_offset: ArgFloatHint = jax_static_field(default=0)
    inter: int = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: int = jax_static_field(default=ResizeMethod.NEAREST)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Random offsets
            ran = sample_float(key, self.corner_jitter_ratio, shape=(4, 2))
            offsets = ran * jnp.array(
                [(h, w), (h, -w), (-h, w), (-h, -w)], dtype=jnp.float32
            )
            offsets = sample_float(key, self.corner_jitter_offset, shape=(4, 2))
            # Define source points (corners)
            src_pts = jnp.array(
                [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], dtype=jnp.float32
            )
            dst_pts = src_pts + offsets
            # Compute perspective transform
            M = _get_perspective_transform(src_pts, dst_pts)
            # Warp
            return applied_rotation_matrix(
                x, M, mode="perspective", inter=self.inter, inter_mask=self.inter_mask
            )
        return x


class ShiftScaleRotate(Augment):
    """
    Apply random shift, scale, and rotation to the image, mask, and points using an affine transform.
    """

    p: float = jax_static_field(default=0.5)
    shift_ratio: ArgFloatHint = jax_static_field(default=(-0.0625, 0.0625))  # m -1, M 1
    scale_ratio: ArgFloatHint = jax_static_field(default=(-0.2, 0.0))
    rotate_limit: ArgFloatHint = jax_static_field(default=(-5, 5))
    inter: int = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: int = jax_static_field(default=ResizeMethod.NEAREST)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            # Determine dimensions
            h, w = x.get_bounds_hw()
            # Sample parameters
            shift_x = sample_float(key, self.shift_ratio) * w
            shift_y = sample_float(key, self.shift_ratio) * h
            scale = 1 + sample_float(key, self.scale_ratio)
            angle = sample_float(key, self.rotate_limit)
            # Compute affine matrix
            M = _get_rotation_matrix_2d((w / 2, h / 2), angle, scale)
            M[:, 2] += jnp.asarray([shift_x, shift_y])
            # Warp
            return applied_rotation_matrix(
                x, M, mode="affine", inter=self.inter, inter_mask=self.inter_mask
            )
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    def _main():
        for Aug in [
            FlipHorizontal,
            FlipVertical,
            RotateBounded,
            Rotate180,
            PerspectiveWarp,
            ShiftScaleRotate,
        ]:
            Aug().quick_test()

    _main()
