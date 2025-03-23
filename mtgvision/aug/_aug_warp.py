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

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax.scipy.ndimage import map_coordinates

from mtgvision.aug import AugItems, Augment, AugPrngHint
from mtgvision.aug._base import jax_static_field
from mtgvision.aug._util_args import ArgFloatHint, sample_float
from mtgvision.aug._util_jax import ResizeMethod

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def to_homogeneous(coords):
    """Convert 2D coordinates to homogeneous coordinates by appending ones."""
    return jnp.concatenate([coords, jnp.ones_like(coords[..., :1])], axis=-1)


def generate_grid(shape):
    """Generate a grid of (y, x) coordinates for the given shape."""
    h, w = shape
    y = jnp.arange(h)
    x = jnp.arange(w)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack([yy, xx], axis=-1)


def apply_transform(coords, M):
    """Apply a 3x3 transformation matrix to coordinates."""
    homo_coords = to_homogeneous(coords)
    transformed_homo = jnp.dot(homo_coords, M.T)
    transformed = transformed_homo[..., :2] / (transformed_homo[..., 2:] + 1e-10)
    return transformed


def warp_2d_image(image, M_inv, output_shape, order=1):
    """Warp an image using the inverse transformation matrix."""
    assert image.ndim == 2
    h, w = output_shape
    grid = generate_grid((h, w))  # (h, w, 2), [y, x]
    src_coords = apply_transform(grid, M_inv)  # (h, w, 2), [y_src, x_src]
    src_y = src_coords[..., 0]
    src_x = src_coords[..., 1]
    warped = map_coordinates(
        image, [src_y, src_x], order=order, mode="constant", cval=0.0
    )
    return warped


def warp_rgb_image(image, M_inv, output_shape, order=1):
    assert image.ndim == 3
    assert image.shape[-1] == 3
    return jnp.dstack(
        [
            warp_2d_image(image[:, :, 0], M_inv, output_shape, order),
            warp_2d_image(image[:, :, 1], M_inv, output_shape, order),
            warp_2d_image(image[:, :, 2], M_inv, output_shape, order),
        ]
    )


def warp_mask_image(image, M_inv, output_shape, order=1):
    assert image.ndim == 3
    assert image.shape[-1] == 1
    return warp_2d_image(image[:, :, 0], M_inv, output_shape, order)[:, :, None]


def transform_points(points, M):
    """Transform points using the forward transformation matrix."""
    homo_points = to_homogeneous(points)
    transformed_homo = jnp.dot(homo_points, M.T)
    transformed = transformed_homo[:, :2] / (transformed_homo[:, 2:] + 1e-10)
    return transformed


def _get_rotation_matrix_2d(center, angle, scale):
    """Compute a 2D affine rotation matrix, similar to cv2.getRotationMatrix2D."""
    cx, cy = center
    angle = jnp.deg2rad(angle)
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    M = jnp.array(
        [
            [cos * scale, -sin * scale, cx * (1 - cos * scale) + cy * sin * scale],
            [sin * scale, cos * scale, cy * (1 - cos * scale) - cx * sin * scale],
        ],
        dtype=jnp.float32,
    )
    return M


def _get_perspective_transform(src_pts, dst_pts):
    """Compute a perspective transform matrix, similar to cv2.getPerspectiveTransform."""
    A = jnp.vstack([src_pts.T, jnp.ones((1, 4), dtype=jnp.float32)])
    B = jnp.vstack([dst_pts.T, jnp.ones((1, 4), dtype=jnp.float32)])
    M, _, _, _ = jnp.linalg.lstsq(A.T, B.T, rcond=None)
    return M.T


def applied_rotation_matrix(
    x: AugItems,
    M: jnp.ndarray,
    mode: str,
    inter: ResizeMethod,
    inter_mask: ResizeMethod,
) -> AugItems:
    """Apply an affine or perspective transformation to AugItems."""
    h, w = x.get_bounds_hw()
    if mode == "affine":
        M_full = jnp.vstack([M, jnp.array([0, 0, 1], dtype=M.dtype)])
    elif mode == "perspective":
        M_full = M
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute inverse for warping
    M_inv = jnp.linalg.inv(M_full)

    # Warp image
    im = None
    if x.has_image:
        order = 1 if inter == ResizeMethod.LINEAR else 0
        im = warp_rgb_image(x.image, M_inv, (h, w), order=order)

    # Warp mask
    mask = None
    if x.has_mask:
        order_mask = 0 if inter_mask == ResizeMethod.NEAREST else 1
        mask = warp_mask_image(x.mask, M_inv, (h, w), order=order_mask)

    # Transform points
    points = None
    if x.has_points:
        points = transform_points(x.points, M_full)

    return x.override(image=im, mask=mask, points=points)


# ========================================================================= #
# AUG                                                                       #
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
            im = jnp.fliplr(x.image) if x.has_image else None
            mask = jnp.fliplr(x.mask) if x.has_mask else None
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points = points.at[:, 0].set(w - 1 - points[:, 0])
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
            im = jnp.flipud(x.image) if x.has_image else None
            mask = jnp.flipud(x.mask) if x.has_mask else None
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points = points.at[:, 1].set(h - 1 - points[:, 1])
            return x.override(image=im, mask=mask, points=points)
        return x


@register_dataclass
@dataclass(frozen=True)
class RotateBounded(Augment):
    """
    Rotate the image, mask, and points within a degree range, keeping output size fixed.
    """

    p: float = jax_static_field(default=0.5)
    deg: ArgFloatHint = jax_static_field(default=(0, 360))  # min -360, max 360
    inter: ResizeMethod = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: ResizeMethod = jax_static_field(default=ResizeMethod.NEAREST)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            deg = sample_float(key, self.deg)
            h, w = x.get_bounds_hw()
            cy, cx = h / 2, w / 2
            M = _get_rotation_matrix_2d((cx, cy), deg, 1.0)
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
            h, w = x.get_bounds_hw()
            im = jnp.rot90(x.image, k=2) if x.has_image else None
            mask = jnp.rot90(x.mask, k=2) if x.has_mask else None
            points = x.points.copy() if x.has_points else None
            if points is not None:
                points = points.at[:, 0].set(w - 1 - points[:, 0])
                points = points.at[:, 1].set(h - 1 - points[:, 1])
            return x.override(image=im, mask=mask, points=points)
        return x


@register_dataclass
@dataclass(frozen=True)
class PerspectiveWarp(Augment):
    """
    Apply a random perspective warp by distorting the four corners.
    """

    p: float = jax_static_field(default=0.5)
    corner_jitter_ratio: ArgFloatHint = jax_static_field(default=(-0.1, 0.1))
    corner_jitter_offset: ArgFloatHint = jax_static_field(default=0)
    inter: ResizeMethod = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: ResizeMethod = jax_static_field(default=ResizeMethod.NEAREST)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            h, w = x.get_bounds_hw()
            ran = sample_float(key, self.corner_jitter_ratio, shape=(4, 2))
            offsets = ran * jnp.array(
                [(h, w), (h, -w), (-h, w), (-h, -w)], dtype=jnp.float32
            )
            offsets += sample_float(key, self.corner_jitter_offset, shape=(4, 2))
            src_pts = jnp.array(
                [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], dtype=jnp.float32
            )
            dst_pts = src_pts + offsets
            M = _get_perspective_transform(src_pts, dst_pts)
            return applied_rotation_matrix(
                x, M, mode="perspective", inter=self.inter, inter_mask=self.inter_mask
            )
        return x


@register_dataclass
@dataclass(frozen=True)
class ShiftScaleRotate(Augment):
    """
    Apply random shift, scale, and rotation using an affine transform.
    """

    p: float = jax_static_field(default=0.5)
    shift_ratio: ArgFloatHint = jax_static_field(
        default=(-0.0625, 0.0625)
    )  # min -1, max 1
    scale_ratio: ArgFloatHint = jax_static_field(default=(-0.2, 0.0))
    rotate_limit: ArgFloatHint = jax_static_field(default=(-5, 5))
    inter: ResizeMethod = jax_static_field(default=ResizeMethod.LINEAR)
    inter_mask: ResizeMethod = jax_static_field(default=ResizeMethod.NEAREST)

    def _apply(self, key: AugPrngHint, x: AugItems) -> AugItems:
        if x.has_image or x.has_mask or x.has_points:
            h, w = x.get_bounds_hw()
            shift_x = sample_float(key, self.shift_ratio) * w
            shift_y = sample_float(key, self.shift_ratio) * h
            scale = 1 + sample_float(key, self.scale_ratio)
            angle = sample_float(key, self.rotate_limit)
            M = _get_rotation_matrix_2d((w / 2, h / 2), angle, scale)
            M = M.at[:, 2].set(M[:, 2] + jnp.array([shift_x, shift_y], dtype=M.dtype))
            return applied_rotation_matrix(
                x, M, mode="affine", inter=self.inter, inter_mask=self.inter_mask
            )
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    def _main():
        with jax.disable_jit(disable=False):
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
