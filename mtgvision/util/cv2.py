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


from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from mtgvision.util.image import img_uint8

# a point as accepted by `as_point`: either an ndarray or a plain (x, y) sequence
type Point = np.ndarray | Sequence[float]

# ============================================================================ #
# CV2 Shape Helper Functions                                                   #
# Arrays dont directly contain points, but contains arrays of single elements  #
# ============================================================================ #


def cv2_poly_is_convex(pts: np.ndarray) -> bool:
    if len(pts) < 3:
        raise Exception("Need at least 3 pts")
    total, i = 0, 0
    points = list(np.array(pts).reshape((-1, 2)))
    for (ax, ay), (bx, by), (cx, cy) in zip(
        points, points[1:] + points[:1], points[2:] + points[:2]
    ):
        dx1 = bx - ax
        dy1 = by - ay
        dx2 = cx - bx
        dy2 = cy - by
        total += -1 if (dx1 * dy2 - dy1 * dx2 < 0) else 1
        i += 1
        if abs(total) != i:
            return False
    return True


def cv2_quad_flip_upright(quad: np.ndarray) -> np.ndarray:
    assert len(quad) == 4
    shape = quad.shape
    quad = quad.reshape((-1, 2))
    p0, p1, p2, p3 = quad
    m01, m12, m23, m30 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2, (p3 + p0) / 2
    d1, d2 = np.linalg.norm(m01 - m23), np.linalg.norm(m12 - m30)
    # should result in: tl, bl, br, tr
    if d1 > d2:
        quad[[0, 1, 2, 3]] = quad[[1, 2, 3, 0]]
    return quad.reshape(shape)


def cv2_poly_expand(poly: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    assert len(poly) > 0
    shape = poly.shape
    poly = poly.reshape((-1, 2))
    center = np.average(poly, axis=0)
    poly += np.round((poly - center) * ratio).astype(np.int32)
    return poly.reshape(shape)


def cv2_poly_center(poly: np.ndarray) -> np.ndarray:
    assert len(poly) > 0
    return np.average(poly.reshape((-1, 2)), axis=0)


# ============================================================================ #
# CV2 Image Helper Functions                                                   #
# ============================================================================ #


def cv2_warp_imgs_onto(
    img: np.ndarray,
    cards: Iterable[np.ndarray],
    bounds: Iterable[np.ndarray],
) -> np.ndarray:
    img = img.copy()
    for card, bound in zip(cards, bounds):
        scnvs = img.shape
        scard = card.shape

        card = img_uint8(card)

        src_pts = np.array(
            [(0, 0), (0, scard[0]), (scard[1], scard[0]), (scard[1], 0)],
            dtype=np.float32,
        )
        dst_pts = np.array([p[0] for p in bound], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warp = cv2.warpPerspective(card, M, (scnvs[1], scnvs[0]))
        cv2.fillConvexPoly(img, points=dst_pts.astype(np.int32), color=(0, 0, 0))
        img = cv2.bitwise_or(warp, img)
    return img


def cv2_draw_contours(
    image: np.ndarray,
    contours: Iterable[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> None:
    for contour in contours:
        cv2.drawContours(image, [contour], -1, color, thickness)


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def as_color(color: tuple[int, int, int] | int) -> tuple[int, int, int]:
    """Broadcast a grey level to BGR. cv2 stubs require a sequence, not a scalar."""
    if isinstance(color, int):
        return (color, color, color)
    return color


def as_point(point: Point) -> tuple[int, int]:
    """cv2 stubs type points as a 2-tuple of ints, not an ndarray."""
    x, y = np.asarray(point).astype(int).reshape(2).tolist()
    return int(x), int(y)


def lerp_color(
    color1: tuple[int, int, int] | int,
    color2: tuple[int, int, int] | int,
    t: float,
) -> tuple[int, int, int]:
    a = as_color(color1)
    b = as_color(color2)
    return (
        int(a[0] * (1 - t) + b[0] * t),
        int(a[1] * (1 - t) + b[1] * t),
        int(a[2] * (1 - t) + b[2] * t),
    )


def cv2_draw_poly(
    frame: np.ndarray,
    points: np.ndarray,
    c: tuple[int, int, int] | int = (255, 0, 0),
    color_mod: tuple[int, int, int] | int | None = None,
) -> None:
    poly_color = as_color(c) if color_mod is None else lerp_color(c, color_mod, 0.5)
    cv2.polylines(
        frame,
        [np.asarray(points).astype(int)],
        isClosed=True,
        color=poly_color,
        thickness=1,
    )


def cv2_draw_arrow(
    frame: np.ndarray,
    start: Point,
    end: Point,
    c: tuple[int, int, int] | int = (0, 0, 255),
    color_mod: tuple[int, int, int] | int | None = None,
) -> None:
    arrow_color = as_color(c) if color_mod is None else lerp_color(c, color_mod, 0.5)
    cv2.arrowedLine(
        frame,
        as_point(start),
        as_point(end),
        color=arrow_color,
        thickness=1,
    )


def cv2_draw_text(
    frame: np.ndarray,
    text: str,
    center: Point,
    c: tuple[int, int, int] | int = (0, 0, 255),
    color_mod: tuple[int, int, int] | int | None = None,
    font_scale: float = 0.25,
) -> None:
    color = as_color(c) if color_mod is None else lerp_color(c, color_mod, 0.5)
    cv2.putText(
        frame,
        text,
        as_point(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
    )
