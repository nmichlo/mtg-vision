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


import numpy as np
import cv2

from mtgvision.util.image import img_uint8


# ============================================================================ #
# CV2 Shape Helper Functions                                                   #
# Arrays dont directly contain points, but contains arrays of single elements  #
# ============================================================================ #


def cv2_poly_is_convex(pts):
    if len(pts) < 3:
        raise Exception("Need at least 3 pts")
    total, i, pts = 0, 0, list(np.array(pts).reshape((-1, 2)))
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:] + pts[:1], pts[2:] + pts[:2]):
        dx1 = bx - ax
        dy1 = by - ay
        dx2 = cx - bx
        dy2 = cy - by
        total += -1 if (dx1 * dy2 - dy1 * dx2 < 0) else 1
        i += 1
        if abs(total) != i:
            return False
    return True


def cv2_quad_flip_upright(quad):
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


def cv2_poly_expand(poly, ratio=0.05):
    assert len(poly) > 0
    shape = poly.shape
    poly = poly.reshape((-1, 2))
    center = np.average(poly, axis=0)
    poly += np.round((poly - center) * ratio).astype(np.int32)
    return poly.reshape(shape)


def cv2_poly_center(poly):
    assert len(poly) > 0
    return np.average(poly.reshape((-1, 2)), axis=0)


# ============================================================================ #
# CV2 Image Helper Functions                                                   #
# ============================================================================ #


def cv2_warp_imgs_onto(img, cards, bounds):
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


def cv2_draw_contours(image, contours, color=(0, 255, 0), thickness=1):
    for contour in contours:
        cv2.drawContours(image, [contour], -1, color, thickness)
