import functools
import math
import os
import random
import warnings
from pathlib import Path
from typing import Literal, Optional, TypedDict

import albumentations as A
import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon
from tqdm import tqdm

from mtgvision.encoder_datasets import (
    CocoValImages,
    IlsvrcImages,
    SyntheticBgFgMtgImages,
)
from mtgvision.util.image import imread_float, imshow_loop, imwrite, round_rect_mask
from mtgvision.util.random import seed_all

# ========================================================================= #
# CV2 helper                                                                #
# ========================================================================= #


def corner_jitter_2d(
    pts: np.ndarray, jitter_ratio: float, center: tuple[float, float] = None
) -> np.ndarray:
    assert pts.shape[-1] == 2
    if center is None:
        center = np.mean(pts, axis=0)
    # get jitter
    deltas = np.linalg.norm((pts - center), axis=-1)
    deltas *= np.random.uniform(1 - jitter_ratio, 1 + jitter_ratio, size=deltas.shape)
    # recompute corners based on new deltas
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = np.stack(
        [
            center[0] + deltas * np.cos(angles),
            center[1] + deltas * np.sin(angles),
        ],
        axis=-1,
    )
    return pts


def rotate_2d(
    pts: np.ndarray, deg: float, center: tuple[float, float] = (0, 0), scale: float = 1
) -> np.ndarray:
    M = np.identity(3)
    m = cv2.getRotationMatrix2D(center, deg, scale)
    M[:2, :] = m
    return apply_transform_2d(pts, M)


def translate_2d(pts: np.ndarray, dx: float, dy: float) -> np.ndarray:
    assert pts.shape[-1] == 2
    return pts + np.array([dx, dy])


def apply_transform_2d(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    assert M.shape == (3, 3)
    assert pts.shape[-1] == 2
    pts = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1)
    pts = pts @ M.T
    pts = pts[..., :2] / pts[..., 2:3]
    return pts


def apply_transform_2d_img(
    img: np.ndarray,
    M: np.ndarray,
    out_size_hw: tuple[int, int] | int | None = None,
) -> np.ndarray:
    if out_size_hw is None:
        out_size_hw = img.shape[:2]
    else:
        out_size_hw = get_hw(out_size_hw)
    return cv2.warpPerspective(img, M, out_size_hw[::-1], flags=cv2.INTER_LINEAR)


def get_rotate_over_output_transform(
    in_size_hw: tuple[int, int] | int,
    deg: float,
    out_size_hw: tuple[int, int] | int,
    mode: Literal["cover", "inside"] = "cover",
    scale_factor: float = 1.0,
):
    h, w = get_hw(in_size_hw)
    oh, ow = get_hw(out_size_hw)
    # get scale
    if mode == "cover":
        # scale the image to always exactly cover the entire background no matter the
        # rotation
        scale = math.hypot(oh / max(ow, oh), ow / max(ow, oh)) * max(oh, ow) / min(h, w)
    elif mode == "inside":
        # scale the image to always exactly be inside the entire background no matter
        # the rotation
        scale = min(oh, ow) / math.hypot(w, h)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # rotate around center and scale to larger than bg_size
    M0 = cv2.getRotationMatrix2D((w // 2, h // 2), deg, scale * scale_factor)
    M0 = np.concatenate([M0, np.array([[0, 0, 1]])], axis=0)
    # translate to center of out image
    M1 = np.array(
        [
            [1, 0, (ow - w) // 2],
            [0, 1, (oh - h) // 2],
            [0, 0, 1],
        ]
    )
    M = M1 @ M0
    assert M.shape == (3, 3)
    return M


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


class Sample(TypedDict, total=False):
    # shape (h, w, 3), float32, [0, 1]
    image: np.ndarray
    # shape (h, w), float32, [0, 1]
    mask: np.ndarray
    # shape (-1, 2), float32, [0, 1]
    # * usually 4 corners so shape (N, 4, 2), can also be segment masks (N, M, 2)
    keypoints: np.ndarray
    # shape (N,)
    keypoints_labels: np.ndarray


def get_hw(hw: tuple[int, int] | int | Sample) -> tuple[int, int]:
    if isinstance(hw, (int, float)):
        return (hw, hw)
    elif isinstance(hw, (list, tuple)):
        h, w = hw[:2]
        return h, w
    raise ValueError(f"Invalid type: {type(hw)}")


# ========================================================================= #
# Augments                                                                  #
# ========================================================================= #


def rotate_over_output(
    x: "Sample",
    *,
    deg: float,
    out_size_hw: tuple[int, int] | int,
    mode: Literal["cover", "inside"] = "cover",
    scale_factor: float = 1.0,
) -> "Sample":
    # get inputs
    image = x["image"]
    mask = x.get("mask")
    pts = x.get("keypoints")
    # checks
    in_size_hw = get_hw(image.shape)
    out_size_hw = get_hw(out_size_hw)
    # get transform
    M = get_rotate_over_output_transform(
        in_size_hw=in_size_hw,
        deg=deg,
        out_size_hw=out_size_hw,
        mode=mode,
        scale_factor=scale_factor,
    )
    # transform
    out: Sample = {}
    if image is not None:
        assert image.ndim == 3 and image.shape[-1] == 3
        out["image"] = apply_transform_2d_img(image, M, out_size_hw=out_size_hw)
    if mask is not None:
        assert image.ndim == 2 and mask.shape == image.shape[:2]
        out["mask"] = apply_transform_2d_img(mask, M, out_size_hw=out_size_hw)
    if pts is not None:
        assert pts.shape[-1] == 2
        out["keypoints"] = apply_transform_2d(pts, M)
    # done
    return out


# ========================================================================= #
# Make - BG                                                                 #
# ========================================================================= #


def make_background(bg: np.ndarray, bg_size: tuple[int, int]) -> np.ndarray:
    """
    Load and rotate/resize a random background image to encompass the target shape
    """
    bg = rotate_over_output(
        {"image": bg}, deg=np.random.randint(0, 360), out_size_hw=bg_size, mode="cover"
    )
    # augment colors and things like that
    return bg["image"]


def make_aug_background(bg: np.ndarray, bg_size: tuple[int, int]) -> np.ndarray:
    bg = make_background(bg, bg_size)
    bg = get_bg_transform_light()(image=bg)["image"]
    bg = get_bg_transform(extra=True)(image=bg)["image"]
    return bg


# ========================================================================= #
# Make - Card                                                               #
# ========================================================================= #


def make_card_with_mask(
    card: np.ndarray,
    *,
    keypoint_margin_ratio: float = 0.03,
    keypoint_size_ratio: float = 0.5,
    corner_radius_ratio: float = 0.046,
    kind: Literal["obb", "seg"] = "obb",
) -> Sample:
    """
    Generate a transformed card image with its mask and transformed keypoints.
    """

    # Load random card image (RGB, not RGBA)
    h, w = card.shape[:2]

    # Generate rounded rectangle mask
    mask = round_rect_mask((h, w), radius_ratio=corner_radius_ratio)

    def _box(lft, top, rht, bot, margin=0.0, mlr=1.0, mrr=1.0, mtr=1.0, mbr=1.0):
        return [
            (lft + margin * mlr, top + margin * mtr),
            (rht - margin * mrr, top + margin * mtr),
            (rht - margin * mrr, bot - margin * mbr),
            (lft + margin * mlr, bot - margin * mbr),
        ]

    if kind == "obb":
        # Define initial corner keypoints
        # - we can't get exact orientation, even with rotated bounding box
        #   so we specify top and bottom regions with different classes
        #   so we can compute this later.
        r = keypoint_size_ratio
        m = keypoint_margin_ratio * max(w, h)
        keypoints = np.asarray(
            [
                _box(0, 0, w, h, margin=0),  # card
                _box(0, 0, w, r * h, margin=m, mbr=0.5),  # top
                _box(0, (1 - r) * h, w, h, margin=m, mtr=0.5),  # bottom
            ]
        )
    elif kind == "seg":
        # make polygon with cutout from the bottom so we can figure out orientation
        card_box = _box(0, 0, w, h)
        bottom_indent = _box(w * 0.4, h * 0.5, w * 0.6, h * 1.1)
        card_box = Polygon(card_box).difference(Polygon(bottom_indent))
        # remove last point
        segment_points = np.asarray(card_box.exterior.coords)
        assert np.allclose(segment_points[0], segment_points[-1])
        segment_points = segment_points[:-1]
        # round polygons, e.g. make corners less sharp. This can be done with
        keypoints = np.asarray([segment_points])
    else:
        raise KeyError(f"invalid: {kind}")

    # done!
    return {
        "image": card,
        "mask": mask,
        "keypoints": keypoints,
        "keypoints_labels": np.arange(len(keypoints)),
        "bbox": np.asarray(_box(0, 0, w, h)),  # used for colision later
    }


# ========================================================================= #
# Make - Composite Card                                                     #
# ========================================================================= #


def place_card_on_background_get_transform(
    # card
    card_sample: Sample,
    # bg
    bg: np.ndarray,
    bg_existing_polygons: list[Polygon],
    *,
    # scaling
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.9,
    size_sample_mode: Literal["uniform", "log_uniform"] = "log_uniform",
    # collision
    min_visible: float = 0.5,
    min_visible_edge: Optional[float] = 1.0,
    no_contains: bool = True,
    jitter_ratio: float = 0.25,
    max_attempts: int = 10,
) -> np.ndarray | None:
    """
    Place a card on the background using rejection sampling to minimize overlap.
    """
    bh, bw = get_hw(bg.shape)
    ch, cw = get_hw(card_sample["image"].shape)
    card_diag = int(math.hypot(ch, cw))

    if min_visible_edge is None:
        min_visible_edge = min_visible
    min_visible_edge = max(min_visible, min_visible_edge)

    # try place card on image
    for _ in range(max_attempts):
        # random card center somewhere in bg, allow some edge overlap considering
        # rotating
        edge_pad = card_diag // 2
        edge_ovr = int(card_diag * (1 - min_visible_edge))
        cx = random.randint(0 + edge_pad - edge_ovr, bw - edge_pad + edge_ovr)
        cy = random.randint(0 + edge_pad - edge_ovr, bh - edge_pad + edge_ovr)
        # get random rotation
        deg = np.random.uniform(0, 360)
        # get random scaling
        min_area = bh * bw * min_area_ratio
        max_area = bh * bw * max_area_ratio
        if size_sample_mode == "uniform":
            target_area = np.random.uniform(min_area, max_area)
        elif size_sample_mode == "log_uniform":
            target_area = np.exp(np.random.uniform(np.log(min_area), np.log(max_area)))
        else:
            raise KeyError(size_sample_mode)
        # get scale
        scale = target_area / (ch * cw)
        # print(full_scale, min_area, max_area, target_area, scale)
        # calculate transform to warp card into position
        src_pts = np.asarray([(0, 0), (cw, 0), (cw, ch), (0, ch)])
        dst_pts = src_pts.copy()
        dst_pts = corner_jitter_2d(dst_pts, jitter_ratio=jitter_ratio)
        dst_pts = rotate_2d(dst_pts, deg=deg, center=(cw / 2, ch / 2), scale=scale)
        dst_pts = translate_2d(
            dst_pts, dx=cx - (cw / 2) * scale, dy=cy - (ch / 2) * scale
        )
        # transform matrix
        M = cv2.getPerspectiveTransform(
            src_pts.astype(np.float32), dst_pts.astype(np.float32)
        )
        # warp points
        keypoints = card_sample["keypoints"]  # shape (-1, 4, 2), first [0] is card
        keypoints = apply_transform_2d(keypoints, M)
        # get shapely polygons
        img_bound = Polygon([(0, 0), (bw, 0), (bw, bh), (0, bh)])
        card_bound = Polygon(keypoints[0])
        # check intersection of card with image bounds area is at least 1 - max_overlap
        card_visible = img_bound.intersection(card_bound)
        if card_visible.area / card_bound.area < min_visible_edge:
            continue
        # check exclusion/difference of card with other cards
        visible = True
        for p in (Polygon(p) for p in bg_existing_polygons):
            if card_visible.difference(p).area / card_bound.area < min_visible:
                visible = False
                break
            if p.difference(card_visible).area / p.area < min_visible:
                visible = False
                break
            if no_contains and p.contains(card_visible) or card_visible.contains(p):
                visible = False
        if not visible:
            continue
        # make transform
        return M

    # failed to place card
    return None


# ========================================================================= #
# Transform Helper                                                          #
# ========================================================================= #


def compose(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return A.Compose([fn(*args, **kwargs)], p=1)

    return wrapper


def random_order(*transforms, n=None, replace=False, p=1.0):
    # random order force applies, so need wrapper component to actually do probabilities
    return A.RandomOrder(
        transforms=[A.Sequential([t]) for t in transforms],
        n=len(transforms) if n is None else n,
        replace=replace,
        p=p,
    )


def one_of(*transforms, p=1.0):
    # one of force applies, so need wrapper component to actually do probabilities
    return A.OneOf(
        transforms=[A.Sequential([t]) for t in transforms],
        p=p,
    )


def seq(*transforms, p=1.0):
    return A.Sequential([t for t in transforms], p=p)


# ========================================================================= #
# Transform Makers                                                          #
# ========================================================================= #


@compose
def get_bg_transform_light():
    return random_order(
        A.RandomBrightnessContrast(
            brightness_limit=(-0.4, 0.4),
            contrast_limit=(-0.4, 0.4),
            p=0.5,
        ),
        A.GaussianBlur(sigma_limit=(0, 2), p=0.2),
        A.GaussNoise(std_range=(0.0, 0.1), p=0.2),
        A.Erasing(
            scale=(0.02, 0.2),
            fill=np.random.choice(
                ["random", "random_uniform", 1, 0], p=[0.1, 0.5, 0.2, 0.2]
            ),
            p=0.4,
        ),
        n=3,
    )


@compose
def get_bg_transform(extra: bool = False):
    def _noise_blur(p: float):
        return [
            one_of(
                A.GaussNoise(std_range=(0.0, 0.2), p=p),
                A.ISONoise(color_shift=(0.01, 0.4), p=p),
                A.ShotNoise(scale_range=(0.1, 0.3), p=p),
            ),
            one_of(
                A.GaussianBlur(sigma_limit=(0, 3), p=p),
                A.MedianBlur(blur_limit=(3, 7), p=p),
                A.MotionBlur(blur_limit=(3, 11), p=p),
                A.MotionBlur(blur_limit=(3, 11), p=p),
                A.GlassBlur(sigma=0.5, max_delta=4, iterations=2, p=p / 3 * 2),
            ),
        ]

    return random_order(
        A.RandomBrightnessContrast(
            brightness_limit=(-0.4, 0.4) if not extra else (-0.7, 0.7),
            contrast_limit=(-0.5, 0.5),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=(-30, 30),
            sat_shift_limit=(-40, 40),
            val_shift_limit=(0, 0) if not extra else (-30, 30),
            p=0.5,
        ),
        *_noise_blur(p=0.5),
        *_noise_blur(p=0.1),
        *(
            []
            if not extra
            else [
                A.Erasing(
                    scale=(0.02, 0.4),
                    fill=np.random.choice(
                        ["random", "random_uniform", 1, 0], p=[0.1, 0.1, 0.4, 0.4]
                    ),
                    p=0.4,
                ),
            ]
        ),
        n=4,
    )


@compose
def get_card_transform():
    return random_order(
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.4, -0.4),
            p=0.8,
        ),
        A.HueSaturationValue(
            hue_shift_limit=(-30, 30),
            sat_shift_limit=(-40, 40),
            val_shift_limit=(0, 0),
            p=0.8,
        ),
        A.Erasing(
            scale=(0.02, 0.2),
            fill=np.random.choice(
                ["random", "random_uniform", 1, 0], p=[0.1, 0.5, 0.2, 0.2]
            ),
            p=0.3,
        ),
        n=2,
    )


# ========================================================================= #
# Make - Composite Card                                                     #
# ========================================================================= #


def generate_synthetic_image(
    mtg_ds: SyntheticBgFgMtgImages,
    bg_ds: IlsvrcImages,
    *,
    bg_size_hw: tuple[int, int] | int = 640,
    num_cards_min: int = 1,
    num_cards_max: int = 10,
    card_min_visible_ratio: float = 0.5,
    card_min_visible_ratio_edges: Optional[float] = 1.0,
    card_jitter_ratio: float = 0.25,
    card_min_area_ratio: float = 0.01,
    card_max_area_ratio: float = 0.9,
    card_size_sample_mode: Literal["uniform", "log_uniform"] = "log_uniform",
    card_no_contains: bool = True,
    card_max_place_attempts: int = 10,
    kind: Literal["obb", "seg"] = "obb",
):
    """
    Generate a synthetic image with cards and their rotated bounding box annotations.
    """
    # no warps, lighter
    pre_transform_bg = get_bg_transform_light()
    # only color / noise, no blur
    pre_transform_card = get_card_transform()
    # no warps
    post_transform_bg = get_bg_transform()

    # create BG
    bg = imread_float(bg_ds.ran_path())
    bg = make_background(bg, bg_size_hw)

    # pre-augment-bg
    bg = pre_transform_bg(image=bg)["image"]

    # place cards
    card_samples: list[Sample] = []
    card_Ms: list[np.ndarray] = []
    card_collide: list[np.ndarray] = []
    for i in range(np.random.randint(num_cards_min, num_cards_max)):
        card_sample = make_card_with_mask(
            card=imread_float(mtg_ds.ran_path()),
            kind=kind,
        )
        M = place_card_on_background_get_transform(
            card_sample,
            bg,
            bg_existing_polygons=card_collide,
            min_visible=card_min_visible_ratio,
            min_visible_edge=card_min_visible_ratio_edges,
            jitter_ratio=card_jitter_ratio,
            # scaling
            min_area_ratio=card_min_area_ratio,
            max_area_ratio=card_max_area_ratio,
            size_sample_mode=card_size_sample_mode,
            # collision
            no_contains=card_no_contains,
            max_attempts=card_max_place_attempts,
        )
        if M is None:
            continue
        # transform
        card_sample["image"] = pre_transform_card(image=card_sample["image"])["image"]
        # done
        card_samples.append(card_sample)
        card_Ms.append(M)
        card_collide.append(apply_transform_2d(card_sample["bbox"], M))
    assert len(card_samples) == len(card_Ms)
    assert len(card_samples) == len(card_collide)

    # warp cards onto images, we need to apply to image in reverse order
    # because of overlap checks. Cards added later collide with ALL other cards, cards
    # added first don't check with later cards. So these need to be ON TOP.
    keypoints = []
    keypoints_labels = []
    for sample, M in list(zip(card_samples, card_Ms))[::-1]:
        mask = apply_transform_2d_img(sample["mask"], M, out_size_hw=bg.shape)
        img = apply_transform_2d_img(sample["image"], M, out_size_hw=bg.shape)
        pts = apply_transform_2d(sample["keypoints"], M)
        # merge img onto bg
        bg = mask[:, :, None] * img + (1 - mask[:, :, None]) * bg
        keypoints.extend(pts)
        keypoints_labels.extend(sample["keypoints_labels"])

    # post-augment bg
    bg = post_transform_bg(image=bg)["image"]

    # done!
    return {
        "image": bg,
        "keypoints": np.asarray(keypoints),
        "keypoints_labels": np.asarray(keypoints_labels),
    }


# ========================================================================= #
# Generator                                                                 #
# ========================================================================= #


class Gen:
    def __init__(
        self,
        *,
        bg_size_hw: tuple[int, int] | int = 640,
        num_cards_min: int = 1,
        num_cards_max: int = 10,
        card_min_visible_ratio: float = 0.5,
        card_min_visible_ratio_edges: Optional[float] = 1.0,
        card_jitter_ratio: float = 0.3,
        card_min_area_ratio: float = 0.02,
        card_max_area_ratio: float = 0.9,
        card_size_sample_mode: Literal["uniform", "log_uniform"] = "log_uniform",
        card_no_contains: bool = True,
        card_max_place_attempts: int = 10,
        ratio_bg: Optional[float] = None,
        # 50000 vs 5000
        ilsvrc_vs_coco_sample_weights: tuple[float, float] | None = (1.0, 1.0),
        # segment
        kind: Literal["obb", "seg"] = "obb",
    ):
        # Initialize datasets (replace with your actual classes)
        self.mtg_ds = SyntheticBgFgMtgImages(img_type="small")
        self.bg_ds = IlsvrcImages()
        self.bg2_ds = CocoValImages()

        # params
        self.bg_size_hw = bg_size_hw
        self.num_cards_min = num_cards_min
        self.num_cards_max = num_cards_max
        self.card_min_visible_ratio = card_min_visible_ratio
        self.card_min_visible_ratio_edges = card_min_visible_ratio_edges
        self.card_jitter_ratio = card_jitter_ratio
        self.card_min_area_ratio = card_min_area_ratio
        self.card_max_area_ratio = card_max_area_ratio
        self.card_size_sample_mode = card_size_sample_mode
        self.card_no_contains = card_no_contains
        self.card_max_place_attempts = card_max_place_attempts

        # other data
        self.ratio_bg = ratio_bg
        self.kind = kind

        # get sampling weights
        if ilsvrc_vs_coco_sample_weights is None:
            p = np.asarray([len(self.bg_ds), len(self.bg2_ds)])
        else:
            p = ilsvrc_vs_coco_sample_weights
        self._bg_p = p / np.sum(p)
        self._bg_arr = [self.bg_ds, self.bg2_ds]

    def _get_bg_ds(self):
        idx = np.random.choice(2, p=self._bg_p)
        return self._bg_arr[idx]

    def random_bg(self):
        bg = make_aug_background(
            bg=imread_float(self._get_bg_ds().ran_path()),
            bg_size=self.bg_size_hw,
        )
        return {
            "image": bg,
            "keypoints": [],
            "keypoints_labels": [],
        }

    def random(self):
        if self.ratio_bg and np.random.uniform(0, 1) < self.ratio_bg:
            return self.random_bg()
        sample = generate_synthetic_image(
            self.mtg_ds,
            self._get_bg_ds(),
            bg_size_hw=self.bg_size_hw,
            num_cards_min=self.num_cards_min,
            num_cards_max=self.num_cards_max,
            card_min_visible_ratio=self.card_min_visible_ratio,
            card_min_visible_ratio_edges=self.card_min_visible_ratio_edges,
            card_jitter_ratio=self.card_jitter_ratio,
            card_min_area_ratio=self.card_min_area_ratio,
            card_max_area_ratio=self.card_max_area_ratio,
            card_size_sample_mode=self.card_size_sample_mode,
            card_no_contains=self.card_no_contains,
            card_max_place_attempts=self.card_max_place_attempts,
            kind=self.kind,
        )
        return sample

    def debug_show_loop(self, n: int = None):
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
        ]
        count = 0
        while n is None or count < n:
            sample = self.random()
            # draw keypoints to image
            img = sample["image"]
            for bbox, label in zip(sample["keypoints"], sample["keypoints_labels"]):
                for (x0, y0), (x1, y1) in zip(bbox, np.roll(bbox, 1, axis=0)):
                    cv2.line(
                        img, (int(x0), int(y0)), (int(x1), int(y1)), color=colors[label]
                    )
            # done
            imshow_loop(img, "synthetic")
            count += 1


# ========================================================================= #
# Yolo                                                                      #
# ========================================================================= #


def create_yolo_obb_dataset(
    generator: Gen,
    *,
    output_dir: str,
    num_train: int = 20000,
    num_val_ratio: float = 0.1,
    num_test_ratio: float = 0.1,
    ext: Literal["png", "jpg"] = "jpg",
):
    """
    Generate a YOLO dataset with synthetic images and annotations.
    """
    output_dir = Path(output_dir)

    if output_dir.exists() and len(list(output_dir.iterdir())) > 0:
        raise FileExistsError(f"not clean... {output_dir}")

    # Create output directories
    yaml_file = os.path.join(output_dir, "mtg_obb.yaml")
    img_dir = output_dir / "images"  # / "train" or "val" or "test"
    label_dir = output_dir / "labels"  # / "train" or "val" or "test"
    img_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)

    # generate yaml dataset descriptor
    with open(yaml_file, "w") as fp:
        yaml.safe_dump(
            {
                "path": ".",
                "train": str(img_dir / "train"),
                "val": str(img_dir / "val"),
                "test": str(img_dir / "test"),
                "names": {
                    0: "card",
                    1: "card_top",
                    2: "card_bottom",
                },
            },
            fp,
        )

    num_val = int(num_val_ratio * num_train)
    num_test = int(num_test_ratio * num_train)

    # Generate train dataset
    for name, num in [("train", num_train), ("val", num_val), ("test", num_test)]:
        # make dirs
        curr_img_dir = img_dir / name
        curr_label_dir = label_dir / name
        curr_img_dir.mkdir(exist_ok=True, parents=True)
        curr_label_dir.mkdir(exist_ok=True, parents=True)
        # generate data
        for i in tqdm(range(num), desc=f"Generating {name} Dataset"):
            save_sample(
                generator.random(),
                i=i,
                label_dir=curr_label_dir,
                img_dir=curr_img_dir,
                ext=ext,
            )


def save_sample(
    sample: Sample,
    i: int,
    label_dir: str | Path,
    img_dir: str | Path,
    ext: Literal["png", "jpg"] = "jpg",
):
    img = sample["image"]
    kps = sample["keypoints"]
    kps_labels = sample["keypoints_labels"]

    # checks
    assert len(kps) == len(kps_labels)
    assert img.ndim == 3
    assert img.shape == (640, 640, 3)

    # make obb annotations from keypoints and keypoints_labels
    # https://docs.ultralytics.com/datasets/obb/
    annotations = []
    for j, (pts, label) in enumerate(zip(kps, kps_labels)):
        pts /= img.shape[:2][::-1]  # points are (w, h), not (h, w) like shape
        if np.any(pts < 0) or np.any(pts > 1):
            warnings.warn(
                f"points for image {i} are out of bounds, yolo will consider these "
                "invalid... other libs might not. Set `card_min_visible_ratio_edges=0` "
                "and use mosaic/translation during training instead."
            )
        annotations.append(f"{label} {' '.join(map(str, pts.flatten()))}")

    # Save annotations
    label_path = os.path.join(label_dir, f"image_{i:04d}.txt")
    with open(label_path, "w") as f:
        for ann in annotations:
            f.write(ann + "\n")

    # Save image
    # Placeholder: assume converts float32 [0, 1] to uint8 [0, 255]
    img_path = os.path.join(img_dir, f"image_{i:04d}.{ext}")
    imwrite(img_path, sample["image"])


# ========================================================================= #
# Test                                                                      #
# ========================================================================= #


if __name__ == "__main__":
    # seed_all(42)
    # gen = Gen(card_min_visible_ratio=0.5, card_min_visible_ratio_edges=0.5, num_train=10000)
    # create_yolo_obb_dataset(gen, output_dir="../data/yolo_mtg_dataset", ext="png")

    # seed_all(42)
    # gen = Gen()
    # create_yolo_obb_dataset(gen, output_dir="../data/yolo_mtg_dataset_v2", ext="jpg")

    # seed_all(42)
    # gen = Gen(card_min_visible_ratio=0.4, card_jitter_ratio=0.7)
    # create_yolo_obb_dataset(gen, output_dir="../data/yolo_mtg_dataset_v2_tune", num_train=2500, num_val=250, ext="jpg")

    # seed_all(42)
    # gen = Gen(card_min_visible_ratio=0.4, card_min_visible_ratio_edges=0.75, card_jitter_ratio=0.7)
    # create_yolo_obb_dataset(gen, output_dir="../data/yolo_mtg_dataset_v2_tune_B", num_train=5000, ext="jpg")

    # seed_all(42)
    # gen = Gen(card_min_visible_ratio=0.5, card_min_visible_ratio_edges=0.75, card_jitter_ratio=0.7, ratio_bg=0.1)
    # create_yolo_obb_dataset(gen, output_dir="../data/yolo_mtg_dataset_v3", num_train=10000, ext="jpg")

    seed_all(42)
    gen = Gen(
        card_min_visible_ratio=0.5,
        card_min_visible_ratio_edges=0.0,
        card_jitter_ratio=0.7,
        ratio_bg=0.1,
        kind="seg",
    )
    create_yolo_obb_dataset(
        gen, output_dir="../data/yolo_mtg_dataset_seg", num_train=10000, ext="jpg"
    )
