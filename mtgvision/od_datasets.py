import math
import os
from typing import Literal, TypedDict

from tqdm import tqdm

from mtgvision.encoder_datasets import IlsvrcImages, SyntheticBgFgMtgImages
from mtgvision.util.image import imread_float, imshow_loop, imwrite, round_rect_mask


def init_dir(base, sub):
    os.makedirs(os.path.join(base, sub), exist_ok=True)
    return os.path.join(base, sub)


import random
import numpy as np
import cv2
from shapely.geometry import Polygon
import albumentations as A


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
        # scale the image to always exactly cover the entire background no matter the rotation
        scale = math.hypot(oh / max(ow, oh), ow / max(ow, oh)) * max(oh, ow) / min(h, w)
    elif mode == "inside":
        # scale the image to always exactly be inside the entire background no matter the rotation
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
    # * usually 4 corners so shape (N*4, 2) which can be reshaped to (N, 4, 2)
    keypoints: np.ndarray


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


# ========================================================================= #
# Make - Card                                                               #
# ========================================================================= #


def make_card_with_mask(
    card: np.ndarray,
    *,
    keypoint_margin_ratio: float = 0.03,
    keypoint_size_ratio: float = 0.5,
    corner_radius_ratio: float = 0.046,
) -> Sample:
    """
    Generate a transformed card image with its mask and transformed keypoints.
    """

    def _box(l, t, r, b, m=0, mlr=1, mrr=1, mtr=1, mbr=1):
        return [
            (l + m * mlr, t + m * mtr),
            (r - m * mrr, t + m * mtr),
            (r - m * mrr, b - m * mbr),
            (l + m * mlr, b - m * mbr),
        ]

    # Load random card image (RGB, not RGBA)
    h, w = card.shape[:2]

    # Generate rounded rectangle mask
    mask = round_rect_mask((h, w), radius_ratio=corner_radius_ratio)

    # Define initial corner keypoints
    # - we can't get exact orientation, even with rotated bounding box
    #   so we specify top and bottom regions with different classes
    #   so we can compute this later.
    r = keypoint_size_ratio
    m = keypoint_margin_ratio * max(w, h)
    keypoints = np.asarray(
        [
            _box(0, 0, w, h, m=0),  # card
            _box(0, 0, w, r * h, m=m, mbr=0.5),  # top
            _box(0, (1 - r) * h, w, h, m=m, mtr=0.5),  # bottom
        ]
    )
    return {
        "image": card,
        "mask": mask,
        "keypoints": keypoints,
        "keypoints_labels": np.arange(len(keypoints)),
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
    # scaling
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.9,
    size_sample_mode: Literal["uniform", "log_uniform"] = "log_uniform",
    # collision
    min_visible: float = 0.5,
    min_visible_edge: float = None,
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
        # random card center somewhere in bg, allow some edge overlap considering rotating
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


def generate_synthetic_image(
    mtg_ds: SyntheticBgFgMtgImages,
    bg_ds: IlsvrcImages,
    *,
    bg_size_hw: tuple[int, int] | int = 640,
    num_cards_min: int = 1,
    num_cards_max: int = 10,
    min_card_visible: float = 0.5,
    min_card_visible_edge: float = None,
    card_jitter_ratio: float = 0.25,
):
    """
    Generate a synthetic image with cards and their rotated bounding box annotations.
    """
    # augments
    pre_transform_bg = A.RandomOrder(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.5), contrast_limit=(-0.5, -0.5), p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-30, 30),
                sat_shift_limit=(-40, 40),
                val_shift_limit=(0, 20),
                p=0.3,
            ),
            A.GaussianBlur(sigma_limit=(0, 2), p=0.1),
            A.GaussNoise(std_range=(0.0, 0.1), p=0.1),
        ],
        n=3,
    )
    post_transform_bg = A.RandomOrder(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.5), contrast_limit=(-0.5, -0.5), p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-30, 30),
                sat_shift_limit=(-40, 40),
                val_shift_limit=(0, 20),
                p=0.3,
            ),
            A.GaussianBlur(sigma_limit=(0, 2), p=0.2),
            A.GaussNoise(std_range=(0.0, 0.3), p=0.2),
        ]
    )

    # create BG
    bg = imread_float(bg_ds.ran_path())
    bg = make_background(bg, bg_size_hw)

    # pre-augment-bg
    bg = pre_transform_bg(image=bg)["image"]

    # place cards
    card_samples: list[Sample] = []
    card_Ms: list[np.ndarray] = []
    card_polys: list[np.ndarray] = []
    for i in range(np.random.randint(num_cards_min, num_cards_max)):
        card_sample = make_card_with_mask(card=imread_float(mtg_ds.ran_path()))
        M = place_card_on_background_get_transform(
            card_sample,
            bg,
            bg_existing_polygons=card_polys,
            min_visible=min_card_visible,
            min_visible_edge=min_card_visible_edge,
            jitter_ratio=card_jitter_ratio,
        )
        if M is None:
            continue
        card_samples.append(card_sample)
        card_Ms.append(M)
        card_polys.append(apply_transform_2d(card_sample["keypoints"][0], M))
    assert len(card_samples) == len(card_Ms)
    assert len(card_samples) == len(card_polys)

    # warp cards onto images, we need to apply to image in reverse order
    # because of overlap checks. Cards added later collide with ALL other cards, cards
    # added first don't check with later cards. So these need to be ON TOP.
    keypoints = []
    keypoints_labels = []
    for sample, M, poly in list(zip(card_samples, card_Ms, card_polys))[::-1]:
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
        min_card_visible: float = 0.5,
        min_card_visible_edge: float = None,
        card_jitter_ratio: float = 0.25,
    ):
        # Initialize datasets (replace with your actual classes)
        self.mtg_ds = SyntheticBgFgMtgImages(img_type="small")
        self.bg_ds = IlsvrcImages()

        # params
        self.bg_size_hw = bg_size_hw
        self.num_cards_min = num_cards_min
        self.num_cards_max = num_cards_max
        self.min_card_visible = min_card_visible
        self.min_card_visible_edge = min_card_visible_edge
        self.card_jitter_ratio = card_jitter_ratio

    def random(self):
        sample = generate_synthetic_image(
            self.mtg_ds,
            self.bg_ds,
            bg_size_hw=self.bg_size_hw,
            num_cards_min=self.num_cards_min,
            num_cards_max=self.num_cards_max,
            min_card_visible=self.min_card_visible,
            min_card_visible_edge=self.min_card_visible_edge,
            card_jitter_ratio=self.card_jitter_ratio,
        )
        return sample

    def debug_show_loop(self, n: int = None):
        count = 0
        while n is None or count < n:
            sample = self.random()
            imshow_loop(sample["image"], "synthetic")
            count += 1


# ========================================================================= #
# Yolo                                                                      #
# ========================================================================= #


def create_yolo_dataset(
    generator: Gen,
    *,
    output_dir: str,
    num_images: int = 1000,
):
    """
    Generate a YOLO dataset with synthetic images and annotations.

    Args:
        output_dir: Directory to save the dataset.
        num_images: Number of images to generate.
        min_cards: Minimum number of cards per image.
        max_cards: Maximum number of cards per image.
        bg_size: Tuple of (height, width) for the output images.
    """

    # Create output directories
    img_dir = init_dir(output_dir, "images")
    label_dir = init_dir(output_dir, "labels")

    # Generate dataset
    for i in tqdm(range(num_images), desc="Generating dataset"):
        sample = generator.random()

        # Save image
        img_path = os.path.join(img_dir, f"image_{i:04d}.png")
        imwrite(
            img_path, img
        )  # Placeholder: assume converts float32 [0, 1] to uint8 [0, 255]

        # make annotations from keypoints and keypoints_labels
        raise NotImplementedError

        # Save annotations
        label_path = os.path.join(label_dir, f"image_{i:04d}.txt")
        with open(label_path, "w") as f:
            for ann in annotations:
                f.write(" ".join(map(str, ann)) + "\n")


def corners_to_xyxy(keypoints: np.ndarray) -> np.ndarray:
    # keypoints shape is (-1, 4, 2)
    # [[(x0, y0), (x1, y1), (x2, y2), (x3, y3)], ...]
    mins = np.min(keypoints, axis=-2, keepdims=True)  # -> (-1, 1, 2)
    maxs = np.max(keypoints, axis=-2, keepdims=True)  # -> (-1, 1, 2)
    xyxy = np.concatenate([mins, maxs], axis=-2)  # -> (-1, 2, 2)
    return xyxy.reshape(-1, 4)


# ========================================================================= #
# Test                                                                      #
# ========================================================================= #


if __name__ == "__main__":
    gen = Gen()
    gen.debug_show_loop(n=10)

    # create_yolo_dataset("./yolo_mtg_dataset", num_images=10, bg_size=(640, 640))
