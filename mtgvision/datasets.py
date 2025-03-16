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


import abc
import uuid
from os import PathLike
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from math import ceil
import random
import albumentations as A
from albumentations.core.composition import TransformsSeqType
from tqdm import tqdm


from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace


import functools
from typing import TypeVar, Tuple


# Assuming these are custom utility modules; replace with actual imports if different
import mtgvision.util.random as uran


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
        return uimg.flip(
            img,
            horr=horr and (random.random() >= 0.5),
            vert=vert and (random.random() >= 0.5),
        )

    @staticmethod
    def rotate_bounded(img, deg_min=0, deg_max=360):
        return uimg.rotate_bounded(
            img, deg_min + np.random.random() * (deg_max - deg_min)
        )

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
            noisy = uimg.noise_speckle(img, strength=0.3)
        elif noise_type == "gaussian":
            noisy = uimg.noise_gaussian(img, mean=0, var=0.05)
        elif noise_type == "pepper":
            noisy = uimg.noise_salt_pepper(img, strength=0.1, svp=0.5)
        elif noise_type == "poisson":
            noisy = uimg.noise_poisson(img, peak=0.8, amount=0.5)
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
            img[:, :, i] = uimg.img_clip(r * img[:, :, i])
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
# Vars                                                                      #
# ========================================================================= #


DATASETS_ROOT = Path(__file__).parent.parent.parent / "mtg-dataset/mtgdata/data"

print(f"DATASETS_ROOT={DATASETS_ROOT}")


class _SomeOf(A.SomeOf):
    # https://github.com/albumentations-team/albumentations/issues/2474
    def __init__(
        self,
        transforms: TransformsSeqType,
        n: int = 1,
        replace: bool = False,
        p: float = 1,
    ):
        # wrap each transform in with a wrapper that removes the kwargs "force_apply" from __call__
        transforms = [A.Compose([t], p=1.0) for t in transforms]
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)


class _RandomOrder(A.RandomOrder):
    # https://github.com/albumentations-team/albumentations/issues/2474
    def __init__(
        self,
        transforms: TransformsSeqType,
        n: int = 1,
        replace: bool = False,
        p: float = 1,
    ):
        # wrap each transform in with a wrapper that removes the kwargs "force_apply" from __call__
        transforms = [A.Compose([t], p=1.0) for t in transforms]
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)


def compose(*args, p=1.0):
    return A.Compose(list(args), p=p)


def some_of(*args, n=None, p=1.0):
    return _SomeOf(list(args), n=n or len(args), p=p)


def random_order(*args, n=None, p=1.0):
    return _RandomOrder(list(args), n=n or len(args), p=p)


def one_of(*args, p=1.0):
    return A.OneOf(list(args), p=p)


# ========================================================================= #
# Dataset - Random                                                          #
# ========================================================================= #


class _BaseImgDataset(abc.ABC):
    """
    Base class for random datasets. This class provides a simple interface
    for generating random images.
    """

    @classmethod
    def _load_image(cls, path_or_img):
        if isinstance(path_or_img, (str, Path)):
            return uimg.imread_float32(path_or_img)
        else:
            return uimg.img_float32(path_or_img)

    @abc.abstractmethod
    def __getitem__(self, item) -> np.ndarray: ...

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def ran(self) -> np.ndarray:
        return self._load_image(self.ran_path())

    @abc.abstractmethod
    def ran_path(self): ...

    def get(self, idx) -> np.ndarray:
        return self.__getitem__(idx)


# ========================================================================= #
# Dataset - IlsvrcImages                                                    #
# ========================================================================= #


class IlsvrcImages(_BaseImgDataset):
    """
    Dataset of ILSVRC 2010 Images. This is a small dataset of random images from
    different categories. This is intended to be used as background images for
    the MTG Dataset.
    """

    def __init__(self):
        root = DATASETS_ROOT / "ilsvrc" / "2010"
        val = root / "val"

        if not root.is_dir():
            print(
                f"MAKE SURE YOU HAVE DOWNLOADED THE ILSVRC 2010 TEST DATASET TO: {root}\n"
                f" - The images must all be located within: {val}\n"
                f" - For example: {val / 'ILSVRC2010_val_00000001.JPEG'}\n"
                f"The image versions of the ILSVRC Datasets are for educational purposes only, and cannot be redistributed.\n"
                f"Please visit: www.image-net.org to obtain the download links.\n",
            )
        self._paths: "list[str]" = sorted(ufls.get_image_paths(root, prefixed=True))

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, item):
        return self._load_image(self._paths[item])

    def __iter__(self):
        yield from (self._load_image(path) for path in self._paths)

    def ran_path(self) -> str:
        return random.choice(self._paths)


# ========================================================================= #
# Dataset - MTG Images                                                      #
# ========================================================================= #


class MtgImages(_BaseImgDataset):
    """
    Dataset of Magic The Gathering Card Images. This generates Synthetic data of
    slightly distorted images of cards with random backgrounds.

    This is intended to mimic the results that would be fed into an embedding model
    as the result of some detection or segmentation task.
    """

    def __init__(self, img_type=ScryfallImageType.small, predownload=False):
        self._ds = ScryfallDataset(
            img_type=img_type,
            data_root=Path(__file__).parent.parent.parent / "mtg-dataset/mtgdata/data",
            force_update=False,
            download_mode="now" if predownload else "none",
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, item):
        return self._load_image(self._ds[item])

    def __iter__(self):
        yield from (self._load_image(c.dl_and_open_im_resized()) for c in self._ds)

    def ran_path(self) -> str:
        return random.choice(self._ds).download()

    # CUSTOM

    def get_card_by_id(self, id_: uuid.UUID | str) -> ScryfallCardFace:
        return self._ds.get_card_by_id(id_)

    def get_image_by_id(self, id_: uuid.UUID | str):
        return self._load_image(self.get_card_by_id(id_).dl_and_open_im_resized())

    def card_iter(self) -> Iterator[ScryfallCardFace]:
        yield from self._ds


# ========================================================================= #
# Dataset - Synthetic Background & Foreground MTG Images                    #
# ========================================================================= #

SizeHW = tuple[int, int]
PathOrImg = str | np.ndarray | PathLike

# ========================================================================= #
# VARS & UTILITIES                                                          #
# ========================================================================= #

T = TypeVar("T")


def ensure_float32(fn: T) -> T:
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


@ensure_float32
def my_flip_horr(data: dict) -> dict:
    """Flip image and mask horizontally."""
    img = uimg.flip_horr(data["image"])
    mask = uimg.flip_horr(data["mask"])
    return {"image": img, "mask": mask}


@ensure_float32
def my_flip_vert(data: dict) -> dict:
    """Flip image and mask vertically."""
    img = uimg.flip_vert(data["image"])
    mask = uimg.flip_vert(data["mask"])
    return {"image": img, "mask": mask}


@ensure_float32
def my_rotate_bounded(data: dict, deg: float) -> dict:
    """Rotate image and mask by a specified degree with bounding."""
    img = uimg.rotate_bounded(data["image"], deg)
    mask = uimg.rotate_bounded(data["mask"], deg)
    return {"image": img, "mask": mask}


@ensure_float32
def my_upsidedown(data: dict) -> dict:
    """Rotate image and mask 180 degrees."""
    img = np.rot90(data["image"], k=2)
    mask = np.rot90(data["mask"], k=2)
    return {"image": img, "mask": mask}


@ensure_float32
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


@ensure_float32
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


@ensure_float32
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


@ensure_float32
def my_tint(data: dict, amount=0.15) -> dict:
    """Apply random tint to image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    for i in range(3):
        r = 1 + amount * (2 * np.random.random() - 1)
        img[:, :, i] = uimg.img_clip(r * img[:, :, i])
    return {"image": img, "mask": mask}


@ensure_float32
def my_blur(data: dict, n_max=7) -> dict:
    """Apply blur to image; mask unchanged."""
    img = Mutate.blur(data["image"], n_max=n_max)  # Assuming uimg.blur exists
    mask = data["mask"]
    return {"image": img, "mask": mask}


@ensure_float32
def my_image_compression(data: dict, quality_range=(98, 100)) -> dict:
    """Apply JPEG compression to image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", (img * 255).astype(np.uint8), encode_param)
    img_transformed = cv2.imdecode(encimg, cv2.IMREAD_COLOR).astype(np.float32) / 255
    return {"image": img_transformed, "mask": mask}


@ensure_float32
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


@ensure_float32
def my_noise(data: dict, var=0.05) -> dict:
    """Add Gaussian noise to image; mask unchanged."""
    img = uimg.noise_gaussian(data["image"], mean=0, var=var)
    mask = data["mask"]
    return {"image": img, "mask": mask}


@ensure_float32
def my_gamma(data: dict, gamma_limit=(80, 120)) -> dict:
    """Adjust gamma of image; mask unchanged."""
    img = data["image"].copy()
    mask = data["mask"]
    gamma = random.uniform(gamma_limit[0], gamma_limit[1]) / 100
    img_transformed = np.clip(img**gamma, 0, 1).astype(np.float32)
    return {"image": img_transformed, "mask": mask}


# ========================================================================= #
# AUGMENTATION PIPELINES & MAIN CLASS                                       #
# ========================================================================= #


class SyntheticBgFgMtgImages:
    def __init__(
        self,
        ds_mtg=None,  # Replace with actual type, e.g., MtgImages
        ds_bg=None,  # Replace with actual type, e.g., IlsvrcImages
        *,
        half_upsidedown: bool = False,
        default_x_size_hw: SizeHW = (192, 128),
        default_y_size_hw: SizeHW = (192, 128),
    ):
        # Placeholder datasets; replace with actual implementations
        self.ds_mtg = ds_mtg if ds_mtg is not None else ScryfallDataset()
        self.ds_bg = (
            ds_bg if ds_bg is not None else ScryfallDataset()
        )  # Adjust as needed

        self._default_x_size_hw = default_x_size_hw
        self._default_y_size_hw = default_y_size_hw

        # Upside-down augmentation
        if half_upsidedown:
            self._aug_upsidedown = uran.ApplyFn(my_upsidedown, p=0.5)
        else:
            self._aug_upsidedown = uran.NoOp()

        # Background augmentations (image only)
        self._aug_bg = uran.ApplySequence(
            uran.ApplyFn(my_flip_horr, p=0.5),
            uran.ApplyFn(my_flip_vert, p=0.5),
            uran.ApplyFn(
                lambda data: my_rotate_bounded(data, deg=random.uniform(0, 360))
            ),
            uran.ApplyFn(my_tint, amount=0.15, p=0.9),
        )

        # Foreground augmentations (image and mask)
        self._aug_fg = uran.ApplyShuffled(
            uran.ApplySequence(
                uran.ApplyFn(
                    my_shift_scale_rotate,
                    shift_limit=(-0.0625, 0.0625),
                    scale_limit=(-0.2, 0.0),
                    rotate_limit=(-5, 5),
                ),
                uran.ApplyFn(my_perspective, scale=(0, 0.075)),
                p=0.8,
            ),
            uran.ApplyFn(my_erasing, scale=(0.2, 0.7), p=0.2),
            uran.ApplyFn(my_tint, amount=0.15, p=0.5),
        )

        # Virtual augmentations (image only)
        self._aug_vrtl = uran.ApplyShuffled(
            uran.ApplyFn(
                uran.ApplyShuffled(
                    uran.ApplyFn(my_blur, n_max=7, p=0.5),
                    uran.ApplyFn(my_image_compression, quality_range=(98, 100), p=0.25),
                    uran.ApplyFn(my_downscale, scale_range=(0.2, 0.9), p=0.25),
                ),
                p=0.5,
            ),
            uran.ApplyFn(my_noise, var=0.05, p=0.5),
            uran.ApplyOne(
                uran.ApplyFn(my_tint, amount=0.15),
                uran.ApplyFn(my_gamma, gamma_limit=(80, 120)),
                p=1.0,
            ),
        )

    # IMAGE LOADING METHODS

    @classmethod
    def _get_img(cls, path_or_img: PathOrImg) -> np.ndarray:
        """Load an image as float32."""
        return (
            uimg.imread_float32(path_or_img)
            if isinstance(path_or_img, (str, PathLike))
            else uimg.img_float32(path_or_img)
        )

    def _get_card(self, card_path_or_img: PathOrImg | None) -> np.ndarray:
        """Get a random card image or load the specified one."""
        if card_path_or_img is None:
            return self.ds_mtg.ran()  # Adjust based on actual dataset method
        return self._get_img(card_path_or_img)

    def _get_bg(self, bg_path_or_img: PathOrImg | None) -> np.ndarray:
        """Get a random background image or load the specified one."""
        if bg_path_or_img is None:
            return self.ds_bg.ran()  # Adjust based on actual dataset method
        return self._get_img(bg_path_or_img)

    # AUGMENTED IMAGE METHODS

    def make_target_card(
        self, card_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> np.ndarray:
        """Create a target card image without background."""
        card = self._get_card(card_path_or_img)
        ret = uimg.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size_hw=size_hw or self._default_y_size_hw,
        )
        return ret

    def _make_aug_card_and_mask(
        self, path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate an augmented card and its mask."""
        card = self._get_card(path_or_img)
        # Apply upside-down augmentation to card only initially
        card = self._aug_upsidedown({"image": card, "mask": np.ones_like(card)})[
            "image"
        ]
        card = uimg.resize(card, size_hw=size_hw or self._default_x_size_hw)
        mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        data = {"image": card, "mask": mask}
        transformed = self._aug_fg(data)
        return transformed["image"], transformed["mask"]

    def _make_aug_bg(
        self, bg_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> np.ndarray:
        """Generate an augmented background."""
        bg = self._get_bg(bg_path_or_img)
        data = {"image": bg, "mask": np.ones_like(bg)}  # Dummy mask
        transformed = self._aug_bg(data)
        bg_aug = transformed["image"]
        bg_aug = uimg.crop_to_size(bg_aug, size_hw or self._default_x_size_hw)
        return bg_aug

    def make_synthetic_input_card(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
    ) -> np.ndarray:
        """Create a synthetic input card with background."""
        size_hw = size_hw or self._default_x_size_hw
        # Foreground (card) and mask
        fg, fg_mask = self._make_aug_card_and_mask(card_path_or_img, size_hw=size_hw)
        # Background
        bg = self._make_aug_bg(bg_path_or_img, size_hw=size_hw)
        # Merge foreground and background
        synthetic_card = uimg.rgb_mask_over_rgb(fg, fg_mask, bg)
        # Apply virtual augmentations
        data = {
            "image": synthetic_card,
            "mask": np.ones_like(synthetic_card),
        }  # Dummy mask
        transformed = self._aug_vrtl(data)
        synthetic_card_aug = transformed["image"]
        # Size check
        assert synthetic_card_aug.shape[:2] == size_hw, (
            f"Expected size_hw={size_hw}, got={synthetic_card_aug.shape[:2]}"
        )
        return synthetic_card_aug

    def make_synthetic_input_and_target_card_pair(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        x_size_hw: SizeHW | None = None,
        y_size_hw: SizeHW | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a pair of synthetic input and target cards."""
        card = self._get_card(card_path_or_img)
        x = self.make_synthetic_input_card(card, bg_path_or_img, size_hw=x_size_hw)
        y = self.make_target_card(card, size_hw=y_size_hw)
        return x, y


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":
    ds = SyntheticBgFgMtgImages(
        ds_mtg=MtgImages(),
        ds_bg=IlsvrcImages(),
        default_x_size_hw=(192, 128),
        default_y_size_hw=(192, 128),
        half_upsidedown=True,
    )

    for i in tqdm(range(10)):
        x, y = ds.make_synthetic_input_and_target_card_pair()

    # 100%|██████████| 1000/1000 [00:16<00:00, 60.01it/s]
    for i in tqdm(range(1000)):
        x, y = ds.make_synthetic_input_and_target_card_pair()

        # uimg.imshow_loop(x, "x")
        # uimg.imshow_loop(y, "y")
