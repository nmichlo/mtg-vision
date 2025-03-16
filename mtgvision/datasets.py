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
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from math import ceil
import random
from tqdm import tqdm

from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.random as uran
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace


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
PathOrImg = str | np.ndarray


class SyntheticBgFgMtgImages:
    def __init__(
        self,
        ds_mtg: MtgImages | None = None,
        ds_bg: IlsvrcImages | None = None,
        *,
        half_upsidedown: bool = False,
        default_x_size_hw: SizeHW = (192, 128),
        default_y_size_hw: SizeHW = (192, 128),
    ):
        self.ds_mtg = ds_mtg if (ds_mtg is not None) else MtgImages()
        self.ds_bg = ds_bg if (ds_bg is not None) else IlsvrcImages()

        self._default_x_size_hw = default_x_size_hw
        self._default_y_size_hw = default_y_size_hw

        # Upside down augment
        if half_upsidedown:
            self._aug_upsidedown = uran.ApplyChance(Mutate.upsidedown, p=0.5)
        else:
            self._aug_upsidedown = uran.NoOp()

        # Make augments
        self._aug_bg = uran.ApplyShuffled(
            uran.ApplyChance(uimg.flip_horr, p=0.5),
            uran.ApplyFn(uimg.flip_vert, p=0.5),
            uran.ApplyFn(Mutate.rotate_bounded, deg_min=0, deg_max=360, p=1.0),
            uran.ApplyFn(Mutate.tint, amount=0.15, p=1.0),
            uran.ApplyFn(Mutate.fade_black, amount=0.5, p=1.0),
            uran.ApplyFn(Mutate.fade_white, amount=0.33, p=1.0),
        )

        self._aug_fg = uran.ApplyShuffled(
            uran.ApplySequence(
                uran.ApplyFn(Mutate.warp, warp_ratio=0.15, warp_ratio_min=-0.05),
                uran.ApplyFn(Mutate.rotate_bounded, deg_min=-5, deg_max=5),
                # A.ShiftScaleRotate + A.Perspective
                p=0.8,
            ),
            # A.Erasing(p=0.2, scale=(0.2, 0.7), fill="random_uniform",),
            # A.ColorJitter(p=0.5, brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0),
            uran.ApplyFn(Mutate.tint, amount=0.15, p=0.5),
        )

        self._aug_vrtl = uran.ApplyShuffled(
            uran.ApplyShuffled(
                uran.ApplyOne(
                    uran.ApplyFn(Mutate.blur, n_max=7, p=1.0),
                ),
                p=0.5,
            ),
            uran.ApplyOne(
                uran.ApplyFn(Mutate.noise, amount=0.75, p=1.0),
                p=0.5,
            ),
            uran.ApplyOne(
                uran.ApplyFn(Mutate.tint, amount=0.15, p=1.0),
                uran.ApplyFn(Mutate.fade_black, amount=0.5, p=1.0),
                uran.ApplyFn(Mutate.fade_white, amount=0.33, p=1.0),
                p=1.0,
            ),
        )

    # GET IMAGES - NO AUGMENTS

    @classmethod
    def _get_img(cls, path_or_img):
        return (
            uimg.imread_float32(path_or_img)
            if isinstance(path_or_img, str)
            else uimg.img_float32(path_or_img)
        )

    def _get_card(self, card_path_or_img: PathOrImg | None) -> np.ndarray:
        if card_path_or_img is None:
            return self.ds_mtg.ran()
        return self._get_img(card_path_or_img)

    def _get_bg(self, bg_path_or_img: PathOrImg | None) -> np.ndarray:
        if bg_path_or_img is None:
            return self.ds_bg.ran()
        return self._get_img(bg_path_or_img)

    # AUGMENTED

    def make_target_card(
        self, card_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> np.ndarray:
        card = self._get_card(card_path_or_img)
        ret = uimg.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size_hw=size_hw or self._default_y_size_hw,
        )
        return ret

    def _make_aug_card_and_mask(
        self, path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # create card and resize
        card = self._get_card(path_or_img)
        card = self._aug_upsidedown(card)
        # crop card and mask
        # mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        # augment
        # seed = random.randint(0, 2 ** 32 - 1)
        card = self._aug_fg(card)
        card = uimg.resize(card, size_hw=size_hw or self._default_x_size_hw)
        # mask = self._aug_fg(np.repeat(mask[:, :, None], 3, axis=-1))  # TODO: wrong
        # done
        mask = np.average(card, axis=-1)
        return card, mask

    def _make_aug_bg(
        self, bg_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> np.ndarray:
        bg = self._get_bg(bg_path_or_img)
        # augment
        bg = self._aug_bg(bg)
        # resize
        bg = uimg.crop_to_size(bg, size_hw or self._default_x_size_hw)
        return bg

    def make_synthetic_input_card(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
    ) -> np.ndarray:
        size_hw = size_hw or self._default_x_size_hw
        # fg - card
        fg, fg_mask = self._make_aug_card_and_mask(card_path_or_img, size_hw=size_hw)
        # bg
        bg = self._make_aug_bg(bg_path_or_img, size_hw=size_hw)
        # merge
        synthetic_card = uimg.rgb_mask_over_rgb(fg, fg_mask, bg)
        synthetic_card = self._aug_vrtl(synthetic_card)
        # checks
        assert synthetic_card.shape[:2] == size_hw, (
            f"Expected size_hw={size_hw}, got={synthetic_card.shape[:2]}"
        )
        return synthetic_card

    def make_synthetic_input_and_target_card_pair(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        x_size_hw: SizeHW | None = None,
        y_size_hw: SizeHW | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # ensure the same card is used for both x and y
        card = self._get_card(card_path_or_img)
        # only inputs are flipped
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

    # 100%|██████████| 1000/1000 [00:13<00:00, 74.33it/s]
    for i in tqdm(range(1000)):
        x, y = ds.make_synthetic_input_and_target_card_pair()

        # uimg.imshow_loop(x, "x")
        # uimg.imshow_loop(y, "y")
