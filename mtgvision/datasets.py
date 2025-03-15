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
import albumentations as A

from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace

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
    def __getitem__(self, item) -> np.ndarray:
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    def ran(self) -> np.ndarray:
        return self._load_image(self.ran_path())

    @abc.abstractmethod
    def ran_path(self):
        ...

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
    ):
        self.ds_mtg = ds_mtg if (ds_mtg is not None) else MtgImages()
        self.ds_bg = ds_bg if (ds_bg is not None) else IlsvrcImages()
        # Make augments
        self._aug_bg = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(0, 360), p=1.0),
            A.Perspective(scale=(0.05, 0.15), p=1.0),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.5),

        ])



    # OLD
    _RAN_BG = uran.ApplyShuffled(
        uran.ApplyOrdered(Mutate.flip, Mutate.rotate_bounded, Mutate.warp_inv),
        uran.ApplyChoice(Mutate.fade_black, Mutate.fade_white, None),
    )
    _RAN_FG = uran.ApplyOrdered(
        uran.ApplyShuffled(
            Mutate.warp,
            uran.ApplyChoice(Mutate.fade_black, Mutate.fade_white, None),
        )
    )
    _RAN_VRTL = uran.ApplyShuffled(
        uran.ApplyChoice(Mutate.blur, None),
        uran.ApplyChoice(Mutate.noise, None),
        uran.ApplyChoice(Mutate.tint, None),
        uran.ApplyChoice(Mutate.fade_black, Mutate.fade_white, None),
    )

    @staticmethod
    def make_cropped(
        path_or_img: PathOrImg,
        size_hw: SizeHW | None = None,
        half_upsidedown: bool = False,
    ):
        card = (
            uimg.imread_float(path_or_img)
            if isinstance(path_or_img, str)
            else path_or_img
        )
        ret = uimg.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size_hw=size_hw,
        )
        return (
            uran.ApplyChoice(Mutate.upsidedown, None)(ret) if half_upsidedown else ret
        )

    @classmethod
    def make_masked(cls, path_or_img: PathOrImg) -> np.ndarray:
        card = (
            uimg.imread_float(path_or_img)
            if isinstance(path_or_img, str)
            else path_or_img
        )
        mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        ret = cv2.merge(
            (
                card[:, :, 0],
                card[:, :, 1],
                card[:, :, 2],
                mask,
            )
        )
        return ret

    @classmethod
    def make_bg(cls, bg_path_or_img: PathOrImg, size_hw: SizeHW) -> np.ndarray:
        bg = (
            uimg.imread_float(bg_path_or_img)
            if isinstance(bg_path_or_img, str)
            else bg_path_or_img
        )
        bg = cls._RAN_BG(bg)
        bg = uimg.crop_to_size(bg, size_hw)
        return bg

    @classmethod
    def make_virtual(
        cls,
        card_path_or_img: PathOrImg,
        bg_path_or_img: PathOrImg,
        size_hw: SizeHW,
        half_upsidedown: bool = False,
    ) -> np.ndarray:
        card = (
            uimg.imread_float(card_path_or_img)
            if isinstance(card_path_or_img, str)
            else card_path_or_img
        )
        card = (
            uran.ApplyChoice(Mutate.upsidedown, None)(card) if half_upsidedown else card
        )
        # fg - card
        fg = cls.make_masked(card)
        fg = uimg.crop_to_size(fg, size_hw, pad=True)
        fg = cls._RAN_FG(fg)
        # bg
        bg = cls.make_bg(bg_path_or_img, size_hw)
        # merge
        virtual = uimg.rgba_over_rgb(fg, bg)
        virtual = cls._RAN_VRTL(virtual)
        assert virtual.shape[:2] == size_hw
        return virtual

    @classmethod
    def make_virtual_pair(
        cls,
        card_path_or_img: PathOrImg,
        bg_path_or_img: PathOrImg,
        x_size_hw: SizeHW,
        y_size_hw: SizeHW,
        half_upsidedown: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        card = (
            uimg.imread_float(card_path_or_img)
            if isinstance(card_path_or_img, str)
            else card_path_or_img
        )
        # only inputs are flipped
        x = cls.make_virtual(
            card, bg_path_or_img, size_hw=x_size_hw, half_upsidedown=half_upsidedown
        )
        y = cls.make_cropped(card, size_hw=y_size_hw)
        return x, y


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":

    virtual = SyntheticBgFgMtgImages(
        ds_mtg=MtgImages(),
        ds_bg=IlsvrcImages(),
        default_x_size_hw=(192, 128),
        default_y_size_hw=(192, 128),
        half_upsidedown=True,
    )

    while True:
        x, y = virtual.make_virtual_pair()
        uimg.imshow_loop(x, "x")
        uimg.imshow_loop(y, "y"


# ========================================================================= #
# Dataset - Multi-Card YOLO Dataset                                        #
# ========================================================================= #

