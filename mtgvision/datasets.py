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
from albumentations.core.composition import TransformsSeqType

from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace

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
            self._aug_upsidedown = compose(
                A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), p=0.5
            )
        else:
            self._aug_upsidedown = A.NoOp()

        # Make augments
        self._aug_bg = compose(
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(p=1.0, limit=(0, 360)),
            A.ColorJitter(p=0.9, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        )

        self._aug_fg = random_order(
            compose(
                A.ShiftScaleRotate(
                    p=1.0,
                    shift_limit=(-0.0625, 0.0625),
                    scale_limit=(-0.2, 0.0),
                    rotate_limit=(-5, 5),
                    interpolation=cv2.INTER_CUBIC,
                    mask_interpolation=cv2.INTER_CUBIC,
                ),
                A.Perspective(
                    p=1.0,
                    scale=(0, 0.075),
                    interpolation=cv2.INTER_CUBIC,
                    mask_interpolation=cv2.INTER_CUBIC,
                ),
                p=0.8,
            ),
            A.Erasing(
                p=0.2,
                scale=(0.2, 0.7),
                fill="random_uniform",
            ),
            A.ColorJitter(p=0.5, brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0),
        )

        self._aug_vrtl = random_order(
            random_order(
                one_of(
                    A.GaussianBlur(p=1.0, blur_limit=7),
                    A.MotionBlur(p=1.0, blur_limit=7),
                    A.MedianBlur(p=1.0, blur_limit=7),
                    p=0.5,
                ),
                A.ImageCompression(p=0.25, quality_range=(98, 100)),
                A.Downscale(p=0.25, scale_range=(0.2, 0.9)),
                p=0.5,
            ),
            one_of(
                A.ISONoise(p=1.0, intensity=(0.1, 0.5)),
                A.RandomFog(p=1.0),
                A.ShotNoise(p=1.0),
                A.AdditiveNoise(p=1.0),
                # A.MultiplicativeNoise(p=0.5),
                A.GaussNoise(p=1.0),
                A.SaltAndPepper(p=1.0),
                p=0.5,
            ),
            one_of(
                A.ColorJitter(
                    p=1.0, brightness=0.8, contrast=0.5, saturation=0.3, hue=0.1
                ),
                A.RandomGamma(p=1.0, gamma_limit=(80, 120)),
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
        card = self._aug_upsidedown(image=card)["image"]
        card = uimg.resize(card, size_hw=size_hw or self._default_x_size_hw)
        # crop card and mask
        mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        # augment
        transformed = self._aug_fg(image=card, mask=mask)
        return transformed["image"], transformed["mask"]

    def _make_aug_bg(
        self, bg_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> np.ndarray:
        bg = self._get_bg(bg_path_or_img)
        # augment
        bg = self._aug_bg(image=bg)["image"]
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
        synthetic_card = self._aug_vrtl(image=synthetic_card)["image"]
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

    while True:
        x, y = ds.make_synthetic_input_and_target_card_pair()
        uimg.imshow_loop(x, "x")
        # uimg.imshow_loop(y, "y")
