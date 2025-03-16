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

import numpy as np
from math import ceil
import random
from tqdm import tqdm


from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace


from typing import Tuple


# Assuming these are custom utility modules; replace with actual imports if different
import mtgvision.util.random as uran


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
PathOrImg = str | np.ndarray | PathLike


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
            self._aug_upsidedown = uran.ApplyFn(uimg.my_upsidedown, p=0.5)
        else:
            self._aug_upsidedown = uran.NoOp()

        # Background augmentations (image only)
        self._aug_bg = uran.ApplySequence(
            uran.ApplyFn(uimg.my_flip_horr, p=0.5),
            uran.ApplyFn(uimg.my_flip_vert, p=0.5),
            uran.ApplyFn(
                lambda data: uimg.my_rotate_bounded(data, deg=random.uniform(0, 360))
            ),
            uran.ApplyFn(uimg.my_tint, amount=0.15, p=0.9),
        )

        # Foreground augmentations (image and mask)
        self._aug_fg = uran.ApplyShuffled(
            uran.ApplySequence(
                uran.ApplyFn(
                    uimg.my_shift_scale_rotate,
                    shift_limit=(-0.0625, 0.0625),
                    scale_limit=(-0.2, 0.0),
                    rotate_limit=(-5, 5),
                ),
                uran.ApplyFn(uimg.my_perspective, scale=(0, 0.075)),
                p=0.8,
            ),
            uran.ApplyFn(uimg.my_erasing, scale=(0.2, 0.7), p=0.2),
            uran.ApplyFn(uimg.my_tint, amount=0.15, p=0.5),
        )

        # Virtual augmentations (image only)
        self._aug_vrtl = uran.ApplyShuffled(
            uran.ApplyFn(
                uran.ApplyShuffled(
                    uran.ApplyFn(uimg.my_blur, n_max=7, p=0.5),
                    uran.ApplyFn(
                        uimg.my_image_compression, quality_range=(98, 100), p=0.25
                    ),
                    uran.ApplyFn(uimg.my_downscale, scale_range=(0.2, 0.9), p=0.25),
                ),
                p=0.5,
            ),
            uran.ApplyFn(uimg.my_noise, var=0.05, p=0.5),
            uran.ApplyOne(
                uran.ApplyFn(uimg.my_tint, amount=0.15),
                uran.ApplyFn(uimg.my_gamma, gamma_limit=(80, 120)),
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
        # 270 it/s
        # x = ds._get_card(None)  # 1050 it/s
        # y = ds._get_bg(None)  # 320 it/s

        # 70 it/s
        x, y = ds.make_synthetic_input_and_target_card_pair()

        # uimg.imshow_loop(x, "x")
        # uimg.imshow_loop(y, "y")
