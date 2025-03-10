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

import uuid
from pathlib import Path

import cv2
import numpy as np
from math import ceil
import random

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
# Dataset - IlsvrcImages                                                    #
# ========================================================================= #


class IlsvrcImages:
    """
    Dataset of ILSVRC 2010 Images. This is a small dataset of random images from
    different categories. This is intended to be used as background images for
    the MTG Dataset.
    """

    ILSVRC_SET_TYPES = ["val", "test", "train"]

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
        self._paths = sorted(ufls.get_image_paths(root, prefixed=True))

    def _load_image(self, path):
        return uimg.imread_float(path)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, item):
        return self._load_image(self._paths[item])

    def __iter__(self):
        for card in self._paths:
            yield self._load_image(card)

    def ran(self) -> np.ndarray:
        return self._load_image(random.choice(self._paths))

    def get(self, idx) -> np.ndarray:
        return self[idx]


# ========================================================================= #
# Dataset - MTG Images                                                      #
# ========================================================================= #

SizeHW = tuple[int, int]
PathOrImg = str | np.ndarray


class SyntheticBgFgMtgImages:
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
        self._cards: list[ScryfallCardFace] = sorted(self._ds, key=lambda x: x.id)

    def get_card_by_id(self, id_: uuid.UUID | str) -> ScryfallCardFace:
        if isinstance(id_, str):
            id_ = uuid.UUID(id_)
        # list is sorted, so binary search for ID, checking: `lazy.resource.id` for each lazy in `self._lazies`
        low = 0
        high = len(self._cards) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_val = self._cards[mid].id
            if mid_val < id_:
                low = mid + 1
            elif mid_val > id_:
                high = mid - 1
            else:
                return self._cards[mid]
        raise KeyError(f"Card with ID: {id_} not found!")

    def get_image_by_id(self, id_: uuid.UUID | str):
        return self._load_card_image(self.get_card_by_id(id_))

    @classmethod
    def _load_card_image(cls, card: ScryfallCardFace):
        return uimg.img_float32(card.dl_and_open_im_resized())

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, item):
        return self._load_card_image(self._cards[item])

    def __iter__(self):
        for card in self._cards:
            yield self._load_card_image(card)

    def ran(self) -> np.ndarray:
        return self._load_card_image(random.choice(self._cards))

    def get(self, idx) -> np.ndarray:
        return self[idx]

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
# MAIN - TEST                                                               #
# ========================================================================= #


if __name__ == "__main__":
    orig = SyntheticBgFgMtgImages(img_type="normal")
    ilsvrc = IlsvrcImages()

    while True:
        _o = orig.ran()
        _l = ilsvrc.ran()

        x, y = SyntheticBgFgMtgImages.make_virtual_pair(
            _o, _l, (192, 128), (192, 128), True
        )

        uimg.imshow_loop(x, "asdf")
        uimg.imshow_loop(y, "asdf")
