"""
Datasets for training a denoising auto-encoder or contrastive learning model
to embed synthetic wraped and distorted versions of magic cards to produce
embeddings invariant to their distortions.

Using similar techniques to facial recognition.
"""

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
from typing import Iterator, Literal

import cv2
import numpy as np
from math import ceil
import random

from tqdm import tqdm

from mtgdata import ScryfallDataset, ScryfallImageType
from mtgvision.util.image import ensure_float32
import mtgvision.util.image as uimg
import mtgvision.util.random as uran
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace


# ========================================================================= #
# RANDOM TRANSFORMS                                                         #
# ========================================================================= #


class Mutate:
    """
    A collection of random image transformations.
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
    def warp(img, warp_ratio=0.2, warp_ratio_min=-0.15):
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

    @staticmethod
    @ensure_float32
    def brightness_contrast(img, brightness=0.2, contrast=0.2):
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        beta = np.random.uniform(-brightness, brightness)
        img = alpha * img + beta
        return np.clip(img, 0, 1)

    # @staticmethod
    @ensure_float32
    # def color_jitter(img, hue=0.1, saturation=0.1, value=0.1):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     h = h + np.random.uniform(-hue, hue) * 180
    #     s = s * (1 + np.random.uniform(-saturation, saturation))
    #     v = v * (1 + np.random.uniform(-value, value))
    #     hsv = cv2.merge([h, s, v])
    #     return cv2.cvtColor(np.clip(hsv, 0, 255), cv2.COLOR_HSV2BGR)

    @staticmethod
    @ensure_float32
    def gaussian_noise(img, mean=0, sigma=0.25):
        noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        return np.clip(img + noise, 0, 1)

    @staticmethod
    @ensure_float32
    def salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
        noisy = img.copy()
        num_salt = np.ceil(salt_prob * img.size)
        num_pepper = np.ceil(pepper_prob * img.size)
        # Add salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        noisy[coords[0], coords[1], :] = 1
        # Add pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy

    @staticmethod
    @ensure_float32
    def sharpen(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    # @staticmethod
    # def elastic_transform(img, alpha=36, sigma=6):
    #     random_state = np.random.RandomState(None)
    #     shape = img.shape
    #     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #     x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    #     indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    #     return map_coordinates(img, indices, order=1).reshape(shape)

    @staticmethod
    @ensure_float32
    def cutout(img, num_holes=8, max_h_size=8, max_w_size=8):
        h, w, _ = img.shape
        for _ in range(num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - max_h_size // 2, 0, h)
            y2 = np.clip(y + max_h_size // 2, 0, h)
            x1 = np.clip(x - max_w_size // 2, 0, w)
            x2 = np.clip(x + max_w_size // 2, 0, w)
            img[y1:y2, x1:x2, :] = 0
        return img

    @staticmethod
    @ensure_float32
    def random_erasing(
        img,
        *,
        scale_min_max: tuple[float, float] = (0.2, 0.5),  # [0, 1]
        aspect_min_max: tuple[float, float] = (1, 3),  # [1, inf]
        color: Literal["random", "uniform_random", "zeros", "ones", "mean"] = (
            "random",
            "uniform_random",
            "zeros",
            "ones",
            "mean",
        ),
        inside: bool = False,
    ):
        h, w = img.shape[:2]
        # scale
        scale = np.random.uniform(*scale_min_max)
        target_area = scale * (h * w)
        # aspect
        aspect_ratio = np.random.uniform(*aspect_min_max)
        if np.random.random() < 0.5:
            aspect_ratio = 1 / aspect_ratio

        block_w = int((target_area / aspect_ratio) ** 0.5)
        block_h = int((target_area * aspect_ratio) ** 0.5)
        # get coords
        if inside:
            (mx, Mx), (my, My) = (
                (block_w // 2, w - block_w // 2),
                (block_h // 2, h - block_h // 2),
            )
        else:
            (mx, Mx), (my, My) = (
                (0 - block_w // 2, w + block_w // 2),
                (0 - block_h // 2, h + block_h // 2),
            )

        if Mx <= mx:
            return img
        if My <= my:
            return img

        cx = np.random.randint(mx, Mx)
        cy = np.random.randint(my, My)

        # clamp to valid ranges
        block_x0 = np.maximum(0, cx - block_w // 2)
        block_y0 = np.maximum(0, cy - block_h // 2)
        block_x1 = np.minimum(w, cx + block_w // 2)
        block_y1 = np.minimum(h, cy + block_h // 2)
        block_h = block_y1 - block_y0
        block_w = block_x1 - block_x0

        if isinstance(color, (tuple, list)):
            color = random.choice(color)
        if color == "uniform_random":
            c = np.random.uniform(0, 1, (img.shape[-1],))
        elif color == "random":
            c = np.random.uniform(0, 1, (block_h, block_w, img.shape[-1]))
        elif color == "zeros":
            c = np.zeros((img.shape[-1],))
        elif color == "mean":
            c = img[block_y0:block_y1, block_x0:block_x1, :].mean(axis=(0, 1))
        elif color == "ones":
            c = np.ones((img.shape[-1],))
        else:
            raise ValueError(f"Invalid color choice: {color}")

        # set
        img[block_y0:block_y1, block_x0:block_x1, :] = c
        return img

    @staticmethod
    @ensure_float32
    def affine_transform(img, angle=0, translate=(0, 0), scale=1, shear=0):
        rows, cols, _ = img.shape
        center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M = np.vstack([M, [0, 0, 1]])
        shear_matrix = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]])
        M = np.dot(shear_matrix, M)
        M[0, 2] += translate[0]
        M[1, 2] += translate[1]
        return cv2.warpAffine(img, M[:2, :], (cols, rows))

    @staticmethod
    @ensure_float32
    def perspective_transform(img, strength=0.1):
        rows, cols, _ = img.shape
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        pts2 = np.float32(
            [
                [
                    np.random.uniform(-strength, strength) * cols,
                    np.random.uniform(-strength, strength) * rows,
                ],
                [
                    cols + np.random.uniform(-strength, strength) * cols,
                    np.random.uniform(-strength, strength) * rows,
                ],
                [
                    np.random.uniform(-strength, strength) * cols,
                    rows + np.random.uniform(-strength, strength) * rows,
                ],
                [
                    cols + np.random.uniform(-strength, strength) * cols,
                    rows + np.random.uniform(-strength, strength) * rows,
                ],
            ]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (cols, rows))


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

    def _get_download_message(self, root: Path, subdir: Path) -> str:
        return (
            f"MAKE SURE YOU HAVE DOWNLOADED THE ILSVRC 2010 TEST DATASET TO: {root}\n"
            f" - The images must all be located within: {subdir}\n"
            f" - For example: {subdir / 'ILSVRC2010_val_00000001.JPEG'}\n"
            f"The image versions of the ILSVRC Datasets are for educational purposes only, and cannot be redistributed.\n"
            f"Please visit: www.image-net.org to obtain the download links.\n"
        )

    def __init__(
        self,
        root: str | Path = DATASETS_ROOT / "ilsvrc" / "2010",
        subdir: str | Path = "val",
    ):
        root = Path(root)
        if subdir is not None:
            subdir = Path(subdir)
            if subdir.is_absolute():
                raise ValueError("subdir must be a relative path")
            subdir = root / subdir
        else:
            subdir = root
        if not root.is_dir():
            print(self._get_download_message(root, subdir))
        self._paths: "list[str]" = sorted(ufls.get_image_paths(root, prefixed=True))

    def _load_image(self, path):
        return uimg.imread_float(path)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, item):
        return self._load_image(self._paths[item])

    def __iter__(self):
        for card in self._paths:
            yield self._load_image(card)

    def ran_path(self) -> str:
        return random.choice(self._paths)

    def ran(self) -> np.ndarray:
        return self._load_image(self.ran_path())

    def get(self, idx) -> np.ndarray:
        return self[idx]


class CocoValImages(IlsvrcImages):
    def _get_download_message(self, root: Path, subdir: Path) -> str:
        return (
            f"MAKE SURE YOU HAVE DOWNLOADED THE COCO 2017 VAL DATASET TO: {root}\n"
            f" - The images must all be located within: {subdir}\n"
            f" - For example: {subdir / '000000000139.jpg'}\n"
            f"Please visit: `cocodataset.org` to obtain the download links.\n"
        )

    def __init__(
        self,
        root: str | Path = DATASETS_ROOT / "coco" / "2017",
        subdir: str | Path = "val2017",
    ):
        super().__init__(root=root, subdir=subdir)


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

    def get_card_by_id(self, id_: uuid.UUID | str) -> ScryfallCardFace:
        return self._ds.get_card_by_id(id_)

    def get_image_by_id(self, id_: uuid.UUID | str):
        return self._load_card_image(self.get_card_by_id(id_))

    @classmethod
    def _load_card_image(cls, card: ScryfallCardFace):
        return uimg.img_float32(card.dl_and_open_im_resized())

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, item):
        return self._load_card_image(self._ds[item])

    def __iter__(self):
        for card in self._ds:
            yield self._load_card_image(card)

    def card_iter(self) -> Iterator[ScryfallCardFace]:
        for card in self._ds:
            yield card

    def ran(self) -> np.ndarray:
        return self._load_card_image(self.ran_card())

    def ran_path(self) -> str:
        return str(self.ran_card().download())

    def ran_card(self) -> ScryfallCardFace:
        return random.choice(self._ds)

    def get(self, idx) -> np.ndarray:
        return self[idx]

    _RAN_BG = uran.ApplyShuffled(
        uran.ApplyOrdered(Mutate.flip, Mutate.rotate_bounded, Mutate.warp_inv),
        uran.ApplyChoice(
            Mutate.fade_black, Mutate.fade_white, Mutate.brightness_contrast, None
        ),
        # uran.ApplyChoice(Mutate.color_jitter, None),
    )
    _RAN_FG = uran.ApplyOrdered(
        uran.ApplyChoice(
            Mutate.warp,
            Mutate.affine_transform,
            Mutate.perspective_transform,
        ),
        uran.ApplyChoice(
            Mutate.fade_black, Mutate.fade_white, Mutate.brightness_contrast, None
        ),
    )
    _RAN_VRTL = uran.ApplyShuffled(
        uran.ApplyChoice(Mutate.blur, None),
        uran.ApplyChoice(Mutate.sharpen, None),
        uran.ApplyChoice(
            Mutate.noise,
            Mutate.gaussian_noise,
            Mutate.salt_pepper_noise,
            Mutate.random_erasing,
            Mutate.cutout,
            None,
        ),
        uran.ApplyChoice(
            uran.ApplyChoice(
                Mutate.noise,
                Mutate.gaussian_noise,
                Mutate.salt_pepper_noise,
                Mutate.random_erasing,
                Mutate.cutout,
                None,
            ),
            None,
        ),
        uran.ApplyChoice(Mutate.tint, None),
        uran.ApplyChoice(
            Mutate.fade_black, Mutate.fade_white, Mutate.brightness_contrast, None
        ),
        uran.ApplyChoice(Mutate.random_erasing, None),
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
    mtg = SyntheticBgFgMtgImages(img_type="small")
    ilsvrc = IlsvrcImages()

    for i in tqdm(range(10)):
        _o = mtg.ran()
        _l = ilsvrc.ran()
        x, y = SyntheticBgFgMtgImages.make_virtual_pair(
            _o, _l, (192, 128), (192, 128), True
        )

    # 100%|██████████| 1000/1000 [00:10<00:00, 94.77it/s]
    for i in tqdm(range(1000)):
        _o = mtg.ran()
        _l = ilsvrc.ran()

        x, y = SyntheticBgFgMtgImages.make_virtual_pair(
            _o, _l, (192, 128), (192, 128), True
        )

        uimg.imshow_loop(x, "asdf")
        # uimg.imshow_loop(y, "asdf")
