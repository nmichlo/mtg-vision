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
import contextlib
import uuid
from os import PathLike
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import jax.numpy as jnp
from math import ceil
import random

import jax.random
import numpy as np
from tqdm import tqdm

from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace

import mtgvision.aug as A
from mtgvision.aug import AugItems

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
    def __getitem__(self, item) -> jnp.ndarray: ...

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def ran(self) -> jnp.ndarray:
        return self._load_image(self.ran_path())

    @abc.abstractmethod
    def ran_path(self): ...

    def get(self, idx) -> jnp.ndarray:
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
PathOrImg = str | jnp.ndarray | PathLike


# ========================================================================= #
# AUGMENTATION PIPELINES & MAIN CLASS                                       #
# ========================================================================= #


class SyntheticBgFgMtgImages:
    def __init__(
        self,
        ds_mtg=None,
        ds_bg=None,
        *,
        half_upsidedown: bool = False,
        default_x_size_hw: SizeHW = (192, 128),
        default_y_size_hw: SizeHW = (192, 128),
    ):
        self.ds_mtg = ds_mtg if ds_mtg is not None else MtgImages()
        self.ds_bg = ds_bg if ds_bg is not None else IlsvrcImages()
        self._default_x_size_hw = default_x_size_hw
        self._default_y_size_hw = default_y_size_hw

        # Define augmentations with the new system
        # Upside-down augmentation
        self._aug_upsidedown = A.Rotate180(p=0.5) if half_upsidedown else A.NoOp()

        # Background augmentations (image only)
        self._aug_bg = A.SomeOf(
            augments=(
                A.FlipHorizontal(p=0.5),
                A.FlipVertical(p=0.5),
                A.RotateBounded(deg=(0, 360), p=1.0),
                A.ColorTint(tint=(-0.15, 0.15), p=0.9),
            ),
            n=(1, 4),
        )

        # Foreground augmentations (image and mask)
        self._aug_fg = A.AllOf(
            augments=(
                # A.AllOf(
                #     augments=(
                #         A.ShiftScaleRotate(
                #             shift_ratio=(-0.0625, 0.0625),
                #             scale_ratio=(-0.1, 0.0),
                #             rotate_limit=(-5, 5),
                #             p=1.0,
                #         ),
                #         A.PerspectiveWarp(
                #             corner_jitter_ratio=(-0.075, 0.075),
                #             p=1.0,
                #         ),
                #     ),
                #     p=0.95,
                # ),
                A.ColorTint(tint=(-0.15, 0.15), p=0.5),
            ),
        )

        # Virtual augmentations (image only)
        self._aug_vrtl = A.SomeOf(
            augments=(
                A.BlurGaussian(sigma=(0, 2), kernel_size=7, p=0.5),
                A.BlurJpegCompression(quality=(98, 100), p=0.25),
                A.BlurDownscale(p=0.25),
                # noise
                A.NoiseAdditiveGaussian(strength=(0, 0.1), p=1.0),
                # A.NoisePoison(peak=(0.01, 1), p=1.0),
                A.NoiseSaltPepper(strength=(0, 0.1), p=1.0),
                A.NoiseMultiplicativeGaussian(strength=(0, 0.1), p=1.0),
                A.RandomErasing(scale=(0.2, 0.7), p=1.0, color="uniform_random"),
                # color
                A.ColorTint(tint=(-0.15, 0.15), p=1.0),
                A.ColorGamma(gamma=(0.8, 1.2), p=1.0),
                A.ColorBrightness(brightness=(-0.2, 0.2), p=1.0),
                A.ColorExposure(exposure=(-0.2, 0.2), p=1.0),
            ),
            n=(1, 3),
        )

        # quick test
        # self._aug_upsidedown.quick_test()
        # self._aug_bg.quick_test()
        # self._aug_fg.quick_test()
        # self._aug_vrtl.quick_test()

        # JIT EVERYTHING
        # self._aug_upsidedown
        # self._aug_bg
        # self._aug_fg
        # self._aug_vrtl

        self._key = jax.random.key(42)

    # IMAGE LOADING METHODS

    @staticmethod
    @jax.jit
    def _to_float_img(img) -> jnp.ndarray:
        """Load an image as float32."""
        if isinstance(img, jnp.ndarray):
            if img.dtype == jnp.uint8:
                return img.astype(jnp.float32) / 255.0
            elif img.dtype == jnp.float32:
                return img
            else:
                raise ValueError(f"Unsupported dtype: {img.dtype}")
        else:
            raise ValueError(f"Unsupported type: {type(img)}")

    @classmethod
    def _load(cls, path_or_img: PathOrImg | None, ran_fn) -> jnp.ndarray:
        img = path_or_img
        if img is None:
            img = ran_fn()
        if isinstance(img, (str, Path)):
            img = cv2.imread(img, cv2.IMREAD_COLOR_RGB)  # much faster than Image.open
            img = np.asarray(img, dtype=jnp.float32) / 255.0  # much faster than jnp???
            return img
        assert img.dtype in (np.float32, jnp.float32)
        return img

    def _get_card(self, card_path_or_img: PathOrImg | None) -> jnp.ndarray:
        """Get a random card image or load the specified one."""
        return self._load(card_path_or_img, self.ds_mtg.ran_path)

    def _get_bg(self, bg_path_or_img: PathOrImg | None) -> jnp.ndarray:
        """Get a random background image or load the specified one."""
        return self._load(bg_path_or_img, self.ds_bg.ran_path)

    def next_key(self, key=None):
        if key is None:
            self._key, key = jax.random.split(self._key)
        return key

    # AUGMENTED IMAGE METHODS

    def make_target_card(
        self, card_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> jnp.ndarray:
        """Create a target card image without background."""
        card = self._get_card(card_path_or_img)
        ret = uimg.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size_hw=size_hw or self._default_y_size_hw,
        )
        return ret

    def _make_aug_card_and_mask(
        self,
        path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
        key=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate an augmented card and its mask."""
        key = self.next_key(key)
        card = self._get_card(path_or_img)
        card = self._aug_upsidedown(key, AugItems(image=card)).image
        card = uimg.resize(card, size_hw=size_hw or self._default_x_size_hw)
        mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)[:, :, None]
        items = self._aug_fg(key, AugItems(image=card, mask=mask))
        return items.image, items.mask[:, :, 0]

    def _make_aug_bg(
        self,
        bg_path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
        key=None,
    ) -> jnp.ndarray:
        """Generate an augmented background."""
        key = self.next_key(key)
        bg = self._get_bg(bg_path_or_img)
        bg_aug = self._aug_bg(key, AugItems(image=bg)).image
        bg_aug = uimg.crop_to_size(bg_aug, size_hw or self._default_x_size_hw)
        return bg_aug

    def make_synthetic_input_card(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
        key=None,
    ) -> jnp.ndarray:
        """Create a synthetic input card with background."""
        key = self.next_key(key)
        size_hw = size_hw or self._default_x_size_hw
        # Foreground (card) and mask
        fg, fg_mask = self._make_aug_card_and_mask(
            card_path_or_img,
            size_hw=size_hw,
            key=key,
        )
        # Background
        bg = self._make_aug_bg(bg_path_or_img, size_hw=size_hw)
        # Merge foreground and background
        synthetic_card = uimg.rgb_mask_over_rgb(fg, fg_mask, bg)
        synthetic_card_aug = self._aug_vrtl(key, AugItems(image=synthetic_card)).image
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
        key=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a pair of synthetic input and target cards."""
        card = self._get_card(card_path_or_img)
        x = self.make_synthetic_input_card(
            card, bg_path_or_img, size_hw=x_size_hw, key=key
        )
        y = self.make_target_card(card, size_hw=y_size_hw)
        return x, y


@contextlib.contextmanager
def timer(name: str):
    import time

    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.4f}s")


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        ds = SyntheticBgFgMtgImages(
            ds_mtg=MtgImages(),
            ds_bg=IlsvrcImages(),
            default_x_size_hw=(192, 128),
            default_y_size_hw=(192, 128),
            half_upsidedown=True,
        )

        # for i in tqdm(range(10)):
        #     x, y = ds.make_synthetic_input_and_target_card_pair()

        # 100%|██████████| 1000/1000 [00:16<00:00, 60.01it/s]
        for i in tqdm(range(10)):
            # 300 it/s
            x = ds._get_card(None)  # 1150 it/s
            y = ds._get_bg(None)  # 440 it/s

            # 2 it/s
            x, y = ds.make_synthetic_input_and_target_card_pair(x, y)
