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

import numpy as np
from math import ceil
import random
import imgaug.augmenters as iaa
from imgaug import SegmentationMapsOnImage
from tqdm import tqdm


from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.files as ufls
from mtgdata.scryfall import ScryfallCardFace

# polyfill
np.bool = bool

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
    def _load_image(cls, path_or_img) -> np.ndarray[np.uint8]:
        if isinstance(path_or_img, (str, Path)):
            return uimg.imread_uint8(path_or_img)
        else:
            return uimg.img_uint8(path_or_img)

    @abc.abstractmethod
    def __getitem__(self, item) -> np.ndarray: ...

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def ran(self) -> np.ndarray[np.uint8]:
        return self._load_image(self.ran_path())

    @abc.abstractmethod
    def ran_path(self): ...

    def get(self, idx) -> np.ndarray[np.uint8]:
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
    Dataset of Magic The Gathering Card Images. This generates synthetic data of
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
            self._aug_upsidedown = iaa.Sometimes(
                0.5,
                iaa.Sequential(
                    [
                        iaa.Fliplr(1.0),  # Horizontal flip
                        iaa.Flipud(1.0),  # Vertical flip
                    ]
                ),
            )
        else:
            self._aug_upsidedown = iaa.Noop()

        # Background augmentation
        self._aug_bg = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5, iaa.Fliplr(1.0)
                ),  # Horizontal flip with 50% probability
                iaa.Sometimes(
                    0.5, iaa.Flipud(1.0)
                ),  # Vertical flip with 50% probability
                iaa.Affine(
                    rotate=(0, 360)
                ),  # Random rotation between 0 and 360 degrees
                iaa.Sometimes(
                    0.9,
                    iaa.Sequential(
                        [  # Color jitter with 90% probability
                            iaa.Multiply((0.5, 1.5)),  # Brightness adjustment
                            iaa.LinearContrast((0.5, 1.5)),  # Contrast adjustment
                            iaa.MultiplySaturation((0.5, 1.5)),  # Saturation adjustment
                            iaa.AddToHue((-51, 51)),  # Hue adjustment, approx. 0.2*255
                        ]
                    ),
                ),
            ]
        )

        # Foreground (card) augmentation
        self._aug_fg = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.8,
                    iaa.Sequential(
                        [
                            iaa.Affine(
                                translate_percent={
                                    "x": (-0.0625, 0.0625),
                                    "y": (-0.0625, 0.0625),
                                },
                                scale=(0.8, 1.0),  # Scale down by up to 20%
                                rotate=(-5, 5),
                                order=3,  # Cubic interpolation
                            ),
                            iaa.PerspectiveTransform(scale=(0, 0.075), keep_size=True),
                        ]
                    ),
                ),
                iaa.Sometimes(
                    0.2,
                    iaa.Cutout(
                        nb_iterations=1,
                        size=(0.2, 0.7),
                        fill_mode="constant",
                        cval=(0, 255),  # Random uniform fill
                    ),
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.Sequential(
                        [
                            iaa.Multiply((0.7, 1.3)),  # Brightness
                            iaa.LinearContrast((0.7, 1.3)),  # Contrast
                        ]
                    ),
                ),
            ],
            random_order=True,
        )

        # Virtual (synthetic image) augmentation
        self._aug_vrtl = iaa.Sequential(
            [
                iaa.Sequential(
                    [
                        iaa.Sometimes(
                            0.5,
                            iaa.OneOf(
                                [
                                    iaa.GaussianBlur(
                                        sigma=(0, 3.0)
                                    ),  # Approx. blur_limit=7
                                    iaa.MotionBlur(k=7),
                                    iaa.MedianBlur(k=7),
                                ]
                            ),
                        ),
                        iaa.Sometimes(0.25, iaa.JpegCompression(compression=(0, 2))),
                        # iaa.Sometimes(0.25, iaa.Sequential([
                        #     iaa.Resize((0.2, 0.9), interpolation="linear"),
                        #     iaa.Resize(default_x_size_hw, interpolation="linear")
                        # ]))
                    ],
                    random_order=True,
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf(
                        [
                            iaa.AdditiveGaussianNoise(
                                scale=(0, 0.1 * 255)
                            ),  # Approx. ISONoise
                            # iaa.SimplexNoiseAlpha(),  # Placeholder for fog
                            iaa.AdditivePoissonNoise(lam=(0, 40)),  # Shot noise
                            iaa.AdditiveGaussianNoise(
                                scale=(0, 0.1 * 255)
                            ),  # Additive noise
                            iaa.AdditiveGaussianNoise(
                                scale=(0, 0.1 * 255)
                            ),  # Gaussian noise
                            iaa.SaltAndPepper(p=(0, 0.05)),
                        ]
                    ),
                ),
                iaa.OneOf(
                    [
                        iaa.Sequential(
                            [
                                iaa.Multiply((0.2, 1.8)),  # Brightness
                                iaa.LinearContrast((0.5, 1.5)),  # Contrast
                                iaa.MultiplySaturation((0.7, 1.3)),  # Saturation
                                iaa.AddToHue((-25, 25)),  # Hue, approx. 0.1*255
                            ]
                        ),
                        iaa.GammaContrast(gamma=(0.8, 1.2)),  # Gamma adjustment
                    ]
                ),
            ],
            random_order=True,
        )

    # GET IMAGES - NO AUGMENTS

    @classmethod
    def _get_img(cls, path_or_img) -> np.ndarray[np.uint8]:
        return (
            uimg.imread_uint8(path_or_img)
            if isinstance(path_or_img, str)
            else uimg.img_uint8(path_or_img)
        )

    def _get_card(self, card_path_or_img: PathOrImg | None) -> np.ndarray[np.uint8]:
        if card_path_or_img is None:
            return self.ds_mtg.ran()
        return self._get_img(card_path_or_img)

    def _get_bg(self, bg_path_or_img: PathOrImg | None) -> np.ndarray[np.uint8]:
        if bg_path_or_img is None:
            return self.ds_bg.ran()
        return self._get_img(bg_path_or_img)

    # AUGMENTED

    def make_target_card(
        self,
        card_path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
        float32: bool = True,
    ) -> np.ndarray[np.uint8] | np.ndarray[np.float32]:
        card = self._get_card(card_path_or_img)
        ret = uimg.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size_hw=size_hw or self._default_y_size_hw,
        )
        if float32:
            ret = uimg.img_float32(ret)
        return ret

    def _make_aug_card_and_mask(
        self, path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> (
        tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]
        | tuple[np.ndarray[np.float32], np.ndarray[np.float32]]
    ):
        # create card and resize
        card = self._get_card(path_or_img)
        card = self._aug_upsidedown.augment_image(card)
        card = uimg.resize(card, size_hw=size_hw or self._default_x_size_hw)
        # crop card and mask
        mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        # augment
        segmap = SegmentationMapsOnImage(mask, shape=card.shape)
        card, segmap = self._aug_fg.augment(image=card, segmentation_maps=segmap)
        segmap: SegmentationMapsOnImage
        mask = segmap.arr
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[:, :, 0]
            else:
                raise ValueError(
                    f"Expected mask to have shape (H, W) or (H, W, 1), got {mask.shape}"
                )
        assert mask.dtype == np.int32
        mask = mask.astype(np.uint8)
        return card, mask

    def _make_aug_bg(
        self, bg_path_or_img: PathOrImg | None = None, size_hw: SizeHW | None = None
    ) -> np.ndarray[np.uint8]:
        bg = self._get_bg(bg_path_or_img)
        # augment
        bg = self._aug_bg.augment_image(bg)
        # resize
        bg = uimg.crop_to_size(bg, size_hw or self._default_x_size_hw)
        return bg

    def make_synthetic_input_card(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        size_hw: SizeHW | None = None,
        float32: bool = True,
    ) -> np.ndarray[np.uint8] | np.ndarray[np.float32]:
        size_hw = size_hw or self._default_x_size_hw
        # fg - card
        fg, fg_mask = self._make_aug_card_and_mask(card_path_or_img, size_hw=size_hw)
        # bg
        bg = self._make_aug_bg(bg_path_or_img, size_hw=size_hw)
        # merge
        synthetic_card = uimg.rgb_mask_over_rgb(fg, fg_mask, bg)
        synthetic_card = self._aug_vrtl.augment_image(synthetic_card)
        # checks
        assert synthetic_card.shape[:2] == size_hw, (
            f"Expected size_hw={size_hw}, got={synthetic_card.shape[:2]}"
        )
        if float32:
            synthetic_card = uimg.img_float32(synthetic_card)
        return synthetic_card

    def make_synthetic_input_and_target_card_pair(
        self,
        card_path_or_img: PathOrImg | None = None,
        bg_path_or_img: PathOrImg | None = None,
        x_size_hw: SizeHW | None = None,
        y_size_hw: SizeHW | None = None,
        float32: bool = True,
    ) -> (
        tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]
        | tuple[np.ndarray[np.float32], np.ndarray[np.float32]]
    ):
        # ensure the same card is used for both x and y
        card = self._get_card(card_path_or_img)
        # only inputs are flipped
        x = self.make_synthetic_input_card(
            card, bg_path_or_img, size_hw=x_size_hw, float32=float32
        )
        y = self.make_target_card(card, size_hw=y_size_hw, float32=float32)
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

    # 100%|██████████| 1000/1000 [00:16<00:00, 61.01it/s]
    for i in tqdm(range(1000)):
        x, y = ds.make_synthetic_input_and_target_card_pair()

        # uimg.imshow_loop(x, "x")
        # uimg.imshow_loop(y, "y")
