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
from pathlib import Path

import cv2
import numpy as np
from math import ceil
import random
import os
from tqdm import tqdm
import platform
import re

from mtgdata import ScryfallDataset, ScryfallImageType
import mtgvision.util.image as uimg
import mtgvision.util.values as uval
import mtgvision.util.random as uran
import mtgvision.util.files as ufls
import mtgvision.util.lazy as ulzy


# ========================================================================= #
# RANDOM TRANSFORMS                                                         #
# ========================================================================= #


# TODO: replace with skimage functions

class Mutate:

    @staticmethod
    def flip(img, horr=True, vert=True):
        return uimg.flip(img, horr=horr and (random.random() >= 0.5), vert=vert and (random.random() >= 0.5))

    @staticmethod
    def rotate_bounded(img, deg_min=0, deg_max=360):
        return uimg.rotate_bounded(img, deg_min + np.random.random() * (deg_max - deg_min))

    @staticmethod
    def upsidedown(img):
        return np.rot90(img, k=2)

    @staticmethod
    def warp(img, warp_ratio=0.15, warp_ratio_min=-0.05):
        # [top left, top right, bottom left, bottom right]
        (h, w) = (img.shape[0]-1, img.shape[1]-1)
        src_pts = uimg.fxx((0, 0), (0, w), (h, 0), (h, w))
        ran = warp_ratio_min + np.random.rand(4, 2) * (abs(warp_ratio-warp_ratio_min)*0.5)
        dst_pts = uimg.fxx(ran * uimg.fxx((h, w), (h, -w), (-h, w), (-h, -w)) + src_pts)
        # transform matrix
        transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warp
        return cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]))

    @staticmethod
    def warp_inv(img, warp_ratio=0.5, warp_ratio_min=0.25):
        return Mutate.warp(img, warp_ratio=-warp_ratio, warp_ratio_min=-warp_ratio_min)

    @staticmethod
    def noise(img, amount=0.75):
        noise_type = random.choice(['speckle', 'gaussian', 'pepper', 'poisson'])
        if noise_type == 'speckle':
            noisy = uimg.noise_speckle(img, strength=0.3)
        elif noise_type == 'gaussian':
            noisy = uimg.noise_gaussian(img, mean=0, var=0.05)
        elif noise_type == 'pepper':
            noisy = uimg.noise_salt_pepper(img, strength=0.1, svp=0.5)
        elif noise_type == 'poisson':
            noisy = uimg.noise_poisson(img, peak=0.8, amount=0.5)
        else:
            raise Exception('Invalid Choice')
        ratio = np.random.random()*amount
        img[:,:,:3] = (ratio)*noisy[:,:,:3] + (1-ratio)*img[:,:,:3]
        return img

    @staticmethod
    def blur(img, n_max=3):
        n = np.random.randint(0, (n_max-1)//2+1) * 2 + 1
        return cv2.GaussianBlur(img, (n, n), 0)

    @staticmethod
    def tint(img, amount=0.15):
        for i in range(3):
            r = 1 + amount * (2*np.random.random()-1)
            img[:,:,i] = uimg.clip(r * img[:, :, i])
        return img

    @staticmethod
    def fade_white(img, amount=0.33):
        ratio = np.random.random()*amount
        img[:,:,:3] = (ratio)*1 + (1-ratio)*img[:,:,:3]
        return img

    @staticmethod
    def fade_black(img, amount=0.5):
        ratio = np.random.random()*amount
        img[:,:,:3] = (ratio)*0 + (1-ratio)*img[:,:,:3]
        return img


# ========================================================================= #
# Vars                                                                      #
# ========================================================================= #


# if 'darwin' in platform.system().lower():
#     DATASETS_ROOT = os.getenv('DATASETS_ROOT', os.path.join(os.environ['HOME'], 'Downloads/datasets'))
# else:
#     DATASETS_ROOT = os.getenv('DATASETS_ROOT', '/datasets')

DATASETS_ROOT = Path(__file__).parent.parent.parent / 'mtg-dataset/mtgdata/data'

print(f'DATASETS_ROOT={DATASETS_ROOT}')


# ========================================================================= #
# Dataset - Base                                                            #
# ========================================================================= #


class Dataset(ufls.Folder):
    def __init__(self, name, datasets_root=None):
        _dataset_root = ufls.init_dir(uval.default(datasets_root, DATASETS_ROOT), name)
        self._dataset_name = name
        super().__init__(_dataset_root)

    @property
    def name(self):
        return self._dataset_name


# ========================================================================= #
# Dataset - IlsvrcImages                                                    #
# ========================================================================= #


class IlsvrcImages(ulzy.LazyList):

    ILSVRC_SET_TYPES = ['val', 'test', 'train']

    def __init__(self):
        root = Dataset('ilsvrc').cd('2010')
        if not os.path.isdir(root.path):
            print("MAKE SURE YOU HAVE DOWNLOADED THE ILSVRC 2010 TEST DATASET TO: {}".format(root.path), 'yellow')
            print(" - The images must all be located within: {}".format(root.to('val')), 'yellow')
            print(" - For example: {}".format(root.to('val', 'ILSVRC2010_val_00000001.JPEG')), 'yellow')
            print("The image versions of the ILSVRC Datasets are for educational purposes only, and cannot be redistributed.", 'yellow')
            print("Please visit: www.image-net.org to obtain the download links.", 'yellow')
        super().__init__([ulzy.Lazy(file, uimg.imread) for file in ufls.get_image_paths(root.path, prefixed=True)])


# ========================================================================= #
# Dataset - MTG Images                                                      #
# ========================================================================= #


class MtgImages(ulzy.LazyList):

    def __init__(self, img_type=ScryfallImageType.small, predownload=False):
        print(Path(__file__).parent.parent.parent / 'mtg-dataset/mtgdata/data')
        print(Path(__file__).parent.parent.parent / 'mtg-dataset/mtgdata/data')
        print(Path(__file__).parent.parent.parent / 'mtg-dataset/mtgdata/data')
        self._ds = ScryfallDataset(
            img_type=img_type,
            data_root=Path(__file__).parent.parent.parent / 'mtg-dataset/mtgdata/data',
            force_update=False,
            download_mode='now' if predownload else 'none',
        )
        # open PIL.Image.Image
        super().__init__(self._make_lazy_cards())

    @classmethod
    def _get_card(cls, card):
        return np.asarray(card.dl_and_open_im_resized(), dtype='float32') / 255

    def _make_lazy_cards(self):
        return [
            ulzy.Lazy(card, self._get_card) for card in self._ds
        ]

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
    def make_cropped(path_or_img, size=None, half_upsidedown=False):
        card = uimg.imread(path_or_img) if (type(path_or_img) == str) else path_or_img
        ret = uimg.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size=size
        )
        return uran.ApplyChoice(Mutate.upsidedown, None)(ret) if half_upsidedown else ret

    @staticmethod
    def make_masked(path_or_img):
        card = uimg.imread(path_or_img) if (type(path_or_img) == str) else path_or_img
        mask = uimg.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        ret = cv2.merge((
            card[:, :, 0],
            card[:, :, 1],
            card[:, :, 2],
            mask,
        ))
        return ret

    @staticmethod
    def make_bg(bg_path_or_img, size):
        bg = uimg.imread(bg_path_or_img) if (type(bg_path_or_img) == str) else bg_path_or_img
        bg = MtgImages._RAN_BG(bg)
        bg = uimg.crop_to_size(bg, size)
        return bg

    @staticmethod
    def make_virtual(card_path_or_img, bg_path_or_img, size, half_upsidedown=False):
        card = uimg.imread(card_path_or_img) if (type(card_path_or_img) == str) else card_path_or_img
        card = uran.ApplyChoice(Mutate.upsidedown, None)(card) if half_upsidedown else card
        # fg - card
        fg = MtgImages.make_masked(card)
        fg = uimg.crop_to_size(fg, size, pad=True)
        fg = MtgImages._RAN_FG(fg)
        # bg
        bg = MtgImages.make_bg(bg_path_or_img, size)
        # merge
        virtual = uimg.rgba_over_rgb(fg, bg)
        virtual = MtgImages._RAN_VRTL(virtual)
        assert virtual.shape[:2] == size
        return virtual

    @staticmethod
    def make_virtual_pair(card_path_or_img, bg_path_or_img, x_size, y_size, half_upsidedown=False):
        card = uimg.imread(card_path_or_img) if (type(card_path_or_img) == str) else card_path_or_img
        # only inputs are flipped
        x = MtgImages.make_virtual(card, bg_path_or_img, size=x_size, half_upsidedown=half_upsidedown)
        y = MtgImages.make_cropped(card, size=y_size)
        return x, y


# ========================================================================= #
# LOCAL MTG FILES                                                           #
# ========================================================================= #


# TODO: merge with above
class MtgLocalFiles(object):

    def __init__(self, img_type=ScryfallImageType.small, x_size=(320, 224), y_size=(204, 144), pregen=False, force=False):
        img_type = ScryfallImageType(img_type)

        self.ds_orig = Dataset('mtg').cd(img_type.value)
        self.ds_warp = Dataset('mtg_warp').cd(img_type.value, '{}_{}'.format(x_size[0], x_size[1]))
        self.ds_crop = Dataset('mtg_crop').cd(img_type.value, '{}_{}'.format(y_size[0], y_size[1]))

        self.x_size = x_size
        self.y_size = y_size

        self.paths = sorted(ufls.get_image_paths(self.ds_orig.path, prefixed=False))
        self.paths_warp = set(ufls.get_image_paths(self.ds_warp.path, prefixed=False))
        self.paths_crop = set(ufls.get_image_paths(self.ds_crop.path, prefixed=False))

        self.ilsvrc = IlsvrcImages()

        if pregen:
            MtgImages(img_type=img_type, predownload=pregen)
            print('Pre-generating:')
            if force or (len(self.paths_warp) < len(self.paths)) or (len(self.paths_crop) < len(self.paths)):
                for rel in tqdm(self.paths, desc='{}x [warp] {} [crop] {}'.format(len(self.paths), x_size, y_size), total=len(self.paths)):
                    self._gen_save_warp(rel, force=force, regen_rate=0)
                    self._gen_save_crop(rel, force=force)

    def __getitem__(self, i):
        return self._gen_save_crop(self.paths[i])

    def get_uuid(self, i):
        # eg: apc/apc__01f891ca-4e6a-4710-b1cf-5dabb5e1ad93__whirlpool-warrior__small.jpg
        name = os.path.basename(self.paths[i])
        uuid = re.match('.*?__(.*?-.*?-.*?-.*?-.*?)__.*?__.*?.jpg', name).group(1)
        return uuid

    def __len__(self):
        return len(self.paths)

    def _gen_save_warp(self, rel, force=False, save=True, orig_ratio=0.1, regen_rate=0.1, half_upsidedown=True):
        if force or not save or (rel not in self.paths_warp) or random.random() < regen_rate:
            card = uimg.imread(self.ds_orig.to(rel))  # some cards are not the same size
            if random.random() < orig_ratio:
                x = MtgImages.make_cropped(card, self.x_size, half_upsidedown=half_upsidedown)
            else:
                x = MtgImages.make_virtual(card, self.ilsvrc.ran(), size=self.x_size, half_upsidedown=half_upsidedown)
            if save:
                uimg.imwrite(self.ds_warp.to(rel, init=True, is_file=True), x)
                self.paths_warp.add(rel)
            return x
        else:
            return uimg.imread(self.ds_warp.to(rel))

    def _gen_save_crop(self, rel, force=False, save=True):
        if force or not save or (rel not in self.paths_crop):
            card = uimg.imread(self.ds_orig.to(rel))  # some cards are not the same size
            y = MtgImages.make_cropped(card, size=self.y_size)
            if save:
                uimg.imwrite(self.ds_crop.to(rel, init=True, is_file=True), y)
                self.paths_crop.add(rel)
            return y
        else:
            return uimg.imread(self.ds_crop.to(rel))

    def gen_warp_crop(self, rel_path=None, force=False, orig_ratio=0.1, regen_rate=0.1, half_upsidedown=False, save=True):
        if rel_path is None:
            rel_path = random.choice(self.paths)
        x = self._gen_save_warp(rel_path, force=force, orig_ratio=orig_ratio, half_upsidedown=half_upsidedown, regen_rate=regen_rate, save=save)
        y = self._gen_save_crop(rel_path, force=force, save=save)
        return x, y

    def gen_warp_crop_set(self, n, orig_ratio=0.1, regen_rate=0.1, half_upsidedown=False, save=True):
        random.shuffle(self.paths)
        existing = len([None for r in self.paths[:n] if r not in self.paths_warp or r not in self.paths_crop])
        results = (self.gen_warp_crop(r, orig_ratio=orig_ratio, regen_rate=regen_rate, half_upsidedown=half_upsidedown, save=save) for r in self.paths[:n])
        xs, ys = list(zip(*tqdm(results, desc='{} of {} | {} -> {}'.format(existing, n, self.x_size, self.y_size), total=n)))
        xs = np.array(xs, dtype=np.float16)
        ys = np.array(ys, dtype=np.float16)

        return xs, ys

    def gen_warp_crop_orig_set(self, n, orig_ratio=0.1, regen_rate=0.1, half_upsidedown=False, save=True, chosen=None):
        random.shuffle(self.paths)
        if chosen is None:
            chosen = []
        for path in chosen:
            if path not in self.paths:
                print('Invalid Chosen Path: {}'.format(path))
        chosen = (chosen + self.paths[:n])[:n]
        print(chosen)
        existing = len([None for r in chosen if r not in self.paths_warp or r not in self.paths_crop])
        results = ((*self.gen_warp_crop(r, orig_ratio=orig_ratio, regen_rate=regen_rate, half_upsidedown=half_upsidedown, save=save), MtgImages.make_cropped(self.ds_orig.to(r), self.x_size)) for r in chosen)
        xs, ys, os = list(zip(*tqdm(results, desc='{} of {} | {} -> {}'.format(existing, n, self.x_size, self.y_size), total=n)))
        xs = np.array(xs, dtype=np.float16)
        ys = np.array(ys, dtype=np.float16)
        os = np.array(os, dtype=np.float16)
        return xs, ys, os


# ========================================================================= #
# MAIN - TEST                                                               #
# ========================================================================= #


if __name__ == "__main__":
    orig = MtgImages(img_type='normal')
    ilsvrc = IlsvrcImages()

    while True:
        o = orig.ran()
        l = ilsvrc.ran()

        x, y = MtgImages.make_virtual_pair(o, l, (192, 128), (192, 128), True)

        uimg.imshow_loop(x, 'asdf')
        uimg.imshow_loop(y, 'asdf')
