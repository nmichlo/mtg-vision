#  MIT License
#
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import cv2
import numpy as np
from math import ceil
import random
from termcolor import cprint
import util
from util import default, ApplyChoice, ApplyOrdered, ApplyShuffled, JsonCache, Proxy, asrt_in, asrt
import os
from tqdm import tqdm
from mtgtools.MtgDB import MtgDB
import platform
import re


# ========================================================================= #
# RANDOM TRANSFORMS                                                         #
# ========================================================================= #


# TODO: replace with skimage functions

class Mutate:

    @staticmethod
    def flip(img, horr=True, vert=True):
        return util.flip(img, horr=horr and (random.random() >= 0.5), vert=vert and (random.random() >= 0.5))

    @staticmethod
    def rotate_bounded(img, deg_min=0, deg_max=360):
        return util.rotate_bounded(img, deg_min + np.random.random()*(deg_max - deg_min))

    @staticmethod
    def upsidedown(img):
        return np.rot90(img, k=2)

    @staticmethod
    def warp(img, warp_ratio=0.15, warp_ratio_min=-0.05):
        # [top left, top right, bottom left, bottom right]
        (h, w) = (img.shape[0]-1, img.shape[1]-1)
        src_pts = util.fxx((0, 0), (0, w), (h, 0), (h, w))
        ran = warp_ratio_min + np.random.rand(4, 2) * (abs(warp_ratio-warp_ratio_min)*0.5)
        dst_pts = util.fxx(ran * util.fxx((h, w), (h, -w), (-h, w), (-h, -w)) + src_pts)
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
            noisy = util.noise_speckle(img, strength=0.3)
        elif noise_type == 'gaussian':
            noisy = util.noise_gaussian(img, mean=0, var=0.05)
        elif noise_type == 'pepper':
            noisy = util.noise_salt_pepper(img, strength=0.1, svp=0.5)
        elif noise_type == 'poisson':
            noisy = util.noise_poisson(img, peak=0.8, amount=0.5)
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
            img[:,:,i] = util.clip(r * img[:,:,i])
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


if 'darwin' in platform.system().lower():
    DATASETS_ROOT = os.getenv('DATASETS_ROOT', os.path.join(os.environ['HOME'], 'Downloads/datasets'))
else:
    DATASETS_ROOT = os.getenv('DATASETS_ROOT', '/datasets')

print('DATASETS_ROOT={}'.format(DATASETS_ROOT))


# ========================================================================= #
# Dataset - Base                                                            #
# ========================================================================= #


class Dataset(util.Folder):
    def __init__(self, name, datasets_root=None):
        _dataset_root = util.init_dir(default(datasets_root, DATASETS_ROOT), name)
        self._dataset_name = name
        super().__init__(_dataset_root)

    @property
    def name(self):
        return self._dataset_name


# ========================================================================= #
# Dataset - MTG Info                                                        #
# ========================================================================= #


class MtgHandler(object):
    def __init__(self, force_update=False, force_update_scryfall=False, force_update_mtgio=False):
        dataset = Dataset('mtgtools')

        print(dataset)
        print(dataset.to('mtg.db'))

        self.db = MtgDB(dataset.to('mtg.db'))
        if force_update_scryfall or force_update or len(self.db.root.scryfall_sets) <= 0:
            self.db.scryfall_update()
        if force_update_mtgio or force_update:
            self.db.mtgio_update()

        print('Getting UUIDs')
        with JsonCache('./cache/mtgtools_uuids.json') as data:
            if 'uuids' not in data or force_update_scryfall or force_update:
                data['uuids'] = {}
                for i, card in enumerate(tqdm(self.scryfall_cards, desc='Caching Card UUIDs')):
                    data['uuids'][card.id] = i
            self.scryfall_uuids_to_index = data['uuids']
            print('UUIDS FOUND: {} of total {}'.format(len(self.scryfall_uuids_to_index), len(self.scryfall_cards)))
        print('Got UUIDs')

    @property
    def scryfall_sets(self):
        return self.db.root.scryfall_sets

    @property
    def mtgio_sets(self):
        return self.db.root.mtgio_sets

    @property
    def scryfall_cards(self):
        return self.db.root.scryfall_cards

    @property
    def mtgio_cards(self):
        return self.db.root.mtgio_cards

    def scryfall_card_from_uuid(self, uuid):
        if uuid in self.scryfall_uuids_to_index:
            return self.scryfall_cards[self.scryfall_uuids_to_index[uuid]]
        return None

    IMG_TYPES = ['small', 'normal', 'large', 'png', 'art_crop', 'border_crop']

    def scryfall_cards_paths_uris(self, img_type='small', force=False):
        asrt_in(img_type, MtgHandler.IMG_TYPES)
        root = Dataset('mtg').cd(img_type, init=True)
        with JsonCache('./cache/mtg_uris_{}.json'.format(img_type), refresh=force) as uris:
            if 'resources' not in uris:
                resources = []
                for s in tqdm(self.scryfall_sets, desc='Building Image URI Cache [{}] ({})'.format(img_type, root)):
                    if s.api_type != 'scryfall':
                        continue
                    for card in s:
                        uri_file = MtgHandler._card_uri_to_file(card, folder=s.code, img_type=img_type)
                        if uri_file is not None:
                            resources.append(uri_file)
                print('URIS FOUND: {} of {}'.format(len(resources), len(self.scryfall_cards)))
                uris['resources'] = resources
            return [(u, root.to(f)) for u, f in uris['resources']]

    @staticmethod
    def _card_uri_to_file(card, folder, img_type):
        if card.image_uris is None:
            return None
        uri = card.image_uris[img_type]
        if uri is None:
            return None
        # filename:
        ext = os.path.splitext(uri)[1].split('?')[0]
        name = re.sub('[^-a-z]', '', card.name.lower().replace(" ", "-"))
        file = os.path.join(folder, '{}__{}__{}__{}{}'.format(card.set, card.id, name, img_type, ext))
        return uri, file


# ========================================================================= #
# Dataset - IlsvrcImages                                                    #
# ========================================================================= #


class IlsvrcImages(util.LazyList):

    ILSVRC_SET_TYPES = ['val', 'test', 'train']

    def __init__(self):
        root = Dataset('ilsvrc').cd('2010')
        if not os.path.isdir(root.path):
            cprint("MAKE SURE YOU HAVE DOWNLOADED THE ILSVRC 2010 TEST DATASET TO: {}".format(root.path), 'yellow')
            cprint(" - The images must all be located within: {}".format(root.to('val')), 'yellow')
            cprint(" - For example: {}".format(root.to('val', 'ILSVRC2010_val_00000001.JPEG')), 'yellow')
            cprint("The image versions of the ILSVRC Datasets are for educational purposes only, and cannot be redistributed.", 'yellow')
            cprint("Please visit: www.image-net.org to obtain the download links.", 'yellow')
        super().__init__([util.Lazy(file, util.imread) for file in util.get_image_paths(root.path, prefixed=True)])


# ========================================================================= #
# Dataset - MTG Images                                                      #
# ========================================================================= #


class MtgImages(util.LazyList):

    def __init__(self, img_type='normal', predownload=False, handler=None):
        if handler is None:
            handler = MtgHandler()
        resources = handler.scryfall_cards_paths_uris(img_type=img_type) #, force=predownload)

        prox = Proxy('cache', default_threads=128, default_attempts=10, logger=tqdm.write)
        if predownload:
            download = [(u, f) for u, f in resources if not os.path.exists(f)]
            print('PREDOWNLOADING DOWNLOADING: {} of {}'.format(len(download), len(resources)))
            dirs = [util.init_dir(d) for d in { os.path.dirname(f)  for u, f in download } if not os.path.exists(d)]
            print('MADE DIRECTORIES: {}'.format(len(dirs)))
            prox.downloadThreaded(download)
            super().__init__([util.Lazy(file, util.imread) for uri, file in resources])
        else:
            super().__init__([util.LazyFile(uri, file, prox.download, util.imread) for uri, file in resources])

    _RAN_BG = ApplyShuffled(
        ApplyOrdered(Mutate.flip, Mutate.rotate_bounded, Mutate.warp_inv),
        ApplyChoice(Mutate.fade_black, Mutate.fade_white, None),
    )
    _RAN_FG = ApplyOrdered(
        ApplyShuffled(
            Mutate.warp,
            ApplyChoice(Mutate.fade_black, Mutate.fade_white, None),
        )
    )
    _RAN_VRTL = ApplyShuffled(
        ApplyChoice(Mutate.blur, None),
        ApplyChoice(Mutate.noise, None),
        ApplyChoice(Mutate.tint, None),
        ApplyChoice(Mutate.fade_black, Mutate.fade_white, None),
    )

    @staticmethod
    def make_cropped(path_or_img, size=None, half_upsidedown=False):
        card = util.imread(path_or_img) if (type(path_or_img) == str) else path_or_img
        ret = util.remove_border_resized(
            img=card,
            border_width=ceil(max(0.02 * card.shape[0], 0.02 * card.shape[1])),
            size=size
        )
        return ApplyChoice(Mutate.upsidedown, None)(ret) if half_upsidedown else ret

    @staticmethod
    def make_masked(path_or_img):
        card = util.imread(path_or_img) if (type(path_or_img) == str) else path_or_img
        ret = cv2.merge((
            card[:, :, 0],
            card[:, :, 1],
            card[:, :, 2],
            util.round_rect_mask(card.shape[:2], radius_ratio=0.05)
        ))
        return ret

    @staticmethod
    def make_bg(bg_path_or_img, size):
        bg = util.imread(bg_path_or_img) if (type(bg_path_or_img) == str) else bg_path_or_img
        bg = MtgImages._RAN_BG(bg)
        bg = util.crop_to_size(bg, size)
        return bg

    @staticmethod
    def make_virtual(card_path_or_img, bg_path_or_img, size, half_upsidedown=False):
        card = util.imread(card_path_or_img) if (type(card_path_or_img) == str) else card_path_or_img
        card = ApplyChoice(Mutate.upsidedown, None)(card) if half_upsidedown else card
        # fg - card
        fg = MtgImages.make_masked(card)
        fg = util.crop_to_size(fg, size, pad=True)
        fg = MtgImages._RAN_FG(fg)
        # bg
        bg = MtgImages.make_bg(bg_path_or_img, size)
        # merge
        virtual = util.rgba_over_rgb(fg, bg)
        virtual = MtgImages._RAN_VRTL(virtual)
        assert virtual.shape[:2] == size
        return virtual

    @staticmethod
    def make_virtual_pair(card_path_or_img, bg_path_or_img, x_size, y_size, half_upsidedown=False):
        card = util.imread(card_path_or_img) if (type(card_path_or_img) == str) else card_path_or_img
        # only inputs are flipped
        return MtgImages.make_virtual(card, bg_path_or_img, size=x_size, half_upsidedown=half_upsidedown), MtgImages.make_cropped(card, size=y_size)


# ========================================================================= #
# LOCAL MTG FILES                                                           #
# ========================================================================= #


# TODO: merge with above
class MtgLocalFiles(object):

    def __init__(self, img_type='small', x_size=(320, 224), y_size=(204, 144), pregen=False, force=False):
        self.ds_orig = Dataset('mtg').cd(img_type)
        self.ds_warp = Dataset('mtg_warp').cd(img_type, '{}_{}'.format(x_size[0], x_size[1]))
        self.ds_crop = Dataset('mtg_crop').cd(img_type, '{}_{}'.format(y_size[0], y_size[1]))

        self.x_size = x_size
        self.y_size = y_size

        self.paths = sorted(util.get_image_paths(self.ds_orig.path, prefixed=False))
        self.paths_warp = set(util.get_image_paths(self.ds_warp.path, prefixed=False))
        self.paths_crop = set(util.get_image_paths(self.ds_crop.path, prefixed=False))

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
            card = util.imread(self.ds_orig.to(rel))  # some cards are not the same size
            if random.random() < orig_ratio:
                x = MtgImages.make_cropped(card, self.x_size, half_upsidedown=half_upsidedown)
            else:
                x = MtgImages.make_virtual(card, self.ilsvrc.ran(), size=self.x_size, half_upsidedown=half_upsidedown)
            if save:
                util.imwrite(self.ds_warp.to(rel, init=True, is_file=True), x)
                self.paths_warp.add(rel)
            return x
        else:
            return util.imread(self.ds_warp.to(rel))

    def _gen_save_crop(self, rel, force=False, save=True):
        if force or not save or (rel not in self.paths_crop):
            card = util.imread(self.ds_orig.to(rel))  # some cards are not the same size
            y = MtgImages.make_cropped(card, size=self.y_size)
            if save:
                util.imwrite(self.ds_crop.to(rel, init=True, is_file=True), y)
                self.paths_crop.add(rel)
            return y
        else:
            return util.imread(self.ds_crop.to(rel))

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
    ilsvrc = orig or IlsvrcImages()

    while True:
        o = orig.ran()
        l = ilsvrc.ran()

        x, y = MtgImages.make_virtual_pair(o, l, (192, 128), (192, 128), True)

        util.imshow_loop(x, 'asdf')
        util.imshow_loop(y, 'asdf')