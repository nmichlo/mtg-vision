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

import os
import time
import traceback
import tensorflow as tf
from keras.models import load_model
from tqdm import tqdm
import util
from datasets import DATASETS_ROOT, MtgLocalFiles, MtgHandler, MtgImages
import numpy as np
import cv2 as cv2
from sklearn.neighbors import NearestNeighbors


# ========================================================================= #
# Parallel                                                                  #
# ========================================================================= #
from model_from_scratch import Model


class VideoStream(util.ThreadLoop):
    def __init__(self, mutator=None, src=0):
        super().__init__()
        self._mutator = mutator
        self._stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self._stream.read()
        self.grabbed, self.frame = False, np.zeros((1, 1))

    def update(self):
        (grabbed, frame) = self._stream.read()
        if callable(self._mutator):
            frame = self._mutator(frame)
        self.grabbed, self.frame = grabbed, frame


class TfSessionThreadLoop(util.ThreadLoop):
    def __init__(self, update):
        super().__init__(update=update)

    def loop(self):
        try:
            with tf.Session():
                super().loop()
        except Exception as e:
            traceback.print_exc()
            print()


# ========================================================================= #
# CARD BLOB DETECTOR                                                        #
# ========================================================================= #


class CardDetector:

    def __init__(self, scale=4, card_size=(310, 223), expand_ratio=0.075):
        self.scale = scale
        self.scale2 = self.scale ** 2
        self.card_h, self.card_w = card_size
        self.card_area = self.card_w * self.card_h / 4
        self.expand_ratio = expand_ratio

    def detect(self, image):
        scaled = cv2.resize(image, None, fx=(1.0 / self.scale), fy=(1.0 / self.scale), interpolation=cv2.INTER_AREA)
        blur = cv2.bilateralFilter(scaled, 5, 17, 17)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

        (c, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(c, key=lambda contour: -cv2.contourArea(contour))

        cards = []
        for contour in contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if (len(approx) == 4) and (cv2.contourArea(approx) > self.card_area / self.scale2):
                if util.cv2_poly_is_convex(approx):
                    approx = util.cv2_quad_flip_upright(approx * self.scale)
                    approx = util.cv2_poly_exapand(approx, ratio=self.expand_ratio)
                    cards.append(approx)
        cards = sorted(cards, key=lambda card: util.cv2_poly_center(card)[0])

        # TODO: fix, this returns an array of tuples contained in single arrays
        return cards

    def extract(self, image, approx_bounds):
        cards = []
        for i, bound in enumerate(approx_bounds):
            if(bound is None) or (len(bound) != 4):
                return np.zeros([self.card_h, self.card_w], np.int8)

            # should match: tl, bl, br, tr
            w, h = (self.card_w, self.card_h)
            dst_pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
            src_pts = np.float32(bound).reshape((-1, 2))

            transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
            card = cv2.warpPerspective(image, transform, (w, h))

            cards.append(card)
        return cards


# ========================================================================= #
# CARD MATCHER                                                              #
# ========================================================================= #


class CardMatcher(object):

    def __init__(self, model_file):
        self._init = False
        self._null = True

        self.use_custom_model = False

        self._model_file = model_file

    def initialise(self, print_summary=False):
        if self.init:
            raise Exception('Already Initialised')

        print('Init Matcher: START')

        model_file = os.path.join('./trained', self._model_file)
        cache_file = os.path.splitext(model_file)[0] + '.small.vectors.np'

        print('Init Matcher: Loading Model')
        self.model = load_model(model_file)
        if print_summary:
            self.model.summary()

        print('Init Matcher: Generating Custom Model from Model')
        self.model_custom = Model.from_keras(self.model)

        print('Init Matcher: Loading Layers')
        in_lyr = self.model.layers[0]
        enc_lyr = next(l for l in self.model.layers if 'encoded' in l.name)
        out_lyr = self.model.layers[-1]

        self.encoder = tf.keras.backend.function([in_lyr.input], [enc_lyr.output])

        self.x_size, self.y_size = in_lyr.input_shape[1:3], out_lyr.output_shape[1:3]
        self.enc_n = enc_lyr.output_shape[1] * enc_lyr.output_shape[2] * enc_lyr.output_shape[3] # why cant i use sum() here?

        print('Init Matcher: Loading Mtg Handler')
        self.mtgdb = MtgHandler()
        MtgImages(img_type='small', predownload=True, handler=self.mtgdb)

        print('Init Matcher: Loading Images')
        self.images = MtgLocalFiles(img_type='small', x_size=self.x_size, y_size=self.x_size)

        def regen():
            print('Init Matcher: Loading Images Keys')
            keys = []
            for i in tqdm(range(len(self.images)), desc='Init Matcher: Generating Keys'):
                try:
                    keys.append(self.encode(self.images[i]))
                except Exception as e:
                    traceback.print_exc()
                    print()
            with open(util.init_dir(cache_file, is_file=True), 'wb') as file:
                file.write(np.array(keys, dtype=np.float32).tobytes())

        try:
            if not os.path.exists(cache_file):
                regen()
            if os.path.exists(cache_file):
                print('Init Matcher: Loaded Images Keys')
                with open(util.init_dir(cache_file, is_file=True), 'rb') as file:
                    try:
                        self.keys = np.frombuffer(file.read(), dtype=np.float32).reshape((len(self.images), self.enc_n))
                        print('LOADED KEYS: {}'.format(len(self.keys)))
                    except Exception as e:
                        traceback.print_exc()
                        print('Failed to load keys from: {} REGENERATIING!'.format(cache_file))
                        regen()
        except Exception as e:
            traceback.print_exc()
            print()
            raise e

        print('Init Matcher: DONE')

        # DONE INITIALISING
        self._init = True
        self._null = False

    @property
    def init(self):
        return self._init

    @property
    def null(self):
        return self._null

    def _perpare_feedforward_input(self, img):
        if img.shape[:2] != self.x_size:
            print('resized')
            img = cv2.resize(img, self.x_size)
        img = util.fxx(img)
        return img

    def feedforward(self, img):
        img = self._perpare_feedforward_input(img)
        if self.use_custom_model:
            out = self.model_custom.forward(np.array([img], dtype=np.float32))[0]
        else:
            out = self.model.predict(np.array([img], dtype=np.float32))[0]
        return out

    def feedforward_all(self, imgs):
        return [self.feedforward(img) for img in imgs]

    def encode(self, img):
        img = self._perpare_feedforward_input(img)
        if self.use_custom_model:
            out = self.model_custom.forward(np.array([img], dtype=np.float32), stop_at_name='encoded')[0].reshape((-1))
        else:
            out = self.encoder(np.array([img], dtype=np.float32))[0].reshape((-1))
        return out

    def encode_all(self, imgs):
        return [self.encode(img) for img in imgs]

    def match(self, imgs, k):
        if len(imgs) < 1:
            return [], []
        encoded = self.encode_all(imgs)
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        dists, indices = knn.fit(self.keys).kneighbors(encoded, )
        return dists, indices

    def matched_images(self, knn_indices):
        return [[self.images[i] for i in c] if hasattr(c, '__iter__') else self.images[c] for c in knn_indices]

    def matched_info(self, knn_indices):
        def get(i):
            uuid = self.images.get_uuid(i)
            card = self.mtgdb.scryfall_card_from_uuid(uuid)
            return card
        return [[get(i) for i in c] if hasattr(c, '__iter__') else get(c) for c in knn_indices]


# ========================================================================= #
# VIDEO STREAMS                                                             #
# ========================================================================= #


class Streams:

    def __init__(self, video_scale=0.7):
        self.init = False
        self.video_scale = video_scale

        self.card_bounds = []
        self.card_images = []
        self.match_knn = []
        self.match_knn_all = []
        self.match_knn_dists = []
        self.match_knn_dists_all = []
        self.match_images = []
        self.match_pred = []
        self.match_info = []
        self.match_info_all = []
        self.match_steps_completed = 1

        self.detector = None
        self.matcher = None

        self.video_stream = None
        self.detect_stream = None
        self.match_stream = None

    def global_mutate(self, img):
        return cv2.resize(img, (0, 0), fx=self.video_scale, fy=self.video_scale, interpolation=cv2.INTER_AREA)

    def detect_cards(self):
        if self.video_stream.grabbed:
            img = self.video_stream.frame
            self.card_bounds = self.detector.detect(img)
            self.card_images = self.detector.extract(img, self.card_bounds)

    def match_cards(self):
        if self.matcher.null:
            self.matcher.initialise()
        if self.video_stream.grabbed:
            try:
                if len(self.card_images) > 0:
                    # KNN
                    self.match_knn_dists_all, self.match_knn_all = self.matcher.match(self.card_images, k=3)
                    self.match_knn_dists, self.match_knn = [a[0] for a in self.match_knn_dists_all], [a[0] for a in self.match_knn_all]
                    # MTGTOOLS INFO
                    self.match_info_all = self.matcher.matched_info(self.match_knn_all)
                    self.match_info = [a[0] for a in self.match_info_all]
                    # CARDS
                    self.match_images = self.matcher.matched_images(self.match_knn)
                    # NN FEEDFORWARD
                    self.match_pred = self.matcher.feedforward_all(self.card_images)
                    # Increment
                    self.match_steps_completed += 1
            except Exception as e:
                traceback.print_exc()
                print()

    def start(self, model_name, x_size):
        if self.init:
            raise Exception('Already Initialised')

        self.detector = CardDetector(4, card_size=x_size, expand_ratio=0.1)
        self.matcher = CardMatcher(model_name)
        self.video_stream = VideoStream(self.global_mutate)
        self.detect_stream = util.ThreadLoop(self.detect_cards)
        self.match_stream = TfSessionThreadLoop(self.match_cards)
        self.video_stream.start()
        self.detect_stream.start()
        self.match_stream.start()
        # DONE
        self.init = True
        return self

    def stop(self):
        self.video_stream.stop()
        self.detect_stream.stop()
        self.match_stream.stop()
        # DONE
        self.init = False
        return self


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


MODEL_NAME = 'weights.03-0.5411.hdf5'

def main():
    # VARS
    x_size, y_size = (192, 128), (192 // 2, 128 // 2)

    S = Streams().start(
        model_name=os.getenv('MODEL', MODEL_NAME),
        x_size=x_size
    )

    DIST_THRESH = 0.03

    # Matcher Loading Bar
    match_steps_completed, match_time, last_time = S.match_steps_completed, 5, 0
    def update_loading_bar():
        nonlocal match_steps_completed, match_time, last_time
        if S.matcher.init:
            if match_steps_completed != S.match_steps_completed:
                match_steps_completed = S.match_steps_completed
                t = time.time()
                match_time = 0.6 * match_time + 0.4 * (t - last_time)
                last_time = t
            else:
                t, (h, w) = time.time(), img.shape[:2]
                match_ratio = np.clip((t - last_time) / match_time, 0, 1)
                cv2.line(img, (0, h - 1), (int(w * match_ratio), h - 1), clr_c, 2)
                text(img, (5, h-10), 'Matching: {:3d}% {:5d}ms'.format(int(match_ratio*100), int(match_time*1000)), color=clr_c)
        else:
            h, w = img.shape[:2]
            text(img, (5, h-10), 'Initialising Matcher...', color=clr_r)
            last_time = time.time()

    # HELPER
    clr_r, clr_g, clr_b = (0, 0, 255), (0, 255, 0), (255, 0, 0)
    clr_y, clr_m, clr_c = (0, 255, 255), (255, 0, 255), (255, 255, 0)
    clr_black, clr_white = (0, 0, 0), (255, 255, 255)

    def text(img, pos, *text, color=clr_g):
        string = ' '.join((str(t) for t in text))
        cv2.putText(img, string, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=color, lineType=2)

    # MAIN THREAD LOOP
    VIEW = 1
    while True:
        try:
            img = S.video_stream.frame.copy()

            # LOADING BAR
            update_loading_bar()
            mode, color = 'Raw Feed', clr_r if S.matcher.null else clr_b
            inner_bounds = [util.cv2_poly_exapand(b.copy(), -0.125) for b in S.card_bounds]

            # Modes
            if VIEW == 0:
                mode = 'Bounding Boxes'
            elif VIEW == 1:
                img = util.cv2_warp_imgs_onto(img, S.card_images, S.card_bounds)
                mode = 'Detected Cards'
                util.cv2_draw_contours(img, inner_bounds, color=clr_b)
            elif VIEW == 2:
                img = util.cv2_warp_imgs_onto(img, S.match_pred, inner_bounds)
                mode = 'Representations'
            elif VIEW == 3:
                img = util.cv2_warp_imgs_onto(img, S.match_images, inner_bounds)
                mode = 'Matched Cards'

            # draw card bounds
            for i, contour in enumerate(S.card_bounds):
                c = clr_b
                if (i < len(S.match_knn_dists)) and (S.match_knn_dists[i] >= DIST_THRESH):
                    c = clr_r
                cv2.drawContours(img, [contour], -1, c, 1)

            # draw card bounds node indices
            for i, bound in enumerate(S.card_bounds):
                for j, b in enumerate(bound.reshape((-1, 2))):
                    text(img, tuple(b - (7, 0)), str(i)+':'+str(j), color=clr_black)

            # draw card names
            for i, (bound, card) in enumerate(zip(S.card_bounds, S.match_info)):
                text(img, tuple(bound[1][0] + (0, 15)), card, color=clr_m)
                text(img, tuple(bound[1][0] + (0, 30)), card.type_line, color=clr_m)
                text(img, tuple(bound[1][0] + (0, 45)), card.mana_cost, color=clr_m)

            # draw card match distances
            for i, (bound, dist) in enumerate(zip(S.card_bounds, S.match_knn_dists)):
                text(img, tuple(bound[1][0] + (0, 60)), 'Dist: {:.4f} {}'.format(dist, '' if dist < 0.03 else 'Inverted?'), color=clr_y)

            # Info
            db_size = -1 if S.matcher.null else len(S.matcher.images)
            text(img, (15, 15), 'Database Size:', db_size, color=color)
            text(img, (15, 30), 'Mode:', mode, color=color)
            text(img, (15, 45), 'FeedForward:', 'Custom' if S.matcher.use_custom_model else 'Tensorflow', color=color)

            # Show
            window_name = 'camera'
            cv2.imshow(window_name, img)

            # Wait
            key = cv2.waitKey(1) & 0xFF

            if (key == 27) or key == ord('q') or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
                cv2.destroyAllWindows()
                S.stop()
                break
            elif (key == ord(' ')):
                VIEW = (VIEW + 1) % 4
            elif (key == ord('0')):
                VIEW = 0
            elif (key == ord('1')):
                VIEW = 1
            elif (key == ord('2')):
                VIEW = 2
            elif (key == ord('3')):
                VIEW = 3
            elif (key == ord('=')):
                S.matcher.use_custom_model = False
            elif (key == ord('-')):
                S.matcher.use_custom_model = True

        except Exception as e:
            traceback.print_exc()
            print()


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


if __name__ == "__main__":
    main()
