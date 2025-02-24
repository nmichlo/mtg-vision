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

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
import os
import time
import cv2
from mtg_vision._old import util
from mtg_vision._old.datasets import MtgLocalFiles
from mtg_vision._old.util import GLOBAL_RAN, JsonPickler
from mtg_vision._old.unused.viola_jones_classifiers import SampleImg, ClassifierCascade

# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #

if __name__ == "__main__":

    train = True
    detect = False
    show_all = False
    show_mean = False

    if train:

        train_x, train_y = MtgLocalFiles.gen_set_virtual_classed(128, (24, 24))

        for x, y in zip(train_x, train_y):
            util.imshow_loop(x, 'positive' if y == 1 else 'negative')

        pos = [SampleImg(x) for x, y in zip(train_x, train_y) if y == 1]
        neg = [SampleImg(x) for x, y in zip(train_x, train_y) if y == 0]
        del train_x
        del train_y

        #train
        GLOBAL_RAN.reset(777)
        cascade = ClassifierCascade()

        # cascade.train(pos, neg, min_dr=0.97, max_fp=0.65, target_fp=0.000001, num_ran=None, subset_size_ratios=(0.01, 0.001), num_best_features=100) #subset
        cascade.train(pos, neg, min_dr=0.99, max_fp=0.5, target_fp=0.000001, num_ran=5000, subset_size_ratios=None, num_best_features=None) #random
        # cascade.train(pos, neg, min_dr=0.95, max_fp=0.5, target_fp=0.001, num_ran=None, subset_size_ratios=None, num_best_features=None) #brute

        #save
        path = util.init_dir('model', 'cascade-{}.json'.format(int(time.time())), is_file=True)
        string = JsonPickler.dump(cascade, path)
        print("Saved classifier: {}".format(path))

        #test
        # test_pos = SampleImg.loadImageList(my_pos_list_1, cvt_to_sample=True, train_samples=True)
        # test_neg = SampleImg.loadImageList(my_neg_list_1, cvt_to_sample=True, train_samples=True)
        # cascade.test(test_pos, test_neg)
        # test_pos = SampleImg.loadImageList(my_pos_list_1, cvt_to_sample=True, train_samples=False)
        # test_neg = SampleImg.loadImageList(my_neg_list_1, cvt_to_sample=True, train_samples=False)
        # cascade.test(test_pos, test_neg)

    if detect:

        #load
        path = os.path.join(dir, sorted([name for name in os.listdir("./cascades/") if name.strip().endswith(".json")], reverse=True)[0] if load_cascade is None else load_cascade)
        cascade = JsonPickler.load(path)
        print("Loaded classifier: {}".format(path))

        #visualise
        capture = cv2.VideoCapture(0)
        while(True):
            img = cv2.flip(capture.read()[1], 1)

            if show_all:
                for bound in cascade.detect(img, scale_interval=0.75, step=0.1, min_size=300):
                    x0, y0, x1, y1 = bound
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0))

            if show_mean:
                for bound in cascade.detectAverage(img, scale_interval=0.75, step=0.1, min_size=300, dist_thresh=1):
                    print("Detected")
                    x0, y0, x1, y1 = bound
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255))

            cv2.imshow("cam", img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
