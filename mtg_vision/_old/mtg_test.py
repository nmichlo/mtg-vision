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
from tqdm import tqdm

from datasets import MtgLocalFiles
from mtg_detect import CardMatcher
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    # MTG DETECTOR

    with tf.Session() as sess:
        matcher = CardMatcher('weights.03-0.5411.hdf5')
        matcher.initialise()

        n = 5
        total = len(matcher.images.paths)
        w_top = np.zeros(n)

        for i in tqdm(range(len(matcher.images.paths)), desc='Testing Matcher'):
            warp, crop = matcher.images.gen_warp_crop(matcher.images.paths[i], save=False)
            dists, [wis] = matcher.match([warp], k=n)
            matched = False
            for j, wi in enumerate(wis):
                if wi == i:
                    matched = True
                    w_top[j:] += 1
            if not matched:
                print('Not Matched: {} [{}]'.format(i, matcher.images.paths[i]))
            if i % 100 == 0:
                print('[ACCURACY {}]:'.format(i), np.around(w_top / (i + 1) * 100, decimals=2))
            del warp, crop

        print('[ACCURACY ALL]:', w_top / total * 100)



