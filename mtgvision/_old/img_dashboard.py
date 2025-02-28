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

import itertools
import os
import uuid
from multiprocessing import Process
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import random
from datasets import MtgImages, MtgLocalFiles
from util import image2base64, safe_imread, get_image_paths, imwrite, tqdm
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from sklearn.neighbors import NearestNeighbors, VALID_METRICS


# ========================================================================= #
# CONSTS                                                                    #
# ========================================================================= #

DASHBOARD_DIR = './cache/dashboard'

# ========================================================================= #
# IMAGE DASH - PLOTLY DASH                                                  #
# ========================================================================= #

# start server by running this file
class ImageDashboard:

    def __init__(self, root_dir='./cache/dashboard', row_height=128, refresh_ms=1000):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        self.row_height = row_height

        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, 'input')
        self.output_dir = os.path.join(self.root_dir, 'output')
        self.orig_dir = os.path.join(self.root_dir, 'orig')
        self.match_dir = os.path.join(self.root_dir, 'match')

        if not os.path.isdir(self.input_dir):
            os.makedirs(self.input_dir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.orig_dir):
            os.makedirs(self.orig_dir)
        if not os.path.isdir(self.match_dir):
            os.makedirs(self.match_dir)

        self.refresh_ms = refresh_ms
        self.app = None

    def remove_images(self):
        for path in get_image_paths(self.input_dir, prefixed=True):
            os.remove(path)
        for path in get_image_paths(self.output_dir, prefixed=True):
            os.remove(path)
        for path in get_image_paths(self.orig_dir, prefixed=True):
            os.remove(path)
        for path in get_image_paths(self.match_dir, prefixed=True):
            os.remove(path)

    def save_image_pair(self, x, y, z, m=None, i=0):
        name = '{}_{}.jpg'.format(i, uuid.uuid4())
        imwrite(os.path.join(self.input_dir, name), x)
        imwrite(os.path.join(self.output_dir, name), y)
        imwrite(os.path.join(self.orig_dir, name), z)
        if m is not None:
            imwrite(os.path.join(self.match_dir, name), m)

    def save_image_pairs(self, xs, ys, zs, ms=None):
        for i, t in enumerate(zip(xs, ys, zs, ms or itertools.repeat(None, len(xs)))):
            self.save_image_pair(*t, i)

    def run(self, debug=False, blocking=False):
        self.app = dash.Dash('image-dashboard')
        self.app.layout = html.Div([
            html.Div(id='image-list'),
            dcc.Interval(
                id='interval-component',
                interval=self.refresh_ms,  # in milliseconds
            )
        ])

        @self.app.callback(Output('image-list', 'children'), [Input('interval-component', 'n_intervals')])
        def update_metrics(n):
            img_style = {'width': 'auto', 'height': '{}px'.format(self.row_height)}
            # dirs
            inputs = get_image_paths(self.input_dir, prefixed=True)
            outputs = get_image_paths(self.output_dir, prefixed=True)
            origs = get_image_paths(self.orig_dir, prefixed=True)
            matches = get_image_paths(self.match_dir, prefixed=True)
            # display
            elem = [
                html.Div([
                    html.Img(src="data:image/png;base64,{}".format(image2base64(safe_imread(path), 'png').decode()), style=img_style)
                    for path in paths
                ])
                for paths in (zip(inputs, outputs, origs, matches) if len(matches) > 0 else zip(inputs, outputs, origs))
            ] if (len(inputs) > 0 and len(outputs) > 0) else html.Plaintext('No Images Found')
            # return
            return elem

        if not blocking:
            process = Process(target=self.app.run_server, kwargs={'debug': debug})
            process.start()  # Will run "foo"
            return process
        else:
            self.app.run_server(debug=debug)
            return self


# ========================================================================= #
# IMAGE DASH CALLBACK                                                        #
# ========================================================================= #


class DashboardCallback(tf.keras.callbacks.Callback):

    def __init__(self, x_test, y_test, img_dash=ImageDashboard()):
        super().__init__()
        self.img_dash = img_dash
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        n = 15
        predictions = self.model.predict(self.x_test[:n])
        self.img_dash.remove_images()
        self.img_dash.save_image_pairs(self.x_test[:n], predictions, self.y_test[:n])


# ========================================================================= #
# TENSORBOARD IMAGE CALLBACK                                                #
# ========================================================================= #


class TensorBoardOutputImages(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, x_orig, log_dir='./cache/tensorboard/'):
        super().__init__()
        self.log_dir = log_dir
        self.x_test = x_test
        self.y_test = y_test
        self.x_orig = x_orig

        summaries_x = tf.Summary(value=[tf.Summary.Value(tag='img_{:02d}/x'.format(i), image=self.make_image(x)) for i, x in enumerate(self.x_test)])
        summaries_y = tf.Summary(value=[tf.Summary.Value(tag='img_{:02d}/y'.format(i), image=self.make_image(y)) for i, y in enumerate(self.y_test)])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summaries_x, 0)
        writer.add_summary(summaries_y, 0)
        writer.close()

    def on_epoch_end(self, epoch, logs={}):
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(self.make_images_summary(self.model.predict(self.x_test), 'img', 'p'), epoch)
        writer.add_summary(self.make_images_summary(self.model.predict(self.x_orig), 'img', 'o'), epoch)
        writer.close()

    @staticmethod
    def make_images_summary(img_iter, folder='img', name='p'):
        return tf.Summary(value=[
            tf.Summary.Value(tag='{}_{:02d}/{}'.format(folder, i, name), image=TensorBoardOutputImages.make_image(p))
            for i, p in enumerate(img_iter)
        ])

    @staticmethod
    def make_image(tensor):
        # TODO: this can be replaced by newer methods.
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape
        if tensor.dtype in [np.float16, np.float32, np.float64]:
            tensor = (255 * tensor).astype(np.uint8)
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)


# ========================================================================= #
# TENSORBOARD MATCH CALLBACK                                                #
# ========================================================================= #


class TensorBoardMatch(tf.keras.callbacks.Callback):
    def __init__(self, encoding_layer, n, x_size, y_size, N=9999999, log_dir='./cache/tensorboard/'):
        super().__init__()
        self.log_dir = log_dir
        self.encoding_layer = encoding_layer
        self.n = n
        self.N = N
        self.x_size = x_size
        self.y_size = y_size

        self.indices = None
        self.count = 0

    def on_train_end(self, epoch, logs={}):
        print('Matching Images:')
        print(epoch)
        print(logs)

        writer = tf.summary.FileWriter(self.log_dir)

        encoder = tf.keras.backend.function([self.model.layers[0].input], [self.encoding_layer])
        images = MtgLocalFiles(img_type='small', x_size=self.x_size, y_size=self.x_size)

        if self.indices is None:
            i = list(range(self.N))
            random.shuffle(i)
            self.indices = i[:self.n]

            actual_imgs = [images[i] for i in self.indices]
            writer.add_summary(TensorBoardOutputImages.make_images_summary(actual_imgs, 'match', 'a'), self.count)

        x_imgs = [images._gen_save_warp(images.paths[i]) for i in self.indices]
        x_keys = [encoder(np.array([i]))[0].reshape((-1)) for i in tqdm(x_imgs)]
        all_keys = [encoder(np.array([images[a]]))[0].reshape((-1)) for a in tqdm(range(min(self.N, len(images))), desc='Generating Image Keys')]

        knn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(all_keys)
        dists, indices = knn.kneighbors(x_keys)
        match_imgs = [images[i[0]] for i in indices]

        # writer.add_summary(TensorBoardOutputImages.make_images_summary(x_imgs, 'match', 'x'), epoch)
        writer.add_summary(TensorBoardOutputImages.make_images_summary(match_imgs, 'match', 'm'), self.count)
        writer.close()

        self.count += 1


# ========================================================================= #
# IMAGE DASH - PLOTLY DASH - ENTRYPOINT                                     #
# ========================================================================= #


if __name__ == "__main__":
    img_dash = ImageDashboard()
    img_dash.run(debug=False, blocking=True)
