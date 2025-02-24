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

import inspect
import os

from keras.models import load_model
import keras
import numpy as np
from tqdm import tqdm
import util
from datasets import MtgImages, IlsvrcImages
from numba import jit
from termcolor import cprint


# ========================================================================= #
# LAYERS                                                                    #
# ========================================================================= #


@jit('float32[:,:,:,:](float32[:,:,:,:], float32[:,:,:,:], float32[:], UniTuple(uint8, 2), UniTuple(uint8, 2))')
def forward_Conv2D(data: np.ndarray, kernel: np.ndarray, bias: np.ndarray, strides=(1, 1), dilation_rate=(1, 1)):
    (images, i_height, i_width, i_filters) = data.shape
    (k_height, k_width, k_prev_filters, k_filters) = kernel.shape
    (b_filters,) = bias.shape

    assert strides[0] == 1 and strides[1] == 1, 'Only strides of (1, 1) are supported!'
    assert dilation_rate[0] == 1 and dilation_rate[1] == 1, 'Only dilation rates of (1, 1) are supported!'
    assert i_filters == k_prev_filters, 'Input and Kernel length mismatch'
    assert k_filters == b_filters, 'Bias and Kernel length mismatch'
    assert k_height == k_width, 'Kernels must best square'
    assert k_height % 2 == 1 and k_width % 2 == 1, 'Kernels must be odd in size'

    pad = (k_height - 1) // 2
    padded = np.zeros((images, i_height + 2*pad, i_width + 2*pad, i_filters), dtype=np.float32)
    padded[:, pad:-pad, pad:-pad, :] = data

    out = np.zeros((images, i_height, i_width, k_filters), dtype=np.float32)

    for i in range(images):                                   # data   | out in  X
        # SINGLE IMAGE
        for f in range(k_filters):                            # kernel | out X   bias
            for y in range(i_height):                         # data   | out X   X
                for x in range(i_width):                      # data   | out X   X
                    # CONVOLVE
                    sum = 0.0
                    for kx in range(k_height):                # kernel | X   in  X
                        for ky in range(k_width):             # kernel | X   in  X
                            for kf in range(k_prev_filters):  # data   | X   in  X
                                sum += kernel[ky,kx,kf,f] * padded[i, y+ky, x+kx, kf]
                    out[i, y, x, f] = sum + bias[f]
    return out


@jit('float32[:,:,:,:](float32[:,:,:,:], UniTuple(uint8, 2), UniTuple(uint8, 2))')
def forward_MaxPool2D(data: np.ndarray, pool_size: tuple, strides=(2, 2)) -> np.ndarray:
        n, H, W, f = data.shape
        ph, pw = pool_size

        assert ph == 2 and pw == 2, "Pooling Size Must be (2, 2)"
        assert H % ph == 0 and W % pw == 0, 'Size of layer must be divisible by pool size'
        assert strides[0] == 2 and strides[1] == 2, "Stride size must match pool size (2, 2)"

        if (H % ph != 0) or (W % pw != 0):
            raise Exception("Dims not divisible")

        h = H // ph
        w = W // pw

        pool = np.zeros((n, h, w, f), dtype=np.float32)

        for i in range(n):
            for l in range(f):
                for j in range(h):
                    for k in range(w):
                        a = data[i, j*ph:(j+1)*ph, k*pw:(k+1)*pw, l]
                        pool[i, j, k, l] = a.max()

        return pool


@jit('float32[:,:,:,:](float32[:,:,:,:], float32[:], float32[:], float32[:], float32[:])')
def forward_BatchNormalization(data, gamma, beta, moving_mean, moving_variance):

    (images, i_height, i_width, i_filters) = data.shape

    assert len(gamma) == len(beta) and len(beta) == len(moving_mean) and len(moving_mean) == len(moving_variance), "BatchNorm Variable length mismatch"
    assert len(gamma) == i_filters, "BatchNorm and Data length mismatch"

    norm = (data - moving_mean) / np.sqrt(moving_variance)
    out = gamma * norm + beta
    return out


@jit('float32[:,:,:,:](float32[:,:,:,:], UniTuple(uint8, 2), string)')
def forward_UpSampling2D(data: np.ndarray, size: tuple, interpolation='nearest') -> np.ndarray:
        n, H, W, f = data.shape
        ph, pw = size

        assert ph == 2 and pw == 2, "Pooling Size Must be 2x2"
        assert interpolation == 'nearest', 'Only Nearest Interpolation Supported'

        if (H % ph != 0) or (W % pw != 0):
            raise Exception("Dims not divisible")

        h = H * ph
        w = W * pw

        pool = np.zeros((n, h, w, f), dtype=np.float32)

        for i in range(n):
            for l in range(f):
                for j in range(h):
                    for k in range(w):
                        pool[i, j*ph:(j+1)*ph, k*pw:(k+1)*pw, l] = data[i, j, k, l]

        return pool


@jit('float32[:,:,:,:](float32[:,:,:,:], float32)')
def forward_LeakyReLU(data: np.ndarray, alpha=0.01):
    # return np.where(data > 0, data, data * 0.01)
    out = np.copy(data)
    (images, i_height, i_width, i_filters) = data.shape
    for i in range(images):
        for y in range(i_height):
            for x in range(i_width):
                for f in range(i_filters):
                    if out[i,y,x,f] < 0:
                        out[i,y,x,f] *= alpha
    return out


@jit('float32[:,:,:,:](float32[:,:,:,:])')
def forward_Sigmoid(data: np.ndarray):
    return 1.0 / (1.0 + np.exp(-data))


# ========================================================================= #
# LAYER WRAPPER                                                             #
# ========================================================================= #


class Layer(object):

    def __init__(self, function, weights, weight_names, input_shape, output_shape, config):
        self.function = function
        self.weights = weights
        self.weight_names = weight_names

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.config = config

        self.name = config['name']
        print(self.name, config)
        self.activation = config['activation'] if 'activation' in config else 'linear'

        assert self.activation in ['linear', 'sigmoid'], 'Activation unsupported: {}'.format(self.activation)

        self.func_arg_names_all = inspect.getfullargspec(function.py_func).args
        self.func_arg_names = [arg for arg in self.func_arg_names_all if arg in config]
        self.args = [config[arg] for arg in self.func_arg_names]

        assert len(self.weight_names) == len(self.weights)

    def forward(self, data: np.ndarray):
        return self.function(data, *self.weights, *self.args)

    def __str__(self):
        return "{:30s} | {} -- {}".format(
            self.name,
            ' '.join(["{}: {}".format(n, w.shape) for w, n in zip(self.weights, self.weight_names)]),
            ' '.join(["{}: {}".format(n, v) for n, v in zip(self.func_arg_names, self.args)])
        )

    def __repr__(self):
        return str(self)


# ========================================================================= #
# MODEL                                                                     #
# ========================================================================= #


class Model(object):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, data: np.ndarray, stop_at_name=None) -> np.ndarray:
        for l in tqdm(self.layers, 'FeedForward'):
            tqdm.write('Feeding Through Layer: {}'.format(l))
            data = l.forward(data)
            # ACTIVATE
            if l.activation == 'linear':
                pass
            elif l.activation == 'sigmoid':
                data = forward_Sigmoid(data)
            else:
                raise Exception('Unsupported Activation')
            # STOP
            if stop_at_name is not None and stop_at_name in l.name.lower():
                break
        return data

    KERAS_TO_LAYER = {
        keras.layers.BatchNormalization: forward_BatchNormalization,
        keras.layers.Conv2D: forward_Conv2D,
        keras.layers.MaxPool2D: forward_MaxPool2D,
        keras.layers.UpSampling2D: forward_UpSampling2D,
        keras.layers.LeakyReLU: forward_LeakyReLU,
    }

    @classmethod
    def from_keras(cls, model: any=None) -> 'Model':
        if type(model) == str:
            if os.path.exists(model):
                model = load_model(model, compile=False)
            else:
                raise Exception('Model file does not exist: {}'.format(model))

        layers = []
        for l, w in zip(model.layers, model.get_weights()):
            if type(l) not in Model.KERAS_TO_LAYER:
                raise Exception('Layer Type Not Supported: {}'.format(type(l)))
            layers.append(Layer(
                function=Model.KERAS_TO_LAYER[type(l)],
                weights=l.get_weights(),                                     # weights = trainable_weights + non_trainable_weights
                weight_names=[w.name[len(l.name)+1:-2] for w in l.weights],  # name = <layer_name>/<name>:0
                input_shape=l.input_shape,
                output_shape=l.output_shape,
                config=l.get_config(),
            ))

        return Model(layers=layers)

    def __str__(self):
        return "\n".join([str(l) for l in self.layers])

    def __repr__(self):
        return str(self)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == "__main__":

    cprint('[Converting Model]', 'yellow')
    model = Model.from_keras('./trained/weights.03-0.5411.hdf5')

    orig = MtgImages(img_type='normal')
    ilsvrc = orig or IlsvrcImages()

    while True:
        o = orig.ran()
        l = ilsvrc.ran()

        x, y = MtgImages.make_virtual_pair(o, l, (192, 128), (192//2, 128//2), True)

        util.imshow_loop(x, 'asdf')
        util.imshow_loop(y, 'asdf')
        out = model.forward(np.array([x], dtype=np.float32))[0]
        util.imshow_loop(out, 'asdf')

