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

from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, Input, Reshape, Flatten, AvgPool2D, MaxPool2D, UpSampling2D, Conv2DTranspose
import keras as keras


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def begin_layer(f):
    def wrapper(*args, **kwargs):
        self = args[0]
        reset = False
        if not self._begun_layer:
            self._begun_layer = True
            self._j = 0
            reset = True
        f(*args, **kwargs)
        if reset:
            self._i += 1
            self._begun_layer = False
        return self
    return wrapper


def join(itr, sep='x'):
    return sep.join([str(i) for i in itr])

def p(string):
    return (string + '_') if string is not None else ''


# ========================================================================= #
# MODEL BUILDER                                                             #
# ========================================================================= #


class ModelBuilderTf:

    def __init__(self, xsize, ysize):
        self._model = keras.models.Sequential()
        self._xsize = xsize
        self._ysize = ysize
        # self._loss = loss
        # self._optimizer = optimizer
        self._i = 0
        self._j = 0
        self._begun_layer = False

    def add(self, layer):
        name = '{:02d}-{:02d}_{}'.format(self._i, self._j, layer.name)
        try:
            # not all layers are the same, some replace .name with properties that cannot be set.
            setattr(layer, 'name', name) # TODO: there must be a better way
        except:
            setattr(layer, '_name', name)
        self._model.add(layer)
        self._j += 1
        return self

    def __add__(self, layer):
        return self.add(layer)

    def __len__(self):
        return len(self._model.layers)

    def __getitem__(self, item):
        return self._model.layers[item]

    def last(self):
        return self[-1]

    def get_model(self):
        ret = self._model
        self._model = None
        return ret

    @begin_layer
    def conv_in(self, filters, kernel_size=(3, 3)):
        self + Conv2D(filters, kernel_size, padding='same', name='Conv2D_input_{}_{}'.format(filters, join(kernel_size)), input_shape=(self._xsize[0], self._xsize[1], 3))
        self + BatchNormalization(name='BatchNorm')
        self + LeakyReLU(0.01, name='LeakyReLU_0.01')

    @begin_layer
    def conv(self, filters, kernel_size=(3, 3), name=None, sigmoid=False):
        self + Conv2D(filters, kernel_size, padding='same', activation=None if not sigmoid else 'sigmoid', name='{}Conv2D_{}_{}'.format(p(name) if sigmoid else '', filters, join(kernel_size)))
        if not sigmoid:
            self + BatchNormalization(name='BatchNorm')
            self + LeakyReLU(0.01, name='{}LeakyReLU_0.01'.format(p(name)))

    @begin_layer
    def conv_out(self, kernel_size=(3, 3)):
        self + Conv2D(3, kernel_size, activation='sigmoid', padding='same', name='Output_Conv2D_{}_{}'.format(3, join(kernel_size)))

    @begin_layer
    def pool(self, pool_size=(2, 2), name=None):
        self + MaxPool2D(pool_size=pool_size, name='{}Encode_MaxPool_{}'.format(p(name), join(pool_size)))

    @begin_layer
    def grow(self, pool_size=(2, 2)):
        self + UpSampling2D(pool_size, interpolation='nearest', name='Decode_UpSample2D_{}'.format(join(pool_size)))

    @begin_layer
    def encode(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
        self.conv(filters, kernel_size)
        self.pool(pool_size)

    @begin_layer
    def decode(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
        self.grow(pool_size)
        self.conv(filters, kernel_size)

    @begin_layer
    def dense(self, size, name=None):
        from numpy import prod
        units = int(prod(size))
        self + Dense(units)
        self + LeakyReLU(0.01, name='{}Decode_LeakyReLU_0.01'.format(p(name)))
        if size != units:
            self.reshape(size)

    @begin_layer
    def dense_out(self, size):
        from numpy import prod
        units = int(prod(size))
        self + Dense(units, activation='sigmoid')
        if size != units:
            self.reshape(size)

    @begin_layer
    def flatten(self, name=None):
        self + Flatten(name='{}Flatten'.format(p(name)))

    @begin_layer
    def reshape(self, size, name=None):
        self + Reshape(size, name='{}Reshape_{}'.format(p(name), join(size)))