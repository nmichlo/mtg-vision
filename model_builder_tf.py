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

from tensorflow.keras import layers, models
from functools import partial
import tensorflow as tf

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
        with tf.name_scope('stage_{}'.format(self._i)):
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




conv1x1 = partial(layers.Conv2D, kernel_size=1, activation='relu')
conv3x3 = partial(layers.Conv2D, kernel_size=3, padding='same', activation='relu')
conv1x3 = partial(layers.Conv2D, kernel_size=(1, 3), padding='same', activation='relu')
conv3x1 = partial(layers.Conv2D, kernel_size=(3, 1), padding='same', activation='relu')
conv1x5 = partial(layers.Conv2D, kernel_size=(1, 5), padding='same', activation='relu')
conv5x1 = partial(layers.Conv2D, kernel_size=(5, 1), padding='same', activation='relu')
conv5x5 = partial(layers.Conv2D, kernel_size=5, padding='same', activation='relu')


# ========================================================================= #
# MODEL BUILDER                                                             #
# ========================================================================= #


# class ModelBuilderTf:
#
#     def __init__(self, xsize, ysize):
#         self._model = keras.models.Sequential()
#         self._xsize = xsize
#         self._ysize = ysize
#         # self._loss = loss
#         # self._optimizer = optimizer
#         self._i = 0
#         self._j = 0
#         self._begun_layer = False
#
#     def add(self, layer):
#         name = '{:02d}-{:02d}_{}'.format(self._i, self._j, layer.name)
#         try:
#             # not all layers are the same, some replace .name with properties that cannot be set.
#             setattr(layer, 'name', name) # TODO: there must be a better way
#         except:
#             setattr(layer, '_name', name)
#         self._model.add(layer)
#         self._j += 1
#         return self
#
#     def __add__(self, layer):
#         return self.add(layer)
#
#     def __len__(self):
#         return len(self._model.layers)
#
#     def __getitem__(self, item):
#         return self._model.layers[item]
#
#     def last(self):
#         return self[-1]
#
#     def get_model(self):
#         ret = self._model
#         self._model = None
#         return ret
#
#     @begin_layer
#     def conv_in(self, filters, kernel_size=(3, 3)):
#         self + Conv2D(filters, kernel_size, padding='same', name='Conv2D_input_{}_{}'.format(filters, join(kernel_size)), input_shape=(self._xsize[0], self._xsize[1], 3))
#         self + BatchNormalization(name='BatchNorm')
#         self + LeakyReLU(0.01, name='LeakyReLU_0.01')
#
#     @begin_layer
#     def conv(self, filters, kernel_size=(3, 3), name=None, sigmoid=False):
#         self + Conv2D(filters, kernel_size, padding='same', activation=None if not sigmoid else 'sigmoid', name='{}Conv2D_{}_{}'.format(p(name) if sigmoid else '', filters, join(kernel_size)))
#         if not sigmoid:
#             self + BatchNormalization(name='BatchNorm')
#             self + LeakyReLU(0.01, name='{}LeakyReLU_0.01'.format(p(name)))
#
#     @begin_layer
#     def conv_out(self, kernel_size=(3, 3)):
#         self + Conv2D(3, kernel_size, activation='sigmoid', padding='same', name='Output_Conv2D_{}_{}'.format(3, join(kernel_size)))
#
#     @begin_layer
#     def pool(self, pool_size=(2, 2), strides=None, name=None):
#         self + MaxPool2D(pool_size=pool_size, strides=strides, name='{}Encode_MaxPool_{}'.format(p(name), join(pool_size)))
#
#     @begin_layer
#     def grow(self, pool_size=(2, 2)):
#         self + UpSampling2D(pool_size, interpolation='nearest', name='Decode_UpSample2D_{}'.format(join(pool_size)))
#
#     @begin_layer
#     def encode(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
#         self.conv(filters, kernel_size)
#         self.pool(pool_size)
#
#     @begin_layer
#     def decode(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
#         self.grow(pool_size)
#         self.conv(filters, kernel_size)
#
#     @begin_layer
#     def dense(self, size, name=None):
#         from numpy import prod
#         units = int(prod(size))
#         self + Dense(units)
#         self + LeakyReLU(0.01, name='{}Decode_LeakyReLU_0.01'.format(p(name)))
#         if size != units:
#             self.reshape(size)
#
#     @begin_layer
#     def dense_out(self, size):
#         from numpy import prod
#         units = int(prod(size))
#         self + Dense(units, activation='sigmoid')
#         if size != units:
#             self.reshape(size)
#
#     @begin_layer
#     def flatten(self, name=None):
#         self + Flatten(name='{}Flatten'.format(p(name)))
#
#     @begin_layer
#     def reshape(self, size, name=None):
#         self + Reshape(size, name='{}Reshape_{}'.format(p(name), join(size)))




class ModelBuilderTf:

    def __init__(self, xsize, ysize):
        self._xsize = xsize
        self._ysize = ysize
        self._l = 0

        self._i = 0
        self._j = 0
        self._begun_layer = False

        self._input = layers.Input((*xsize[:2], 3))
        self._last = self._input

    def add(self, layer):
        name = '{:02d}-{:02d}_{}'.format(self._i, self._j, layer.name)
        try:
            # not all layers are the same, some replace .name with properties that cannot be set.
            setattr(layer, 'name', name)  # TODO: there must be a better way
        except:
            setattr(layer, '_name', name)
        self._last = layer
        self._j += 1
        self._l += 1
        return self

    def __add__(self, layer):
        return self.add(layer)

    def __len__(self):
        return self._l

    @property
    def first(self):
        return self._input

    @property
    def last(self):
        return self._last

    def get_model(self):
        return models.Model(self.first, [self.last])

    @begin_layer
    def conv_in(self, filters, kernel_size=(3, 3)):
        self + layers.Conv2D(filters, kernel_size, padding='same', input_shape=(self._xsize[0], self._xsize[1], 3))(self.last)
        self + layers.BatchNormalization()(self.last)
        self + layers.LeakyReLU(0.01)(self.last)

    @begin_layer
    def conv(self, filters, kernel_size=(3, 3), sigmoid=False):
        self + layers.Conv2D(filters, kernel_size, padding='same', activation=None if not sigmoid else 'sigmoid')(self.last)
        if not sigmoid:
            self + layers.BatchNormalization()(self.last)
            self + layers.LeakyReLU(0.01)(self.last)

    @begin_layer
    def conv_out(self, kernel_size=(5, 5)):
        self + layers.Conv2D(3, kernel_size, activation='sigmoid', padding='same')(self.last)

    @begin_layer
    def pool(self, pool_size=(2, 2), strides=None):
        self + layers.MaxPool2D(pool_size=pool_size, strides=strides)(self.last)

    @begin_layer
    def grow(self, pool_size=(2, 2)):
        self + layers.UpSampling2D(pool_size, interpolation='nearest')(self.last)

    @begin_layer
    def encode(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
        self.conv(filters, kernel_size)(self.last)
        self.pool(pool_size)(self.last)

    @begin_layer
    def decode(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
        self.grow(pool_size)(self.last)
        self.conv(filters, kernel_size)(self.last)

    @begin_layer
    def dense(self, size):
        from numpy import prod
        units = int(prod(size))
        self + layers.Dense(units)(self.last)
        self + layers.LeakyReLU(0.01)(self.last)
        if size != units:
            self.reshape(size)

    @begin_layer
    def dense_out(self, size):
        from numpy import prod
        units = int(prod(size))
        self + layers.Dense(units, activation='sigmoid')(self.last)
        if size != units:
            self.reshape(size)

    @begin_layer
    def flatten(self):
        self + layers.Flatten()(self.last)

    @begin_layer
    def reshape(self, size):
        self + layers.Reshape(size)(self.last)

    @begin_layer
    def incept(self, c1, c3_1, c3, c5_1, c5, pp):
        # with tf.variable_scope('1x1'):
        conv1 = conv1x1(c1)(self.last)
        # with tf.variable_scope('3x3'):
        conv3_1 = conv1x1(c3_1)(self.last)
        conv3_1 = conv3x1(c3)(conv3_1)
        conv3 = conv1x3(c3)(conv3_1)
        # with tf.variable_scope('5x5'):
        conv5_1 = conv1x1(c5_1)(self.last)
        conv5_1 = conv1x5(c5)(conv5_1)
        conv5 = conv5x1(c5)(conv5_1)
        # with tf.variable_scope('pool'):
        pool_conv = conv1x1(pp)(self.last)
        pool = layers.AvgPool2D(3, strides=1, padding='same')(pool_conv)
        self + layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])

# ========================================================================= #
# INCEPTION - https://medium.com/datadriveninvestor/five-powerful-cnn-architectures-b939c9ddd57b
# ========================================================================= #

# def inception_module(in_tensor, c1, c3_1, c3, c5_1, c5, pp):
#     with tf.variable_scope('1x1'):
#         conv1 = conv1x1(c1)(in_tensor)
#
#     with tf.variable_scope('3x3'):
#         conv3_1 = conv1x1(c3_1)(in_tensor)
#         conv3 = conv3x3(c3)(conv3_1)
#
#     with tf.variable_scope('5x5'):
#         conv5_1 = conv1x1(c5_1)(in_tensor)
#         conv5 = conv5x5(c5)(conv5_1)
#
#     with tf.variable_scope('pool'):
#         pool_conv = conv1x1(pp)(in_tensor)
#         pool = layers.MaxPool2D(3, strides=1, padding='same')(pool_conv)
#
#     merged = layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])
#     return merged

# def aux_clf(in_tensor):
#     avg_pool = layers.AvgPool2D(5, 3)(in_tensor)
#     conv = conv1x1(128)(avg_pool)
#     flattened = layers.Flatten()(conv)
#     dense = layers.Dense(1024, activation='relu')(flattened)
#     dropout = layers.Dropout(0.7)(dense)
#     out = layers.Dense(1000, activation='softmax')(dropout)
#     return out
#
# def inception_net(in_shape=(224,224,3), n_classes=1000, opt='sgd'):
#     in_layer = layers.Input(in_shape)
#
#     conv1 = layers.Conv2D(64, 7, strides=2, activation='relu', padding='same')(in_layer)
#     pad1 = layers.ZeroPadding2D()(conv1)
#     pool1 = layers.MaxPool2D(3, 2)(pad1)
#     conv2_1 = conv1x1(64)(pool1)
#     conv2_2 = conv3x3(192)(conv2_1)
#     pad2 = layers.ZeroPadding2D()(conv2_2)
#     pool2 = layers.MaxPool2D(3, 2)(pad2)
#
#     inception3a = inception_module(pool2, 64, 96, 128, 16, 32, 32)
#     inception3b = inception_module(inception3a, 128, 128, 192, 32, 96, 64)
#     pad3 = layers.ZeroPadding2D()(inception3b)
#     pool3 = layers.MaxPool2D(3, 2)(pad3)
#
#     inception4a = inception_module(pool3, 192, 96, 208, 16, 48, 64)
#     inception4b = inception_module(inception4a, 160, 112, 224, 24, 64, 64)
#     inception4c = inception_module(inception4b, 128, 128, 256, 24, 64, 64)
#     inception4d = inception_module(inception4c, 112, 144, 288, 32, 48, 64)
#     inception4e = inception_module(inception4d, 256, 160, 320, 32, 128, 128)
#     pad4 = layers.ZeroPadding2D()(inception4e)
#     pool4 = layers.MaxPool2D(3, 2)(pad4)
#
#     aux_clf1 = aux_clf(inception4a)
#     aux_clf2 = aux_clf(inception4d)
#
#     inception5a = inception_module(pool4, 256, 160, 320, 32, 128, 128)
#     inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128)
#     pad5 = layers.ZeroPadding2D()(inception5b)
#     pool5 = layers.MaxPool2D(3, 2)(pad5)
#
#     avg_pool = layers.GlobalAvgPool2D()(pool5)
#     dropout = layers.Dropout(0.4)(avg_pool)
#     preds = layers.Dense(1000, activation='softmax')(dropout)
#
#     model = Model(in_layer, [preds, aux_clf1, aux_clf2])
#     model.compile(loss="categorical_crossentropy", optimizer=opt,
# 	              metrics=["accuracy"])
#     return model
#
# if __name__ == '__main__':
#     model = inception_net()
#     print(model.summary())







if __name__ == '__main__':

    b = ModelBuilderTf((192, 128), (192, 128))

    b.conv(64, (7, 7))
    b.pool((3, 3), 2)
    # b.conv(64, (1, 1))
    # b.conv(96, (3, 3))
    # b.pool((3, 3), 2)

    # b.incept(32, 16, 32, 16, 32, 16)
    # b.pool((3, 3), 2)
    # b.incept(32, 16, 32, 16, 32, 16)
    # b.pool((3, 3), 2)
    # b.incept(32, 16, 32, 16, 32, 16)
    # b.pool((3, 3), 2)
    b.incept(128, 128, 192, 32, 96, 64)
    b.pool((3, 3), 2)

    m = b.get_model()
    print(m.summary())