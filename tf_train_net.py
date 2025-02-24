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

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras.models import load_model
from tensorflow.python import Session
from model_builder_tf import ModelBuilderTf
import numpy as np
import os
import util
import time
from datasets import DATASETS_ROOT, MtgImages, MtgLocalFiles
from img_dashboard import TensorBoardOutputImages, TensorBoardMatch

# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


def create_model(x_size, y_size, batch_size):
    assert len(x_size) == 3
    assert len(y_size) == 3

    b = ModelBuilderTf(x_size, y_size)

    # ================================= #
    # http://ruder.io/optimizing-gradient-descent/index.html#whichoptimizertochoose # tl;dr: adam [RMSprop with momentum] is probs best choice
    # https://www.jeremyjordan.me/convnet-architectures/
    # ================================= #

    # b.conv_in(64, (11, 11))
    # b.conv(80, (9, 9))
    # b.pool()  # (192, 128) -> (96, 64)
    # b.conv(96, (7, 7))
    # b.conv(128, (5, 5))
    # b.pool()  # (96, 64) -> (48, 32)
    # b.conv(128)
    # b.conv(192)
    # b.pool()  # (48, 32) -> (24, 16)
    # b.conv(192)
    # b.conv(256)
    # b.pool()  # (24, 16) -> (12, 8)
    #
    # b.conv(10, name='encoded')
    # encoding_layer = b.last()
    #
    # b.grow()  # (12, 8) -> (24, 16)
    # b.conv(256)
    # b.conv(192)
    # b.grow()  # (24, 16) -> (48, 32)
    # b.conv(192)
    # b.conv(128)
    # b.grow()  # (48, 32) -> (96, 64)
    # b.conv(128)
    # b.conv(96)
    # b.conv(80, (5, 5))
    # b.conv(64, (7, 7))
    # b.conv(48, (9, 9))
    #
    # b.conv_out()

    # ================================= #

    # a = [24, 36, 48, 60, 72,  96, 120]
    # b = [32, 48, 64, 80, 96, 128, 160]

    def incept(a, A):
        b.incept(A, a, A, a, A, a)

    # b.conv(32, 5)
    b.conv(48, 5)
    b.conv(48, 3)
    b.pool()  # (192, 128) -> (96, 64)
    incept(48, 64)
    b.pool()  # (96, 64) -> (48, 32)
    incept(48, 64)
    b.pool()  # (48, 32) -> (24, 16)
    incept(48, 64)
    b.pool()  # (24, 16) -> (12, 8)
    incept(60, 80)
    b.pool()  # (12, 8) -> (6, 4)
    incept(72, 96)
    b.pool()  # (6, 4) -> (3, 2)
    incept(128, 256)
    b.pool((3, 2))  # (3, 2) -> (1, 1)

    b.conv(960, 1) #, name='encoded')
    encoding_layer = b.last

    b.grow((3, 2))  # (1, 1) -> (3, 2)
    b.conv(128, 3)
    b.grow()  # (3, 2) -> (6, 4)
    b.conv(96, 3)
    b.grow()  # (6, 4) -> (12, 8)
    b.conv(80, 3)
    b.grow()  # (12, 8) -> (24, 16)
    b.conv(64, 3)
    b.grow()  # (24, 16) -> (48, 32)
    b.conv(64, 3)
    b.grow()  # (48, 32) -> (96, 64)
    b.conv(64, 3)
    b.grow()  # (96, 64) -> (192, 128)
    b.conv(48, 3)
    b.conv(48, 5)

    b.conv_out()

    # ================================= #
    return b.get_model(), encoding_layer

# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #

# # COMMAND USED TO RUN ON LINUX, USING NVIDIA DOCKER CONTAINER
# # docker build ./ -t mtg-net && nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --user $(id -u):$(id -g) -v "$HOME/downloads/datasets:/datasets/" -v "$HOME/downloads/tmp/pycharm_project_106:/workspace" mtg-net python tf_train_net.py

if __name__ == "__main__":
    # CONSTS
    # x_size, y_size = (192, 128), (192//2, 128//2)
    x_size, y_size = (192, 128), (192, 128)
    # x_size, y_size = (640, 448), (640, 448)

    # VARS
    print('Dataset Root:', DATASETS_ROOT)

    opt_runs = os.getenv('RUNS', 512)
    opt_epochs = os.getenv('EPOCHS', 32)
    opt_samples = os.getenv('SAMPLES', 4096)
    opt_batch_size = os.getenv('BATCH_SIZE', 16)
    opt_test_size = os.getenv('TEST_SIZE', 128)
    opt_vis_size = os.getenv('VIS_SIZE', 20)
    opt_jit = os.getenv('JIT', None)
    opt_model = os.getenv('MODEL', None)

    # CONFIG
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    if opt_jit is not None: config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # START TIME IN SECONDS
    time_str = time.strftime("%Y-%m-%d__%H-%M-%S")

    # SESSION : TODO: replace with new approach
    with Session(config=config) as sess:
        # MODEL:
        if opt_model is not None:
            print('Loading: {}'.format(opt_model))
            model = load_model(opt_model, compile=False)
            encoding_layer = next(l for l in model.layers if 'encoded' in l.name)
            print('Loaded: {} [{}]'.format(opt_model, encoding_layer))
        else:
            print('Making:')
            model, encoding_layer = create_model((*x_size, 3), (*y_size, 3), batch_size=opt_batch_size)
            print('Made:')

        # COMPILE & LOSS:
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        # model.compile(loss='mse', optimizer=optimizer)
        model.summary()

        # DATA:
        dataset = MtgLocalFiles(img_type='small', x_size=x_size, y_size=y_size)
        out_x, out_y, out_o = dataset.gen_warp_crop_orig_set(opt_vis_size, save=False)

        # CALLBACKS:
        log_dir = './cache/tensorboard/{}'.format(time_str)
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=opt_batch_size, write_graph=True, write_grads=False, write_images=False),
            # ReduceLROnPlateau(factor=0.95, patience=8, cooldown=5),
            TensorBoardOutputImages(x_test=out_x, y_test=out_y, x_orig=out_o, log_dir=log_dir),
            # TensorBoardMatch(encoding_layer, n=opt_vis_size, x_size=x_size, y_size=y_size, log_dir=log_dir)
        ]

        # TRAIN:
        for run in range(opt_runs):
            print('\nRUN: {:02d}'.format(run))
            x_train, y_train = dataset.gen_warp_crop_set(opt_samples + opt_test_size, save=False)

            model.fit(
                x_train, y_train,
                validation_split=opt_test_size/(opt_samples+opt_test_size),
                epochs=opt_epochs,
                batch_size=opt_batch_size,
                callbacks=callbacks + [
                    ModelCheckpoint(filepath=util.init_dir('./model/{}'.format(time_str)) + '/weights.{:02d}'.format(run) + '.{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=False, period=opt_epochs),

                ]
            )
            del x_train
            del y_train
