
# MTG CARD DETECTOR

## Overview

1. Card shape detection using cv2.adaptiveThreshold then cv2.findContours and various contour filtering techniques including concavity checking and reorientation.
2. Contours are extracted into cards using cv2.warpPerspective
3. Keys are generated for each detected card using the middle thinnest encoding layer in the Convolutional Auto Encoder (12x8x10 reshaped to 960x vector).
4. Detected Card Keys are matched to the entire database of card keys using cosine distance for K-Nearest-Neighbour
5. Info for matched cards is retrieved using mtgtools

## Running:

The entry point for this app is:

```bash
$ python3 mtg_detect.py

# If the above does not directly work, try running the following first.
$ python3 mtg_init_data
```

Options are available by pressing keys:

- '1' Overlay Extracted Cards From Blob Detector
- '2' Overlay Auto Encoder Output
- '3' Overlay Matched Card & Info
- '0' Draw Bound From Blob Detector
- '-' Use Custom Numba FeedForward Implementation (Extremely Slow)
- '=' Use Tensorflow for FeedForward (Fast)

Default model is included in: ./trained/weights.03-0.5411.hdf5

Default Card Encodings are found at: ./trained/weights.03-0.5411.small.vectors.np (This might not work across systems and may need to be deleted and thus regenerated, due to my terrible caching implementation) (Binary Numpy Array)

## CNN Implementation:

Tensorflow is used to speedup inference, but custom implementation of FeedForward is available at model_from_scratch.py (See keys Above)

The custom Model class uses jit compilation with numba for operations on each layer, and it only supports a limited subset of keras operations for a sequential model: LeakyReLU, Sigmoid, BatchNorm, Conv2D (1x1 strides), MaxPool2D (2x2 pooling), UpScaling2D (2x2 upsample, nearest-neigbour)

# MTG CARD TRAINER

## Overview

The CNN architecture is as follows, but could be a lot better:

```python
b = ModelBuilderTf(x_size=(192, 128), y_size=(96, 64))

# Input images are 192x128x3 (3=rgb)
b.conv_in(64, (11, 11))
b.conv(80, (9, 9))
b.pool()  # (192, 128) -> (96, 64)
b.conv(96, (7, 7))
b.conv(128, (5, 5))
b.pool()  # (96, 64) -> (48, 32)
b.conv(128)
b.conv(192)
b.pool()  # (48, 32) -> (24, 16)
b.conv(192)
b.conv(256)
b.pool()  # (24, 16) -> (12, 8)

b.conv(10, name='encoded') 
# Card is now encoded with 960 value vector representation (12x8x10=960)

b.grow()  # (12, 8) -> (24, 16)
b.conv(256)
b.conv(192)
b.grow()  # (24, 16) -> (48, 32)
b.conv(192)
b.conv(128)
b.grow()  # (48, 32) -> (96, 64)
b.conv(128)
b.conv(96)
b.conv(80, (5, 5))
b.conv(64, (7, 7))
b.conv(48, (9, 9))

b.conv_out() # sigmoid convolution with 3 of 3x3 filers, one each for RGB
# Output images are 96x64x3 (3=rgb)
```

## Running

(Optional) Start TensorBoard, make sure to run this within your project root.

```bash
$ tensorboard --logdir="./cache/tensorboard" --port 8050
```

Example command to start training of neural network in tf_train_net.py,
if your project is currently in $HOME/downloads/tmp/pycharm_project_106:/workspace.
Make sure to run this within your project root.

```bash
$ docker build ./ -t mtg-net

$ nvidia-docker run -it --rm \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --user $(id -u):$(id -g) \
    -v "$HOME/downloads/datasets:/datasets/" \
    -v "$HOME/downloads/tmp/pycharm_project_106:/workspace" \
    mtg-net python tf_train_net.py
```

The model is saved every run by default to ./model/<time>



# MTG CARD TESTS

To run the tests, and get a top-K error rate run:

```bash
$ python3 mtg_test.py
# It is recomended to instead run this the same way as tf_train_net.py in the above section, using NVIDIA docker.
```

