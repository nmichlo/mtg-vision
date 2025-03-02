#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #

def join(itr, sep='x'):
    return sep.join([str(i) for i in itr])

def p(string):
    return (string + '_') if string is not None else ''

# PyTorch conv helpers with explicit in_channels
conv1x1 = partial(nn.Conv2d, kernel_size=1)
conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1)
conv1x3 = partial(nn.Conv2d, kernel_size=(1, 3), padding=(0, 1))
conv3x1 = partial(nn.Conv2d, kernel_size=(3, 1), padding=(1, 0))
conv1x5 = partial(nn.Conv2d, kernel_size=(1, 5), padding=(0, 2))
conv5x1 = partial(nn.Conv2d, kernel_size=(5, 1), padding=(2, 0))
conv5x5 = partial(nn.Conv2d, kernel_size=5, padding=2)

class ModelBuilder(nn.Module):
    def __init__(self, x_size, y_size):
        super(ModelBuilder, self).__init__()
        self.x_size = x_size  # e.g., (1, 3, 192, 128) in NCHW
        self.y_size = y_size  # e.g., (1, 3, 192, 128) in NCHW

        self._l = 0  # Layer counter
        self._i = 0  # Stage counter
        self._j = 0  # Layer index within stage
        self._begun_layer = False

        # Stage 0: conv(64, (7, 7))
        self.conv_0_0 = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False)
        self.bn_0_1 = nn.BatchNorm2d(64)
        self.relu_0_2 = nn.LeakyReLU(0.01, inplace=True)

        # Stage 1: pool((3, 3), 2)
        self.pool_1_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2: incept(128, 128, 192, 32, 96, 64)
        # Input to inception is 64 channels from previous pool
        self.incept_1x1 = conv1x1(in_channels=64, out_channels=128, bias=False)
        self.incept_3x3_1 = conv1x1(in_channels=64, out_channels=128, bias=False)
        self.incept_3x3_2 = conv3x1(in_channels=128, out_channels=192, bias=False)
        self.incept_3x3_3 = conv1x3(in_channels=192, out_channels=192, bias=False)
        self.incept_5x5_1 = conv1x1(in_channels=64, out_channels=32, bias=False)
        self.incept_5x5_2 = conv1x5(in_channels=32, out_channels=96, bias=False)
        self.incept_5x5_3 = conv5x1(in_channels=96, out_channels=96, bias=False)
        self.incept_pool_1 = conv1x1(in_channels=64, out_channels=64, bias=False)
        self.incept_pool_2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.incept_relu = nn.ReLU(inplace=True)  # Inception uses ReLU

        # Stage 3: pool((3, 3), 2)
        self.pool_3_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._init_weights()
        self.encoded = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _begin_layer(self, stage_name):
        if not self._begun_layer:
            self._begun_layer = True
            self._j = 0
            return True
        return False

    def _end_layer(self, reset=False):
        if reset:
            self._i += 1
            self._begun_layer = False

    def forward(self, x):
        # Input shape: (1, 3, 192, 128) if NCHW, or (1, 192, 128, 3) if NHWC
        if x.size(1) != 3:
            if x.size(3) == 3:
                x = x.permute(0, 3, 1, 2)
                # Shape: (1, 3, 192, 128)

        # Stage 0: conv(64, (7, 7))
        reset = self._begin_layer('stage_0')
        x = self.conv_0_0(x)
        # Shape: (1, 64, 192, 128) - 7x7 conv with padding=3 keeps spatial size
        x = self.bn_0_1(x)
        # Shape: (1, 64, 192, 128) - BN preserves shape
        x = self.relu_0_2(x)
        # Shape: (1, 64, 192, 128) - ReLU preserves shape
        self._l += 3
        self._j += 3
        self._end_layer(reset)

        # Stage 1: pool((3, 3), 2)
        reset = self._begin_layer('stage_1')
        x = self.pool_1_0(x)
        # Shape: (1, 64, 96, 64) - 3x3 pool with stride=2 halves H and W (192/2=96, 128/2=64)
        self._l += 1
        self._j += 1
        self._end_layer(reset)

        # Stage 2: incept(128, 128, 192, 32, 96, 64)
        reset = self._begin_layer('stage_2')
        # 1x1 branch
        conv1 = self.incept_1x1(x)
        conv1 = self.incept_relu(conv1)
        # Shape: (1, 128, 96, 64) - 1x1 conv changes channels to 128
        # 3x3 branch
        conv3 = self.incept_3x3_1(x)
        conv3 = self.incept_relu(conv3)
        # Shape: (1, 128, 96, 64) - 1x1 conv to 128 channels
        conv3 = self.incept_3x3_2(conv3)
        conv3 = self.incept_relu(conv3)
        # Shape: (1, 192, 96, 64) - 3x1 conv to 192 channels
        conv3 = self.incept_3x3_3(conv3)
        conv3 = self.incept_relu(conv3)
        # Shape: (1, 192, 96, 64) - 1x3 conv keeps 192 channels
        # 5x5 branch
        conv5 = self.incept_5x5_1(x)
        conv5 = self.incept_relu(conv5)
        # Shape: (1, 32, 96, 64) - 1x1 conv to 32 channels
        conv5 = self.incept_5x5_2(conv5)
        conv5 = self.incept_relu(conv5)
        # Shape: (1, 96, 96, 64) - 1x5 conv to 96 channels
        conv5 = self.incept_5x5_3(conv5)
        conv5 = self.incept_relu(conv5)
        # Shape: (1, 96, 96, 64) - 5x1 conv keeps 96 channels
        # Pool branch
        pool = self.incept_pool_1(x)
        pool = self.incept_relu(pool)
        # Shape: (1, 64, 96, 64) - 1x1 conv to 64 channels
        pool = self.incept_pool_2(pool)
        # Shape: (1, 64, 96, 64) - 3x3 avg pool with stride=1 keeps spatial size
        x = torch.cat([conv1, conv3, conv5, pool], dim=1)
        # Shape: (1, 480, 96, 64) - Concat: 128 + 192 + 96 + 64 = 480 channels
        self._l += 9  # 8 convs + 1 pool
        self._j += 9
        self.encoded = x
        self._end_layer(reset)

        # Stage 3: pool((3, 3), 2)
        reset = self._begin_layer('stage_3')
        x = self.pool_3_0(x)
        # Shape: (1, 480, 48, 32) - 3x3 pool with stride=2 halves H and W (96/2=48, 64/2=32)
        self._l += 1
        self._j += 1
        self._end_layer(reset)

        return x

def create_model(x_size, y_size):
    assert len(x_size) == 4 and len(y_size) == 4
    model_x_size = (x_size[0], x_size[3], x_size[1], x_size[2])  # Convert to NCHW
    model_y_size = (y_size[0], y_size[3], y_size[1], y_size[2])  # Convert to NCHW
    model = ModelBuilder(model_x_size, model_y_size)
    return model, model.encoded


if __name__ == '__main__':
    x_size = (16, 192, 128, 3)  # NHWC format
    y_size = (16, 48, 32, 480)  # NHWC format, matches TF output

    model, encoding_layer = create_model(x_size, y_size)
    device = torch.device("mps")  # MPS for Apple Silicon
    model = model.to(device)
    dummy_input = torch.randn(16, 192, 128, 3).to(device)  # NHWC format

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # Benchmark
    with torch.no_grad():
        for i in tqdm(range(100)):
            output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")  # (1, 192, 128, 3) NHWC
    print(f"Output shape: {output.shape}")    # (1, 480, 48, 32) NCHW
    print(f"Encoding shape: {model.encoded.shape}")  # (1, 480, 96, 64) NCHW