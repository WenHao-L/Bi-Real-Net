# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from mindspore import nn, ops, Parameter
from mindspore.common import dtype as mstype

__all__ = ['birealnet18', 'birealnet34']


# if os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
#     BatchNorm2d = nn.SyncBatchNorm
# else:
BatchNorm2d = nn.BatchNorm2d


class AdaptiveAvgPool2d(nn.Cell):
    """AdaptiveAvgPool2d"""
    def __init__(self):
        super(AdaptiveAvgPool2d, self).__init__()
        self.mean = ops.ReduceMean(True)

    def construct(self, x):
        x = self.mean(x[:, :, 0:7, 0:7 ], (-2,-1))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, pad_mode="pad", has_bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, pad_mode="valid", has_bias=False)


class BinaryActivation(nn.Cell):

    def __init__(self):
        super(BinaryActivation, self).__init__()

    def construct(self, x):
        out_forward = ops.Sign()(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * ops.Cast()(mask1, mstype.float32) + (x*x + 2*x) * (1-ops.Cast()(mask1, mstype.float32))
        out2 = out1 * ops.Cast()(mask2, mstype.float32) + (-x*x + 2*x) * (1-ops.Cast()(mask2, mstype.float32))
        out3 = out2 * ops.Cast()(mask3, mstype.float32) + 1 * (1- ops.Cast()(mask3, mstype.float32))
        out =  ops.stop_gradient(out_forward) -  ops.stop_gradient(out3) + out3

        return out


class HardBinaryConv(nn.Cell):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = Parameter(ops.UniformReal()((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.conv2d = ops.Conv2D(out_channel=out_chn, kernel_size=3, stride=self.stride, pad=self.padding, pad_mode="pad")
        self.mean = ops.ReduceMean(keep_dims=True)


    def construct(self, x):
        real_weights = ops.Reshape()(self.weights, self.shape)
        scaling_factor = self.mean(self.mean(self.mean(ops.Abs()(real_weights), 3), 2), 1)
        # print(scaling_factor, flush=True)
        scaling_factor = ops.stop_gradient(scaling_factor)
        binary_weights_no_grad = scaling_factor * ops.Sign()(real_weights)
        cliped_weights = ops.clip_by_value(real_weights, -1.0, 1.0)
        binary_weights = ops.stop_gradient(binary_weights_no_grad) - ops.stop_gradient(cliped_weights) + cliped_weights
        # print(binary_weights, flush=True)
        y = self.conv2d(x, binary_weights)

        return y

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.binary_activation(x)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class BiRealNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad", has_bias=False)
        self.bn1 = BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d()
        self.fc = nn.Dense(512 * block.expansion, num_classes, has_bias=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                BatchNorm2d(num_features=planes * block.expansion)
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = ops.Reshape()(x, (x.shape[0], -1))
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model

