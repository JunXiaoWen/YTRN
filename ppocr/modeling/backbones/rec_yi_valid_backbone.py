import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from math import sqrt
from ppocr.modeling.backbones.rec_yi_sequenceModule import sequenceModule, SequenceModule, AdvFeatureFusion, CAFModule
import matplotlib.pyplot as plt
import time

from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, make_divisible

from paddle.vision.ops import DeformConv2D



from paddle import ParamAttr

# __all__ = ["YBN"]


class DeformConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, group, isMask=False):
        super(DeformConv, self).__init__()
        self.isMask = isMask
        self.offset = nn.Conv2D(in_channels=in_channels, out_channels=2*kernel_size[0]*kernel_size[1], kernel_size=kernel_size,
                                padding=padding, stride=1)
        self.conv = DeformConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                                 groups=group, padding=padding)
        if isMask:
            self.mask = nn.Conv2D(in_channels=in_channels, out_channels=kernel_size[0] * kernel_size[1], kernel_size=kernel_size,
                                  padding=padding, stride=1)

    def forward(self, input):
        offset = self.offset(input)
        if self.isMask:
            mask = self.mask(input)
        else:
            mask = None

        conv = self.conv(input, offset=offset, mask=mask)
        return conv






class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm(num_channels=out_channels, act=None)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = self.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x

class GroupBatchNorm2D(nn.Layer):
    def __init__(self, c_num:int,
                 group_num: int = 16,
                 eps: float = 1e-10):
        super(GroupBatchNorm2D, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = self.create_parameter([c_num, 1, 1], None,
                                          dtype='float32',
                                          is_bias=False,
                                          default_initializer=nn.initializer.Assign(paddle.ones(shape=[c_num, 1, 1], dtype='float32')))
        self.beta = self.create_parameter(attr=None,
                                          shape=[c_num, 1, 1],
                                          dtype='float32',
                                          is_bias=False,
                                          default_initializer=nn.initializer.Assign(paddle.zeros(shape=[c_num, 1, 1], dtype='float32')))
        self.eps = eps
    def forward(self, x):
        N, C, H, W = x.shape
        x = paddle.reshape(x, shape=(N, self.group_num, -1))
        mean = paddle.mean(x, axis=2, keepdim=True)
        std = paddle.std(x, axis=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = paddle.reshape(x, shape=(N, C, H, W))
        return x * self.gamma + self.beta

class SRU(nn.Layer):
    def __init__(self,
                 op_channels: int,
                 group_num: int = 16,
                 gate_threshold: float = 0.5
                 ):
        super(SRU, self).__init__()
        self.gn = GroupBatchNorm2D(op_channels, group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / paddle.sum(self.gn.gamma, axis=0)
        reweight = self.sigmoid(gn_x * w_gamma)
        info_mask = reweight > self.gate_threshold
        info_mask = info_mask.astype("float32")
        noninfo_mask = reweight <= self.gate_threshold
        noninfo_mask = noninfo_mask.astype("float32")
        x_1 = info_mask * x * reweight
        x_2 = noninfo_mask * x * reweight
        out = self.reconstruct(x_1, x_2)
        return out

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = paddle.split(x_1, 2, axis=1)
        x_21, x_22 = paddle.split(x_2,  2, axis=1)
        return paddle.concat([x_11+x_22, x_12+x_21], axis=1)


class CRU(nn.Layer):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 0.5,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super(CRU, self).__init__()
        self.up_channel = up_channel = int(alpha*op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2D(up_channel, up_channel//squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        self.squeeze2 = nn.Conv2D(low_channel, low_channel//squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        # up branch
        self.GWC = nn.Conv2D(up_channel//squeeze_radio, op_channel,
                             kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size//2, groups=group_size)
        self.PWC1 = nn.Conv2D(up_channel//squeeze_radio, op_channel,
                              kernel_size=1, bias_attr=False)
        # low branch
        self.PWC2 = nn.Conv2D(low_channel//squeeze_radio,
                              op_channel - low_channel // squeeze_radio,
                              kernel_size=1, bias_attr=False)
        self.advavg = nn.AdaptiveAvgPool2D(1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Split
        up, low = paddle.split(x, [self.up_channel, self.low_channel], axis=1)
        up, low = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = paddle.concat([self.PWC2(low), low], axis=1)
        # Fuse
        out = paddle.concat([Y1, Y2], axis=1)
        out = self.softmax(self.advavg(out)) * out
        out1, out2 = paddle.split(out, 2, axis=1)
        return out1 + out2
class SCRU(nn.Layer):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 0.5,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 group_num: int = 16,
                 gate_threshold: float = 0.5
                 ):
        super(SCRU, self).__init__()
        self.gn = GroupBatchNorm2D(op_channel, group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()
        self.up_channel = up_channel = op_channel
        self.low_channel = low_channel = op_channel
        self.squeeze1 = nn.Conv2D(up_channel, up_channel//squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        self.squeeze2 = nn.Conv2D(low_channel, low_channel//squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        # up branch
        self.GWC = nn.Conv2D(up_channel//squeeze_radio, op_channel,
                             kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size//2, groups=group_size)
        self.PWC1 = nn.Conv2D(up_channel//squeeze_radio, op_channel,
                              kernel_size=1, bias_attr=False)
        # low branch
        self.PWC2 = nn.Conv2D(low_channel//squeeze_radio,
                              op_channel - low_channel // squeeze_radio,
                              kernel_size=1, bias_attr=False)
        self.advavg = nn.AdaptiveAvgPool2D(1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # split


        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / paddle.sum(self.gn.gamma, axis=0)
        reweight = self.sigmoid(gn_x * w_gamma)
        info_mask = reweight > self.gate_threshold
        info_mask = info_mask.astype("float32")
        noninfo_mask = reweight <= self.gate_threshold
        noninfo_mask = noninfo_mask.astype("float32")
        up = info_mask * x * reweight
        low = noninfo_mask * x * reweight

        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = paddle.concat([self.PWC2(low), low], axis=1)
        # Fuse
        out = paddle.concat([Y1, Y2], axis=1)
        out = self.softmax(self.advavg(out)) * out
        out1, out2 = paddle.split(out, 2, axis=1)
        return out1 + out2

class FEM(nn.Layer):
    def __init__(self,
                 op_channel: int,
                 alpha: float = (4/8),
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 ):
        super(FEM, self).__init__()

        self.alpha = alpha
        self.gap = nn.AdaptiveAvgPool2D((1, 1))
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2D(up_channel, up_channel // squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        self.squeeze2 = nn.Conv2D(low_channel, low_channel // squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        # up branch
        self.GWC1 = nn.Conv2D(up_channel // squeeze_radio, op_channel, kernel_size=(3, 3),
                               padding=(1, 1), groups=group_size)
        self.GWC2 = nn.Conv2D(up_channel // squeeze_radio, op_channel, kernel_size=(3, 1),
                               padding=(1, 0), groups=group_size)
        self.GWC3 = nn.Conv2D(up_channel // squeeze_radio, op_channel, kernel_size=(1, 3),
                               padding=(0, 1), groups=group_size)
        self.PWC3 = nn.Conv2D(up_channel // squeeze_radio, op_channel, kernel_size=1, padding=0, bias_attr=False)

        # low branch
        self.PWC2 = nn.Conv2D(low_channel // squeeze_radio,
                              op_channel - low_channel // squeeze_radio,
                              kernel_size=1, bias_attr=False)
        self.advavg = nn.AdaptiveAvgPool2D(1)
        self.softmax = nn.Softmax()
    def forward(self, x):
        # Split
        up, low = paddle.split(x, [self.up_channel, self.low_channel], axis=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC1(up) + self.GWC2(up) + self.GWC3(up) + self.PWC3(up)
        Y2 = paddle.concat([self.PWC2(low), low], axis=1)
        # Fuse
        out = paddle.concat([Y1, Y2], axis=1)
        att = self.softmax(self.advavg(out))
        out = att * out
        out1, out2 = paddle.split(out, 2, axis=1)
        return out2 + out1 + x




class SpatialAttentionModule(nn.Layer):
    def __init__(self, in_channels, kernel_size):
        super(SpatialAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, stride=1, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = paddle.max(x, axis=1, keepdim=True)
        avg_pool = paddle.mean(x, axis=1, keepdim=True)
        pool = paddle.concat([max_pool, avg_pool], axis=1)
        attention = self.conv(pool)
        attention = self.sigmoid(attention)
        return x * attention




class DownSampleBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding,
                 is_shortcut=False,
                 is_attention=True,
                 attention_kernel=3,
                 groups=1):
        super(DownSampleBlock, self).__init__()
        self.stride = stride
        self.is_shortcut = is_shortcut
        self.is_attention = is_attention
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu'
        )
        self.sam = SpatialAttentionModule(in_channels=out_channels, kernel_size=3)
        self.shortcut = nn.Sequential(
            ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, groups=groups,
                        stride=stride, if_act=False)

        )
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels=out_channels, out_channels=out_channels, padding=padding, kernel_size=3, stride=stride),
            nn.BatchNorm2D(out_channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        conv = self.conv0(x)
        if self.is_attention:
            conv = self.sam(conv)
        if self.is_shortcut:
            short = self.shortcut(x)
        conv = self.conv1(conv)
        return self.relu(conv)


class BottleNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding,
                 if_se=True):
        super(BottleNeck, self).__init__()
        self.conv = FEM(in_channels)


    def forward(self, x):
        conv = self.conv(x)
        # out = self.relu(paddle.add(x, conv))
        return conv




class VYTRN(nn.Layer):
    def __init__(self, in_channels=7, out_channels=512):
        super(VYTRN, self).__init__()
        self.out_channels = out_channels
        # self.conv = nn.Sequential(
        #     DownSampleBlock(in_channels=7, out_channels=32, stride=1, padding=1, groups=1, is_attention=False),
        #     DownSampleBlock(in_channels=32, out_channels=64, stride=1, padding=1, groups=1, is_attention=False),  # 32, 256
        #     BottleNeck(in_channels=64, out_channels=64, stride=1, padding=1, if_se=True),
        #     DownSampleBlock(in_channels=64, out_channels=128, stride=(2, 2), padding=1, groups=1, is_attention=True),  # 16, 256
        #     BottleNeck(in_channels=128, out_channels=128, stride=1, padding=1, if_se=True),
        #     DownSampleBlock(in_channels=128, out_channels=256, stride=2, padding=1, groups=1, is_attention=True),  # 8, 128
        #     BottleNeck(in_channels=256, out_channels=256, stride=1, padding=1, if_se=True),
        #     DownSampleBlock(in_channels=256, out_channels=512, stride=2, padding=1, groups=1, is_attention=True),  # 4, 64
        #     BottleNeck(in_channels=512, out_channels=512, stride=1, padding=1, if_se=True),
        # )
        self.pool = nn.AdaptiveAvgPool2D((1, 32))
        self.seq_branch = SequenceModule()  # sequenceModule()
        # self.advFusion = CAFModule(512, 4)
    def forward(self, x, seq):
        x = x.astype("float32")
        seq = seq.astype("float32")
        seq = seq.transpose((0, 2, 1))

        # conv = self.conv(x)
        # conv = self.pool(conv)

        seq_out = self.seq_branch(seq)
        # out = self.advFusion(conv, seq_out)


        return seq_out

if __name__ == '__main__':
    x = paddle.randn((1, 7, 32, 256), dtype='float32')
    seq = paddle.randn((1, 4, 400), dtype='float32')
    # conv = paddle.randn((32, 64, 32, 256), dtype='float32')


    ytrn = VYTRN()
    paddle.summary(ytrn, ((32, 7, 32, 256), (32, 448, 4)))

    # FLOPs = paddle.flops(ytrn, [1, 7, 32, 256], custom_ops={},
    #                      print_detail=True)
    # print(FLOPs)











