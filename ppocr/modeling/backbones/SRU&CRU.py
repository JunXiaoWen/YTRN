import paddle
from paddle import nn
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np



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

        return x_1, x_2
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

    def forward(self,x):
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

class Sequence_Attention(nn.Layer):
    # input : batch * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v):
        super(Sequence_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.softmax = nn.Softmax()
        self.norm_face = 1 / sqrt(dim_k)

    def forward(self, x, seq):
        x = paddle.squeeze(x, 2) # B, C, W
        x = paddle.transpose(x, [0, 2, 1])  # B, W, C
        Q = self.q(x)  # Q:batch_size * seq_len * dim_k
        K = self.k(x)
        V = self.v(x)
        atten = self.softmax(paddle.bmm(Q, K.transpose((0,2,1)) * self.norm_face))
        output = paddle.bmm(atten, V)
        output = paddle.transpose(output, [0, 2, 1])
        output = paddle.unsqueeze(output, 2)
        return atten, output


class FEM(nn.Layer):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 0.5,
                 squeeze_radio: int = 2,
                 group_size: int = 1,
                 group_kernel_size: int = 3,
                 gate_threshold: float = 0.5,
                 lamda: float = 1e-5,
                 ):
        super(FEM, self).__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.gate_threshold = gate_threshold
        self.gap = nn.AdaptiveAvgPool2D((1, 1))
        self.sigmoid = nn.Sigmoid()
        up_channel = int(op_channel * alpha)
        low_channel = op_channel - up_channel
        self.up_channel = up_channel
        self.low_channel = low_channel
        self.squeeze1 = nn.Conv2D(up_channel, up_channel // squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        self.squeeze2 = nn.Conv2D(low_channel, low_channel // squeeze_radio,
                                  kernel_size=1, bias_attr=False)
        # up branch
        self.GWC1 = DeformConv(up_channel // squeeze_radio, up_channel // squeeze_radio, kernel_size=(3, 3),
                               padding=(1, 1)
                               , group=group_size, isMask=False)

        self.GWC2 = nn.Conv2D(up_channel // squeeze_radio, up_channel // squeeze_radio,
                              kernel_size=(3, 1), stride=1,
                              padding=(1, 0), groups=group_size)
        self.GWC3 = nn.Conv2D(up_channel // squeeze_radio, up_channel // squeeze_radio,
                              kernel_size=(1, 3), stride=1,
                              padding=(0, 1), groups=group_size)
        self.PWC1 = nn.Conv2D(up_channel // squeeze_radio, up_channel // squeeze_radio,
                              kernel_size=1, bias_attr=False, groups=group_size)
        self.PWC3 = nn.Conv2D((up_channel // squeeze_radio) * 4, op_channel, kernel_size=1, padding=0)

        # low branch
        self.PWC2 = nn.Conv2D(low_channel // squeeze_radio,
                              op_channel - low_channel // squeeze_radio,
                              kernel_size=1, bias_attr=False)
        self.advavg = nn.AdaptiveAvgPool2D(1)
        self.softmax = nn.Softmax()

    def splitChannel(self, X, W):
        batch, channel, height, weight = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        w = paddle.squeeze(W)

        sortedIndex = paddle.argsort(w)

        large_mask = sortedIndex >= X.shape[1] * self.alpha
        low_mask = ~large_mask

        large_indices = paddle.nonzero(large_mask, as_tuple=False)
        low_indices = paddle.nonzero(low_mask, as_tuple=False)

        X_small = paddle.gather_nd(X, index=low_indices)
        X_large = paddle.gather_nd(X, index=large_indices)

        new_channel = int(X_small.shape[0] // batch)

        X_small = paddle.reshape(X_small, [batch, new_channel, height, weight])
        X_large = paddle.reshape(X_large, [batch, new_channel, height, weight])

        return X_large, X_small

    def forward(self, x):
        weight = self.gap(x)
        weight = self.sigmoid(weight)
        up, low = self.splitChannel(x, weight)

        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = paddle.concat([self.GWC1(up), self.GWC2(up), self.GWC3(up), self.PWC1(up)], axis=1)
        Y1 = self.PWC3(Y1)
        Y2 = paddle.concat([self.PWC2(low), low], axis=1)
        # Fuse
        out = paddle.concat([Y1, Y2], axis=1)
        att = self.softmax(self.advavg(out))
        out = att * out
        out1, out2 = paddle.split(out, 2, axis=1)
        return x + out1 + out2


class CrossAttentionModule(nn.Layer):
    def __init__(self, embed_dim, num_heads, need_weights=False, dropout=0.1):
        super(CrossAttentionModule, self).__init__()

        self.cross_atten = nn.MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                 need_weights=need_weights, dropout=dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.outNorm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        q = self.norm_q(query)
        k_v = self.norm_kv(key_value)
        src = self.cross_atten(query=q, key=k_v, value=k_v)
        return self.outNorm(src + k_v)

class CAFModule(nn.Layer):
    def __init__(self, embed_dim, num_heads, type='concat'):
        super(CAFModule, self).__init__()
        self.seq_query_atten = CrossAttentionModule(embed_dim, num_heads=num_heads)
        self.conv_query_atten = CrossAttentionModule(embed_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        if type == 'concat':
            self.conv = nn.Conv2D(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1, stride=1)
        else:
            self.conv = nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1)

    def forward(self, conv_feature, seq_feature):
        conv_feature = conv_feature.squeeze(2)
        seq_feature = seq_feature.squeeze(2)
        conv_feature = paddle.transpose(conv_feature, [0, 2, 1])
        seq_feature = paddle.transpose(seq_feature, [0, 2, 1])
        seq_query_conv = self.seq_query_atten(seq_feature, conv_feature)
        conv_query_seq = self.conv_query_atten(conv_feature, seq_feature)
        add_fusion = paddle.add(seq_query_conv, conv_query_seq) + conv_feature
        add_fusion = self.norm(add_fusion)
        add_fusion = paddle.transpose(add_fusion, [0, 2, 1])
        add_fusion = add_fusion.unsqueeze(2)
        return self.conv(add_fusion)


class CAFModuleV2(nn.Layer):
    def __init__(self, embed_dim, num_heads, type='concat'):
        super(CAFModuleV2, self).__init__()
        self.seq_query_atten = CrossAttentionModule(embed_dim, num_heads=num_heads)
        # self.conv_query_atten = CrossAttentionModule(embed_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        if type == 'concat':
            self.conv = nn.Conv2D(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1, stride=1)
        else:
            self.conv = nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1)

    def forward(self, conv_feature, seq_feature):
        conv_feature = conv_feature.squeeze(2)
        seq_feature = seq_feature.squeeze(2)
        conv_feature = paddle.transpose(conv_feature, [0, 2, 1])
        seq_feature = paddle.transpose(seq_feature, [0, 2, 1])
        seq_query_conv = self.seq_query_atten(seq_feature, conv_feature)
        # conv_query_seq = self.conv_query_atten(conv_feature, seq_feature)
        add_fusion = paddle.add(seq_query_conv, conv_feature)
        add_fusion = self.norm(add_fusion)
        add_fusion = paddle.transpose(add_fusion, [0, 2, 1])
        add_fusion = add_fusion.unsqueeze(2)
        return self.conv(add_fusion)

















if __name__ == '__main__':
    # x = paddle.randn((2, 512, 32, 160), dtype='float32')
    # simCru = SRU(512)
    # print(simCru(x).shape)
    conv = paddle.randn((2, 512, 1, 32), dtype='float32')
    seq = paddle.randn((2, 512, 32), dtype='float32')
    model = CAFModule(embed_dim=512, num_heads=2, type='add')
    paddle.summary(model, ((2, 512, 1, 32), (2, 512, 32)))


    # x = paddle.randn((2, 512, 1, 25), dtype='float32')
    # seq = paddle.randn((2, 512, 25), dtype='float32')





    # model = Sequence_Attention(512, 512, 512)
    # atten, output = model(x, seq)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # heatmap = ax.imshow(atten[0].numpy(), cmap='Reds', aspect='auto')
    #
    # # 显示数值标签
    # # if annot:
    # #     for i in range(20):
    # #         for j in range(20):
    # #             text = ax.text(j, i, format(atten[0][i, j], fmt), ha="center", va="center", color="black")
    #
    # cbar = plt.colorbar(heatmap)
    # plt.tight_layout()
    # plt.show()


    pass