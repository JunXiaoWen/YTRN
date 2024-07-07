import paddle
from paddle import nn
from math import sqrt

class LSTMBlock(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMBlock, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           direction='bidirectional', dropout=dropout)

    def forward(self, x):
        y, h = self.rnn(x)
        return h, y
class ConvBlock(nn.Layer):
    def __init__(self, in_channel, out_channel, stride=2, groups=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1D(out_channel),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels=out_channel, out_channels=out_channel, groups=groups, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1D(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1D(out_channel),
        )
        self.relu = nn.PReLU()
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return self.relu(paddle.add(y1, y2))

class sequenceModule(nn.Layer):
    def __init__(self):
        super(sequenceModule, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_channel=4, out_channel=8, stride=1, groups=1),
            ConvBlock(in_channel=8, out_channel=16, stride=2, groups=1),
        )
        self.rnn1 = LSTMBlock(16, hidden_size=16, num_layers=2, dropout=0.1)  # 200, 32
        self.conv2 = ConvBlock(in_channel=32, out_channel=64, stride=2, groups=1)  # 100, 64
        self.rnn2 = LSTMBlock(64, hidden_size=64, num_layers=2, dropout=0.1)  # 100, 128
        self.conv3 = ConvBlock(in_channel=128, out_channel=256, stride=2, groups=1)  # 50, 256
        self.rnn3 = LSTMBlock(256, 256, num_layers=2, dropout=0.1)  # 50, 512
        self.conv4 = ConvBlock(in_channel=512, out_channel=512, stride=2, groups=1)  # 25, 512
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = paddle.transpose(conv1, [0, 2, 1])  # -> B W C
        rnn1_y, rnn1_h = self.rnn1(conv1)
        rnn1_h = paddle.transpose(rnn1_h, [0, 2, 1])  # -> B C W
        conv2 = self.conv2(rnn1_h)
        conv2 = paddle.transpose(conv2, [0, 2, 1])  # -> B W C
        rnn2_y, rnn2_h = self.rnn2(conv2)
        rnn2_h = paddle.transpose(rnn2_h, [0, 2, 1])  # -> B C W
        conv3 = self.conv3(rnn2_h)
        conv3 = paddle.transpose(conv3, [0, 2, 1])  # -> B W C
        rnn3_y, rnn3_h = self.rnn3(conv3)
        rnn3_h = paddle.transpose(rnn3_h, [0, 2, 1])  # -> B C W
        conv4 = self.conv4(rnn3_h)
        # out = self.pool(conv4)
        out = paddle.unsqueeze(conv4, 2)
        return out




class BlockA(nn.Layer):
    def __init__(self, in_channel, out_channel, groups=1):
        super(BlockA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, groups=groups, kernel_size=3, padding=2,
                      stride=1, dilation=2),
            nn.BatchNorm1D(out_channel),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels=out_channel, out_channels=out_channel, groups=groups, kernel_size=3, padding=2,
                      stride=1, dilation=2),
            nn.BatchNorm1D(out_channel)
        )
        self.relu = nn.GELU()
    def forward(self, x):
        y = self.conv(x)
        return self.relu(paddle.add(y, x))

class BlockB(nn.Layer):
    def __init__(self, in_channel, out_channel, stride=2, groups=1):
        super(BlockB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1D(out_channel),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1D(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, groups=groups
                      ,stride=stride),
            nn.BatchNorm1D(out_channel),
        )
        self.relu = nn.GELU()
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return self.relu(paddle.add(y1, y2))




class SequenceModule(nn.Layer):
    def __init__(self, in_channels=4, **kwargs):
        super(SequenceModule, self).__init__()
        self.conv = nn.Sequential(
            BlockB(in_channel=4, out_channel=64, stride=2),   # 200
            BlockA(in_channel=64, out_channel=64, groups=1),
            BlockB(in_channel=64, out_channel=128, stride=2),  # 100
            BlockA(in_channel=128, out_channel=128, groups=1),
            BlockB(in_channel=128, out_channel=256, stride=2),  # 50
            BlockA(in_channel=256, out_channel=256, groups=1),
        )
        self.pool = nn.AdaptiveAvgPool1D(32)
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=1, dropout=0.2, direction='bidirectional')
    def forward(self, x):
        # x [b, c, w]
        y = self.conv(x)
        y = self.pool(y)
        y = paddle.transpose(y, [0, 2, 1])  # B, C, W -->  B, W, C
        h, _ = self.rnn(y)
        y = paddle.transpose(h, [0, 2, 1])  # B, W, C -->  B, C, W
        y = y.unsqueeze(2)
        return y

class AdvFeatureFusion(nn.Layer):
    def __init__(self, conv_dim, seq_dim):
        super(AdvFeatureFusion, self).__init__()
        self.conv_project = nn.Linear(in_features=conv_dim, out_features=conv_dim)
        self.seq_project = nn.Linear(in_features=seq_dim, out_features=seq_dim)
        self.conv = nn.Conv2D(in_channels=conv_dim + seq_dim, out_channels=conv_dim, kernel_size=1, padding=0, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, conv, seq):
        # shape: B, C, H, W
        # conv = input[0]
        # seq = input[1]
        conv = conv.squeeze(2)
        seq = seq.squeeze(2)
        conv = paddle.transpose(conv, (0, 2, 1))
        seq = paddle.transpose(seq, (0, 2, 1))
        conv = self.conv_project(conv) # B W C
        seq = self.seq_project(seq)
        conv = paddle.transpose(conv, (0, 2, 1))
        seq = paddle.transpose(seq, (0, 2, 1))
        conv = conv.unsqueeze(2)
        seq = seq.unsqueeze(2)
        concat = paddle.concat([conv, seq], 1)
        # att = self.softmax(self.avgpool(self.conv(concat)))
        # return att * conv

        # att = self.sigmoid(self.avgpool(self.conv(concat)))
        gate = self.sigmoid(self.conv(concat))
        # out1 = att * conv
        out2 = paddle.multiply(gate, conv)
        return out2


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
    def __init__(self, embed_dim, num_heads, type='add'):
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
        conv_query_seq = self.conv_query_atten(conv_feature, seq_feature)

        add_fusion = conv_query_seq + conv_feature  # paddle.concat([conv_query_seq, conv_feature], axis=2)
        # add_fusion = self.norm(add_fusion)
        add_fusion = paddle.transpose(add_fusion, [0, 2, 1])
        add_fusion = add_fusion.unsqueeze(2)
        return add_fusion


if __name__ == '__main__':
    conv = paddle.randn((2, 512, 1, 25), dtype='float32')
    seq = paddle.randn((2, 512, 1, 25), dtype='float32')
    # model = AdvFeatureFusion(512, 512)
    # seq_branch = SequenceModule()
    # # model.eval()
    # # # out = model(conv, seq)
    # paddle.summary(model, ((2, 512, 1, 32), (2, 512, 1, 32)))
    # paddle.summary(seq_branch, ((1, 4, 1000)))

    caf = CAFModule(512, 8)
    paddle.summary(caf, ((1, 512, 1, 32), (1, 512, 1, 32)))















