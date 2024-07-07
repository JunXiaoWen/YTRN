import paddle
import paddle.nn as nn


class BlockA(nn.Layer):
    def __init__(self, in_channel, out_channel, groups=1):
        super(BlockA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, groups=groups, kernel_size=3, padding=1,
                      stride=1),
            nn.BatchNorm1D(out_channel),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels=out_channel, out_channels=out_channel, groups=groups, kernel_size=3, padding=2,
                      stride=1, dilation=2),
            nn.BatchNorm1D(out_channel)
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        y = self.conv(x)
        return self.relu(paddle.add(y, x))


class BlockB(nn.Layer):
    def __init__(self, in_channel, out_channel, stride=2, groups=1):
        super(BlockB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1D(out_channel),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1D(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, groups=groups
                      , stride=stride),
            nn.BatchNorm1D(out_channel),
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return self.relu(paddle.add(y1, y2))


class TCRN(nn.Layer):
    def __init__(self, in_channels=6, out_channels=400):
        super(TCRN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            BlockB(in_channel=6, out_channel=64, stride=1),
            BlockA(in_channel=64, out_channel=64),
            BlockB(in_channel=64, out_channel=128),
            BlockA(in_channel=128, out_channel=128),
            BlockB(in_channel=128, out_channel=200),
            BlockA(in_channel=200, out_channel=200),
        )
        self.rnn1 = nn.LSTM(input_size=200, hidden_size=200, num_layers=1, direction='bidirectional')
        self.downsample = nn.MaxPool1D(kernel_size=2, stride=2)
        self.rnn2 = nn.LSTM(input_size=400, hidden_size=200, num_layers=1, direction='bidirectional')

    def forward(self, conv, seq):
        seq = seq.astype("float32")
        # seq = seq.transpose((0, 2, 1))
        conv_out = self.conv(seq)
        conv_out = paddle.transpose(conv_out, [0, 2, 1])  # [B, L, C]
        h, y = self.rnn1(conv_out)
        h = paddle.transpose(h, [0, 2, 1])  # [B, C, L]
        h = self.downsample(h)
        h = paddle.transpose(h, [0, 2, 1])  # [B, L, C]
        h2, y2 = self.rnn2(h)
        h2 = paddle.transpose(h2, [0, 2, 1])  # [B, C, L]
        h2 = h2.unsqueeze(2)
        return h2


if __name__ == '__main__':
    tcrn = TCRN(6, 400)
    paddle.summary(tcrn, ((4, 7, 32, 256), (4, 1200, 6)))

    pass
