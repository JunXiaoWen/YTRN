import paddle
import paddle.nn as nn


class BlockA(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(BlockA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1D(in_channels, out_channels, stride=1, padding=1,
                      kernel_size=3),
            nn.BatchNorm1D(out_channels),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1D(out_channels, out_channels, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm1D(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv1D(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm1D(out_channels)
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        conv = self.conv(x)
        shortcut = self.shortcut(x)
        out = conv + shortcut
        out = self.relu(out)

        return out


class BlockB(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(BlockB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1D(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1D(in_channels),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1D(in_channels),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1D(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1D(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1D(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm1D(out_channels)
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        conv = self.conv(x)
        shortcut = self.shortcut(x)
        out = paddle.add(conv, shortcut)
        out = self.relu(out)
        return out


class ACRN(nn.Layer):
    def __init__(self, in_channels, out_channels=400):
        super(ACRN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            BlockA(in_channels=6, out_channels=32),
            BlockA(in_channels=32, out_channels=64),
            BlockB(in_channels=64, out_channels=128),
            BlockB(in_channels=128, out_channels=256),
        )
        self.rnn1 = nn.LSTM(input_size=256, hidden_size=320, num_layers=1, direction='bidirectional')
        self.downsample1 = nn.Conv1D(in_channels=640, out_channels=256, kernel_size=1, stride=1)
        self.rnn2 = nn.LSTM(input_size=256, hidden_size=400, num_layers=1, direction='bidirectional')
        self.downsample2 = nn.Conv1D(in_channels=800, out_channels=400, kernel_size=1, stride=1)
        self.mha1 = nn.MultiHeadAttention(embed_dim=400, num_heads=8)
        self.mha2 = nn.MultiHeadAttention(embed_dim=400, num_heads=8)

    def forward(self, x, seq):
        seq = seq.astype('float32')  # [B C W]
        # seq = seq.transpose((0, 2, 1))

        conv = self.conv(seq)

        conv = paddle.transpose(conv, [0, 2, 1])  # [B W C]
        h1, _ = self.rnn1(conv)  # [B W C]
        h1 = paddle.transpose(h1, [0, 2, 1])  # [B C W]
        h1 = self.downsample1(h1)  # [B C:256 W]
        h1 = paddle.transpose(h1, [0, 2, 1])  # [B W C]
        h2, _ = self.rnn2(h1)
        h2 = paddle.transpose(h2, [0, 2, 1])  # [B C W]
        h2 = self.downsample2(h2)  # [B C:256 W]
        h2 = paddle.transpose(h2, [0, 2, 1])  # [B W C]

        mha1_out = self.mha1(query=h2, key=h2, value=h2)
        mha2_out = self.mha2(query=mha1_out, key=mha1_out, value=mha1_out)
        mha2_out = mha2_out.transpose((0, 2, 1))
        mha2_out = mha2_out.unsqueeze(2)

        return mha2_out


if __name__ == '__main__':
    x = paddle.randn((1, 6, 448), dtype='float32')
    model = ACRN(in_channels=6, out_channels=400)
    paddle.summary(model, ((1, 7, 32, 256), (1, 1000, 4)))
