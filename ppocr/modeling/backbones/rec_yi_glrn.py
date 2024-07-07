import paddle
import  paddle.nn as nn

def transpose(x):
    return paddle.transpose(x, [0, 2, 1])


class GLRN(nn.Layer):
    def __init__(self, in_channels=4, out_channels=512):
        super(GLRN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1_1 = nn.Conv1D(in_channels=6, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.layerNorm1_1 = nn.LayerNorm(128)
        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=128, dropout=0.1),
            1
        )
        self.conv1_2 = nn.Conv1D(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.layerNorm1_2 = nn.LayerNorm(128)

        self.conv2_1 = nn.Conv1D(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.layerNorm2_1 = nn.LayerNorm(512)
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, dropout=0.1),
            1
        )
        self.conv2_2 = nn.Conv1D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.layerNorm2_2 = nn.LayerNorm(512)

    def forward(self, x, seq):
        seq = seq.astype('float32')  # [B C W]
        # seq = paddle.transpose(seq, [0, 2, 1])
        conv1 = self.conv1_1(seq)  # [B C W]
        conv1 = self.layerNorm1_2(transpose(conv1))  # [B W C]
        conv1 = self.transformer1(conv1)  # [B W C]
        conv1 = self.conv1_2(transpose(conv1))  # [B C W]
        conv1 = self.layerNorm1_2(transpose(conv1))  # [B W C]
        conv2 = self.conv2_1(transpose(conv1))  # [B C W]
        conv2 = self.layerNorm2_1(transpose(conv2))  # [B W C]
        conv2 = self.transformer2(conv2)  # [B W C]
        conv2 = self.conv2_2(transpose(conv2))  # [B C W]
        conv2 = self.layerNorm2_2(transpose(conv2))  # [B W C]

        return paddle.unsqueeze(transpose(conv2), 2)


if __name__ == '__main__':
    # seq = paddle.randn((1, 6, 400), dtype='float32')
    # x = paddle.randn((1, 7, 32, 256), dtype='float32')
    # model = GLRN()
    # # print(model(x, seq).shape)
    # paddle.summary(model, ((1, 7, 32, 256), (1, 448, 6)))

    ctc = nn.Sequential(
        nn.Linear(512, 1165)
    )
    paddle.summary(ctc,(1, 32, 512))
