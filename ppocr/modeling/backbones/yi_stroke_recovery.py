import paddle
import paddle.nn as nn
import pysdtw


class TRNN(nn.Layer):
    def __init__(self, in_channels=1, out_channels=1024):
        super(TRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv1DTranspose(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=0),  # 40
            nn.Conv1DTranspose(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0),  # 8 0
            nn.Conv1DTranspose(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0),  # 160
            nn.Conv1DTranspose(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0),  # 320
            nn.Conv1DTranspose(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),  # 320
        )

    def forward(self, x):
        out = self.conv(x)
        return out


if __name__ == '__main__':
    x = paddle.randn((4, 512, 20))
    model = TRNN()
    print(model(x).shape)