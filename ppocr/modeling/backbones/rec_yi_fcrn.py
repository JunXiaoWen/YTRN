import paddle
import paddle.nn as nn

class ResidualLSTM(nn.Layer):
    def __init__(self, input_size, hidden_size, direction='bidirectional'):
        super(ResidualLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, direction=direction)

    def forward(self, x):
        h, _ = self.rnn(x)
        return h + x


class FCRN(nn.Layer):
    def __init__(self, in_channels=7, out_channels=1024):
        super(FCRN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv0 = nn.Sequential(
            nn.Conv2D(in_channels=7, out_channels=32, kernel_size=3, padding=(0, 1), stride=1),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=(0, 1)),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(0, 1)),
            nn.BatchNorm2D(128),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=(0, 1)),
            nn.BatchNorm2D(128),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=(0, 1)),
            nn.BatchNorm2D(256),
        )
        self.conv1 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(in_channels=256, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2D(512),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2D(512),
            nn.Conv2D(in_channels=512, out_channels=1024, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2D(1024)
        )
        self.rnn1 = nn.Sequential(
            ResidualLSTM(1024, 512),
            ResidualLSTM(1024, 512),
            ResidualLSTM(1024, 512),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2D(512),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2D(512),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2D(512),
            nn.Conv2D(in_channels=512, out_channels=1024, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2D(1024),
        )
        self.rnn2 = nn.Sequential(
            ResidualLSTM(input_size=1024, hidden_size=512),
            ResidualLSTM(input_size=1024, hidden_size=512),
            ResidualLSTM(input_size=1024, hidden_size=512),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Linear(in_features=1024, out_features=1024)
        )


    def forward(self, x, seq):
        x = x.astype("float32")
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv1 = paddle.squeeze(conv1, 2)
        conv2 = self.conv2(conv0)
        conv2 = paddle.squeeze(conv2, 2)
        conv1 = paddle.transpose(conv1, [0, 2, 1])
        conv2 = paddle.transpose(conv2, [0, 2, 1])
        rnn1 = self.rnn1(conv1)
        rnn2 = self.rnn2(conv2)
        rnn = paddle.concat([rnn1, rnn2], axis=2)
        linear_out = self.linear(rnn)
        linear_out = paddle.transpose(linear_out, [0, 2, 1])
        linear_out = linear_out.unsqueeze(2)
        return linear_out





if __name__ == '__main__':
    x = paddle.randn((1, 7, 126, 576))
    model = FCRN()
    model.eval()
