import paddle
import paddle.nn as nn




class DBLSTM(nn.Layer):
    def __init__(self, in_channels=7, out_channels=512):
        super(DBLSTM, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Dropout2D(0.1),
            nn.Conv2D(in_channels=7, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 16, 128
            nn.Dropout2D(0.3),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2D(128),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 8, 64
            nn.Dropout2D(0.3),
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2D(256),
            nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1)),  # 4, 64
            nn.Dropout2D(0.3),
            nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2D(512),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 4, 64
            nn.Dropout2D(0.3),
            nn.Conv2D(in_channels=512, out_channels=32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2D(32),
        )
        self.rnn1 = nn.LSTM(input_size=64, hidden_size=128, time_major=False, direction='bidirectional')
        self.rnn2 = nn.LSTM(input_size=256, hidden_size=128, time_major=False, direction='bidirectional')

    def forward(self, x, seq):
        x = x.astype('float32')
        out = self.conv(x)
        out = paddle.split(out,[1, 1], axis=2)
        out = paddle.concat([out[0], out[1]], axis=1)
        out = out.squeeze(2)
        out = out.transpose([0, 2, 1])
        rnn1, h1 = self.rnn1(out)
        rnn2, h2 = self.rnn2(rnn1)
        out = paddle.transpose(rnn2, [0, 2, 1])
        out = out.unsqueeze(2)
        return out


class CharNet(nn.Layer):
    def __init__(self, in_channel=7, out_channels=512):
        super(CharNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2D(0.1),
            nn.Conv2D(in_channels=7, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Dropout2D(0.3),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2D(128),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Dropout2D(0.3),
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2D(256),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Dropout2D(0.3)
        )
        self.conv1_1 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv1_2 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.conv1_3 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv1_4 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.conv1_5 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, groups=4)
        self.bn1 = nn.BatchNorm2D(512)
        self.pool = nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1))
        self.conv2_1 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, groups=4)
        self.conv2_2 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=(1, 3), padding=(0, 1), stride=1, groups=4)
        self.conv2_3 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=(3, 1), padding=(1, 0), stride=1, groups=4)
        self.conv2_4 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, groups=4)
        self.bn2 = nn.BatchNorm2D(1024)
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1, groups=16),
            nn.BatchNorm2D(512),
            nn.Conv2D(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2D(128),
            nn.Conv2D(in_channels=128, out_channels=32, kernel_size=1, padding=0, stride=1)
        )
        self.rnn1 = nn.LSTM(input_size=64, hidden_size=128, time_major=False, direction='bidirectional')
        self.rnn2 = nn.LSTM(input_size=256, hidden_size=128, time_major=False, direction='bidirectional')
        # self.rnn1 = nn.LSTM()

    def forward(self, x, seq):
        x = x.astype('float32')
        out = self.conv(x)
        conv1_1 = self.conv1_1(out)
        conv1_2 = self.conv1_2(out)
        conv1_3 = self.conv1_3(out)
        conv1_4 = self.conv1_4(out)
        conv1 = paddle.concat([conv1_1, conv1_2, conv1_3, conv1_4], axis=1)
        conv1 = self.bn1(conv1)
        conv1_5 = self.conv1_5(conv1)
        conv1_5 = self.pool(conv1_5)
        conv2_1 = self.conv2_1(conv1_5)
        conv2_2 = self.conv2_2(conv1_5)
        conv2_3 = self.conv2_3(conv1_5)
        conv2_4 = self.conv2_4(conv1_5)
        conv2 = paddle.concat([conv2_1, conv2_2, conv2_3, conv2_4], axis=1)
        conv3 = self.conv3(conv2)
        out = paddle.split(conv3, [1, 1], axis=2)
        out = paddle.concat([out[0], out[1]], axis=1)
        out = out.squeeze(2)
        out = out.transpose([0, 2, 1])
        rnn1, h1 = self.rnn1(out)
        rnn2, h2 = self.rnn2(rnn1)
        out = paddle.transpose(rnn2, [0, 2, 1])
        out = out.unsqueeze(2)
        return out











if __name__ == '__main__':
    model = DBLSTM()
    x = paddle.randn((32, 7, 32, 256), dtype='float32')
    seq = paddle.randn((32, 400, 4), dtype='float32')
    # print(model(x).shape)
    paddle.summary(model, ((32, 7, 32, 256), (32, 400, 4)))