import paddle
import paddle.nn as nn

class CRNN(nn.Layer):
    def __init__(self, in_channels):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(64, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2D(2, 2)
        self.conv3 = nn.Conv2D(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2D(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2D((1, 2), 2)
        self.conv5 = nn.Conv2D(256, 512, 3, 1, 1)
        self.bn1 = nn.BatchNorm(512, act=None)
        self.conv6 = nn.Conv2D(512, 512, 3, 1, 1)
        self.bn2 = nn.BatchNorm(512, act=None)
        self.pool4 = nn.AdaptiveAvgPool2D((2, 21))
        self.conv7 = nn.Conv2D(512, 512, 2, 1, 0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool3(out)
        out = self.conv5(out)
        out = self.bn1(out)
        out = self.conv6(out)
        out = self.bn2(out)
        out = self.pool4(out)
        out = self.conv7(out)
        return out



if __name__ == '__main__':
    x = paddle.randn((1, 3, 32, 256))
    crnn = CRNN(3)
    out = crnn(x)
    print(out.shape)
    paddle.summary(crnn, (1, 3, 32, 256))
    pass

