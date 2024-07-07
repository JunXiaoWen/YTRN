import paddle
import paddle.nn as nn


if __name__ == '__main__':
    x = paddle.randn((2, 512, 1, 20), dtype='float32')
    conv1 = nn.Conv2DTranspose(512, 512, (1, 3), stride=4, padding=(0, 1))
    print(conv1(x).shape)

