import paddle
import paddle.nn as nn


def distill(input):
    B, W, C = input.shape[0], input.shape[1] // 2, input.shape[2] * 2
    output = paddle.zeros((B, W, C))
    for i in range(W):
        output[:, i, :] = paddle.concat([input[:, 2*i, :], input[:, 2*i + 1, :]], axis=1)
    return output


class DistillGRU(nn.Layer):
    def __init__(self, in_channels=4, out_channels=512):
        super(DistillGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.distillGRU_1 = nn.GRU(input_size=4, hidden_size=50, direction='forward', time_major=False)
        self.projection1 = nn.Linear(in_features=100, out_features=100, bias_attr=False)
        self.distillGRU_2 = nn.GRU(input_size=100, hidden_size=256, direction='forward', time_major=False)
        self.projection2 = nn.Linear(in_features=512, out_features=512, bias_attr=False)
        self.normGRU = nn.GRU(input_size=512, hidden_size=512, direction='forward', time_major=False, num_layers=2)


    def forward(self, conv, x):  # input [B, W, C]
        x = x.astype("float32")
        x = paddle.transpose(x, [0, 2, 1])
        out1, h1 = self.distillGRU_1(x)   #   B, 400, 50
        out1 = distill(out1)  # [B, W, C]  B, 200, 100
        out1 = self.projection1(out1)  # B, 200, 100
        out2, h2 = self.distillGRU_2(out1)  # B, 200, 256
        out2 = distill(out2)  # Batch, 100, 512
        out2 = self.projection2(out2)
        out3, h3 = self.normGRU(out2)  # Batch, 100, 512
        out3 = paddle.transpose(out3, [0, 2, 1])
        out3 = out3.unsqueeze(2)
        return out3

if __name__ == '__main__':
    model = DistillGRU()
    model.eval()
    x = paddle.randn((32, 2000, 4), dtype='float32')
    # print(model(x).shape)
    paddle.summary(model, ((32, 7, 32, 256), (32, 400, 4)))