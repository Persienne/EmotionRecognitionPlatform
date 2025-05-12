import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()

        # 定义各层
        self.fc0 = nn.Linear(in_features=input_size, out_features=512)  # 输入到隐藏层的全连接层，输出维度为 512
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)  # 定义双向 LSTM 层
        self.tanh = nn.Tanh()  # 创建 tanh 激活函数的实例
        self.dropout = nn.Dropout(p=0.5)  # 定义 dropout 层，防止过拟合，丢弃 50% 的神经元
        self.fc1 = nn.Linear(in_features=512, out_features=256)  # 隐藏层到下一个全连接层，输出维度为 256
        self.relu1 = nn.ReLU()  # 创建 ReLU 激活函数的实例
        self.fc2 = nn.Linear(in_features=256, out_features=num_class)  # 最终输出层，输出类别的数量

    def forward(self, x):
        x = self.fc0(x)  # 通过第一个全连接层
        x = x.reshape((x.shape[0], 1, x.shape[1]))  # 将输入张量重塑为 (batch_size, seq_length=1, features)

        y, (h, c) = self.lstm(x)  # 将重塑后的输入传入 LSTM 层，得到输出 y 和隐藏状态 h、细胞状态 c
        x = y.squeeze(axis=1)  # 去掉 seq_length 维度，得到形状为 (batch_size, hidden_size * num_directions)

        x = self.tanh(x)  # 应用 tanh 激活函数
        x = self.dropout(x)  # 应用 dropout
        x = self.fc1(x)  # 通过第二个全连接层
        x = self.relu1(x)  # 应用 ReLU 激活函数
        x = self.fc2(x)  # 通过最后的全连接层，得到输出

        return x  # 返回输出张量，表示各类别的 logits
