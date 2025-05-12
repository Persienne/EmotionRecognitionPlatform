from torch import nn


class BaseModel(nn.Module):  # 定义一个名为 BaseModel 的类，继承自 nn.Module
    def __init__(self, input_size=768, num_class=4, hidden_size=256):
        super().__init__()  # 调用父类的构造函数，初始化 nn.Module
        self.relu = nn.ReLU()  # 创建 ReLU 激活函数的实例
        self.pre_net = nn.Linear(input_size, hidden_size)  # 定义从输入层到隐藏层的全连接层
        self.post_net = nn.Linear(hidden_size, num_class)  # 定义从隐藏层到输出层的全连接层

    def forward(self, x):  # 定义前向传播过程
        x = self.relu(self.pre_net(x))  # 将输入 x 传入前置网络，得到隐藏层输出，再通过 ReLU 激活
        x = self.post_net(x)  # 将隐藏层输出传入后置网络，得到最终的输出
        return x
