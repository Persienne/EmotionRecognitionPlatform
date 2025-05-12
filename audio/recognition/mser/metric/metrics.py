import numpy as np
import torch


# 计算准确率
def accuracy(output, label):

    # 对模型输出应用 softmax 函数，得到每个类别的概率分布
    output = torch.nn.functional.softmax(output, dim=-1)

    # 将输出从 GPU 移动到 CPU，并转换为 NumPy 数组
    output = output.data.cpu().numpy()

    # 通过 argmax 获取预测的类别索引，axis=1 表示在每一行中寻找最大值的索引
    output = np.argmax(output, axis=1)

    # 将真实标签从 GPU 移动到 CPU，并转换为 NumPy 数组
    label = label.data.cpu().numpy()

    # 计算准确率：预测正确的样本数除以总样本数
    acc = np.mean((output == label).astype(int))

    return acc  # 返回计算得到的准确率
