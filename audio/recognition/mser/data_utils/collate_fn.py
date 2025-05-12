import numpy as np
import torch


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的样本
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)  # 根据音频长度降序排序
    max_audio_length = batch[0][0].shape[0]  # 获取最长的音频的长度
    batch_size = len(batch)  # 获取batch的大小（样本数量）
    # 以最大的长度创建一个全为0的张量
    inputs = np.zeros((batch_size, max_audio_length), dtype='float32')  # 为输入创建零填充的numpy数组
    input_lens_ratio = []  # 存储每个样本相对于最大长度的比例
    labels = []  # 存储每个样本的标签
    for x in range(batch_size):
        sample = batch[x]  # 获取当前样本
        tensor = sample[0]  # 获取音频数据
        labels.append(sample[1])  # 将标签添加到标签列表中
        seq_length = tensor.shape[0]  # 获取当前样本的音频长度
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length] = tensor[:]  # 将当前样本的音频数据填充到inputs中
        # 计算当前样本长度与最大长度的比例
        input_lens_ratio.append(seq_length/max_audio_length)
    # 转换为numpy数组
    input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
    labels = np.array(labels, dtype='int64')
    # 返回处理后的inputs、labels和输入长度比例，转换为torch张量
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio)
