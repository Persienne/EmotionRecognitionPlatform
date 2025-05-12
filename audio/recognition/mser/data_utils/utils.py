import io
import itertools

import av
import librosa
import numpy as np
import torch


def vad(wav, top_db=20, overlap=200):
    """对音频信号进行语音活动检测（VAD），识别非静音区间。

    Args:
        wav: 输入的音频信号（NumPy 数组）。
        top_db: 设定的阈值，表示静音的分贝值。
        overlap: 当合并音频片段时的重叠样本数。

    Returns:
        包含非静音部分的音频信号。
    """
    # 使用 librosa 的 split 函数获取非静音的音频区间
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav  # 如果没有非静音区间，直接返回原始音频

    wav_output = [np.array([])]  # 初始化输出音频列表
    for sliced in intervals:
        seg = wav[sliced[0]:sliced[1]]  # 切片获取非静音部分
        if len(seg) < 2 * overlap:
            wav_output[-1] = np.concatenate((wav_output[-1], seg))  # 如果片段较短，追加到最后一个片段
        else:
            wav_output.append(seg)  # 否则，作为新的片段添加

    wav_output = [x for x in wav_output if len(x) > 0]  # 过滤掉空片段

    if len(wav_output) == 1:
        wav_output = wav_output[0]  # 如果只有一个非静音片段，直接返回
    else:
        wav_output = concatenate(wav_output)  # 否则，合并所有非静音片段
    return wav_output


def concatenate(wave, overlap=200):
    """合并音频片段，使用重叠区域进行平滑处理。

    Args:
        wave: 输入的非静音音频片段列表。
        overlap: 重叠样本数，用于平滑过渡。

    Returns:
        合并后的音频信号。
    """
    total_len = sum([len(x) for x in wave])  # 计算所有片段的总长度
    unfolded = np.zeros(total_len)  # 初始化合并后的音频数组

    # Equal power crossfade
    window = np.hanning(2 * overlap)  # 创建汉宁窗
    fade_in = window[:overlap]  # 淡入窗口
    fade_out = window[-overlap:]  # 淡出窗口

    end = total_len
    for i in range(1, len(wave)):
        prev = wave[i - 1]  # 上一个片段
        curr = wave[i]  # 当前片段

        if i == 1:
            end = len(prev)
            unfolded[:end] += prev  # 添加第一个片段

        max_idx = 0
        max_corr = 0
        pattern = prev[-overlap:]  # 获取上一个片段的重叠部分
        # 在当前片段中查找与上一个片段重叠部分最匹配的起始位置
        for j in range(overlap):
            match = curr[j:j + overlap]
            corr = np.sum(pattern * match) / (np.sqrt(np.sum(pattern ** 2)) * np.sqrt(np.sum(match ** 2)) + 1e-8)
            if corr > max_corr:  # 如果找到更好的匹配
                max_idx = j
                max_corr = corr

        # 应用淡出效果
        start = end - overlap
        unfolded[start:end] *= fade_out
        end = start + (len(curr) - max_idx)  # 更新合并后的结束位置
        curr[max_idx:max_idx + overlap] *= fade_in  # 应用淡入效果
        unfolded[start:end] += curr[max_idx:]  # 合并当前片段
    return unfolded[:end]  # 返回最终合并的音频信号


def decode_audio(file, sample_rate: int = 16000):
    """读取音频文件，支持多种数据格式并进行重采样。

    Args:
      file: 输入文件的路径或文件类对象。
      sample_rate: 目标采样率。

    Returns:
      一个 float32 的 Numpy 数组，代表音频信号。
    """
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    raw_buffer = io.BytesIO()  # 创建一个字节缓冲区
    dtype = None

    with av.open(file, metadata_errors="ignore") as container:
        frames = container.decode(audio=0)  # 解码音频帧
        frames = _ignore_invalid_frames(frames)  # 忽略无效帧
        frames = _group_frames(frames, 500000)  # 分组处理帧
        frames = _resample_frames(frames, resampler)  # 进行重采样

        for frame in frames:
            array = frame.to_ndarray()  # 将帧转换为 ndarray
            dtype = array.dtype
            raw_buffer.write(array)  # 写入缓冲区

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)  # 从缓冲区读取音频数据

    # 将 s16 格式转换回 f32 格式
    return audio.astype(np.float32) / 32768.0  # 返回 float32 类型的音频信号


def _ignore_invalid_frames(frames):
    """过滤掉无效的音频帧。

    Args:
        frames: 输入的音频帧生成器。

    Yields:
        有效的音频帧。
    """
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue  # 忽略无效数据错误


def _group_frames(frames, num_samples=None):
    """将音频帧分组，以便进行处理。

    Args:
        frames: 输入的音频帧生成器。
        num_samples: 每组帧的样本数量。

    Yields:
        分组后的音频帧。
    """
    fifo = av.audio.fifo.AudioFifo()  # 创建音频 FIFO 队列

    for frame in frames:
        frame.pts = None  # 忽略时间戳检查
        fifo.write(frame)  # 将帧写入 FIFO 队列

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()  # 读取 FIFO 中的帧

    if fifo.samples > 0:
        yield fifo.read()  # 如果 FIFO 中还有帧，读取剩余帧


def _resample_frames(frames, resampler):
    """对音频帧进行重采样。

    Args:
        frames: 输入的音频帧生成器。
        resampler: 用于重采样的 AudioResampler 对象。

    Yields:
        重采样后的音频帧。
    """
    # 添加 None 以冲刷重采样器
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)  # 重采样并返回


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """将整数缓冲区转换为浮点值。

    主要用于将整数值的 wav 数据加载到 numpy 数组中。

    Args:
        x: 输入的整数值数据缓冲区（NumPy 数组）。
        n_bytes: 每个样本的字节数（1, 2, 4）。
        dtype: 目标输出数据类型（默认为 32 位浮点）。

    Returns:
        转换为浮点型的 NumPy 数组。
    """
    # 反转数据的缩放
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # 构建格式字符串
    fmt = "<i{:d}".format(n_bytes)

    # 重新缩放并格式化数据缓冲区
    return scale * np.frombuffer(x, fmt).astype(dtype)  # 返回浮点型数据


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """生成掩码张量，包含填充部分的索引。

    示例：
        lengths = [5, 3, 2]
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)  # 获取批次大小
    max_len = max_len if max_len > 0 else lengths.max().item()  # 获取最大长度
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)  # 创建序列范围
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)  # 扩展序列范围
    seq_length_expand = lengths.unsqueeze(-1)  # 扩展长度
    mask = seq_range_expand >= seq_length_expand  # 生成掩码
    return mask  # 返回掩码张量
