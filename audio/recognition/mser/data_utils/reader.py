import os
import random

import joblib
import numpy as np
from torch.utils.data import Dataset

from mser.data_utils.audio import AudioSegment
from mser.data_utils.featurizer import AudioFeaturizer
from mser.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 audio_featurizer:AudioFeaturizer,
                 scaler_path=None,
                 do_vad=True,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf={},
                 num_speakers=1000,
                 use_dB_normalization=True,
                 target_dB=-20):

        super(CustomDataset, self).__init__()
        assert mode in ['train', 'eval', 'create_data', 'extract_feature']  # 确保模式有效
        self.do_vad = do_vad  # 是否进行语音活动检测
        self.max_duration = max_duration  # 最大音频时长
        self.min_duration = min_duration  # 最小音频时长
        self.mode = mode  # 数据集模式
        self._target_sample_rate = sample_rate  # 目标采样率
        self._use_dB_normalization = use_dB_normalization  # 是否使用分贝归一化
        self._target_dB = target_dB  # 目标分贝值
        self.aug_conf = aug_conf  # 数据增强配置
        self.num_speakers = num_speakers  # 说话人数量
        self.noises_path = None  # 噪声文件路径
        # 获取数据列表
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()  # 按行读取
        # 获取特征器
        self.audio_featurizer = audio_featurizer
        if scaler_path and self.mode != 'create_data':
            self.scaler = joblib.load(scaler_path)  # 加载归一化模型

    def __getitem__(self, idx):
        # 分割数据文件路径和标签
        data_path, label = self.lines[idx].replace('\n', '').split('\t')
        # 如果后缀名为.npy的文件，那么直接读取
        if data_path.endswith('.npy'):
            feature = np.load(data_path)
        else:
            # 读取音频
            audio_segment = AudioSegment.from_file(data_path)
            # 裁剪静音
            if self.do_vad:
                audio_segment.vad()
            # 数据太短不利于训练
            if self.mode == 'train':
                if audio_segment.duration < self.min_duration:
                    return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)  # 递归调用获取下一个索引
            # 重采样
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)
            # 音频增强
            if self.mode == 'train':
                audio_segment = self.augment_audio(audio_segment, **self.aug_conf)
            # 执行音量归一化
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)
            # 裁剪需要的数据
            if self.mode != 'extract_feature' and audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            # 提取音频特征
            feature = self.audio_featurizer(audio_segment.samples, sample_rate=audio_segment.sample_rate)
        # 归一化
        if self.mode != 'create_data' and self.mode != 'extract_feature':
            feature = self.scaler.transform([feature])  # 使用归一化模型进行转换
            feature = feature.squeeze().astype(np.float32)  # 压缩维度并转换数据类型
        return np.array(feature, dtype=np.float32), np.array(int(label), dtype=np.int64)  # 返回特征和标签

    def __len__(self):
        return len(self.lines)  # 返回数据集大小（行数）

    # 音频增强
    def augment_audio(self,
                      audio_segment,  # 音频段
                      speed_perturb=False,  # 是否进行语速扰动
                      volume_perturb=False,  # 是否进行音量扰动
                      volume_aug_prob=0.2,  # 音量扰动的概率
                      noise_dir=None,  # 噪声文件夹路径
                      noise_aug_prob=0.2):  # 噪声增强的概率

        # 语速增强，注意使用语速增强分类数量会大三倍
        if speed_perturb:
            speeds = [1.0, 0.9, 1.1]  # 定义速度变化比
            speed_idx = random.randint(0, 2)  # 随机选择速度变化
            speed_rate = speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)  # 改变音频速度
        # 音量增强
        if volume_perturb and random.random() < volume_aug_prob:
            min_gain_dBFS, max_gain_dBFS = -15, 15  # 定义音量增益范围
            gain = random.uniform(min_gain_dBFS, max_gain_dBFS)  # 随机选择增益值
            audio_segment.gain_db(gain)  # 调整音频音量
        # 获取噪声文件
        if self.noises_path is None and noise_dir is not None:
            self.noises_path = []  # 初始化噪声路径列表
            if noise_dir is not None and os.path.exists(noise_dir):  # 检查噪声目录是否存在
                for file in os.listdir(noise_dir):
                    self.noises_path.append(os.path.join(noise_dir, file))  # 添加噪声文件路径
        # 噪声增强
        if len(self.noises_path) > 0 and random.random() < noise_aug_prob:
            min_snr_dB, max_snr_dB = 10, 50
            # 随机选择一个noises_path中的一个
            noise_path = random.sample(self.noises_path, 1)[0]
            # 读取噪声音频
            noise_segment = AudioSegment.slice_from_file(noise_path)
            # 如果噪声采样率不等于音频段的采样率，则重采样
            if noise_segment.sample_rate != audio_segment.sample_rate:
                noise_segment.resample(audio_segment.sample_rate)
            # 随机生成信噪比值
            snr_dB = random.uniform(min_snr_dB, max_snr_dB)
            # 如果噪声的长度小于audio_segment的长度，则将噪声的前面的部分填充噪声末尾补长
            if noise_segment.duration < audio_segment.duration:
                diff_duration = audio_segment.num_samples - noise_segment.num_samples
                noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
            # 将噪声添加到 音频段 中，并将 信噪比 调整到最小值和最大值之间
            audio_segment.add_noise(noise_segment, snr_dB)
        return audio_segment
