import os

import soundcard
import soundfile


class RecordAudio:
    def __init__(self, channels=1, sample_rate=16000):
        """初始化 RecordAudio 类

        :param channels: 麦克风通道数，默认为 1
        :param sample_rate: 录音采样率，默认为 16000 Hz
        """
        self.channels = channels  # 设置录音通道数
        self.sample_rate = sample_rate  # 设置录音采样率

        # 获取默认麦克风
        self.default_mic = soundcard.default_microphone()  # 获取系统默认的麦克风

    def record(self, record_seconds=3, save_path=None):
        """进行录音

        :param record_seconds: 录音时间，默认 3 秒
        :param save_path: 录音保存的路径，后缀名为 wav
        :return: 录音的 numpy 数据
        """
        print("开始录音......")  # 提示用户开始录音
        num_frames = int(record_seconds * self.sample_rate)  # 计算录音的帧数
        # 使用默认麦克风录音，指定采样率、帧数和通道数
        data = self.default_mic.record(samplerate=self.sample_rate, numframes=num_frames, channels=self.channels)
        audio_data = data.squeeze()  # 将数据展平，去掉多余的维度
        print("录音已结束!")  # 提示用户录音结束

        if save_path is not None:  # 如果提供了保存路径
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建保存目录（如果不存在）
            # 将录音数据保存为 WAV 文件
            soundfile.write(save_path, data=data, samplerate=self.sample_rate)
        return audio_data  # 返回录音的 numpy 数据

