import copy
import io
import os
import random

import numpy as np
import resampy
import soundfile

from mser.data_utils.utils import buf_to_float, vad, decode_audio


class AudioSegment(object):  # 单声道音频段抽象类
    """
    :param samples: 音频样本 [num_samples x num_channels]
    :type samples: ndarray.float32
    :param sample_rate: 音频采样率
    :type sample_rate: int
    :raises TypeError: 如果样本数据类型不是 float 或 int
    """

    def __init__(self, samples, sample_rate):  # 根据样本创建音频段
        """
        样本会被转换为 float32，整型样本会缩放到 [-1, 1]
        """
        self._samples = self._convert_samples_to_float32(samples)  # 转换样本为 float32
        self._sample_rate = sample_rate  # 保存采样率
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)  # 如果是多通道，取平均值

    def __eq__(self, other):  # 返回两个对象是否相等
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):  # 返回两个对象是否不相等
        return not self.__eq__(other)

    def __str__(self):  # 返回该音频的信息
        return (f"{type(self)}: num_samples={self.num_samples}, sample_rate={self.sample_rate}, "
                f"duration={self.duration:.2f}sec, rms={self.rms_db:.2f}dB")

    @classmethod
    def from_file(cls, file):
        """从音频文件创建音频段

        :param file: 文件路径，或者文件对象
        :type file: str, BufferedReader
        :return: 音频片段实例
        :rtype: AudioSegment
        """
        assert os.path.exists(file), f'文件不存在，请检查路径：{file}'  # 确认文件存在
        try:
            samples, sample_rate = soundfile.read(file, dtype='float32')  # 从文件中读取样本和采样率
        except:
            # 支持更多格式数据
            sample_rate = 16000
            samples = decode_audio(file=file, sample_rate=sample_rate)  # 解码音频
        return cls(samples, sample_rate)

    @classmethod
    def slice_from_file(cls, file, start=None, end=None):
        """只加载一小段音频，而不需要将整个文件加载到内存中，这是非常浪费的。

        :param file: 输入音频文件路径或文件对象
        :type file: str|file
        :param start: 开始时间，单位为秒。如果start是负的，则它从末尾开始计算。如果没有提供，这个函数将从最开始读取。
        :type start: float
        :param end: 结束时间，单位为秒。如果end是负的，则它从末尾开始计算。如果没有提供，默认的行为是读取到文件的末尾。
        :type end: float
        :return: AudioSegment输入音频文件的指定片的实例。
        :rtype: AudioSegment
        :raise ValueError: 如开始或结束的设定不正确，例如时间不允许。
        """
        sndfile = soundfile.SoundFile(file)  # 打开音频文件
        sample_rate = sndfile.samplerate  # 获取采样率
        duration = round(float(len(sndfile)) / sample_rate, 3)  # 计算音频持续时间
        start = 0. if start is None else round(start, 3)  # 确定开始时间
        end = duration if end is None else round(end, 3)  # 确定结束时间
        # 从末尾开始计
        if start < 0.0:
            start += duration
        if end < 0.0:
            end += duration
        # 保证数据不越界
        if start < 0.0:
            start = 0.0
        if end > duration:
            end = duration
        if end < 0.0:
            raise ValueError(f"切片结束位置({end} s)越界")
        if start > end:
            raise ValueError(f"切片开始位置({start} s)晚于切片结束位置({end} s)")
        start_frame = int(start * sample_rate)  # 计算起始帧
        end_frame = int(end * sample_rate)  # 计算结束帧
        sndfile.seek(start_frame)  # 定位到起始帧
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')  # 读取音频数据
        return cls(data, sample_rate)  # 返回音频片段实例

    @classmethod
    def from_bytes(cls, data):
        """从包含音频样本的字节创建音频段

        :param data: 包含音频样本的字节
        :type data: bytes
        :return: 音频部分实例
        :rtype: AudioSegment
        """
        samples, sample_rate = soundfile.read(io.BytesIO(data), dtype='float32')  # 从字节读取音频样本
        return cls(samples, sample_rate)  # 返回音频段实例

    @classmethod
    def from_pcm_bytes(cls, data, channels=1, samp_width=2, sample_rate=16000):
        """从包含无格式PCM音频的字节创建音频

        :param data: 包含音频样本的字节
        :type data: bytes
        :param channels: 音频的通道数
        :type channels: int
        :param samp_width: 音频采样的宽度，如np.int16为2
        :type samp_width: int
        :param sample_rate: 音频样本采样率
        :type sample_rate: int
        :return: 音频部分实例
        :rtype: AudioSegment
        """
        samples = buf_to_float(data, n_bytes=samp_width)  # 将PCM字节转换为浮点样本
        if channels > 1:
            samples = samples.reshape(-1, channels)  # 如果多通道，调整形状
        return cls(samples, sample_rate)  # 返回音频段实例

    @classmethod
    def from_ndarray(cls, data, sample_rate=16000):
        """从numpy.ndarray创建音频段

        :param data: numpy.ndarray类型的音频数据
        :type data: ndarray
        :param sample_rate: 音频样本采样率
        :type sample_rate: int
        :return: 音频部分实例
        :rtype: AudioSegment
        """
        return cls(data, sample_rate)  # 创建并返回音频段实例

    @classmethod
    def concatenate(cls, *segments):
        """将任意数量的音频片段连接在一起

        :param *segments: 输入音频片段被连接
        :type *segments: tuple of AudioSegment
        :return: Audio segment instance as concatenating results.
        :rtype: AudioSegment
        :raises ValueError: If the number of segments is zero, or if the
                            sample_rate of any segments does not match.
        :raises TypeError: If any segment is not AudioSegment instance.
        """
        # Perform basic sanity-checks.
        if len(segments) == 0:  # 如果片段为零
            raise ValueError("没有音频片段被给予连接")
        sample_rate = segments[0]._sample_rate  # 获取第一个片段的采样率
        for seg in segments:
            if sample_rate != seg._sample_rate:  # 任何片段的采样率不匹配
                raise ValueError("能用不同的采样率连接片段")
            if type(seg) is not cls:  # 如果任何片段不是 AudioSegment 实例
                raise TypeError("只有相同类型的音频片段可以连接")
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """创建给定持续时间和采样率的静音音频段

        :param duration: 静音的时间，以秒为单位
        :type duration: float
        :param sample_rate: 音频采样率
        :type sample_rate: float
        :return: 给定持续时间的静音AudioSegment实例
        :rtype: AudioSegment
        """
        samples = np.zeros(int(duration * sample_rate))  # 创建静音样本
        return cls(samples, sample_rate)  # 返回...

    def to_wav_file(self, filepath, dtype='float32'):
        """保存音频段到磁盘为wav文件

        :param filepath: WAV文件路径或文件对象，以保存音频段
        :type filepath: str|file
        :param dtype: 音频文件的子类型 Options: 'int16', 'int32',
                      'float32', 'float64'. 默认 'float32'.
        :type dtype: str
        :raises TypeError: 如果 dtype 不被支持
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        subtype_map = {
            'int16': 'PCM_16',
            'int32': 'PCM_32',
            'float32': 'FLOAT',
            'float64': 'DOUBLE'
        }
        soundfile.write(
            filepath,
            samples,
            self._sample_rate,
            format='WAV',
            subtype=subtype_map[dtype])  # 保存为WAV文件

    def superimpose(self, other):
        """将另一个段的样本添加到这个段的样本中(以样本方式添加，而不是段连接)。

        :param other: 包含样品的片段被添加进去
        :type other: AudioSegments
        :raise TypeError: 如果两个片段的类型不匹配
        :raise ValueError: 不能添加不同类型的段
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"不能添加不同类型的段: {type(self)} 和 {type(other)}")  # 检查类型匹配
        if self._sample_rate != other._sample_rate:
            raise ValueError("采样率必须匹配才能添加片段")  # 检查采样匹配
        if len(self._samples) != len(other._samples):
            raise ValueError("段长度必须匹配才能添加段")  # 检查样本长度
        self._samples += other._samples  # 叠加样本

    def to_bytes(self, dtype='float32'):
        """创建包含音频内容的字节字符串

        :param dtype: 导出样本的类型 Options: 'int16', 'int32',
                      'float32', 'float64'. 默认 'float32'.
        :type dtype: str
        :return: 包含音频内容的字节字符串
        :rtype: str
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)  # 转换样本类型
        return samples.tostring()  # 返回字节字符串

    def to(self, dtype='int16'):
        """类型转换

        :param dtype: 导出样本的类型 Options: 'int16', 'int32',
                      'float32', 'float64'. 默认 'float32'.
        :type dtype: str
        :return: 包含指定类型音频内容的 np.ndarray
        :rtype: str
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples

    def gain_db(self, gain):
        """对音频施加分贝增益  就地转换

        :param gain: 要施加到样本上的增益（以分贝为单位）
        :type gain: float|1darray
        """
        self._samples *= 10. ** (gain / 20.)  # 应用增益

    def change_speed(self, speed_rate):
        """通过线性插值改变音频速度

        :param speed_rate: 速度变化率
                           speed_rate > 1.0, 加速音频
                           speed_rate = 1.0, 保持不变
                           speed_rate < 1.0, 减速音频
                           speed_rate <= 0.0, 不允许 raise ValueError.
        :type speed_rate: float
        :raises ValueError: 如果 speed_rate <= 0.0.
        """
        if speed_rate == 1.0:
            return
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = self._samples.shape[0]  # 获取旧长度
        new_length = int(old_length / speed_rate)  # 计算新长度
        old_indices = np.arange(old_length)  # 旧索引
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 新索引
        self._samples = np.interp(new_indices, old_indices, self._samples).astype(np.float32)  # 插值

    def normalize(self, target_db=-20, max_gain_db=300.0):
        """将音频归一化，使其具有所需的有效值(以分贝为单位)

        :param target_db: 目标 RMS 值（以分贝为单位） 该值应小于 0.0 因为 0.0 是满刻度音频
        :type target_db: float
        :param max_gain_db: 在归一化时可以施加的最大增益（以分贝为单位） 防止尝试将一个全为零的信号归一化为 NaN
        :type max_gain_db: float
        :raises ValueError: 如果将段归一化到 target_db 值所需的增益超过 max_gain_db
        """
        gain = target_db - self.rms_db  # 计算所需增益
        if gain > max_gain_db:
            raise ValueError(f"无法将段规范化到{target_db}dB，音频增益{gain}增益已经超过max_gain_db ({max_gain_db}dB)")
        self.gain_db(min(max_gain_db, target_db - self.rms_db))  # 应用增益

    def resample(self, target_sample_rate, filter='kaiser_best'):
        """按目标采样率重新采样音频  就地转换

        :param target_sample_rate: 目标采样率
        :type target_sample_rate: int
        :param filter: 要使用的重采样滤波器，可以选择 {'kaiser_best', 'kaiser_fast'}
        :type filter: str
        """
        # 重新采样
        self._samples = resampy.resample(self.samples, self.sample_rate, target_sample_rate, filter=filter)
        self._sample_rate = target_sample_rate  # 更新采样率

    def pad_silence(self, duration, sides='both'):
        """在这个音频样本上加一段静音  就地采样

        :param duration: 填充的静音长度（以秒为单位）
        :type duration: float
        :param sides: 填充位置:
                     'beginning' - 在开始处添加静音
                     'end' - 在结束处添加静音
                     'both' - 在开始和结束两侧添加静音
        :type sides: str
        :raises ValueError: 如果 sides 不被支持
        """
        if duration == 0.0:  # 如果静音时长为0，直接返回
            return self
        cls = type(self)
        silence = self.make_silence(duration, self._sample_rate)  # 创建静音样本
        # 填充
        if sides == "beginning":
            padded = cls.concatenate(silence, self)
        elif sides == "end":
            padded = cls.concatenate(self, silence)
        elif sides == "both":
            padded = cls.concatenate(silence, self, silence)
        else:
            raise ValueError(f"Unknown value for the sides {sides}")  # 检查填充方向
        self._samples = padded._samples  # 更新样本

    def shift(self, shift_ms):
        """音频偏移。如果shift_ms为正，则随时间提前移位;如果为负，则随时间延迟移位。填补静音以保持持续时间不变。  就地转换

        :param shift_ms: 偏移时间（以毫秒为单位）。如果正值，则提前移位；如果负值，则延迟移位。
        :type shift_ms: float
        :raises ValueError: 如果 shift_ms 超过音频持续时间
        """
        if abs(shift_ms) / 1000.0 > self.duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")  # 检查偏离是否超出持续时间
        shift_samples = int(shift_ms * self._sample_rate / 1000)  # 计算样本偏移
        if shift_samples > 0:
            # 时间提前
            self._samples[:-shift_samples] = self._samples[shift_samples:]  # 提前的时间
            self._samples[-shift_samples:] = 0  # 填补静音
        elif shift_samples < 0:
            # 时间延迟
            self._samples[-shift_samples:] = self._samples[:shift_samples]  #延迟的时间
            self._samples[:-shift_samples] = 0

    def subsegment(self, start_sec=None, end_sec=None):
        """在给定的边界之间切割音频片段  就地转换

        :param start_sec: 子片段的开始时间（以秒为单位）
        :type start_sec: float
        :param end_sec: 子片段的结束时间（以秒为单位）
        :type end_sec: float
        :raise ValueError: 如果 start_sec 或 end_sec 设置不正确，例如越界
        """
        start_sec = 0.0 if start_sec is None else start_sec  # 如果没有提供起始时间，则默认为0
        end_sec = self.duration if end_sec is None else end_sec  # 如果没有提供结束时间，则默认为音频持续时间
        if start_sec < 0.0:
            start_sec = self.duration + start_sec
        if end_sec < 0.0:
            end_sec = self.duration + end_sec
        # 检查起始时间是否越界
        if start_sec < 0.0:
            raise ValueError(f"切片起始位置({start_sec} s)越界")
        if end_sec < 0.0:
            raise ValueError(f"切片结束位置({end_sec} s)越界")
        if start_sec > end_sec:
            raise ValueError(f"切片的起始位置({start_sec} s)晚于结束位置({end_sec} s)")  # 检查起始时间是否晚于结束时间
        if end_sec > self.duration:
            raise ValueError(f"切片结束位置({end_sec} s)越界(> {self.duration} s)")  # 检查结束时间是否超出总时长
        start_sample = int(round(start_sec * self._sample_rate))  # 计算起始样本索引
        end_sample = int(round(end_sec * self._sample_rate))  # 计算结束样本索引
        self._samples = self._samples[start_sample:end_sample]  # 切割样本

    def random_subsegment(self, subsegment_length):
        """随机剪切指定长度的音频片段  就地转换

        :param subsegment_length: 子片段长度（以秒为单位）
        :type subsegment_length: float
        :raises ValueError: 如果子片段的长度大于原始片段
        """
        if subsegment_length > self.duration:
            raise ValueError(" 子片段的长度不得大于原始片段！")
        start_time = random.uniform(0.0, self.duration - subsegment_length)  # 随机选择起始时间
        self.subsegment(start_time, start_time + subsegment_length)  # 切割音频片段

    def add_noise(self, noise, snr_dB, max_gain_db=300.0):
        """以特定的信噪比添加给定的噪声段。如果噪声段比该噪声段长，则从该噪声段中采样匹配长度的随机子段。 就地转换

        :param noise: 要添加的噪声信号
        :type noise: AudioSegment
        :param snr_dB: 信噪比（单位为分贝）
        :type snr_dB: float
        :param max_gain_db: 添加噪声信号之前可以施加的最大增益，防止尝试对零信号施加无限增益
        :type max_gain_db: float
        :raises ValueError: 如果两个音频段之间的采样率不匹配，或者噪声段的持续时间短于原始音频段
        """
        # 检查采样率是否一致
        if noise.sample_rate != self.sample_rate:
            raise ValueError(f"噪声采样率({noise.sample_rate} Hz)不等于基信号采样率({self.sample_rate} Hz)")
        # 检查噪声信号的长度
        if noise.duration < self.duration:
            raise ValueError(f"噪声信号({noise.duration}秒)必须至少与基信号({self.duration}秒)一样长")
        noise_gain_db = min(self.rms_db - noise.rms_db - snr_dB, max_gain_db)  # 计算有效的增益
        noise_new = copy.deepcopy(noise)  # 深拷贝噪声信号
        noise_new.random_subsegment(self.duration)  # 随机选择与原信号相同长度的噪声段
        noise_new.gain_db(noise_gain_db)  # 应用增益
        self.superimpose(noise_new)  # 将噪声叠加到原信号上

    def vad(self, top_db=20, overlap=200):  # 语音活动检测，裁剪音频样本
        self._samples = vad(wav=self._samples, top_db=top_db, overlap=overlap)  # 调用 VAD 函数处理音频样本

    # 裁剪音频
    def crop(self, duration, mode='eval'):  # 裁剪音频至给定时长
        if self.duration > duration:  # 检查当前音频的持续时间是否超过目标持续时间
            if mode == 'train':
                self.random_subsegment(duration)  # 训练模式下随机剪切
            else:
                self.subsegment(end_sec=duration)  # 否则，直接切割到指定时长

    @property
    def samples(self):
        """返回音频样本

        :return: 音频样本
        :rtype: ndarray
        """
        return self._samples.copy()  # 返回样本的副本

    @property
    def sample_rate(self):
        """返回音频采样率

        :return: 音频采样率
        :rtype: int
        """
        return self._sample_rate  # 返回采样率

    @property
    def num_samples(self):
        """返回样品数量

        :return: 样本数量
        :rtype: int
        """
        return self._samples.shape[0]  # 返回样本数量

    @property
    def duration(self):
        """返回音频持续时间

        :return: 音频持续时间（以秒为单位）
        :rtype: float
        """
        return self._samples.shape[0] / float(self._sample_rate)  # 计算并返回持续时间

    @property
    def rms_db(self):
        """返回以分贝为单位的音频均方根能量

        :return: 以分贝为单位的均方根能量
        :rtype: float
        """
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(self._samples ** 2)  # 计算均方值
        if mean_square == 0:
            mean_square = 1  # 避免对零取对数
        return 10 * np.log10(mean_square)  # 转换为分贝

    @staticmethod
    def _convert_samples_to_float32(samples):
        """将样本类型转换为 float32。

        音频样本类型通常是整数或浮点数。
        整数将被缩放到 [-1, 1] 的范围内。
        """
        float32_samples = samples.astype('float32')  # 转换
        if samples.dtype in [np.int8, np.int16, np.int32, np.int64]:
            bits = np.iinfo(samples.dtype).bits  # 获取样本的位数
            float32_samples *= (1. / 2 ** (bits - 1))  # 缩放到[-1,1]
        elif samples.dtype in np.sctypes['float']:
            pass  # 如果是浮点数直接通过
        else:
            raise TypeError(f"Unsupported sample type: {samples.dtype}.")  # 抛出异常
        return float32_samples  # 返回转换后的样本

    @staticmethod
    def _convert_samples_from_float32(samples, dtype):
        """将样本类型从 float32 转换为指定的 dtype。
        音频样本类型通常是整数或浮点数。对于整数类型，float32 将从 [-1, 1] 重新缩放到整数类型支持的最大范围。
        为写入音频文件服务
        """
        dtype = np.dtype(dtype)  # 获取目标数据类型
        output_samples = samples.copy()  # 创建样本副本
        if dtype in [np.int8, np.int16, np.int32, np.int64]:
            bits = np.iinfo(dtype).bits  # 获取目标样本的位数
            output_samples *= (2 ** (bits - 1) / 1.)  # 重新缩放
            min_val = np.iinfo(dtype).min  # 获取最小值
            max_val = np.iinfo(dtype).max  # 获取最大值
            # 限制最小最大值
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        elif samples.dtype in np.sctypes['float']:
            # 获取浮点型最小最大值
            min_val = np.finfo(dtype).min
            max_val = np.finfo(dtype).max
            # 限制
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        else:
            raise TypeError(f"Unsupported sample type: {samples.dtype}.")  # 抛出异常
        return output_samples.astype(dtype)  # 返回转换后的样本
