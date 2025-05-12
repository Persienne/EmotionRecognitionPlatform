import os
import shutil

import librosa
import numpy as np

from mser.utils.logger import setup_logger

logger = setup_logger(__name__)  # 设置日志记录器


class AudioFeaturizer(object):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='Emotion2Vec', method_args={}):
        super().__init__()
        self._method_args = method_args  # 存储方法参数
        self._feature_method = feature_method  # 存储特征方法名称
        self._feature_model = None  # 初始化特征模型
        logger.info(f'使用的特征方法为 {self._feature_method}')  # 记录使用的特征提取方式

    def __call__(self, x, sample_rate: float) -> np.ndarray:  # 根据指定的特征方法提取音频特征
        if self._feature_method == 'CustomFeature':  # 调用自定义特征提取方法
            return self.custom_features(x, sample_rate)
        elif self._feature_method == 'Emotion2Vec':  # 调用Emotion2Vec特征提取方法
            return self.emotion2vec_features(x)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def emotion2vec_features(self, x) -> np.ndarray:
        from mser.utils.emotion2vec_predict import Emotion2vecPredict  # 导入Emotion2Vec预测类
        # 初始化模型
        if self._feature_model is None:
            self._feature_model = Emotion2vecPredict('iic/emotion2vec_base', revision="v2.0.4", use_gpu=True)
        feats = self._feature_model.extract_features(x, self._method_args)  # 提取特征
        return feats  # 返回提取的特征

    @staticmethod
    def custom_features(x, sample_rate: float) -> np.ndarray:  # 提取自定义音频特征
        stft = np.abs(librosa.stft(x))  # 计算短时傅里叶变换（STFT）

        # 提取音高和幅度 fmin 和 fmax 对应于人类语音的最小最大基本频率
        pitches, magnitudes = librosa.piptrack(y=x, sr=sample_rate, S=stft, fmin=70, fmax=400)
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, 1].argmax()  # 找到当前帧中幅度最大的音高
            pitch.append(pitches[index, i])  # 添加该音高到列表

        pitch_tuning_offset = librosa.pitch_tuning(pitches)  # 计算音高调谐偏移量
        pitchmean = np.mean(pitch)  # 计算音高均值
        pitchstd = np.std(pitch)  # 计算音高标准差
        pitchmax = np.max(pitch)  # 计算音高最大值
        pitchmin = np.min(pitch)  # 计算音高最小值

        # 计算频谱质心
        cent = librosa.feature.spectral_centroid(y=x, sr=sample_rate)
        cent = cent / np.sum(cent)  # 归一化
        meancent = np.mean(cent)  # 计算频谱质心均值
        stdcent = np.std(cent)  # 计算频谱质心标准差
        maxcent = np.max(cent)  # 计算频谱质心最大值

        # 计算谱平面均值
        flatness = np.mean(librosa.feature.spectral_flatness(y=x))

        # 使用系数为50的MFCC特征
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)  # 计算MFCC均值
        mfccsstd = np.std(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)  # 计算MFCC标准差
        mfccmax = np.max(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)  # 计算MFCC最大值

        # 计算色谱图特征
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

        # 计算梅尔频率
        mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)

        # ottava对比
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

        # 过零率
        zerocr = np.mean(librosa.feature.zero_crossing_rate(x))

        S, phase = librosa.magphase(stft)  # 获取幅度和相位
        meanMagnitude = np.mean(S)  # 计算幅度均值
        stdMagnitude = np.std(S)  # 计算幅度标准差
        maxMagnitude = np.max(S)  # 计算幅度最大值

        # 均方根能量
        rmse = librosa.feature.rms(S=S)[0]  # 计算均方根能量
        meanrms = np.mean(rmse)  # 计算均方根能量均值
        stdrms = np.std(rmse)  # 计算均方根能量标准差
        maxrms = np.max(rmse)  # 计算均方根能量最大值

        # 将所有提取的特征组合合成一个数组
        features = np.array([
            flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
            maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
            pitch_tuning_offset, meanrms, maxrms, stdrms
        ])

        # 将各个特征连接在一起并转换为float32类型
        features = np.concatenate((features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast)).astype(np.float32)
        return features  # 返回提取的特征数组

    @property
    def feature_dim(self):

        if self._feature_method == 'CustomFeature':
            return 312  # 自定义特征维度
        elif self._feature_method == 'Emotion2Vec':
            return 768  # Emotion2Vec特征维度
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')
