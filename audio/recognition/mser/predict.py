import os
from io import BufferedReader
from typing import List

import joblib
import numpy as np
import torch
import yaml

from mser import SUPPORT_EMOTION2VEC_MODEL
from mser.data_utils.audio import AudioSegment
from mser.data_utils.featurizer import AudioFeaturizer
from mser.models import build_model
from mser.utils.logger import setup_logger
from mser.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class MSERPredictor:
    def __init__(self,
                 configs,
                 use_ms_model=None,
                 model_path='models/BiLSTM_Emotion2Vec/best_model/',
                 use_gpu=True):
        """
        声音分类预测工具
        :param configs: 配置参数
        :param use_ms_model: 使用ModelScope上公开Emotion2vec的模型
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'  # 确保 GPU 可用
            self.device = torch.device("cuda")  # 设置设备为 GPU
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU
            self.device = torch.device("cpu")  # 设置设备为 CPU

        self.use_ms_model = use_ms_model  # 存储是否使用 ModelScope 模型
        # 使用 ModelScope 上的模型
        if use_ms_model is not None:
            # 检查所用模型是否在支持列表中
            assert use_ms_model in SUPPORT_EMOTION2VEC_MODEL, f'没有该模型：{use_ms_model}'
            from mser.utils.emotion2vec_predict import Emotion2vecPredict  # 导入 Emotion2vec 预测工具
            self.predictor = Emotion2vecPredict(use_ms_model, revision=None, use_gpu=use_gpu)  # 初始化预测器
            return

        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:  # 打开配置文件
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)  # 解析 YAML 文件
            print_arguments(configs=configs)  # 打印配置参数

        self.configs = dict_to_object(configs)  # 将字典转换为对象
        # 获取特征提取器
        self._audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                 method_args=self.configs.preprocess_conf.get('method_args', {}))

        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 读取标签列表文件
        self.class_labels = [l.replace('\n', '') for l in lines]  # 去除换行符，保存标签

        # 自动获取分类数量
        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)  # 设置类别数量

        # 获取模型
        self.predictor = build_model(input_size=self._audio_featurizer.feature_dim, configs=self.configs)
        self.predictor.to(self.device)  # 将模型移动到指定设备

        # 加载模型
        if os.path.isdir(model_path):  # 检查模型路径是否为目录
            model_path = os.path.join(model_path, 'model.pth')  # 拼接模型文件路径
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"  # 检查模型文件是否存在

        # 加载模型参数
        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path)  # 从文件加载模型状态字典
        else:
            model_state_dict = torch.load(model_path, map_location='cpu')  # 加载到 CPU

        self.predictor.load_state_dict(model_state_dict)  # 加载模型参数
        print(f"成功加载模型参数：{model_path}")  # 输出加载成功信息
        self.predictor.eval()  # 设置模型为评估模式

        # 加载归一化文件
        self.scaler = joblib.load(self.configs.dataset_conf.dataset.scaler_path)

    def _load_audio(self, audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)  # 从文件路径加载音频
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)  # 从文件对象加载音频
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)  # 从 numpy 数组加载音频
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)  # 从字节加载音频
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')  # 抛出异常

        # 重采样音频
        if audio_segment.sample_rate != self.configs.dataset_conf.dataset.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.dataset.sample_rate)

        # 进行分贝归一化
        if self.configs.dataset_conf.dataset.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.dataset.target_dB)

        # 检查音频时长
        assert audio_segment.duration >= self.configs.dataset_conf.dataset.min_duration, \
            f'音频太短，最小应该为{self.configs.dataset_conf.dataset.min_duration}s，当前音频为{audio_segment.duration}s'

        # 获取特征
        feature = self._audio_featurizer(audio_segment.samples, sample_rate=audio_segment.sample_rate)  # 提取音频特征
        # 归一化特征
        feature = self.scaler.transform([feature])  # 使用加载的归一化器进行归一化
        feature = feature.squeeze().astype(np.float32)  # 压缩维度并转换类型
        return feature  # 返回音频特征

    # 预测一个音频的特征
    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频

        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 结果标签和对应的得分
        """
        if self.use_ms_model is not None:
            labels, scores = self.predictor.predict(audio_data)
            return labels[0], scores[0]
        # 加载音频文件，并进行预处理
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device).unsqueeze(0)
        # 执行预测
        output = self.predictor(input_data)
        result = torch.nn.functional.softmax(output, dim=-1)[0]
        result = result.data.cpu().numpy()
        # 最大概率的label
        lab = np.argsort(result)[-1]
        score = result[lab]
        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data: List, sample_rate=16000):
        """预测一批音频的特征

        :param audios_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 结果标签和对应的得分
        """
        if self.use_ms_model is not None:
            labels, scores = self.predictor.predict(audios_data)
            return labels, scores
        audios_data1 = []
        for audio_data in audios_data:
            # 加载音频文件，并进行预处理
            input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
            audios_data1.append(input_data)
        # 找出音频长度最长的
        batch = sorted(audios_data1, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
        for x in range(batch_size):
            tensor = audios_data1[x]
            seq_length = tensor.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[x, :seq_length] = tensor[:]
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        # 执行预测
        output = self.predictor(inputs)
        results = torch.nn.functional.softmax(output, dim=-1)
        results = results.data.cpu().numpy()
        labels, scores = [], []  # 初始化标签和得分列表
        for result in results:
            lab = np.argsort(result)[-1]  # 获取最大概率索引
            score = result[lab]  # 获取最大概率值
            labels.append(self.class_labels[lab])  # 添加标签
            scores.append(round(float(score), 5))  # 添加得分
        return labels, scores
