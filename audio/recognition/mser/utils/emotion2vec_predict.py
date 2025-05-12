import os
import shutil

import numpy as np
from funasr import AutoModel
from modelscope import snapshot_download

from mser.utils.logger import setup_logger

logger = setup_logger(__name__)  # 设置日志记录器


class Emotion2vecPredict(object):
    def __init__(self, model_id, revision, use_gpu=True):
        emotion2vec_model_dir = 'models/'  # 指定模型存储的目录
        save_model_dir = os.path.join(emotion2vec_model_dir, model_id)  # 构建保存模型的路径
        if not os.path.exists(save_model_dir):  # 如果模型目录不存在
            model_dir = snapshot_download(model_id, revision=revision)  # 下载模型快照
            shutil.copytree(model_dir, save_model_dir)  # 将下载的模型复制到指定目录
        # 加载模型
        self.model = AutoModel(model=save_model_dir,
                               log_level="ERROR",  # 设置日志级别为错误
                               device='cuda' if use_gpu else 'cpu',  # 根据是否使用 GPU 设置设备
                               disable_pbar=True,  # 禁用进度条
                               disable_log=True)  # 禁用日志
        logger.info(f"成功加载模型：{save_model_dir}")  # 记录成功加载模型的信息

    def extract_features(self, x, kwargs):
        res = self.model.generate(input=[x], **kwargs)  # 生成特征
        feats = res[0]["feats"]  # 获取特征
        return feats  # 返回提取的特征

    def predict(self, audio):
        res = self.model.generate(audio, granularity="utterance", extract_embedding=False)  # 生成预测结果
        labels, scores = [], []  # 初始化标签和分数列表
        for result in res:  # 遍历预测结果
            label, score = result["labels"], result["scores"]  # 获取标签和分数
            lab = np.argsort(score)[-1]  # 找到分数最高的索引
            s = score[lab]  # 获取最高分数
            l = label[lab].split("/")[0]  # 获取相应的标签
            labels.append(l)  # 将标签添加到列表
            scores.append(round(float(s), 5))  # 将分数添加到列表并四舍五入
        return labels, scores  # 返回预测的标签和分数
