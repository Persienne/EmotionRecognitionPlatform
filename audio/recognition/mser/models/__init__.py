import importlib

from .base_model import BaseModel  # 从同一包导入 BaseModel 基类
from .bi_lstm import BiLSTM  # 从同一包导入 BiLSTM 模型
from mser.utils.logger import setup_logger  # 从 utils 模块导入日志设置函数

logger = setup_logger(__name__)  # 设置日志记录器，使用当前模块的名称

__all__ = ['build_model']


def build_model(input_size, configs):
    use_model = configs.model_conf.get('model', 'BiLSTM')  # 从配置中获取要使用的模型名称
    model_args = configs.model_conf.get('model_args', {})  # 从配置中获取模型参数
    mod = importlib.import_module(__name__)  # 动态导入当前模块

    # 使用 getattr 获取指定模型类，并传入输入尺寸和模型参数实例化模型
    model = getattr(mod, use_model)(input_size=input_size, **model_args)
    # 日志记录成功创建模型的消息，包括模型类型和参数

    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
