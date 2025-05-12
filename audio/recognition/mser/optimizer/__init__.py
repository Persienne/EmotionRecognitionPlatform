import importlib

from torch.optim import *
from .scheduler import WarmupCosineSchedulerLR
from torch.optim.lr_scheduler import *
from mser.utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = ['build_optimizer', 'build_lr_scheduler']


def build_optimizer(params, configs):  # 创建优化器
    use_optimizer = configs.optimizer_conf.get('optimizer', 'Adam')  # 从配置中获取优化器名称，默认为 'Adam'
    optimizer_args = configs.optimizer_conf.get('optimizer_args', {})  # 从配置中获取优化器参数，默认为空字典
    optim = importlib.import_module(__name__)  # 动态导入当前模块
    optimizer = getattr(optim, use_optimizer)(params=params, **optimizer_args)  # 获取指定优化器并实例化
    logger.info(f'成功创建优化方法：{use_optimizer}，参数为：{optimizer_args}')  # 记录成功创建优化器的日志
    return optimizer


def build_lr_scheduler(optimizer, step_per_epoch, configs):
    use_scheduler = configs.optimizer_conf.get('scheduler', 'WarmupCosineSchedulerLR')  # 从配置获取调度器名称
    scheduler_args = configs.optimizer_conf.get('scheduler_args', {})  # 从配置中获取调度器参数

    # 如果使用 CosineAnnealingLR，并且没有提供 T_max 参数，则计算 T_max
    if configs.optimizer_conf.scheduler == 'CosineAnnealingLR' and 'T_max' not in scheduler_args:
        scheduler_args.T_max = int(configs.train_conf.max_epoch * 1.2) * step_per_epoch
    # 如果使用 WarmupCosineSchedulerLR，并且没有提供 fix_epoch 参数，则设置 fix_epoch
    if configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR' and 'fix_epoch' not in scheduler_args:
        scheduler_args.fix_epoch = configs.train_conf.max_epoch
    # 如果使用 WarmupCosineSchedulerLR，并且没有提供 step_per_epoch 参数，则设置 step_per_epoch
    if configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR' and 'step_per_epoch' not in scheduler_args:
        scheduler_args.step_per_epoch = step_per_epoch
        
    # 动态导入当前模块
    optim = importlib.import_module(__name__)
    scheduler = getattr(optim, use_scheduler)(optimizer=optimizer, **scheduler_args)  # 获取指定调度器并实例化
    logger.info(f'成功创建学习率衰减：{use_scheduler}，参数为：{scheduler_args}')
    return scheduler  # 返回创建的学习率调度器
