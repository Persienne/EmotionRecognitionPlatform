import json
import os
import shutil

import torch

from mser import __version__
from mser.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_pretrained(model, pretrained_model):
    """加载预训练模型

    :param model: 使用的模型
    :param pretrained_model: 预训练模型路径
    """
    # 加载预训练模型
    if pretrained_model is None: return model  # 如果没有提供预训练模型，直接返回当前模型
    if os.path.isdir(pretrained_model):  # 如果提供的是目录，则寻找 'model.pth'
        pretrained_model = os.path.join(pretrained_model, 'model.pth')
    assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"  # 检查模型文件是否存在
    model_state_dict = torch.load(pretrained_model)  # 加载预训练模型的状态字典
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 检查模型是否为分布式并行模型
        model_dict = model.module.state_dict()  # 获取模型的状态字典
    else:
        model_dict = model.state_dict()
        # 过滤不存在的参数
    for name, weight in model_dict.items():  # 遍历模型的参数
        if name in model_state_dict.keys():  # 如果预训练模型中存在该参数
            if list(weight.shape) != list(model_state_dict[name].shape):  # 检查形状是否匹配
                logger.warning(f'{name} not used, shape {list(model_state_dict[name].shape)} '
                               f'unmatched with {list(weight.shape)} in model.')
                model_state_dict.pop(name, None)  # 如果形状不匹配，则从预训练模型中移除该参数
    # 加载权重
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        missing_keys, unexpected_keys = model.module.load_state_dict(model_state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if len(unexpected_keys) > 0:  # 检查是否有未预期的键
        logger.warning('Unexpected key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:  # 检查是否有缺失的键
        logger.warning('Missing key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in missing_keys)))
    logger.info('成功加载预训练模型：{}'.format(pretrained_model))  # 记录成功加载模型的信息
    return model


def load_checkpoint(configs, model, optimizer, amp_scaler, scheduler,
                    step_epoch, save_model_path, resume_model):
    """加载模型

    :param configs: 配置信息
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    :param amp_scaler: 使用的自动混合精度
    :param scheduler: 使用的学习率调整策略
    :param step_epoch: 每个epoch的step数量
    :param save_model_path: 模型保存路径
    :param resume_model: 恢复训练的模型路径
    """
    last_epoch1 = -1
    accuracy1 = 0.

    def load_model(model_path):
        assert os.path.exists(os.path.join(model_path, 'model.pth')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(model_path, 'optimizer.pth')), "优化方法参数文件不存在！"
        state_dict = torch.load(os.path.join(model_path, 'model.pth'))  # 加载模型参数
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 检查模型是否为分布式并行模型
            model.module.load_state_dict(state_dict)  # 加载模型参数
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth')))  # 加载优化器参数
        # 自动混合精度参数
        if amp_scaler is not None and os.path.exists(os.path.join(model_path, 'scaler.pth')):  # 如果存在混合精度文件
            amp_scaler.load_state_dict(torch.load(os.path.join(model_path, 'scaler.pth')))  # 加载混合精度参数
        with open(os.path.join(model_path, 'model.state'), 'r', encoding='utf-8') as f:  # 打开模型状态文件
            json_data = json.load(f)  # 读取 JSON 数据
            last_epoch = json_data['last_epoch'] - 1  # 获取上一个 epoch
            accuracy = json_data['accuracy']  # 获取准确率
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(model_path))
        optimizer.step()  # 更新优化器
        [scheduler.step() for _ in range(last_epoch * step_epoch)]  # 更新学习率
        return last_epoch, accuracy  # 返回上一个 epoch 和准确率

    # 获取最后一个保存的模型
    save_feature_method = configs.preprocess_conf.feature_method  # 获取特征方法
    if configs.preprocess_conf.get('use_hf_model', False):  # 检查是否使用 HF 模型
        save_feature_method = save_feature_method[:-1] if save_feature_method[-1] == '/' else save_feature_method
        save_feature_method = os.path.basename(save_feature_method)  # 提取特征方法的基本名称
    last_model_dir = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'last_model')  # 构建最后模型的路径
    if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                    and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
        if resume_model is not None:  # 如果有恢复模型路径
            last_epoch1, accuracy1 = load_model(resume_model)  # 加载恢复模型
        else:
            try:
                # 自动获取最新保存的模型
                last_epoch1, accuracy1 = load_model(last_model_dir)  # 尝试加载最后模型
            except Exception as e:
                logger.warning(f'尝试自动恢复最新模型失败，错误信息：{e}')  # 记录加载失败的警告
    return model, optimizer, amp_scaler, scheduler, last_epoch1, accuracy1  # 返回模型和状态


# 保存模型
def save_checkpoint(configs, model, optimizer, amp_scaler, save_model_path, epoch_id,
                    accuracy=0., best_model=False):
    """保存模型

    :param configs: 配置信息
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    :param amp_scaler: 使用的自动混合精度
    :param save_model_path: 模型保存路径
    :param epoch_id: 当前epoch
    :param accuracy: 当前准确率
    :param best_model: 是否为最佳模型
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 检查模型是否为分布式并行模型
        state_dict = model.module.state_dict()  # 获取模型状态字典
    else:
        state_dict = model.state_dict()
        # 保存模型的路径
    save_feature_method = configs.preprocess_conf.feature_method  # 获取特征方法
    if configs.preprocess_conf.get('use_hf_model', False):  # 检查是否使用 HF 模型
        save_feature_method = save_feature_method[:-1] if save_feature_method[-1] == '/' else save_feature_method
        save_feature_method = os.path.basename(save_feature_method)  # 提取特征方法的基本名称
    if best_model:  # 如果是最佳模型
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'best_model')  # 构建最佳模型路径
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'epoch_{}'.format(epoch_id))  # 构建当前 epoch 模型路径
    os.makedirs(model_path, exist_ok=True)  # 创建保存路径（如果不存在）
    # 保存模型参数
    torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))  # 保存优化器状态
    torch.save(state_dict, os.path.join(model_path, 'model.pth'))  # 保存模型参数
    # 自动混合精度参数
    if amp_scaler is not None:  # 如果存在混合精度缩放器
        torch.save(amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))  # 保存混合精度参数
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:  # 打开状态文件
        data = {"last_epoch": epoch_id, "accuracy": accuracy, "version": __version__,
                "model": configs.model_conf.model, "feature_method": save_feature_method}
        f.write(json.dumps(data, indent=4, ensure_ascii=False))  # 写入 JSON 数据
    if not best_model:  # 如果不是最佳模型
        last_model_path = os.path.join(save_model_path,
                                       f'{configs.model_conf.model}_{save_feature_method}', 'last_model')  # 构建最后模型路径
        shutil.rmtree(last_model_path, ignore_errors=True)  # 删除旧的最后模型
        shutil.copytree(model_path, last_model_path)  # 复制当前模型为最后模型
        # 删除旧的模型
        old_model_path = os.path.join(save_model_path,
                                      f'{configs.model_conf.model}_{save_feature_method}',
                                      'epoch_{}'.format(epoch_id - 3))  # 构建旧模型路径
        if os.path.exists(old_model_path):  # 如果旧模型存在
            shutil.rmtree(old_model_path)  # 删除旧模型
    logger.info('已保存模型：{}'.format(model_path))  # 记录已保存模型的信息

