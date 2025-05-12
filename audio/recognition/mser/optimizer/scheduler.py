import math
from typing import List


class WarmupCosineSchedulerLR:
    def __init__(self, optimizer, min_lr, max_lr, warmup_epoch, fix_epoch, step_per_epoch):
        self.optimizer = optimizer  # 保存优化器实例
        assert min_lr <= max_lr  # 确保最小学习率小于等于最大学习率
        self.min_lr = min_lr  # 设置最小学习率
        self.max_lr = max_lr  # 设置最大学习率
        self.warmup_step = warmup_epoch * step_per_epoch  # 计算预热阶段的总步数
        self.fix_step = fix_epoch * step_per_epoch  # 计算保持阶段的总步数
        self.current_step = 0.0  # 初始化当前步数

    def set_lr(self, ):
        new_lr = self.clr(self.current_step)  # 计算当前学习率
        for param_group in self.optimizer.param_groups:  # 遍历优化器的参数组
            param_group['lr'] = new_lr  # 更新学习率
        return new_lr  # 返回新的学习率

    def step(self, step=None):
        if step is not None:
            self.current_step = step  # 如果提供了步数，则更新当前步数
        new_lr = self.set_lr()  # 设置新的学习率
        self.current_step += 1  # 增加当前步数
        return new_lr  # 返回新的学习率

    def clr(self, step):
        if step < self.warmup_step:  # 处于预热阶段
            return self.min_lr + (self.max_lr - self.min_lr) * \
                (step / self.warmup_step)  # 线性增加学习率
        elif self.warmup_step <= step < self.fix_step:  # 处于保持阶段
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (step - self.warmup_step) /
                              (self.fix_step - self.warmup_step)))  # 余弦衰减学习率
        else:  # 超过保持阶段，恢复到最小学习率
            return self.min_lr

    def get_last_lr(self) -> List[float]:
        return [self.clr(self.current_step)]  # 返回当前学习率的列表
