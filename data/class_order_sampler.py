'''辅助dataloader的顺序采样器，确保按类别顺序加载数据，用于提取语义特征和图片潜空间统计特征'''

from torch.utils.data.sampler import Sampler
import numpy as np

class ClassOrderSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # 按类别分组样本索引
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        # 按类别顺序展开索引
        self.ordered_indices = []
        for label in sorted(self.class_indices.keys()):
            self.ordered_indices.extend(self.class_indices[label])

    def __iter__(self):
        # 确保每个batch内的样本来自同一类别
        for i in range(0, len(self.ordered_indices), self.batch_size):
            batch_indices = self.ordered_indices[i:i + self.batch_size]
            # 检查是否跨类别（若剩余样本不足batch_size则允许）
            if len(set(self.dataset[idx][1] for idx in batch_indices)) > 1:
                continue  # 跳过跨类别的batch
            yield from batch_indices

    def __len__(self):
        return len(self.ordered_indices)