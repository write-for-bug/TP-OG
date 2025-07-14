
from torch.utils.data import Dataset, Sampler
from typing import  Dict, List
import random



class ClassBalancedSampler(Sampler):
    """
    按类别平衡的采样器，确保每个batch中的label一致

    Args:
        dataset: OODDataset实例
        batch_size: batch大小
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的batch
    """

    def __init__(self, dataset: 'OODDataset', batch_size: int, shuffle: bool = True, drop_last: bool = False):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 按类别组织样本索引
        self.class_to_indices = {}
        for idx, (_, class_name, _, _) in enumerate(dataset.samples):
            if class_name not in self.class_to_indices:
                self.class_to_indices[class_name] = []
            self.class_to_indices[class_name].append(idx)

        # 计算每个类别的batch数量
        self.class_batch_counts = {}
        self.class_remainders = {}
        for class_name, indices in self.class_to_indices.items():
            total_samples = len(indices)
            num_batches = total_samples // batch_size
            remainder = total_samples % batch_size

            if not drop_last and remainder > 0:
                num_batches += 1
                self.class_remainders[class_name] = remainder
            else:
                self.class_remainders[class_name] = 0

            self.class_batch_counts[class_name] = num_batches

        # 验证batch_size不超过最小类别的样本数
        min_samples = min(len(indices) for indices in self.class_to_indices.values()) if self.class_to_indices else 0
        if batch_size > min_samples:
            print(f"警告: batch_size ({batch_size}) 大于最小类别的样本数 ({min_samples})")
            print("这可能导致某些类别无法生成完整的batch")

        # 生成batch索引序列
        self.batch_indices = self._generate_batch_indices()

    def _generate_batch_indices(self) -> List[List[int]]:
        """生成batch索引序列"""
        batch_indices = []

        for class_name, indices in self.class_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)

            # 创建完整的batch
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batch_indices.append(indices[i:i + self.batch_size])

            # 处理剩余的样本（如果drop_last=False）
            remainder = len(indices) % self.batch_size
            if remainder > 0 and not self.drop_last:
                start_idx = len(indices) - remainder
                batch_indices.append(indices[start_idx:])

        # 打乱batch顺序
        if self.shuffle:
            random.shuffle(batch_indices)

        return batch_indices

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)

    def get_class_statistics(self) -> Dict[str, Dict]:
        """获取每个类别的统计信息"""
        stats = {}
        for class_name, indices in self.class_to_indices.items():
            total_samples = len(indices)
            num_batches = self.class_batch_counts[class_name]
            remainder = self.class_remainders[class_name]

            stats[class_name] = {
                'total_samples': total_samples,
                'num_batches': num_batches,
                'remainder': remainder,
                'avg_samples_per_batch': total_samples / num_batches if num_batches > 0 else 0
            }
        return stats

    def get_batch_distribution(self) -> Dict[str, int]:
        """获取batch分布统计"""
        batch_dist = {}
        for batch in self.batch_indices:
            if len(batch) > 0:
                # 获取batch中第一个样本的类别
                _, class_name, _, _ = self.dataset.samples[batch[0]]
                batch_dist[class_name] = batch_dist.get(class_name, 0) + 1
        return batch_dist