'''转为ood数据集加载设计的数据集加载器，兼容imagenet结构和ood结构'''
import os
import json
import yaml
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from typing import Tuple, Dict, List, Optional, Union, Set
from torchvision.transforms import v2 as transforms_v2
from tqdm import tqdm
class OODDataset(Dataset):
    """
    安全关键场景数据集加载器（兼容ImageNet结构）

    Args:
        root (str): 数据集根目录 (e.g., 'PuppyArk')
        split (str): 数据划分 ('train', 'validation', 'test')
        subset (Tuple[str]): 数据子集组合 train/('ID', 'IOOD') 或 validation/('ID', 'OOD')
        transform (Optional[object]): 图像预处理流水线
        return_type (str): 返回类型 ('tensor' 或 'Image')
        fake_ood_dir (Optional[str]): 额外生成的I-OOD数据路径 (仅训练集有效)
    """

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 ood_paths: Optional[List[str]] = None,
                 transform: Optional[object] = None,
                 fake_ood_dir: Optional[str] = None,
                 return_type="tensor"):

        self.root = root
        self.split = split
        self.id_cnt = 0
        self.ood_cnt = 0
        self.ood_sets = [ood_paths] if isinstance(ood_paths, str) else ood_paths
        if transform is None:
          self.transform = transforms_v2.Compose([
            transforms_v2.Resize((256,256)),
            transforms_v2.ToDtype(torch.float32,scale=True)
          ])
        self.fake_ood_dir = fake_ood_dir
        self.loader = default_loader
        assert return_type in ("path","tensor")
        self.return_type=return_type
        # 元数据路径
        self.metadata_dir = os.path.join(root, 'metadata')

        # 加载元数据文件
        try:
            self.risk_table = self._load_metadata('risk_table.yml')
            self.class_descriptions = self._load_metadata('class_descriptions.json')
            self.sample_weights = self._load_metadata('sample_weights.json')
        except Exception as e:
            pass
        # 核心数据结构
        self.samples = []  # (path, class_name, is_ood, risk_value)
        self.class_to_idx = {}
        self.risk_group_to_idx = {}

        # 构建数据集
        self._build_dataset_index()

        # 统计信息
        print(f"Loaded total {len(self)} samples | Classes: {len(self.class_to_idx)} ")
        print(f"Loaded {self.id_cnt} ID samples| Loaded {self.ood_cnt} OOD samples")

    def _load_metadata(self, filename: str) -> dict:
        """加载YAML/JSON元数据文件"""
        path = os.path.join(self.metadata_dir, filename)
        if not os.path.exists(path):
            return {}

        with open(path, 'r') as f:
            if filename.endswith('.json'):
                return json.load(f)
            else:
                return yaml.safe_load(f)

    def _build_dataset_index(self):
        """构建数据集索引核心逻辑"""
        # 步骤1：加载主数据集
        self._load_imagenet_structure()
        # 步骤2：加载ood
        if self.ood_sets is not None:
            for ood_set in self.ood_sets:
                self._load_samples(ood_set, 'OOD')




    def _load_imagenet_structure(self):
        """回退到ImageNet数据结构"""
        base_path = os.path.join(self.root, self.split)
        for class_name in tqdm(os.listdir(base_path),desc=f"load data from {base_path}"):
            class_path = os.path.join(base_path, class_name)
            if not os.path.isdir(class_path):
                continue

            risk_value = self.risk_table.get(class_name, 1.0)#如果需要也可以在risk table中额外编辑ID的risk value提高某项的召回率
            risk_group = 'ID'  # ImageNet结构默认为ID
            self._load_images_from_dir(class_path, class_name, risk_group, risk_value)

    def _load_samples(self, subset_path: str, risk_group: str):
        """加载OOD样本"""

        for class_name in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_path):
                print(f'{class_path} not exists')
                continue

            risk_value = self.risk_table.get(class_name, 1.0)
            self._load_images_from_dir(class_path, class_name, risk_group, risk_value)


    def _load_fake_ood_samples(self):
        """加载额外生成的I-OOD数据"""
        if not os.path.exists(self.fake_ood_dir):
            raise ValueError(f"Fake OOD directory not found: {self.fake_ood_dir}")

        for threat_class in os.listdir(self.fake_ood_dir):
            class_path = os.path.join(self.fake_ood_dir, threat_class)
            if not os.path.isdir(class_path):
                continue

            risk_value = self.risk_table.get(threat_class, 1.0)
            self._load_images_from_dir(class_path, threat_class, 'IOOD', risk_value)

    def _load_images_from_dir(self,
                              dir_path: str,
                              class_name: str,
                              risk_group: str,
                              risk_value: float):
        """从目录加载图像样本"""
        for i,img_name in enumerate(os.listdir(dir_path) ):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(dir_path, img_name)
            self._add_sample(img_path, class_name, risk_group, risk_value)

    def _add_sample(self,
                    path: str,
                    class_name: str,
                    risk_group: str,
                    risk_value: float):
        """添加样本到数据集"""
        # 更新类别索引
        if class_name not in self.class_to_idx:
            self.class_to_idx[class_name] = len(self.class_to_idx)

        # 更新风险组索引
        if risk_group not in self.risk_group_to_idx:
            self.risk_group_to_idx[risk_group] = len(self.risk_group_to_idx)

        # 添加样本
        self.samples.append((path, class_name, risk_group, risk_value))
        if risk_group == 'ID':
            self.id_cnt += 1
        if risk_group == 'OOD':
            self.ood_cnt += 1
    def __getitem__(self, index: int):# -> Tuple[Union[torch.Tensor, Image.Image], Dict]:
        """获取样本（动态返回类型）"""
        path, class_name, risk_group, risk_value = self.samples[index]


        if self.return_type=="path":

          return path,class_name
        elif self.return_type=="tensor":
          img = self.loader(path)
          if self.transform:
              image = self.transform(img)
          image_data = image

          # 构建标签字典
          label_dict = {
              'class_idx': self.class_to_idx[class_name],
              'class_name': class_name,
              'risk_group': risk_group,
              'risk_value': risk_value
          }

          return image_data, label_dict

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布统计"""
        dist = {}
        for _, class_name, _, _ in self.samples:
            dist[class_name] = dist.get(class_name, 0) + 1
        return dist

    def get_risk_group_distribution(self) -> Dict[str, int]:
        """获取风险组分布统计"""
        dist = {}
        for _, _, risk_group, _ in self.samples:
            dist[risk_group] = dist.get(risk_group, 0) + 1
        return dist