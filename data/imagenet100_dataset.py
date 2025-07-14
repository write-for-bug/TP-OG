'''useless'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms_v2
from torchvision.datasets.folder import default_loader
from pprint import pprint
from utils import load_id_name_dict
from typing import Union, Optional,List,Tuple
class ImageNetDataset(Dataset):
    def __init__(self, root, split='train',transform:Optional[transforms_v2.Compose]=None):
        self.root = os.path.join(root,split)
        if transform is None:
          self.transform = transforms_v2.Compose([
            transforms_v2.Resize(256),
            transforms_v2.CenterCrop(224),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        id_name_dict = load_id_name_dict()
        self.classes = []
        self.class_to_idx = []
        self.samples = []  # 初始化样本列表
        # 遍历每个类别目录
        for class_idx, class_dir in enumerate(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_dir)
            # 确保处理的是目录
            if not os.path.isdir(class_path):continue
            class_name = id_name_dict[class_dir]
            self.classes.append(class_name)
            self.class_to_idx.append((class_name,class_idx))
            # 收集图像文件路径
            with os.scandir(class_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.samples.append((entry.path, class_idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, int], Tuple[str, int]] :
        path, target = self.samples[index]
        image = default_loader(path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


