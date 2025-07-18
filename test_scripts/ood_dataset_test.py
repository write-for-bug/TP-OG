import os
from data import OODDataset
from torch.utils.data import  DataLoader
from torchvision.transforms import v2 as transforms_v2
from pprint import pprint
if __name__ == '__main__':
      transform = transforms_v2.Compose([
        transforms_v2.Resize(256),
        transforms_v2.CenterCrop(224),
        transforms_v2.ToTensor(),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      dataset = OODDataset(
        root='./datasets/ImageNet100_full',
        split='train',
        subset=None,
        transform=transform,
        return_type='path'
      )
      loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)
    
      for idx, (_, label) in enumerate(dataset):
        pprint(_)
        pprint(label)
        if idx==4:
          break