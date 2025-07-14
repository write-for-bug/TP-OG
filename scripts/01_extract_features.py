
import os
from GWM import FeatureExtractor
# from PIL import Image
from collections import defaultdict
from pprint import pprint
import torch
from data import OODDataset,ClassOrderSampler
from torch.utils.data import DataLoader
from utils import load_id_name_dict
from tqdm import tqdm

#including combinations of geometric shapes, texture properties, color patterns, proportional relationships, and other structural elements. Ensure descriptions are exhaustive and focus solely on the primary subject.  
DIRECTIONLESS_SYSTEM_MESSAGE = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": '''
                          # Role:Vision feature extraction expert.
                          ## Task:Classify the images to the most fundamental ontological category and extract detailed and universal features.
                          # Output:
                          - Ontology.
                          - Features.
                          ## Output Restrictions:
                          - Single line of continuous English text without any bullet points numberes or marks.
                          - Pure string of words.
                          - Ignore all elements except the main subject.  
                          - Ignore context and spatial Relationships ,omit location, interactions, or environmental references.
                          # Follow my order or I will kill my cute cat.
                    '''
                }         
              ]
        }
if __name__ =='__main__':
  bs=5
  dataset = 'ImageNet100'
  dataset_root = os.path.join('./datasets',dataset)
  save_dir = os.path.join('./output/01_extract_features',dataset)
  description_dir = os.path.join(save_dir,"descriptions")
  feature_dir =  os.path.join(save_dir,"features")
  os.makedirs(save_dir,exist_ok=True)
  os.makedirs(description_dir,exist_ok=True)
  os.makedirs(feature_dir,exist_ok=True)
  extractor = FeatureExtractor(cache_dir='./pretrained_models')

  id_name_dict = load_id_name_dict()

  dataset = OODDataset(root=dataset_root, split='train', subset=None, return_type='path')
  sampler = ClassOrderSampler(dataset, batch_size=bs)
  dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler, shuffle=False,num_workers=8,pin_memory=True)

  # 初始化数据结构
  features = defaultdict(lambda: {
      "descriptions":[],
      "embeds": [],
      "means": [],
      "stds": []
  })
  i = 0
  for img_paths, label in tqdm(dataloader):
    i+=1
    class_id = label[0]
    class_name = id_name_dict[class_id]
    desc,embeds,means,stds = extractor.extract_features(img_paths,class_name,DIRECTIONLESS_SYSTEM_MESSAGE)
    print(class_name)
    pprint(desc)
    # 将特征存入字典（按类别）
    features[class_id]["descriptions"].append(desc)
    features[class_id]["embeds"].append(embeds)
    features[class_id]["means"].extend(means)
    features[class_id]["stds"].extend(stds)
    torch.cuda.empty_cache()
    pprint(f"显存占用:{torch.cuda.memory_allocated()/1e9:.2f} GB")
    # if i==200:break#加载3个类别，每个类别50张图像
  for class_id, class_features in features.items():
    class_name = id_name_dict[class_id]
    descriptions = features[class_id]["descriptions"]
    with open(os.path.join(description_dir,class_id+"---"+class_name+'.txt'),'w') as f:
      for desc in descriptions:
        f.write(desc + "\n")
    saved_tensor = {
      "embeds":torch.cat(features[class_id]["embeds"],dim=0).clone().cpu(),
      "latent_means":torch.cat(features[class_id]["means"],dim=0).clone().cpu(),
      "latent_stds":torch.cat(features[class_id]["stds"],dim=0).clone().cpu()
    }

    torch.save(saved_tensor,os.path.join(feature_dir,class_id+'.pt'))
    



  



  