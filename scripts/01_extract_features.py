import os
from GWM import FeatureExtractor
from collections import defaultdict
from pprint import pprint
import torch
from data import OODDataset,ClassBalancedSampler
from torch.utils.data import DataLoader
from utils import load_id_name_dict,DATASET_PATH_DICT
from tqdm import tqdm
import argparse

def config():
    parser = argparse.ArgumentParser(description='Extract features by clip vision model with  projection to unet.Results will be saved in ./output/01_extract_features')
    parser.add_argument("--dataset",type=str,default='ImageNet100')
    parser.add_argument("--output_dir",type=str,default='01_extract_features')
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--save_file",type=str,default="ImageNet100_features")
    parser.add_argument("--device",type=str,default="cuda:0")
    return parser.parse_args()

if __name__ =='__main__':
  args = config()
  bs=args.batch_size
  device = args.device
  dataset = args.dataset
  id_name_dict=load_id_name_dict()
  save_dir = os.path.join("./output",args.output_dir)
  os.makedirs(save_dir,exist_ok=True)
  save_file=os.path.join(save_dir,f"{args.save_file}.pt")
  extractor = FeatureExtractor(cache_dir='./pretrained_models',device=device)
  try:
    dataset_root = DATASET_PATH_DICT[dataset]
  except Exception as e:
    print(str(e))
  dataset = OODDataset( root=dataset_root, 
                        split='train',
                        subset=None,
                        return_type="path")
  sampler = ClassBalancedSampler(dataset=dataset,
                                  batch_size=bs,
                                  shuffle=False,
                                  drop_last=False)
  dataloader = DataLoader(dataset, 
                          batch_sampler=sampler,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True)
  
  features = defaultdict(list)
  for image_paths, label in tqdm(dataloader):
    class_id = label[0]
    with torch.amp.autocast('cuda'):
      embeds = extractor.extract_features(image_paths).half()
    
    features[class_id].append(embeds.detach().cpu())
  for k,v in features.items():
    features[k] = torch.cat(v,dim=0)
  torch.save(features,save_file)
  print(f"Saving embeds to {save_file}")
  
      


    



  



  