import os
from GWM import FeatureExtractor
import torch

if __name__ =='__main__':

  extractor = FeatureExtractor(cache_dir='./pretrained_models')
  vae = extractor.vae
  vqa_pipe = extractor.vqa_pipe
  text_encoder = extractor.text_encoder
  print(vae.training)
  # print(vqa_pipe.training)
  print(text_encoder.training)
  print(torch.is_inference_mode_enabled())