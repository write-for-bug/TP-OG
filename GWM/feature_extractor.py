from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from typing import List, Union
import torch
from PIL import Image
from transformers import pipeline

class FeatureExtractor:
    def __init__(self,
                 clip_path="openai/clip-vit-large-patch14",
                 clip_vision_model='h94/IP-Adapter',
                 cache_dir='./pretrained_models',
                 device='cuda'):
        self.device = device
        self.cache_dir = cache_dir
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_path,cache_dir=cache_dir)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_vision_model,
                                                                           cache_dir=cache_dir,
                                                                            torch_dtype=torch.float16,
                                                                           subfolder="models/image_encoder").to(device)
    @torch.inference_mode()
    def extract_vision_features(self, image_paths: Union[str, List[str]]) -> torch.Tensor:

        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images = [Image.open(path) for path in image_paths]
        inputs = self.image_processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)
        del images
        image_embeds = self.image_encoder(**inputs).image_embeds

        return image_embeds






