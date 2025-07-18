from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from typing import List, Union
import torch
from PIL import Image
from transformers import pipeline,CLIPTokenizer
from typing import Optional

class FeatureExtractor:
    def __init__(self,
                 clip_path="openai/clip-vit-large-patch14",
                 clip_vision_model='h94/IP-Adapter',
                 i2t_model: str = "Salesforce/blip2-opt-2.7b",
                 cache_dir='./pretrained_models',
                 device='cuda'):
        self.device = device
        self.cache_dir = cache_dir
        self.blip2_pipe = pipeline("image-text-to-text", model=i2t_model,device=device,torch_dtype=torch.float16)

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






if __name__ == "__main__":
    extractor = FeatureExtractor()
    images=[
        "D:/zhanglijie/Projects/TP-OG/datasets/ImageNet100_full/train/n09421951/n09421951_196.JPEG",
        "D:/zhanglijie/Projects/TP-OG/datasets/ImageNet100_full/train/n09421951/n09421951_214.JPEG",
        "D:/zhanglijie/Projects/TP-OG/datasets/ImageNet100_full/train/n09421951/n09421951_347.JPEG"
    ]
    a = extractor._generate_description(images,"sandbar")
    print(a)