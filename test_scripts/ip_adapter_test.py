from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection,CLIPFeatureExtractor,CLIPVisionModelWithProjection,CLIPImageProcessor
# Load model directly
import torch
from PIL import Image
from diffusers.models.embeddings import ImageProjection
from diffusers.utils import load_image
from pprint import pprint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
   
  ip_image = Image.open('test.jpeg')
  imge_processor =  CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14",cache_dir="./pretrained_models/")
  '''图像编码器'''
  image_encoder = CLIPVisionModelWithProjection.from_pretrained( "h94/IP-Adapter",
                                                              cache_dir='./pretrained_models',
                                                              subfolder='models/image_encoder'
                                                              ).to(device)
  image = imge_processor(ip_image, return_tensors="pt").pixel_values.to(device)
  single_image_embeds = image_encoder(image).image_embeds.to(device)
  uncond_image_embeds = torch.zeros_like(single_image_embeds).to(device)

  single_image_embeds = torch.cat([uncond_image_embeds[None,:], single_image_embeds[None,:]], dim=0)
  ip_adapter_image_embeds = [single_image_embeds]

  sdpipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',cache_dir='./pretrained_models',requires_safety_checker=False).to(device)

  # 挂载 IP-Adapter
  sdpipe.load_ip_adapter(
      "h94/IP-Adapter", 
      subfolder="models", 
      weight_name="ip-adapter_sd15.bin",
      cache_dir='./pretrained_models'
  )
  image_embeds1 = sdpipe.prepare_ip_adapter_image_embeds(
                                    ip_image,
                                    ip_adapter_image_embeds=None,
                                    device=device,
                                    num_images_per_prompt=1,
                                    do_classifier_free_guidance=True,
                                    )


  images = sdpipe(
      prompt="a cute rabbit in vacation,4K",
      negative_prompt="blurry, deformed, low resolution",
      num_images_per_prompt= 1,
      ip_adapter_image=ip_image,
      # ip_adapter_image_embeds=ip_adapter_image_embeds,                       # 参考图
      # negative_prompt="blurry, low quality",            # 负面提示
      num_inference_steps=50,
      guidance_scale=8,
      ip_adapter_scale=0.7,  # 控制参考图影响力（0.5~1.2）
      do_classifier_free_guidance=True
  ).images
  for i,image in enumerate(images):
    image.save(f"/content/output_{i}.jpg")


