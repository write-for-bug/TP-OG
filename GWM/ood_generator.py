from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
import torch
class OODGenerator:
  def __init__(self,
            sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            ip_adapter="h94/IP-Adapter",
            adapter_weight_name="ip-adapter_sd15_light.bin",
            cache_dir="./pretrained_models",
            vae_model = "stabilityai/sd-vae-ft-mse",
            device="cuda"):
    vae = AutoencoderKL.from_pretrained(vae_model,cache_dir=cache_dir,torch_dtype=torch.float16)

    self.sd_model = sd_model
    self.sdpipe = StableDiffusionPipeline.from_pretrained(
                      sd_model,
                      vae=vae,
                      cache_dir=cache_dir,
                      torch_dtype=torch.float16,
                      safety_checker=None,
                     ).to(device)
    self.sdpipe._progress_bar_config={"disable": False}
    self.sdpipe.load_ip_adapter(
      ip_adapter,
      subfolder="models",
      weight_name=adapter_weight_name,
      cache_dir=cache_dir
    )

  def generate_images_with_name(self,ip_adapter_image_embeds,class_name,seed=None):
    generator = torch.manual_seed(seed) if seed is not None else None
    images = self.sdpipe(
      prompt=f"8K resolution,Ultra high definition,photorealistic",
      negative_prompt=f"{class_name},blurry",
      num_images_per_prompt=1,
      ip_adapter_image_embeds=ip_adapter_image_embeds,
      num_inference_steps=50,
      guidance_scale=10,
      eta=0.1,
      ip_adapter_scale=0.8,
      height = 512,
      weight = 512,
      generator=generator
    ).images
    return images




  




