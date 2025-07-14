from diffusers import StableDiffusionPipeline
import torch
class OODGenerator:
  def __init__(self,
            sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            ip_adapter="h94/IP-Adapter",
            cache_dir="./pretrained_models",
            device="cuda"):
    self.sdpipe = StableDiffusionPipeline.from_pretrained(
                      sd_model,
                      cache_dir=cache_dir,
                      torch_dtype=torch.float16,
                      safety_checker=None,
                     ).to(device)
    self.sdpipe._progress_bar_config={"disable": True}
    self.sdpipe.load_ip_adapter(
      ip_adapter,
      subfolder="models",
      weight_name="ip-adapter_sd15_light.bin",
      cache_dir=cache_dir
    )

  def generate_images_with_name(self,ip_adapter_image_embeds,class_name,seed=None):
    generator = torch.manual_seed(seed) if seed is not None else None
    images = self.sdpipe(
      prompt=f"8K resolution,Ultra high definition,photorealistic",
      negative_prompt=f"{class_name},blurry",
      num_images_per_prompt=1,
      ip_adapter_image_embeds=ip_adapter_image_embeds,
      num_inference_steps=40,
      guidance_scale=8,
      eta=0.1,
      ip_adapter_scale=0.8,
      height = 256,
      weight = 256,
      generator=generator
    ).images
    return images

if __name__=="__main__":
  from embeds_sampler import EmbedsSampler
  from utils import load_id_name_dict
  import torch
  import os
  from tqdm import tqdm

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  id_name_dict = load_id_name_dict()
  feature_path = './output/01_extract_features/ImageNet100_features.pt'
  es = EmbedsSampler(feature_path,device)
  og = OODGenerator(device=device)

  sampled_embeds = es.density_based_sample_pca( k=100, n_samples=3, n_components=0.95)
  for k,v in tqdm(sampled_embeds.items()):
    class_name = id_name_dict[k]
    # if class_name not in ['Rock crab','Bikini','Volley ball','Water bottle','groom',]:continue
    if class_name not in ['groom','Brambling','Rock crab',"Bikini",'Volleyball','Water bottle' ]: continue
    print(f"Generating fake {class_name} images")
    for i,embeds in enumerate(v):
      images = og.generate_images_with_name(embeds,class_name)
      os.makedirs(f'./temp/ood_samples/{k}---{class_name}',exist_ok=True)
      for j,image in enumerate(images):
        image.save(f"./temp/ood_samples/{k}---{class_name}/fake_{i}.jpeg")


  




