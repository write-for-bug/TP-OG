import torch
from transformers import pipeline,CLIPTextModel,CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from torchvision.transforms import v2 as transforms_v2
from PIL import Image
import os
from tqdm import tqdm
from typing import Dict, Tuple, List,Optional
from pprint import pprint
from torch.cuda.amp import autocast
DIRECTIONLESS_SYSTEM_MESSAGE = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": '''# Role:
                    vision feature extractor. Extracte the shared features in given images.
                    # Output:
                    - The broadest valid abstract category (e.g., 'bottle' to 'container','dog' to 'Quadrupeds','sizzors' to 'tool','computer' to 'electronics')
                    - Universal and detailed physical features(Combination of geometric shapes, texture, color patterns, proportions, etc.)
                    # Requirements:
                    - Exclude words appeared in given class name 
                    - No formatting symbols 
                    - Plain text
                    - Prohibit contextual relationships and spatial relationships
                    '''
                }         
              ]
        }
DIRECTIONAL_SYSTEM_MESSAGE = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": ''
                }
              ]
        }
class FeatureExtractor:
    def __init__(self,
                 vqa_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 sd_model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 tokenizer_path: str =  "openai/clip-vit-large-patch14",
                 clip_path: str = "openai/clip-vit-large-patch14",
                 transform =None,
                 cache_dir: str = None,
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.cache_dir = cache_dir
        # self.vae = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae",cache_dir=cache_dir).to(device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",cache_dir=cache_dir).to(device)
        # self.vae = AutoencoderKL.from_pretrained("hustvl/vavae-imagenet256-f16d32-dinov2",cache_dir=cache_dir).to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path,cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_path,cache_dir=cache_dir).to(device)
        if transform==None:
          self.transforms_v2 = transforms_v2.Compose([
                                            transforms_v2.Resize(512),
                                            transforms_v2.CenterCrop(448)])
        else:
          self.transforms_v2 = transform
        self._init_vqa_pipeline(vqa_model_path)



    def _init_vqa_pipeline(self, model_path: str):
        """初始化VQA模型管道"""
        self.vqa_pipe = pipeline(
            task="image-text-to-text",
            model=model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # cache_dir=self.cache_dir,
            # model_kwargs={"cache_dir":self.cache_dir},
            preprocessor_kwargs={
                "max_length": 512,
                "image_size": 448,
                "padding": "max_length",
                "detail_level": "extreme",
                "disable_dialog_memory": True,
                "cache_dir":self.cache_dir
            }
        )

    def _generate_description(self, images: List[str]|List[Image.Image], class_name: str, sys_template: str) -> str:
        """传入n张图片提取共同的features生成增强语义描述"""
        
        content = [{"type": "image", "image": img} if isinstance(img, Image.Image)
                   else {"type": "image", "image": Image.open(img)}
                   for img in images]
        content.append({"type": "text","text": f"Return the ontology and describe shared features from these images"})
        messages = [sys_template, {"role": "user", "content": content}]
        bad_tokens = ['fundamental','ontological','catgory','shared','features','ontology']
        bad_words_ids = []
        for phrase in bad_tokens:
          token_ids = self.tokenizer(phrase,add_special_tokens=False).input_ids
          bad_words_ids.append(token_ids)
        # print(bad_words_ids)
        description = self.vqa_pipe(messages, #max_new_tokens=64,
                        generate_kwargs = {
                        "max_new_tokens":64,
                        "min_new_tokens":32,
                        "no_repeat_ngram_size":2,
                        "top_p":0.95,
                        "temperature":0.5,
                        # "repeatition_penalty":1.5,
                        "bad_words_ids":bad_words_ids
                        })[0]["generated_text"]
        description = description[2]['content'].replace("\n",".",1).replace("- ","").replace("*","").strip()
        description = ' '.join(description.split())
        return description

    def _encode_latent(self, image: str|Image.Image) -> torch.Tensor:
        """VAE潜空间编码提取"""
        with autocast():
          image = Image.open(image) if isinstance(image,str) else image
          processed_img = self.image_processor.preprocess(image,height=512,width=512,resize_mode='fill').to(self.device)
          
          with torch.no_grad():
              latent_dist = self.vae.encode(processed_img).latent_dist
          del image,processed_img
          return  latent_dist.mean,latent_dist.std  # SD标准缩放因子


    def _encode_prompt(self,
                      prompt,
                      negative_prompt=None,
                      do_classifier_free_guidance=False):
        '''将提示词编码为embeds，支持Negative Prompt（默认batch size=1）'''
        # ====================== 1. 处理Positive Prompt ======================
        # 分词处理
        text_inputs = self.tokenizer(
            prompt,  # 直接处理单个字符串
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        # 注意力掩码（根据配置动态启用）
        use_attention_mask = hasattr(self.text_encoder.config,
                                    "use_attention_mask") and self.text_encoder.config.use_attention_mask
        attention_mask = text_inputs.attention_mask.to(self.device) if use_attention_mask else None

        # 文本编码
        with torch.no_grad():
            text_output = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # 取最后一层CLIP输出
            prompt_embeds = text_output.last_hidden_state if hasattr(text_output, 'last_hidden_state') else text_output[0]
            del text_output
        # ====================== 2. 处理Negative Prompt ======================
        if do_classifier_free_guidance:
            # 处理空Negative Prompt
            if negative_prompt is None:
                negative_prompt = ""

            # Negative Prompt编码
            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_ids = uncond_input.input_ids.to(self.device)

            # 注意力掩码
            uncond_attention_mask = uncond_input.attention_mask.to(self.device) if use_attention_mask else None

            # 文本编码
            with torch.no_grad():
                uncond_output = self.text_encoder(
                    uncond_ids,
                    attention_mask=uncond_attention_mask,
                    output_hidden_states=True
                )
                negative_prompt_embeds = uncond_output.last_hidden_state if hasattr(uncond_output, 'last_hidden_state') else \
                uncond_output[0]

            # 拼接正负Prompt
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
        
    @torch.inference_mode()
    def extract_features(self,
                         image_paths: str|List[str],
                         class_name: str,
                         vqa_prompt_template: str = DIRECTIONLESS_SYSTEM_MESSAGE,
                         ) :
        """
        核心特征提取方法
        Args:
            image_paths: [path1, path2]
            image_batch_size: extract shared semantics features from [image_batch_size] images
        Returns:
            {
                class_id: {
                    "descriptions": [desc1, desc2],
                    "latents": [latent1, latent2],
                    "embeddings": [emb1, emb2]
                }
            }
        """
        
        if isinstance(image_paths,str):image_paths = [image_paths]
         # 1. 图像语义描述提取
        desc = self._generate_description(image_paths, class_name, vqa_prompt_template)
                # 3. 增强提示词嵌入
        prompt_embed = self._encode_prompt(desc,class_name,do_classifier_free_guidance=True).cpu()
        means = []
        stds = []
        for i,path in enumerate(image_paths):         
            # 2. 潜空间编码提取
            mean, std = self._encode_latent(path)
            means.append(mean.half().cpu())
            stds.append(std.half().cpu())
        # latent_means = torch.cat(means,dim=0)
        # latent_stds = torch.cat(stds,dim=0)


        return desc, prompt_embed, means,stds
