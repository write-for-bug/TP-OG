from GWM.ood_generator import OODGenerator
from GWM.embeds_sampler import EmbedsSampler
from utils import load_id_name_dict
import torch
import random
import os
from tqdm import tqdm

import argparse
def config():
    parser = argparse.ArgumentParser(description='Generate fake ood data from extracted features.')
    parser.add_argument("--dataset",type=str,default='ImageNet100')
    parser.add_argument("--output_dir",type=str,default='./output/02_fake_ood')
    parser.add_argument("--fake_num_per_class",type=int,default=2)
    parser.add_argument("--feature_path",type=str,default="./output/01_extract_features/ImageNet100_features.pt")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n_components", type=float, default=0.9)
    parser.add_argument("--sd_model", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--n_class", type=int, default=100)
    parser.add_argument("--noisy_scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--mean_group_size", type=int, default=200)
    return parser.parse_args()
if __name__ == "__main__":
    args = config()
    fake_num_per_class = args.fake_num_per_class
    output_dir = args.output_dir
    dataset = args.dataset
    device = args.device
    id_name_dict = load_id_name_dict()
    feature_path = args.feature_path
    sd_model = args.sd_model
    vae = args.vae
    seed = args.seed
    n_class = args.n_class
    noisy_scale = args.noisy_scale
    mean_group_size = args.mean_group_size
    temperature = args.temperature

    es = EmbedsSampler(feature_path, device)
    og = OODGenerator(sd_model=sd_model,vae_model=vae,device=device)
    k =args.k
    n_components = args.n_components
    # 指定保存根目录
    save_dir = os.path.join(output_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)


    # 选择采样方式
    sampled_embeds = es.density_based_sample_pca(k=k,
                                                 n_samples=fake_num_per_class,
                                                 n_components=n_components,
                                                 noise_scale=noisy_scale,
                                                 temperature=temperature,
                                                 mean_group_size=50,
                                                 )

    sampled_keys = random.sample(list(sampled_embeds.keys()), min(n_class, len(sampled_embeds)))
    for synset in tqdm(sampled_keys):
        v = sampled_embeds[synset]
        class_name = id_name_dict[synset]
        class_dir = os.path.join(save_dir, synset)
        # 检查已存在的图片数量
        existing_count = 0
        if os.path.exists(class_dir):
            existing_count = len([f for f in os.listdir(class_dir) if f.endswith('.jpeg')])

        if existing_count >= fake_num_per_class:
            print(f"\nSkipping {synset}: already has {existing_count} images (target: {fake_num_per_class})")
            continue
            
        # 计算需要生成的数量
        need_generate = fake_num_per_class - existing_count
        print(f"\n{synset}: existing {existing_count}, need generate {need_generate}")
        
        os.makedirs(class_dir, exist_ok=True)

        for i in tqdm(range(need_generate)):
            try:
                # 使用对应的嵌入生成图片
                embeds = v[i+existing_count]
                images = og.generate_images_with_name(embeds, class_name)
                for j, image in enumerate(images):
                    img_path = os.path.join(class_dir, f"{class_name}_{existing_count + i:04d}.jpeg")
                    image.save(img_path)
            except Exception as e:
                info = f"Error generating/saving {synset} image {existing_count + i:04d}: {e}"
                print(info)