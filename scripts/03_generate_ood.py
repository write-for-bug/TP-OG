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
    parser.add_argument("--dataset",type=str,default='ImageNet100_full')
    parser.add_argument("--output_dir",type=str,default='./output/02_fake_ood')
    parser.add_argument("--fake_num_per_class",type=int,default=2)
    parser.add_argument("--feature_path",type=str,default="./output/01_extract_features/ImageNet100_features.pt")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--sd_model", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--n_class", type=int, default=100)
    parser.add_argument("--noise_scale", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--mean_group_size", type=int, default=30)
    parser.add_argument("--filter_percent", type=float, default=0.1)
    return parser.parse_args()
if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False  # 允许CuDNN使用非确定性算法
    torch.backends.cudnn.benchmark = True  # 启用自动优化（引入随机性）

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
    noise_scale = args.noise_scale
    mean_group_size = args.mean_group_size
    temperature = args.temperature
    filter_percent = args.filter_percent

    es = EmbedsSampler(feature_path, device)
    og = OODGenerator(sd_model=sd_model,vae_model=vae,device=device)
    k =args.k
    # 指定保存根目录
    save_dir = os.path.join(output_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)


    # 欧式距离计算密度
    # sampled_embeds = es.density_based_sample_pca(k=k,
    #                                              n_samples=fake_num_per_class,
    #                                              n_components=n_components,
    #                                              noise_scale=noise_scale,
    #                                              temperature=temperature,
    #                                              mean_group_size=50,
    #                                              )
    # 余弦相似度计算密度
    sampled_embeds = es.density_based_sample_cosine(k=k,
                                                    n_samples=fake_num_per_class,
                                                    temperature=temperature,
                                                    mean_group_size=mean_group_size,
                                                    noise_scale=noise_scale,
                                                    target_hit_min=0.0,
                                                    target_hit_max=0.12,
                                                    candidate_batch=100)
    random.seed(seed)
    sampled_keys = random.sample(list(sampled_embeds.keys()), min(n_class, len(sampled_embeds)))
    for synset in tqdm(sampled_keys,position=2):
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

        for i in tqdm(range(need_generate),position=3):
            try:
                # 使用对应的嵌入生成图片
                embeds = v[i+existing_count]
                images = og.generate_images_with_name(embeds, class_name,seed=seed)
                for j, image in enumerate(images):
                    img_path = os.path.join(class_dir, f"{class_name}_{existing_count + i:04d}.jpeg")
                    image.save(img_path)
            except Exception as e:
                info = f"Error generating/saving {synset} image {existing_count + i:04d}: {e}"
                print(info)