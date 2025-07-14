from GWM.ood_generator import OODGenerator
from GWM.embeds_sampler import EmbedsSampler
from utils import load_id_name_dict
import torch
import os
from tqdm import tqdm
import logging

if __name__ == "__main__":
    fake_num_per_class = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id_name_dict = load_id_name_dict()
    feature_path = './output/01_extract_features/ImageNet100_features.pt'
    es = EmbedsSampler(feature_path, device)
    og = OODGenerator(device=device)

    # 选择采样方式
    sampled_embeds = es.density_based_sample_pca(k=200, n_samples=fake_num_per_class, n_components=0.95)

    # 指定保存根目录
    save_root = './temp/ood_samples'
    save_dir = os.path.join(save_root, 'ImageNet100')
    os.makedirs(save_dir, exist_ok=True)

    # 日志配置
    logging.basicConfig(filename='ood_generation_errors.log', level=logging.ERROR)

    for synset, v in tqdm(sampled_embeds.items()):
        class_name = id_name_dict.get(synset)
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
        
        # 只生成需要的数量
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
                logging.error(info)
                print(info)