import os
import json
import tarfile
from tqdm import tqdm
# 路径配置
json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imagenet1k-subset100.json')
tar_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'ILSVRC2012_img_train')
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'ImageNet100', 'train')

# 创建输出目录
os.makedirs(out_dir, exist_ok=True)

# 读取类别列表
with open(json_path, 'r', encoding='utf-8') as f:
    subset = json.load(f)

synsets = list(subset.keys())

for synset in tqdm(synsets):
    tar_file = os.path.join(tar_dir, f'{synset}.tar')
    target_dir = os.path.join(out_dir, synset)
    if not os.path.exists(tar_file):
        print(f"[警告] 未找到类别 {synset} 的tar包: {tar_file}")
        continue
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"[跳过] {synset} 已解压，跳过。")
        continue
    os.makedirs(target_dir, exist_ok=True)
    print(f"解压 {tar_file} 到 {target_dir} ...")
    with tarfile.open(tar_file, 'r') as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(path, members, numeric_owner=numeric_owner)
        safe_extract(tar, path=target_dir)
    print(f"[完成] {synset} 解压完成。")
print("全部完成！")
