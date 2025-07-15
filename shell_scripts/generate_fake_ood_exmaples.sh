#!/bin/bash

# 进入项目根目录（假设脚本在 shell_scripts/ 下）

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
python --version


echo "Running scripts/03_generate_ood.py ..."
python scripts/03_generate_ood.py --dataset ImageNet100 --output_dir "./fake_ood_examples" --fake_num_per_class 10 --seed 0 --n_class 5 \
--k 50 --fake_num_per_class 5 --noisy_scale 0.1 --temperature -3

# 提示信息
echo "Press any key to continue..."

# 捕获单个字符输入
read -n 1 -s -r -p "" key

# 换行（因为输入不会自动换行）
echo