#!/bin/bash

# 进入项目根目录（假设脚本在 shell_scripts/ 下）

echo "current work dir: $(pwd)"

python --version


# 运行测试脚本
echo "Running test_scripts/ood_generator_test.py ..."
python test_scripts/shell_test.py

# 提示信息
echo "Press any key to continue..."

# 捕获单个字符输入
read -n 1 -s -r -p "" key

# 换行（因为输入不会自动换行）
echo