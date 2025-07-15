from utils import Visualizer
import numpy as np
# 1. 初始化可视化器
viz = Visualizer()

# 2. 添加数据组
viz.add_data('silver_tabby', np.random.randn(100, 768), 
            color='royalblue', marker='o', label='Silver Tabby')
viz.add_data('labrador', np.random.randn(120, 768), 
            color='darkorange', marker='s', label='Labrador')
viz.add_data('dog', np.random.randn(80, 768), 
            color='firebrick', marker='^', label='Dog')
viz.add_data('new_dog', np.random.randn(50, 768), 
            color='purple', marker='*', label='New Dog Variant')

# 3. 训练UMAP (仅用前两组)
viz.fit_umap(fit_groups=['silver_tabby', 'labrador'],
            n_neighbors=20,
            min_dist=0.2,
            metric='cosine')

# 4. 投影其他数据
viz.transform_data(transform_groups=['dog', 'new_dog'])

# 5. 可视化 (带凸包和密度)
viz.visualize(
    show_groups=['silver_tabby', 'labrador', 'dog', 'new_dog'],
    convex_hull=True,
    density=True,
    save_path='./umap_visualization.png'
)

# 6. 获取嵌入结果
dog_emb = viz.get_embeddings('dog')
print(f"Dog embeddings shape: {dog_emb.shape}")

# 7. 保存所有嵌入
viz.save_embeddings('./saved_embeddings.npz')