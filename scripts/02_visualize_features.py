import torch
import os
from utils import load_id_name_dict
from pprint import pprint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import umap
from utils import Visualizer
def visualize(X,y,output_dir='/content/temp'):
  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(X[:, 0], X[:, 1], cmap='jet', c=y, alpha=0.6)
  os.makedirs(output_dir,exist_ok=True)
  path = os.path.join(output_dir,'temp.png')
  plt.savefig(path)


if __name__ == "__main__":
  N = 5
  dataset = 'ImageNet100'
  device = 'cpu'
  feature_dir =  os.path.join("./output/01_extract_features",dataset,"features")
  id_name_dict = load_id_name_dict()
  class_features = {}
  for entry in os.scandir(feature_dir):
    if entry.is_dir():continue
    class_id = entry.name.split('.')[0]
    class_features[id_name_dict[class_id]] = torch.load(entry.path,map_location=device)

  # 初始化存储列表
  feature_embeds_list = []
  name_embeds_list = []
  means_list = []
  stds_list = []
  embeds_labels_list = []
  latent_labels_list = []

  for class_idx,(class_name,features) in enumerate( class_features.items() ):
    if class_idx <10:continue
    if class_idx == N+10:break
    feature_embeds = features["embeds"][1::2]           #n/2*77*768
    name_embeds = features["embeds"][0::2] 
    means = features["latent_means"]      #5*4*64*64
    stds = features["latent_stds"]        #5*4*64*64

    feature_embeds_flat = feature_embeds.view(feature_embeds.shape[0], -1)
    name_embeds_flat = name_embeds.view(name_embeds.shape[0], -1)
    means_flat = means.view(means.shape[0], -1)
    stds_flat = stds.view(stds.shape[0], -1)

    feature_embeds_list.append(feature_embeds_flat)
    name_embeds_list.append(name_embeds_flat)
    means_list.append(means_flat)
    stds_list.append(stds_flat)

    embeds_labels_list.extend([class_idx] * feature_embeds_flat.shape[0])
    latent_labels_list.extend([class_idx] * means_flat.shape[0])

  # 3. 跨类别堆叠（沿样本维度拼接）
  feature_embeds_stack = torch.cat(feature_embeds_list, dim=0)  
  name_embeds_stack = torch.cat(name_embeds_list, dim=0)
  means_stack = torch.cat(means_list, dim=0)    
  stds_stack = torch.cat(stds_list, dim=0)   
  means_stack += stds_stack
  embeds_labels = np.array(embeds_labels_list)
  latent_labels = np.array(latent_labels_list)
  print("class_nums:",N)
  print("feature embeds:",feature_embeds_stack.shape)
  print("name embeds:",name_embeds_stack.shape)
  # print(means_stack.shape)
  # print(stds_stack.shape)
  print("label",embeds_labels)
  # print(latent_labels.shape)

  '''umap'''
  reducer = umap.UMAP(
    n_components=2,
    n_neighbors=25,
    min_dist=0.1,
    metric='cosine',
    random_state=42
  )
  train_embeddings = reducer.fit_transform(name_embeds_stack)
  new_embeddings = reducer.transform(feature_embeds_stack)

  print(train_embeddings.shape)
  print(new_embeddings.shape)
  visualize(train_embeddings,embeds_labels)



  quit()

  '''K-means'''
  full_pipe = make_pipeline(
    # StandardScaler(),
    PCA(n_components=0.9),    # 降维到2D便于可视化
    # KMeans(n_clusters=N, random_state=42)
    )
  full_pipe.fit(name_embeds_stack)
  feature_pca = full_pipe.named_steps['pca'].transform(feature_embeds_stack)  # 降维结果
  name_pca = full_pipe.named_steps['pca'].transform(name_embeds_stack[:10])  # 降维结果
  print(feature_pca.shape)
  print(name_pca.shape)
  visualize(feature_pca,embeds_labels)
  quit()


  
# 运行t-SNE（以embeds为例）
  tsne1 = TSNE(n_components=2,n_iter=2000,perplexity=60, random_state=42,learning_rate=200)
  tsne2 =  TSNE(n_components=2,n_iter=2000,perplexity=60, random_state=42,learning_rate=200)
  scaler = StandardScaler()

  embeds_stack = scaler.fit_transform(embeds_stack)

  means_tsne = tsne1.fit_transform(means_stack)
  embeds_tsne = tsne2.fit_transform(embeds_stack)
 
  
  # 绘制结果（假设有类别标签列表 class_labels）
  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(means_tsne[:, 0], means_tsne[:, 1], cmap='jet', c=latent_labels, alpha=0.6)
  # plt.legend(*scatter.legend_elements(), title="Classes")
  plt.title("t-SNE Visualization of Embeds")

  plt.savefig("/content/means_cluster.png")


  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(embeds_tsne[:, 0], embeds_tsne[:, 1], cmap='jet', c=embeds_labels, alpha=0.6)
  # plt.legend(*scatter.legend_elements(), title="Classes")
  plt.title("t-SNE Visualization of Embeds")

  plt.savefig("/content/embeds_clulster.png")






  print('success')









