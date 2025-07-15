import torch
import random
from copy import deepcopy
class EmbedsSampler:
  def __init__(self,feature_path:str,device='cuda'):
    self.device = device
    '''dict{class_name:embeds},embeds shape is [n_samples,1024]'''
    self.vif = torch.load(feature_path, map_location=device, weights_only=False)  # vision features

  def first(self):
    first_embeds = {}
    for k,v in self.vif.items():
      pos_embeds = v[0][None,:]
      neg_embeds = torch.zeros_like(pos_embeds)
      first_embeds[k] = [torch.cat([neg_embeds[None,:],pos_embeds[None,:]],dim=0).to(self.device)]
    return first_embeds

  def mean(self,n_samples=50):
    mean_embeds = {}
    for k,v in self.vif.items():
      pos_embeds = v.mean(dim=0)[None,:]
      neg_embeds = torch.zeros_like(pos_embeds)
      mean_embeds[k] = [[torch.cat([neg_embeds[None,:],pos_embeds[None,:]],dim=0).to(self.device)]]*n_samples
    return mean_embeds

  def high_density(self):
    high_density_embeds = {}
    for k, v in self.vif.items():
      mean = v.mean(dim=0)
      dists = ((v - mean) ** 2).sum(dim=1)
      idx = torch.argmin(dists)
      pos_embeds = v[idx][None, :]
      neg_embeds = torch.zeros_like(pos_embeds)
      high_density_embeds[k] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
    return high_density_embeds

  def high_density_random(self, k=10):
      high_density_embeds = {}
      for class_name, v in self.vif.items():
          mean = v.mean(dim=0)
          dists = ((v - mean) ** 2).sum(dim=1)
          topk_indices = torch.topk(dists, k=min(k, len(dists)), largest=False).indices
          idx = random.choice(topk_indices.tolist())
          pos_embeds = v[idx][None, :]
          neg_embeds = torch.zeros_like(pos_embeds)
          high_density_embeds[class_name] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
      return high_density_embeds

  def high_density_synthetic(self, k=10):
      high_density_embeds = {}
      for class_name, v in self.vif.items():
          mean = v.mean(dim=0)
          dists = ((v - mean) ** 2).sum(dim=1)
          topk_indices = torch.topk(dists, k=min(k, len(dists)), largest=False).indices
          topk_embeds = v[topk_indices]
          region_mean = topk_embeds.mean(dim=0)
          region_std = topk_embeds.std(dim=0)
          # 生成新样本
          pos_embeds = torch.normal(region_mean, region_std)[None, :]
          neg_embeds = torch.zeros_like(pos_embeds)
          high_density_embeds[class_name] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
      return high_density_embeds

  def sample_for_one_class(self,class_name):
      embeds = self.vif[class_name]
      mean = embeds.mean(dim=0)
      std = embeds.std(dim=0)
      pos_embeds = torch.normal(mean, std)[None, :]
      neg_embeds = torch.zeros_like(pos_embeds)
      new = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
      return new
  def synthetic_from_all(self):
      synthetic_embeds = {}
      for class_name, v in self.vif.items():
          mean = v.mean(dim=0)
          std = v.std(dim=0)
          pos_embeds = torch.normal(mean, std*3)[None, :]
          neg_embeds = torch.zeros_like(pos_embeds)
          synthetic_embeds[class_name] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
      return synthetic_embeds

  def high_likelihood_sample(self, max_trials=100, sigma=1.0):
      synthetic_embeds = {}
      for class_name, v in self.vif.items():
          mean = v.mean(dim=0)
          std = v.std(dim=0)
          for _ in range(max_trials):
              sample = torch.normal(mean, std)
              # 判断是否所有维度都在[mean-sigma*std, mean+sigma*std]内
              if torch.all((sample >= mean - sigma * std) & (sample <= mean + sigma * std)):
                  pos_embeds = sample[None, :]
                  neg_embeds = torch.zeros_like(pos_embeds)
                  synthetic_embeds[class_name] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
                  break
          else:
              # 如果max_trials内都没采到，直接用均值
              pos_embeds = mean[None, :]
              neg_embeds = torch.zeros_like(pos_embeds)
              synthetic_embeds[class_name] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
      return synthetic_embeds


  def low_density_kth_transform(self, k=1, noise_scale=0.5):
      synthetic_embeds = {}
      for class_name, v in self.vif.items():
          mean = v.mean(dim=0)
          dists = ((v - mean) ** 2).sum(dim=1)
          sorted_indices = torch.argsort(dists, descending=True)
          if k > len(sorted_indices):
              idx = sorted_indices[-1]
          else:
              idx = sorted_indices[k-1]
          base = v[idx]
          noise = torch.randn_like(base) * noise_scale * base.std()
          pos_embeds = (base + noise)[None, :]  # shape [1, 1024]
          neg_embeds = torch.zeros_like(pos_embeds)  # shape [1, 1024]
          synthetic_embeds[class_name] = [torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)]
      return synthetic_embeds


  def density_based_sample_pca(self, k=50, n_samples=10, n_components=0.9, seed=None, noise_scale=0.1,temperature=1):
      from sklearn.decomposition import PCA
      synthetic_embeds = {}
      for class_name, v in self.vif.items():
          # 降维到n_components维
          pca = PCA(n_components=n_components)
          v_low = torch.tensor(pca.fit_transform(v.cpu().numpy()), device=v.device)
          # 在低维空间做密度估计
          dist_matrix = torch.cdist(v_low, v_low)
          knn_dists, _ = torch.topk(dist_matrix, k=k+1, largest=False)
          knn_dists = knn_dists[:, 1:]
          '''temperature为正:低似然采样，越大越平滑越趋近均匀采样，越低在低密度区域越尖锐
            temperature为负:高似然采样，绝对值越大越平滑越趋近均匀采样，越小在高密度区域越尖锐概率越大
            为正时协变量偏移
            为负时变为语义偏移
          '''
          assert temperature!=0
          density = torch.exp(-knn_dists.mean(dim=1)/temperature)
          prob = density / (density.sum() + 1e-8)
          print(knn_dists.mean().item())
          print(prob.max().item(),prob.min().item(),prob.mean().item())
          # 采样
          generator = torch.Generator(device=self.device).manual_seed(seed)  if seed is not None else None

          idx = torch.multinomial(prob, n_samples, replacement=True, generator=generator)
          sampled = v[idx]  # 返回原始高维样本
          # 对每个采样点加高斯扰动
          result = []
          for s in sampled:
              noise = torch.randn_like(s) * noise_scale * s.std()
              s_noised = s + noise
              pos_embeds = s_noised[None, :]
              neg_embeds = torch.zeros_like(pos_embeds)
              # neg_embeds = torch.randn_like(pos_embeds)
              result.append([torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)])
          synthetic_embeds[class_name] = deepcopy(result)
      return synthetic_embeds

  def distance_based_sample(self, n_samples=1):
      synthetic_embeds = {}
      for class_name, v in self.vif.items():
          mean = v.mean(dim=0)
          dists = ((v - mean) ** 2).sum(dim=1)
          # 直接按距离排序，远的点概率高
          sorted_indices = torch.argsort(dists, descending=True)
          # 取前n_samples个最远的
          idx = sorted_indices[:n_samples]
          sampled = v[idx]
          # ... 拼接逻辑

if __name__=="__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  feature_path = './output/01_extract_features/ImageNet100/ImageNet100.pt'
  es = EmbedsSampler(feature_path,device=device)
  mean = es.first()
  for k,v in mean.items():
    print(k,v[0].shape)
        