import torch
from utils import load_id_name_dict
from copy import deepcopy
from tqdm import tqdm
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

  def _min_max_scale(self,t:torch.Tensor):
      min_val = t.min()
      max_val = t.max()
      return 2 * (t - min_val) / (max_val - min_val) -1

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


  def density_based_sample_pca(self, k=50, n_samples=5, mean_group_size=10, n_components=0.9, seed=None, noise_scale=0.1, temperature=1):
      assert abs(temperature) > 1e-2
      from sklearn.decomposition import PCA
      synthetic_embeds = {}
      for class_name, v in self.vif.items():
          # 降维
          pca = PCA(n_components=n_components)
          v_low = torch.tensor(pca.fit_transform(v.cpu().numpy()), device=v.device)
          # 密度估计
          dist_matrix = torch.cdist(v_low, v_low)
          knn_dists, _ = torch.topk(dist_matrix, k=k+1, largest=False)
          knn_dists = knn_dists[:, 1:]
          density = torch.exp(knn_dists.mean(dim=1) / temperature)
          prob = density / (density.sum() + 1e-8)
          max_val = prob.max().item()
          min_val = prob.min().item()
          mean_val = prob.mean().item()
          print("{:>8.3f} | {:>8.3f} | {:>8.3f}".format(max_val, min_val, mean_val))
          result = []
          for _ in range(n_samples):
              # 每个样本都采 mean_group_size 个嵌入
              idx = torch.multinomial(prob, mean_group_size, replacement=True)
              sampled = v[idx]  # [mean_group_size, 1024]
              mean_embed = sampled.mean(dim=0)
              noise = torch.randn_like(mean_embed) * noise_scale * mean_embed.std()
              s_noised = mean_embed + noise
              pos_embeds = s_noised[None, :]
              neg_embeds = torch.zeros_like(pos_embeds)
              result.append([torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)])
          synthetic_embeds[class_name] = deepcopy(result)
      return synthetic_embeds


  def density_based_sample_cosine(
    self, k=50, n_samples=5, mean_group_size=10, seed=None, noise_scale=0.1, temperature=1,
    filter_percent=0.2, target_hit_min=0.05, target_hit_max=0.1, candidate_batch=20, max_iter=1000
):
    synthetic_embeds = {}
    for class_name, v in tqdm(self.vif.items(),desc="Sampling Embeds...",position=1,):
        # if class_name not in ("n02797295"):continue
        v_norm = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = torch.mm(v_norm, v_norm.t())
        cosine_dist = 1 - cosine_sim

        # 1. 计算原始分布的knn均值
        knn_dists, _ = torch.topk(cosine_dist, k=k + 1, largest=False)
        knn_dists = knn_dists[:, 1:]
        all_knn_means = knn_dists.mean(dim=1)
        sorted_knn_means, _ = torch.sort(all_knn_means, descending=True)
        threshold_idx = max(0, int(len(sorted_knn_means) * filter_percent) - 1)
        edge_threshold = sorted_knn_means[threshold_idx].item()

        density = torch.exp(-self._min_max_scale(all_knn_means) / temperature)
        prob = density / (density.sum() + 1e-8)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        result = []
        cur_noise_scale = noise_scale
        p_par = tqdm(desc=f"Sampling {load_id_name_dict()[class_name]}, noise={cur_noise_scale}",position=2,leave=False)
        while len(result) < n_samples :
            # 批量采样
            batch_embeds = []
            batch_knn_means = []
            for _ in range(candidate_batch):
                idx = torch.multinomial(prob, mean_group_size, replacement=True)
                sampled = v[idx]
                mean_embed = sampled.mean(dim=0)
                noise = torch.randn_like(mean_embed) * cur_noise_scale * mean_embed.std()
                s_noised = mean_embed + noise
                s_noised_norm = s_noised / (s_noised.norm() + 1e-8)
                dists = 1 - torch.mv(v_norm, s_noised_norm)
                knn_mean = torch.topk(dists, k=k, largest=False).values.mean().item()
                batch_embeds.append(s_noised)
                batch_knn_means.append(knn_mean)
            # 命中边缘的
            hits = [(e, m) for e, m in zip(batch_embeds, batch_knn_means) if m >= edge_threshold]
            hit_rate = len(hits) / candidate_batch
            # 动态调整noise_scale
            if hit_rate < target_hit_min:
                cur_noise_scale += 0.02
            elif hit_rate > target_hit_max:
                cur_noise_scale -= 0.04
            # 添加命中样本
            for e, m in hits:
                if len(result) < n_samples:
                    p_par.update(1)
                    pos_embeds = e[None, :]
                    neg_embeds = torch.zeros_like(pos_embeds)
                    result.append([torch.cat([neg_embeds[None, :], pos_embeds[None, :]], dim=0).to(self.device)])

        synthetic_embeds[class_name] = deepcopy(result)

    return synthetic_embeds

 
if __name__=="__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  feature_path = './output/01_extract_features/ImageNet100/ImageNet100.pt'
  es = EmbedsSampler(feature_path,device=device)
  mean = es.first()
  for k,v in mean.items():
    print(k,v[0].shape)
        