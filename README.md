# OOD样本生成的大致思路：
1. 用与训练好的'h94/IP-Adapter' CLIPVisionModelWithProjection提取每个类别的图像特征
2. 对每个类别的聚类进行边缘采样
   - **采样算法(embeds_sampler)**
      - 遍历每个类
        - 计算这个类中所有样本的余弦距离，1-cosine_dist作为样本之间的距离
        - 取前k个最近的样本的嵌入
        - 将这k个嵌入降序排序，计算两个上下限阈值的比例（因为越稀疏embeds的knn越大就越要往边缘去采样l和r就应该更极端，不然很容易生成ID）
          - l_percent = torch.exp(-all_knn_means.mean()*10).item()
          - r_percent = torch.exp(-all_knn_means.mean()*5).item()
        - 根据下面这两个公式计算每个样本被采样的概率，knn越大越稀疏的embeds被采样的概率越大
          - density = torch.exp(-self._min_max_scale(all_knn_means) / temperature)
          - prob = density / (density.sum() + 1e-8)
        - 然后根据上面的概率随机采样n个样本计算均值再加上一个noise作为新的采样向量，共candidate_batch组采样向量
          - 然后计算这个新的向量在原始的嵌入里面的knn距离
          - 保留下在l和r范围之间的向量
          - 计算命中率
        - 命中率过低就逐渐增加noise，过高就减少noise
        - 采样够了samples数量就停止
3. 用下面这几个预训练好的模型替换sd的组件去生成尽可能符合原数据集风格的OOD样本，直接把类名作为negative_prompt传入sdpipe强迫生成远离原本类别的样本
` sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
ip_adapter="h94/IP-Adapter",
adapter_weight_name="ip-adapter_sd15_light.bin",
vae_model = "stabilityai/sd-vae-ft-mse", `
4. 有的时候可能会生成一些far ood，如果需要extreme near的ood可以把Ontology作为正向提示词
- 比如dog可以写成quadruped（四足生物）
- water bottle可以写成Cylinder object或者container
  - 可以手动写或者用一些多模态模型生成


老师可以帮帮忙吗不走平台写不出来论文了QAQ



