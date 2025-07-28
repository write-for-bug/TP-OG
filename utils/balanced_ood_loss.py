from .supcon_loss import SupConLoss
import torch
import torch.nn as nn

class BalancedOODLoss:
    def __init__(self, id_weight=1.0, ood_weight=0.1, supcon_loss:SupConLoss=None,ood_label=100):
        self.id_weight = id_weight
        self.ood_weight = ood_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.supcon_loss = supcon_loss
        self.ood_label = ood_label
    def __call__(self, outputs, labels, features=None):
        # 分离ID和OOD样本
        ood_mask = (labels == self.ood_label)
        id_mask = ~ood_mask
        
        # 计算ID样本的交叉熵损失
        if id_mask.sum() > 0:
            id_outputs = outputs[id_mask]
            id_labels = labels[id_mask]
            id_ce_loss = self.ce_loss(id_outputs, id_labels)
        else:
            id_ce_loss = torch.tensor(0.0).cuda()
        
        # 计算OOD样本的交叉熵损失
        if ood_mask.sum() > 0:
            ood_outputs = outputs[ood_mask]
            ood_labels = labels[ood_mask]
            ood_ce_loss = self.ce_loss(ood_outputs, ood_labels)
        else:
            ood_ce_loss = torch.tensor(0.0).cuda()
        
        # 加权组合交叉熵损失
        ce_loss = self.id_weight * id_ce_loss + self.ood_weight * ood_ce_loss
        
        # 对比学习损失 - 平衡ID和OOD样本
        if features is not None:
            bsz = features.size(0)//2
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features_contrast = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            labels_contrast = labels[:bsz]
            
            # 分离ID和OOD样本进行对比学习
            id_mask_contrast = (labels_contrast != 100)
            ood_mask_contrast = (labels_contrast == 100)
            
            supcon_loss = torch.tensor(0.0).cuda()
            
            # ID样本的对比学习损失
            if id_mask_contrast.sum() > 0:
                id_features = features_contrast[id_mask_contrast]
                id_labels_contrast = labels_contrast[id_mask_contrast]
                id_supcon_loss = self.supcon_loss(id_features, id_labels_contrast)
                supcon_loss += id_supcon_loss*self.id_weight
            
            # OOD样本的对比学习损失（降低权重）
            if ood_mask_contrast.sum() > 0:
                ood_features = features_contrast[ood_mask_contrast]
                ood_labels_contrast = labels_contrast[ood_mask_contrast]
                ood_supcon_loss = self.supcon_loss(ood_features, ood_labels_contrast)
                # 降低OOD样本在对比学习中的权重
                supcon_loss +=  ood_supcon_loss*self.ood_weight
        else:
            supcon_loss = torch.tensor(0.0).cuda()
        
        # 总损失
        total_loss = ce_loss + supcon_loss
        
        return total_loss, {
            'loss_ce': ce_loss.item(),
            'loss_sc': supcon_loss.item()
        }

