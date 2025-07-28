import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from .average_meter import AverageMeter
import torch.nn.functional as F
import torch.nn as nn
from .supcon_loss import SupConLoss
import random

def compute_auroc_fpr95( id_probs, ood_probs):
    # 检查是否有足够的样本
    if len(id_probs) == 0 or len(ood_probs) == 0:
        return float('nan'), float('nan')
    
    y_true = [0] * len(id_probs) + [1] * len(ood_probs)
    y_score = id_probs + ood_probs
    
    if len(set(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_score)
            # FPR@95TPR
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            try:
                fpr95 = fpr[tpr >= 0.95][0]
            except IndexError:
                fpr95 = 1.0
        except Exception as e:
            print(f"Warning: Error computing AUROC/FPR95: {e}")
            auroc = float('nan')
            fpr95 = float('nan')
    else:
        auroc = float('nan')
        fpr95 = float('nan')
    return auroc, fpr95




class TrainEngine:
    def __init__(self, writer,ood_label):
        self.writer = writer
        self.ood_label = ood_label
    def train(self, train_loader, model, optimizer, scheduler, epoch, opt,loss_criterion):
        model.train()
        loss_meter = AverageMeter()

        optimizer.zero_grad()
        id_probs = []
        ood_probs = []
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for idx, (images, label_dict) in pbar:
            labels = label_dict['class_idx'].cuda()
            ood_mask = torch.tensor([x == 'OOD' for x in label_dict['risk_group']]).cuda()

            labels[ood_mask] = self.ood_label
            
            bsz = labels.shape[0]

            # 统一处理所有样本
            images = images.reshape(bsz*2, 3, 224, 224).cuda()
            labels = labels.repeat(2).cuda()
            
            # 前向传播
            proj_features,logits = model(images)
            # 计算损失
            raw_loss, loss_dict = loss_criterion(
                logits, labels,proj_features
            )

            # 梯度累积
            loss = raw_loss / opt.accum_iter
            loss.backward()
            
            if (idx + 1) % opt.accum_iter == 0 or (idx + 1) == len(train_loader):#梯度累加
                loss_meter.update(loss.item(), bsz)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if (idx + 1) % opt.accum_iter != 0:continue
                # 记录详细损失
                global_step = epoch * len(train_loader) + idx
                self.writer.add_scalar('train/loss_total', raw_loss, global_step)
                self.writer.add_scalar('train/loss_ce', loss_dict['loss_ce'], global_step)
                self.writer.add_scalar('train/loss_sc', loss_dict['loss_sc'], global_step)
                self.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            
            # 统计概率
            with torch.no_grad():
                probs_batch = F.softmax(logits, dim=1)
                max_probs, _ = probs_batch.max(dim=1)
                
                # 分离ID和OOD样本的概率
                id_mask = ~ood_mask.repeat(2)
                ood_mask_repeat = ood_mask.repeat(2)
                
                if id_mask.sum() > 0:
                    id_probs.extend(max_probs[id_mask].detach().cpu().numpy().tolist())
                if ood_mask_repeat.sum() > 0:
                    ood_probs.extend(max_probs[ood_mask_repeat].detach().cpu().numpy().tolist())
            
            if idx % 50 == 0:torch.cuda.empty_cache()

            if opt.dry_run:break
        # 计算AUROC和FPR95
        auroc, fpr95 = compute_auroc_fpr95(id_probs, ood_probs)
        self.writer.add_scalar('train/auroc', auroc, epoch)
        self.writer.add_scalar('train/fpr95', fpr95, epoch)
        print(f'[Train] Epoch {epoch} AUROC: {auroc:.4f} FPR@95TPR: {fpr95:.4f}')
        return loss_meter.avg

    def train_acc(self, model, train_loader,opt):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, label_dict in tqdm(train_loader,desc="Train Eval"):
                images,labels = images.cuda(),label_dict['class_idx'].cuda()
                ood_mask = torch.tensor([x == 'OOD' for x in label_dict['risk_group']]).cuda()
                id_mask = ~ood_mask
                
                # 只计算ID样本的准确率
                if id_mask.sum() > 0:
                    id_images = images[id_mask]
                    id_labels = labels[id_mask]
                    id_bsz = id_images.shape[0]
                    
                    # 重复标签以匹配TwoCropTransform的输出
                    id_labels = id_labels.repeat(2).cuda()
                    id_images = id_images.reshape(id_bsz*2, 3, 224, 224).cuda()
                    # 统一使用model()调用
                    proj_features, output = model(id_images)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(id_labels.data).sum().item()
                    total += id_labels.size(0)
                    

                
                if opt.dry_run:break
        return correct / total if total > 0 else 0.0

class TestEngine:
    def __init__(self, writer,ood_label):
        self.writer = writer
        self.ood_label = ood_label
    def test(self, model, test_loader, epoch=None,opt=None):
        model.eval()
        loss_avg = 0.0
        correct = 0
        id_probs = []
        ood_probs = []
        id_labels = []
        ood_labels = []
        total_samples = 0
        with torch.no_grad():
            for data,target in tqdm(test_loader,desc="Test Eval"):
                data = data.to('cuda')
                labels = target['class_idx'].to('cuda')
                ood_mask = torch.tensor([x == 'OOD' for x in target['risk_group']]).cuda()
                id_mask = ~ood_mask
                
                # forward - 统一使用model()调用
                proj_features, output = model(data)
                
                # 只对ID样本计算损失和准确率
                if id_mask.sum() > 0:
                    id_output = output[id_mask]
                    id_labels_batch = labels[id_mask]
                    loss = F.cross_entropy(id_output, id_labels_batch)
                    loss_avg += float(loss.data)
                    
                    # accuracy
                    pred = id_output.data.max(1)[1]
                    correct += pred.eq(id_labels_batch.data).sum().item()
                    total_samples += id_labels_batch.size(0)
                
                # test loss average
                # softmax概率
                probs = F.softmax(output, dim=1)
                max_probs, _ = probs.max(dim=1)

                is_ood = ood_mask
                is_id = id_mask
                id_probs.extend(max_probs[is_id].detach().cpu().numpy().tolist())
                ood_probs.extend(max_probs[is_ood].detach().cpu().numpy().tolist())
                id_labels.extend([0]*is_id.sum().item())
                ood_labels.extend([1]*is_ood.sum().item())
                if opt.dry_run:break
        # 计算AUROC和FPR95
        auroc, fpr95 = compute_auroc_fpr95(id_probs, ood_probs)
        if epoch is not None:
            self.writer.add_scalar('test/loss', loss_avg / len(test_loader), epoch)
            self.writer.add_scalar('test/acc', correct / total_samples if total_samples > 0 else 0.0, epoch)
            self.writer.add_scalar('test/auroc', auroc, epoch)
            self.writer.add_scalar('test/fpr95', fpr95, epoch)
        print(f'[Test] Epoch {epoch} AUROC: {auroc:.4f} FPR@95TPR: {fpr95:.4f}')
        return loss_avg / len(test_loader), correct / total_samples if total_samples > 0 else 0.0, auroc, fpr95

    def test_acc(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, label_dict in tqdm(test_loader, desc="Test Acc"):
                labels = label_dict['class_idx']
                ood_mask = torch.tensor([x == 'OOD' for x in label_dict['risk_group']]).cuda()
                # 只计算ID样本的准确率
                id_mask = ~ood_mask
                
                if id_mask.sum() > 0:
                    id_images = images[id_mask]
                    id_labels = labels[id_mask]
                    id_images = id_images.cuda()
                    id_labels = id_labels.cuda()
                    # 统一使用model()调用
                    proj_features, output = model(id_images)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(id_labels.data).sum().item()
                    total += id_labels.size(0)

        return correct / total if total > 0 else 0.0

