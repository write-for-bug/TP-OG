import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from .average_meter import AverageMeter
import torch.nn.functional as F


def compute_auroc_fpr95( id_probs, ood_probs):
    y_true = [0] * len(id_probs) + [1] * len(ood_probs)
    y_score = id_probs + ood_probs
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        # FPR@95TPR
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        try:
            fpr95 = fpr[tpr >= 0.95][0]
        except IndexError:
            fpr95 = 1.0
    else:
        auroc = float('nan')
        fpr95 = float('nan')
    return auroc, fpr95
class TrainEngine:
    def __init__(self, writer):
        self.writer = writer

    def train(self, train_loader, model, criterion_supcon, optimizer, epoch, opt):
        model.train()

        loss_meter = AverageMeter()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        optimizer.zero_grad()
        id_probs = []
        ood_probs = []
        for idx, (images, label_dict) in pbar:
            labels = label_dict['class_idx']
            ood_mask = torch.tensor([x == 'OOD' for x in label_dict['risk_group']])
            labels[ood_mask] = 100
            del label_dict
            bsz = labels.shape[0]
            labels = labels.repeat(2).cuda()
            images = images.reshape(bsz*2, 3, 224, 224).cuda()
            features_front = model.encoder(images)
            logits = model.classifier(features_front)
            features_front = model.projection(features_front)
            features_front = F.normalize(features_front, dim=1)
            loss_ce = F.cross_entropy(logits, labels)
            f1, f2 = torch.split(features_front, [bsz, bsz], dim=0)
            features_contrast = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_sc = criterion_supcon(features_contrast, labels[:bsz])
            loss = (loss_ce + loss_sc)/opt.accum_iter

            # 梯度累积核心
            loss.backward()
            loss_meter.update(loss.item(), bsz)

            # 统计ID/OOD概率
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                max_probs, _ = probs.max(dim=1)
                is_ood = (labels == 100)
                is_id = ~is_ood
                id_probs.extend(max_probs[is_id].detach().cpu().numpy().tolist())
                ood_probs.extend(max_probs[is_ood].detach().cpu().numpy().tolist())

            if (idx + 1) % opt.accum_iter== 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                # TensorBoard 日志记录
                global_step = epoch * len(train_loader) + idx
                self.writer.add_scalar('train/loss_batch', loss.item(), global_step)
                self.writer.add_scalar('train/loss_ce', loss_ce.item(), global_step)
                self.writer.add_scalar('train/loss_supcon', loss_sc.item(), global_step)
                self.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            torch.cuda.empty_cache()

            if opt.dry_run:
                break
        # 记录 epoch 级别 loss
        self.writer.add_scalar('train/loss_epoch', loss_meter.avg, epoch)

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
                labels = label_dict['class_idx']
                ood_mask = torch.tensor([x == 'OOD' for x in label_dict['risk_group']])
                labels[ood_mask] = 100
                bsz = labels.shape[0]
                labels = labels.repeat(2).cuda()
                images = images.reshape(bsz*2, 3, 224, 224).cuda()
                output = model.encoder(images)
                output = model.classifier(output)
                pred = output.data.max(1)[1]
                correct += pred.eq(labels.data).sum().item()
                total += labels.size(0)
                if opt.dry_run:
                    break
        return correct / total if total > 0 else 0.0

class TestEngine:
    def __init__(self, writer):
        self.writer = writer

    def test(self, model, test_loader, epoch=None,opt=None):
        model.eval()
        loss_avg = 0.0
        correct = 0
        id_probs = []
        ood_probs = []
        id_labels = []
        ood_labels = []
        with torch.no_grad():
            for data,target in tqdm(test_loader,desc="Test Eval"):
                data = data.to('cuda')
                labels = target['class_idx'].to('cuda')
                ood_mask = torch.tensor([x == 'OOD' for x in target['risk_group']])
                labels[ood_mask] = 100
                # forward
                output = model.encoder(data)
                output = model.classifier(output)
                loss = F.cross_entropy(output, labels)
                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(labels.data).sum().item()
                print()
                print(pred)
                print(labels.data)
                print(pred.eq(labels.data))
                # test loss average
                loss_avg += float(loss.data)
                # softmax概率
                probs = F.softmax(output, dim=1)
                max_probs, _ = probs.max(dim=1)
                # 区分ID和OOD
                # 约定: OOD标签为100，ID为0~99
                is_ood = (labels == 100)
                is_id = ~is_ood
                id_probs.extend(max_probs[is_id].detach().cpu().numpy().tolist())
                ood_probs.extend(max_probs[is_ood].detach().cpu().numpy().tolist())
                id_labels.extend([0]*is_id.sum().item())
                ood_labels.extend([1]*is_ood.sum().item())
                if opt.dry_run:
                    break
        # 计算AUROC和FPR95
        auroc, fpr95 = compute_auroc_fpr95(id_probs, ood_probs)
        if epoch is not None:
            self.writer.add_scalar('test/loss', loss_avg / len(test_loader), epoch)
            self.writer.add_scalar('test/acc', correct / len(test_loader.dataset), epoch)
            self.writer.add_scalar('test/auroc', auroc, epoch)
            self.writer.add_scalar('test/fpr95', fpr95, epoch)
        print(f'[Test] Epoch {epoch} AUROC: {auroc:.4f} FPR@95TPR: {fpr95:.4f}')
        return loss_avg / len(test_loader), correct / len(test_loader.dataset), auroc, fpr95

    def test_acc(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, label_dict in tqdm(test_loader, desc="Test Acc"):
                labels = label_dict['class_idx']
                ood_mask = torch.tensor([x == 'OOD' for x in label_dict['risk_group']])
                labels[ood_mask] = 100
                bsz = labels.shape[0]
                images = images.cuda()
                labels = labels.cuda()
                output = model.encoder(images)
                output = model.classifier(output)
                pred = output.data.max(1)[1]
                correct += pred.eq(labels.data).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

