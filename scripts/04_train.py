
import os
from tqdm import tqdm
import argparse
from torchvision.datasets import CIFAR10, CIFAR100
import time
import math
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import numpy as np

from data import OODDataset
from utils import save_model,TwoCropTransform,SupConLoss,TrainEngine,TestEngine,get_scheduler,set_seed
from networks.resnet_largescale import StandardResNet, StandardResNetBase, SupStandardResNet, SupConResNetLargeScale
from networks.resnet_big import StandardResnet_CIFAR, SupStandardResnet_CIFAR

from torch.utils.tensorboard import SummaryWriter




def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # 基础参数
    parser.add_argument('--experiment_name', type=str, default='', help='experiment name')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--dry_run', action='store_true', help='quick debug run')

    # 优化器参数
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--accum_iter', type=int, default=8, help='gradient accumulation steps')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--warm_epochs', type=int, default=10, help='momentum')

    # 模型/数据参数
    parser.add_argument('--model', type=str, default='resnet50', help='model name')
    parser.add_argument('--dataset', type=str, default='ImageNet100',
                        choices=['cifar10', 'cifar100', 'ImageNet100', 'path', 'ImageNet100_baseline'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets/ImageNet100/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='image size for RandomResizedCrop')
    parser.add_argument('--ood_path', type=str, help='out of distribution data path')

    # 方法参数
    parser.add_argument('--method', type=str, default='vision-text', help='method name')

    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')

    opt = parser.parse_args()



    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = 'datasets'
    opt.model_path = f'./save/{opt.dataset}_models'

    # 生成模型名
    opt.model_name = f'{opt.method}_{opt.dataset}_{opt.model}_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}_expname_{opt.experiment_name}'

    # warm-up for large-batch training
    if opt.warm:
        opt.model_name = f'{opt.model_name}_warm'
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt




def set_loader(opt):
    # 1. 均值/方差设定
    if opt.dataset in ['cifar10', 'cifar100']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset in ['ImageNet100', 'ImageNet100_baseline']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError(f'dataset not supported: {opt.dataset}')

    # 2. 数据增强/预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # ±15°随机旋转
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        ], p=0.7),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 3. 数据集构建
    if opt.dataset == 'cifar10':
        train_dataset = CIFAR10(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)
        test_dataset = CIFAR10(root=opt.data_folder, train=False, transform=test_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = CIFAR100(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)
        test_dataset = CIFAR100(root=opt.data_folder, train=False, transform=test_transform)
    elif opt.dataset == 'ImageNet100':
        train_dataset = OODDataset(root=opt.data_folder, split='train', ood_paths=opt.ood_path, transform=TwoCropTransform(train_transform))
        test_dataset = OODDataset(root=opt.data_folder, split='val', transform=test_transform)
    elif opt.dataset == 'ImageNet100_baseline':
        train_dataset = OODDataset(root=opt.data_folder, split='train', transform=TwoCropTransform(train_transform))
        test_dataset = OODDataset(root=opt.data_folder, split='val', transform=test_transform)

    else:
        raise ValueError(f'dataset not supported: {opt.dataset}')

    # 4. DataLoader构建
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.test_batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    return train_loader, test_loader


def set_model(opt):
    if opt.model == 'resnet18' or opt.model == 'resnet34':
        model = SupStandardResnet_CIFAR(name=opt.model, dataset=opt.dataset)
    elif opt.model == 'resnet50' or opt.model == 'resnet101':
        model = SupConResNetLargeScale(name=opt.model,class_num=100)
    elif opt.model == 'resnet50_base':
        model = StandardResNetBase(name='resnet50')
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
            model.classifier = torch.nn.DataParallel(model.classifier)
        model = model.cuda()
        cudnn.benchmark = True
    return model


def main():
    set_seed(0)
    opt = parse_option()
    writer = SummaryWriter(log_dir=f'./runs/{opt.model}_{opt.experiment_name}')
    train_loader, test_loader = set_loader(opt)
    train_engine = TrainEngine(writer)
    test_engine = TestEngine(writer)

    # 对比损失器
    criterion_supcon = SupConLoss(temperature=0.1).cuda()
    model = set_model(opt)
    optimizer = torch.optim.SGD([{'params': model.parameters()}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # 调度器
    scheduler = get_scheduler(opt, optimizer, len(train_loader),warmup_from=opt.learning_rate/10,warmup_to=opt.learning_rate)


    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        loss = train_engine.train(train_loader, model, criterion_supcon, optimizer, epoch, opt)
        scheduler.step()
        train_acc = train_engine.train_acc(model, train_loader,opt)

        print('epoch: {} | learning_rate: {:.6f} | loss: {:.4f} | train_acc: {:.4f}'.format(epoch, optimizer.param_groups[0]['lr'], loss, train_acc))
        writer.add_scalar('train/train_acc', train_acc, epoch)
        test_loss, test_acc, test_auroc, test_fpr95 = test_engine.test(model, test_loader, epoch,opt)
        print('epoch: {} | test loss: {:.4f} | test_acc: {:.4f} | AUROC: {:.4f} | FPR@95TPR: {:.4f}'.format(epoch, test_loss, test_acc, test_auroc, test_fpr95))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    writer.close()


if __name__ == '__main__':
    main()