import os
from tqdm import tqdm
from PIL import Image
import json
import torch
from PIL import Image
from typing import List
import time
import random
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
def load_id_name_dict():
    with open('../imagenet1k-subset100.json') as f:
        res = json.load(f)
    return res


'''存储数据集的路径,OpenOOD的路径结构不一样'''
DATASET_PATH_DICT = {
    "ImageNet100_full": "./datasets/ImageNet100_full",

}

import sys


def in_place_print(*args, **kwargs):
    """原地打印并自动清除"""
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '')
    flush = kwargs.get('flush', True)

    output = sep.join(str(arg) for arg in args) + end
    sys.stdout.write('\r' + output)
    if flush:
        sys.stdout.flush()

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
def train_acc(model, train_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            images = torch.cat([images[0], images[1]], dim=0).cuda()
            labels = labels.repeat(2).cuda()
            output = model.encoder(images)
            output = model.classifier(output)
            # acc
            pred = output.data.max(1)[1]
            correct += pred.eq(labels.data).sum().item()
    return correct / (len(train_loader.dataset) * 2)

def get_scheduler(opt, optimizer,total_batches ,warmup_from,warmup_to):
    warmup_steps = opt.warm_epochs * total_batches
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_from / warmup_to,  # 起始比例
        end_factor=1.0,  # 结束比例
        total_iters=warmup_steps  # 总预热步数
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs * total_batches - warmup_steps,  # 剩余步数
        eta_min=opt.learning_rate * (opt.lr_decay_rate ** 3)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]  # 预热结束后切换
    )
    return scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)







