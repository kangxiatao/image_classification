# -*- coding: utf-8 -*-

"""
Created on 02/09/2022
train_test.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""


import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from easydict import EasyDict as edict
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.data_utils import get_dataloader
from utils.common_utils import (get_logger, makedirs, process_config)
from utils import mail_log

from models import *
from models.vit import ViT
from models.convmixer import ConvMixer


def init_config():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--exp_name', type=str, default='my_test')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--opt', default="sgdm")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default='666')
    parser.add_argument('--aug', action='store_true', help='use randomaug')
    parser.add_argument('--amp', action='store_true', help='enable AMP training')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', default='vgg')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default='180')
    parser.add_argument('--patch', default='4', type=int)
    parser.add_argument('--convkernel', default='8', type=int)
    parser.add_argument('--dp', type=str, default='../../Prune/Data', help='dataset path')
    parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')

    args = parser.parse_args()

    config = edict(vars(args))

    # 输出路径
    summn = [config.exp_name]
    chekn = [config.exp_name]
    summn.append("summary/")
    chekn.append("checkpoint/")
    summary_dir = ["./runs"] + summn
    ckpt_dir = ["./runs"] + chekn
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % config.summary_dir)
    print("=> config.checkpoint_dir: %s" % config.checkpoint_dir)

    # config.send_mail_head = (' -> ' + args.exp_name + '\n')
    # config.send_mail_head = ('Wubba lubba dub dub')
    config.send_mail_head = (' -> ' + config.net + '/' + config.dataset + '\n')
    config.send_mail_str = (mail_log.get_words() + '\n')
    config.send_mail_str += "=> 我能在河边钓一整天的🐟 <=\n"
    for _key in vars(args):
        config.send_mail_str += f'{_key}:{config[_key]},'
    config.send_mail_str += "\n"

    return config


def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    logger = get_logger('log', logpath=config.summary_dir + '/')
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, use_amp):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    _lr = lr_scheduler.get_last_lr()
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (_lr, 0, 0, correct, total))
    writer.add_scalar('train/lr', _lr, epoch)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # with torch.cuda.amp.autocast(enabled=use_amp):
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        _lr = lr_scheduler.get_last_lr()
        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (_lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(net, loader, criterion, epoch, writer):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100. * correct / total

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)
    return acc


def main(config):
    # init logger
    logger, writer = init_logger(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    best_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # preprocessing
    # ====================================== get dataloader ======================================
    trainloader, testloader = get_dataloader(config.dataset, config.bs, 256, 4, config.dp, config)

    # Model
    # ====================================== build/load model ======================================
    print('==> Building model..')
    # net = VGG('VGG19')
    if config.net == 'res18':
        net = ResNet18()
    elif config.net == 'vgg':
        net = VGG('VGG19')
    elif config.net == 'res34':
        net = ResNet34()
    elif config.net == 'res50':
        net = ResNet50()
    elif config.net == 'res101':
        net = ResNet101()
    elif config.net == "convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=config.convkernel, patch_size=1, n_classes=10)
    elif config.net == "vit":
        # ViT for cifar10
        net = ViT(
            image_size=32,
            patch_size=config.patch,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif config.net == "vit_timm":
        import timm

        net = timm.create_model("vit_large_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)

    if config.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(config.pretrained)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)  # make parallel
        cudnn.benchmark = True

    # ========== train =======================
    criterion = nn.CrossEntropyLoss()

    if config.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=config.lr)
    elif config.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=config.lr)
    else:
        # optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=weight_decay)
        optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)

    # use cosine or reduce LR on Plateau scheduling
    if not config.cos:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True,
                                                               min_lr=1e-3 * 1e-5, factor=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.n_epochs)

    for epoch in range(start_epoch, config.n_epochs):
        train(net, trainloader, optimizer, criterion, scheduler, epoch, writer, config.amp)
        test_acc = test(net, testloader, criterion, epoch, writer)

        if test_acc > best_acc and epoch > 10:
            print('Saving..')
            state = {'net': net,
                     'acc': test_acc,
                     'epoch': epoch,
                     'config': config}
            path = os.path.join(config.checkpoint_dir, 'finetune_%s_best.pth.tar' % (config.net))
            torch.save(state, path)
            best_acc = test_acc
            best_epoch = epoch

        if config.cos:
            scheduler.step(epoch - 1)

    logger.info('best acc: %.4f, epoch: %d' % (best_acc, best_epoch))
    print_inf = 'best acc: %.4f, epoch: %d\n' % (best_acc, best_epoch)


    config.send_mail_str += print_inf

    QQmail = mail_log.MailLogs()
    QQmail.sendmail(config.send_mail_str, header=config.send_mail_head)


if __name__ == '__main__':
    config = init_config()
    main(config)


