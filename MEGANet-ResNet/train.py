import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from lib.EGANet import EGANetModel
from utils.dataloader import PolypDataset, get_loader, test_dataset
from utils.loss import DeepSupervisionLoss
from utils.utils import AvgMeter, clip_gradient
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train(train_loader, model, optimizer, epoch, criteria_loss):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    best = 0
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            predicts = model(images)
            loss = criteria_loss(predicts, gts)

            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            clip_gradient(optimizer, opt.grad_norm)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], ' 'loss: {:.4f}'. format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))   

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if (epoch + 1) % 10 == 0: 
        torch.save(model.state_dict(), save_path + 'EGANet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'EGANet-%d.pth'% epoch)


if __name__ == '__main__':
    writer = SummaryWriter()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=0.5, help='gradient clipping norm')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--train_path', type=str,
                        default='/home/bntan/data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='eganet')
    parser.add_argument("--mgpu", type=str, default="false", choices=["true", "false"])
    opt = parser.parse_args()

    model = EGANetModel()

    if opt.mgpu == "true" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    params = model.parameters()

    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.epoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    criteria_loss = DeepSupervisionLoss()

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print(torch.cuda.get_device_name(0))
    print("#"*20, "Start Training", "#"*20)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    for epoch in range(0, opt.epoch):
        train(train_loader, model, optimizer, epoch, criteria_loss)
        scheduler.step()
        

    writer.close()