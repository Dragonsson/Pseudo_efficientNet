
import os
import random
import time
import warnings
from collections import OrderedDict
# from losses import ComLoss
# from torch.nn.modules.loss import _WeightedLoss
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import args
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import PIL
import pandas as pd
from train_data_loader import DF_Dataset, Pseudo_Dataset


best_acc1 = 0
best_acc5 = 0


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker():
    global best_acc1
    global best_acc5
    if torch.cuda.is_available():
        args.gpu = 'cuda:0'
    else:
        raise Exception

    if args.pretrained:
        model = EfficientNet.from_pretrained(args.arch, num_classes=args.num_classes,
                                             path=args.pretrained_path)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
        over_params = {"num_classes": args.num_classes}
        model = EfficientNet.from_name(args.arch, over_params)

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            # loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=args.gpu)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = torch.tensor(checkpoint['best_acc1'])
        best_acc5 = torch.tensor(checkpoint['best_acc5'])
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
            best_acc5 = best_acc5.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        # don't load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = True 

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    image_size = EfficientNet.get_image_size(args.arch)

    all_df, val_df = create_df()
    train_df = all_df[all_df["label"] != "-1"]
    pseudo_df = all_df[all_df["label"]== "-1"]
    dataset_train = DF_Dataset(train_df, num_classes=args.num_classes, transform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, interpolation=PIL.Image.BICUBIC,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))
    # samp = RandomSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0,
                                               pin_memory=True, sampler=None)

    dataset_pseudo = Pseudo_Dataset(pseudo_df, transform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize]))
    pseudo_loader = torch.utils.data.DataLoader(dataset=dataset_pseudo, batch_size=args.batch_size * 4, shuffle=False,
                                                num_workers=0,
                                                pin_memory=True, sampler=None)
    dataset_val = DF_Dataset(val_df, num_classes=args.num_classes, transform=transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize]))
    # samp = RandomSampler(dataset_train)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch_size * 4, shuffle=False,
                                             num_workers=0,
                                             pin_memory=True, sampler=None)

    if args.eval_pth != '':
        print("=> loading checkpoint '{}'".format(args.eval_pth))
        if args.gpu is None:
            checkpoint = torch.load(args.eval_pth)
        else:
            # Map model to be loaded to specified single gpu.
            # loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.eval_pth, map_location=args.gpu)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = torch.tensor(checkpoint['best_acc1'])
        best_acc5 = torch.tensor(checkpoint['best_acc5'])
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
            best_acc5 = best_acc5.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.eval_pth, checkpoint['epoch']))
        validate(val_loader, model, criterion, args)  
        return

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
    print("use Adam and start lr is {}".format(optimizer.param_groups[0]['lr']))

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8, verbose=True, eps=1e-8)

    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 20:
            print("stop")
        train(train_loader, model, criterion, optimizer, epoch)
        acc1, acc5, avg_val_loss = validate(val_loader, model, criterion, focalLoss, args , beta)
        # 因为伪标签的数据集较大，若在正确率很低的时候，进行测试，则收益很低，并且用时很长。
        # 当然也可以前期将possibility降低，同时降低acc1的大小，这样过阈值的伪标签数据才会较多
        # 但较低的阈值带来的较多的伪标签会扰乱分布，可能会引起方差的过大扩大，所以这里对于acc1与possibility需要做权衡
        if acc1.item() > 90:
            labeles, posses, indexes = pseudo(pseudo_loader=pseudo_loader, model=model, possibility=0.96)
            pseudo_df_copy = pseudo_df.copy()
            pseudo_df_copy.loc[:, "label"] = labeles
            possibility_pse = pseudo_df_copy[indexes]
            new_train_df = pd.concat([train_df.copy(), possibility_pse])
            new_train_df = new_train_df[new_train_df["label"] != "-1"]
            # update train_loader
            dataset_train = DF_Dataset(new_train_df, num_classes=args.num_classes, transform=transforms.Compose([
                transforms.RandomResizedCrop(image_size, interpolation=PIL.Image.BICUBIC, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize]))
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True, sampler=None)

        scheduler.step(avg_val_loss)
        # remember best acc@1 and save checkpoint
        is_best = False
        if acc1.item() > best_acc1:
            if epoch > 29:
                print("pass the epoch and the best epoch is {}".format(epoch + 1))
                # print("the acc1 is {},the acc5 is {}".format(acc1.item(), acc5.item()))
                # best_metric = metric
                best_acc1 = acc1.item()
                best_acc5 = acc5.item()
                pth_file_name = os.path.join(args.train_local,
                                             'epoch_{}_acc1{:.3f}_acc5{:.3f}.pth'.format(str(epoch + 1),
                                                                                         acc1, acc5))
                print("save the check point, the best acc1 is {},the best acc5 is {}".format(best_acc1, best_acc5))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'optimizer': optimizer.state_dict(),
                }, is_best, pth_file_name, args)


def pseudo(pseudo_loader, model, possibility=0.9):
    labeles = []
    posses = []
    indexes = []
    s_time = time.time()
    with torch.no_grad():
        for i, pseudo_img in enumerate(pseudo_loader):
            images = pseudo_img["image"]
            # index = pseudo_img["index"]
            # B,P
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            pred_score = F.softmax(output.data, dim=1)
            pred_score = pred_score.topk(1, dim=1, largest=True)
            poss = pred_score.values.view(-1)
            label = pred_score.indices.view(-1)
            p = torch.full_like(poss, possibility)
            index_p = torch.ge(poss, p)
            labeles.extend(label.tolist())
            posses.extend(poss.tolist())
            indexes.extend(index_p.tolist())
    e_time = time.time()
    print("pse time is {}".format(e_time - s_time))
    return labeles, posses, indexes


def train(train_loader, model, criterion, focalLoss, optimizer, epoch, args,beta =1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, t_sample in enumerate(train_loader):
        # measure data loading time
        images = t_sample["image"]
        target = t_sample["label"]
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item() , images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss.backward()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, focalLoss, args,beta =1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, val_img in enumerate(val_loader):
            images = val_img["image"]
            target = val_img["label"]
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} loss {losses.avg:.3f}'
              .format(top1=top1, top5=top5, losses=losses))

    return top1.avg, top5.avg, losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch == 40:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 3
    elif epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    else:
        return


def save_checkpoint(state, is_best, filename, args):
    if not is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_df():
    # 目录层次如下
    # 其中0，1...为分类的目录，用于存放各个分类的图片
    """
    ├─train
    │  ├─0
    │  ├─1  
    ...  
    ├─val
    │  ├─0
    │  ├─1  
    ...
    ├─pseudo
    │  ├─0
    │  ├─1  

    """
    path_t_base = "./datasets/train_val/train"
    t_dir = os.listdir(path_t_base)
    all_train_dir = {"img_loc": [], "type": [], "label": [], "p_label": []}
    for t in t_dir:
        class_img_dir = os.listdir(path_t_base + "/" + t)
        for class_img in class_img_dir:
            all_train_dir["img_loc"].append(path_t_base + "/" + t + "/" + class_img)
            all_train_dir['type'].append("train")
            all_train_dir['label'].append(t)
            all_train_dir["p_label"].append("-1")

    path_v_base = "./datasets/train_val/val"
    v_dir = os.listdir(path_v_base)
    all_val_dir = {"img_loc": [], "type": [], "label": []}
    for v in v_dir:
        class_img_dir = os.listdir(path_v_base + "/" + v)
        for class_img in class_img_dir:
            all_val_dir["img_loc"].append(path_v_base + "/" + v + "/" + class_img)
            all_val_dir['type'].append("test")
            all_val_dir['label'].append(v)

    pseudo_path = "./datasets/train_val/pseudo"
    p_dir = os.listdir(pseudo_path)
    all_pesduo_dir = {"img_loc": [], "type": [], "label": [], "p_label": []}
    for p in p_dir:
        class_img_dir = os.listdir(pseudo_path + "/" + p)
        for class_img in class_img_dir:
            all_pesduo_dir["img_loc"].append(pseudo_path + "/" + p + "/" + class_img)
            all_pesduo_dir['type'].append("pseudo_train")
            all_pesduo_dir['label'].append("-1")
            all_pesduo_dir["p_label"].append(p)

    img_df = pd.DataFrame(all_train_dir)
    pesduo_df = pd.DataFrame(all_pesduo_dir)
    pesduo_df.reset_index(drop=True, inplace=True)
    val_df = pd.DataFrame(all_val_dir)
    all_df = pd.concat([img_df, pesduo_df])
    return all_df, val_df



if __name__ == '__main__':
    main_worker()
