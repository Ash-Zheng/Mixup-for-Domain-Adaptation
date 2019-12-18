import torch
import math
import tqdm
import numpy
import random
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter

from digits.loss import *
from digits.single_test import single_test


# writer = SummaryWriter()  

def one_hot(ten):
    len = ten.size(0)
    wide = ten.size(1)
    flag = torch.zeros(len, 1)
    for i in range(len):
        temp = 0
        # pls = 0
        for j in range(wide):
            if ten[i][j].item() > temp:
                temp = ten[i][j].item()
                # pls = j
            # ten[i][j] = 0.0
        if temp > 0.7:  # 自己设定的mixup开启的阈值
            flag[i] = 1.0
        # ten[i][pls] = 1.0
    return flag


def data_augmentation(image, dev):  # data_augmentation
    batch = image.size(0)
    channel = image.size(1)
    length = image.size(2)
    width = image.size(3)
    tag = random.randint(0, 1)
    result = torch.zeros_like(image)
    if tag == 0:
        noise = torch.randn((batch, channel, length, width)) / 6.0  # 高斯噪声
        noise = noise.to(dev)
        result = image + noise
    if tag == 1:
        for i in range(batch):
            one_img = image[i].cpu()
            PIL_image = transforms.ToPILImage()(one_img)
            transed = transforms.RandomRotation(5)(PIL_image)
            transed = transforms.ToTensor()(transed)
            transed.to(dev)
            result[i] = transed
    return result


def single_train_adapt(train_s_loader, train_t_loader, extractor, classifier, ema_extractor_optimizer,
                       ema_classifier_optimizer, device, epoch, args, val_loader=None):
    # acc = single_test(val_loader, ema_extractor_optimizer.ema_model, ema_classifier_optimizer.ema_model, device)
    # print('Epoch: %d, Accuracy: %f, Best Accuracy: %f' % (epoch + 1, acc))

    best_acc = 0.0
    extractor.train()  # 设置train模式
    classifier.train()

    train_s_iter = iter(train_s_loader)
    train_t_iter = iter(train_t_loader)

    start_step = epoch * len(train_t_loader)
    total_step = args.epochs * len(train_t_loader)  # 总步数

    optim_extractor = optim.Adam(extractor.parameters(), lr=args.lr)  # 使用Adam优化器
    optim_classifier = optim.Adam(classifier.parameters(), lr=args.lr)

    for idx in tqdm.tqdm(range(len(train_t_loader))):
        # for idx in range(len(train_t_loader)):
        # p = (idx + start_step) * 1.0 / total_step
        # mix_coff = 2 - 2 / (1 + math.exp(-10 * p))
        # mix_coff1 = mix_coff
        # mix_coff2 = mix_coff
        # p = 1.0 - p
        # mix_coff = 1 - math.exp(-10 * p) / (math.exp(-10 * p) + 1)
        alpha = 0.75
        # mix_coff = 0.1
        mix_coff1 = numpy.random.beta(alpha, alpha)
        if mix_coff1 < 0.5:
            mix_coff1 = 1 - mix_coff1
        mix_coff2 = numpy.random.beta(alpha, alpha)
        if mix_coff2 < 0.5:
            mix_coff2 = 1 - mix_coff2

        # writer.add_scalar('data/mix_coff', mix_coff1, idx + start_step)
        try:
            s_img1, s_label1 = next(train_s_iter)
            s_img2, s_label2 = next(train_s_iter)
        except:
            train_s_iter = iter(train_s_loader)
            s_img1, s_label1 = next(train_s_iter)
            s_img2, s_label2 = next(train_s_iter)

        try:
            t_img, _ = next(train_t_iter)
        except:
            train_t_iter = iter(train_t_loader)
            t_img, _ = next(train_t_iter)

        s_img1, s_img2 = s_img1.to(device), s_img2.to(device)
        t_img = t_img.to(device)
        s_label1 = torch.zeros(args.batchsize, args.num_classes).scatter_(1, s_label1.view(-1, 1), 1)
        s_label2 = torch.zeros(args.batchsize, args.num_classes).scatter_(1, s_label2.view(-1, 1), 1)
        s_label1 = s_label1.to(device)
        s_label2 = s_label2.to(device)

        optim_extractor.zero_grad()
        optim_classifier.zero_grad()

        t_out = classifier(extractor(t_img))
        guess_label = torch.zeros_like(t_out)
        # comput pseudo labels for target samples

        extractor.eval()  # 设置测试模式
        classifier.eval()
        with torch.no_grad():  # label_guessing
            for i in range(args.aug_times):
                guess_img = data_augmentation(t_img, device)
                guess_out = classifier(extractor(guess_img))
                guess_label += guess_out

            # t_out = classifier(extractor(t_img))
            # t_pred = torch.softmax(t_out, dim=1)  # 求出当前预测值
            guess_label /= args.aug_times
            t_pred = torch.softmax(guess_label, dim=1)
            t_pred_sharp = t_pred ** (1 / args.T)  # 求平方
            t_pse_label = t_pred_sharp / t_pred_sharp.sum(dim=1, keepdim=True)  # sharpen
            t_pse_label = t_pse_label.detach()
            mix_flag = one_hot(t_pse_label)

            if idx % 100 == 99:
                cnt = 0  # 测试有多少样本被mixup
                for i in range(64):
                    if mix_flag[i] > 0:
                        cnt += 1
                print(cnt/64.0)

        extractor.train()  # 设置train模式
        classifier.train()
        # mixup
        mix_img1 = mix_coff1 * s_img1 + (1 - mix_coff1) * t_img
        mix_label1 = mix_coff1 * s_label1 + (1 - mix_coff1) * t_pse_label
        mix_img2 = mix_coff2 * s_img2 + (1 - mix_coff2) * t_img
        mix_label2 = mix_coff2 * s_label2 + (1 - mix_coff2) * t_pse_label

        batchsize = s_img1.size()[0]
        for i in range(batchsize):
            if mix_flag[i].item() < 0.5:
                mix_img1[i] = s_img1[i]
                mix_label1[i] = s_label1[i]
                mix_img2[i] = s_img2[i]
                mix_label2[i] = s_label2[i]

        mix_img1 = mix_coff1 * s_img1 + (1 - mix_coff1) * t_img
        mix_label1 = mix_coff1 * s_label1 + (1 - mix_coff1) * t_pse_label
        mix_img2 = mix_coff2 * s_img2 + (1 - mix_coff2) * t_img
        mix_label2 = mix_coff2 * s_label2 + (1 - mix_coff2) * t_pse_label


        mix_img1_show = vutils.make_grid(mix_img1, normalize=True, scale_each=True)
        mix_img2_show = vutils.make_grid(mix_img2, normalize=True, scale_each=True)
        # writer.add_image('Image/mix_img1', mix_img1_show, idx + start_step)
        # writer.add_image('Image/mix_img2', mix_img2_show, idx + start_step)

        mix_img = torch.cat([mix_img1, mix_img2], dim=0)
        mix_label = torch.cat([mix_label1, mix_label2], dim=0)

        # interleave labeled and unlabeled samples between batches of get correct batchnorm calculation
        # mix_img = list(torch.split(mix_img, args.batchsize))
        # mix_img = interleave(mix_img, batchsize= args.batchsize)

        mix_pred = classifier(extractor(mix_img))
        loss = cls_loss(mix_pred, mix_label) + L2norm_loss(mix_pred, mix_label)
        loss.backward()
        # writer.add_scalar('data/loss', loss.item(), idx + start_step)
        optim_extractor.step()
        optim_classifier.step()
        ema_extractor_optimizer.step()
        ema_classifier_optimizer.step()

        if idx == len(train_t_loader) - 1:
            print('Epoch: %d, Loss: %f' % (epoch + 1, loss.item()))

    if val_loader is not None:
        acc = single_test(val_loader, ema_extractor_optimizer.ema_model, ema_classifier_optimizer.ema_model, device)
        print('Epoch: %d, Accuracy: %f, Best Accuracy: %f' % (epoch + 1, acc, best_acc))

        if acc > best_acc:
            best_acc = acc
