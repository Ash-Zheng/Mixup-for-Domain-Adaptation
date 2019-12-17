import tqdm
import torch
import math
import torch.optim as optim

from utils import *
from domainnet.loss import *
from domainnet.single_test import single_test

def single_train_adapt(train_label_loader, train_unlabel_loader, extractor, classifier, ema_extractor_optimizer,
                       ema_classifier_optimizer, device, epoch, args, val_loader= None):
    best_acc = 0.0
    extractor.train()
    classifier.train()

    train_label_iter = iter(train_label_loader)
    train_unlabel_iter = iter(train_unlabel_loader)

    start_steps = epoch * len(train_unlabel_loader)
    total_steps = args.epochs * len(train_unlabel_loader)

    optim_extractor = optim.Adam(extractor.parameters(), lr= args.lr)
    optim_classifier = optim.Adam(classifier.parameters(), lr= args.lr)

    for idx in tqdm.tqdm(range(len(train_unlabel_loader))):
        p = (idx + start_steps) * 1.0 / total_steps
        mix_coff = 2 - 2 / (1 + math.exp(-10 * p))
        try:
            _, s_img1, s_label1 = next(train_label_iter)
            _, s_img2, s_label2 = next(train_label_iter)
        except:
            train_label_iter = iter(train_label_loader)
            _, s_img1, s_label1 = next(train_label_iter)
            _, s_img2, s_label2 = next(train_label_iter)

        try:
            _, t_img1, t_img2 = next(train_unlabel_iter)
        except:
            train_unlabel_iter = iter(train_unlabel_loader)
            _, t_img1, t_img2 = next(train_unlabel_iter)

        s_img1, s_label1 = s_img1.to(device), s_label1
        s_img2, s_label2 = s_img2.to(device), s_label2
        t_img1, t_img2 = t_img1.to(device), t_img2.to(device)
        s_label1 = torch.zeros(args.batchsize, args.num_classes).scatter_(1, s_label1.view(-1, 1), 1)
        s_label2 = torch.zeros(args.batchsize, args.num_classes).scatter_(1, s_label2.view(-1, 1), 1)
        s_label1 = s_label1.to(device)
        s_label2 = s_label2.to(device)

        optim_extractor.zero_grad()
        optim_classifier.zero_grad()

        # comput pseudo label for target samples
        with torch.no_grad():
            t_out1 = classifier(extractor(t_img1))
            t_out2 = classifier(extractor(t_img2))
            t_pred = (torch.softmax(t_out1, dim= 1) + torch.softmax(t_out2, dim= 1)) / 2
            t_pred_sharp = t_pred ** (1 / args.T)
            t_pse_label = t_pred_sharp / t_pred_sharp.sum(dim= 1, keepdim= True)
            t_pse_label = t_pse_label.detach()

        # mixup
        mix_img1 = mix_coff * s_img1 + (1 - mix_coff) * t_img1
        mix_label1 = mix_coff * s_label1 + (1 - mix_coff) * t_pse_label
        mix_img2 = mix_coff * s_img2 + (1 - mix_coff) * t_img2
        mix_label2 = mix_coff * s_label2 + (1 - mix_coff) * t_pse_label

        mix_img = torch.cat([mix_img1, mix_img2], dim= 0)
        mix_label = torch.cat([mix_label1, mix_label2], dim= 0)

        # interleave labeled and unlabeled samples between batches of get correct batchnorm calculation
        # mix_img = list(torch.split(mix_img, args.batchsize))
        # mix_img = interleave(mix_img, batchsize= args.batchsize)

        mix_pred = classifier(extractor(mix_img))
        loss = cls_loss(mix_pred, mix_label)
        loss.backward()
        optim_extractor.step()
        optim_classifier.step()
        ema_extractor_optimizer.step()
        ema_classifier_optimizer.step()

        if idx == 0 and (epoch + 1) % args.log_interval == 0 :
            print('Epoch: %d, Loss: %f' %(epoch + 1, loss.item()))

    if val_loader is not None:
        acc = single_test(val_loader, ema_extractor_optimizer.ema_model, ema_classifier_optimizer.ema_model, device)
        print('Epoch: %d, Accuracy: %f, Best Accuracy: %f' % (epoch + 1, acc, best_acc))

        if acc > best_acc:
            best_acc = acc



