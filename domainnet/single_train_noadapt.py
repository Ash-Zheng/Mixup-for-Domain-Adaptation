import os
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

from domainnet.loss import cls_loss
from domainnet.single_test import single_test

def single_train_noadapt(train_loader, extractor, classifier,
                         device, epoch, args, val_loader= None):
    best_acc = 0.0
    best_extrac = None
    best_class = None
    extractor.train()
    classifier.train()

    # optim_extractor = optim.Adam(extractor.parameters(), lr= args.lr)
    # optim_classifier = optim.Adam(classifier.parameters(), lr= args.lr)
    optim_extractor = optim.SGD(extractor.parameters(), lr = args.lr)
    optim_classifier = optim.SGD(classifier.parameters(), lr = args.lr)

    with torch.set_grad_enabled(True):
        for idx, (_, s_img, s_label) in tqdm.tqdm(enumerate(train_loader)):
            s_img, s_label = s_img.to(device), s_label.to(device)
            optim_extractor.zero_grad()
            optim_classifier.zero_grad()
            s_feature = extractor(s_img)
            s_output = classifier(s_feature)
            loss = nn.CrossEntropyLoss()(s_output, s_label)
            loss.backward()
            optim_extractor.step()
            optim_classifier.step()

            if (idx+1) % args.log_interval == 0:
                print('Epoch: %d, Batch: %d, Loss: %f' %(epoch + 1, idx + 1, loss.item()))

    if val_loader is not None:
        acc = single_test(val_loader, extractor, classifier, device)
        print('Epoch: %d, Accuracy: %f, Best Accuracy: %f' %(epoch + 1, acc, best_acc))

        if acc > best_acc:
            best_acc = acc
            best_extrac = extractor
            best_class = classifier
            if not os.path.exists(args.pretrain):
                os.makedirs(args.pretrain)
            torch.save(best_extrac.state_dict(), args.pretrain + args.source + '_' + args.target + '_extractor.pth')
            torch.save(best_class.state_dict(), args.pretrain + args.source + '_' + args.target + '_classifier.pth')



