import os
import argparse
import torch
import torch.utils.data as Data
from torchvision import transforms

from dataset.dataset import VisDA2019, VisDA2019UnlabeledPair
from domainnet.model import *
from domainnet.single_train_adapt import single_train_adapt
from domainnet.single_train_noadapt import single_train_noadapt
from domainnet.single_test import single_test
from utils.utils import WeightEMA

parser = argparse.ArgumentParser(description= 'Mixup Domain Adaptation')
parser.add_argument('--seed', type= int, default= 1)
parser.add_argument('--data_root', type= str, default='../data/VisDA2019')
parser.add_argument('--source', type= str, default='infograph')
parser.add_argument('--target', type= str, default='quickdraw')
parser.add_argument('--batchsize', type= int, default= 64)
parser.add_argument('--epochs', type= int, default= 200)
parser.add_argument('--save_model', action= 'store_true', default= False)
parser.add_argument('--snapshot', type= str, default= 'snapshot/')
parser.add_argument('--pretrain', type= str, default= 'pretrain/')
parser.add_argument('--lr', type= float, default= 0.01)
parser.add_argument('--gpu_id', type= int, default= 0)
parser.add_argument('--T', type= float, default= 0.5)
parser.add_argument('--ema_decay', type= float, default= 0.999)
parser.add_argument('--num_classes', type= int, default= 345)
parser.add_argument('--log_interval', type= int, default= 100)
parser.add_argument('--net', type= str, default= 'alexnet')

args = parser.parse_args()
root = args.data_root
seed = args.seed
snapshot = args.snapshot
gpu_id = args.gpu_id
source = args.source
target = args.target
batchsize = args.batchsize
epochs = args.epochs
num_classes = args.num_classes
pretrain = args.pretrain

if not os.path.exists(snapshot):
    os.makedirs(snapshot)

torch.manual_seed(seed)

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:' + str(gpu_id) if use_gpu else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(227),
    transforms.ToTensor(),
    transforms.Normalize([0.717, 0.711, 0.687], [0.334, 0.327, 0.346])
])

val_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize([0.717, 0.711, 0.687], [0.334, 0.327, 0.346])
])

train_label_dataset = VisDA2019(root= root, domain= source, split= 'train', transform= train_transform)
train_unlabel_dataset = VisDA2019UnlabeledPair(root= root, domain= target, split= 'train', transform= train_transform)
val_dataset = VisDA2019(root= root, domain= target, split= 'test', transform= val_transform)
train_dataset = VisDA2019(root= root, domain= source, split= 'test', transform= val_transform)
train_label_loader = Data.DataLoader(train_label_dataset, batch_size= batchsize, shuffle= True, num_workers= 4, drop_last= True)
train_unlabel_loader = Data.DataLoader(train_unlabel_dataset, batch_size= batchsize, shuffle= True, num_workers= 4, drop_last= True)
val_loader = Data.DataLoader(val_dataset, num_workers= 4)
train_loader = Data.DataLoader(train_dataset, num_workers= 4)

# extractor = WideResNetFeature().to(device)
# ema_extractor = WideResNetFeature().to(device)
# classifier = WideResNetClassifier(num_classes= num_classes).to(device)
# ema_classifier = WideResNetClassifier(num_classes= num_classes).to(device)
extractor = AlexNetFeature().to(device)
ema_extractor = AlexNetFeature().to(device)
classifier = AlexNetClassifier(num_classes= num_classes).to(device)
ema_classifier = AlexNetClassifier(num_classes= num_classes).to(device)

for param in ema_extractor.parameters():
    param.detach_()
for param in ema_classifier.parameters():
    param.detach_()

ema_extractor_optimizer = WeightEMA(extractor, ema_extractor, device, args.ema_decay)
ema_classifier_optimizer = WeightEMA(classifier, ema_classifier, device, args.ema_decay)

if os.path.exists(pretrain):
    try:
        extractor.load_state_dict(torch.load(pretrain + source + '_' + target + '_extractor.pth'))
        classifier.load_state_dict(torch.load(pretrain + source +  '_' + target + '_classifier.pth'))
    except Exception as e:
        print(e)

for epoch in range(epochs):
    # single_train_adapt(train_label_loader, train_unlabel_loader, extractor, classifier, ema_extractor_optimizer,
    #                    ema_classifier_optimizer, device, epoch, args, val_loader= None)
    single_train_noadapt(train_label_loader, extractor, classifier, device, epoch, args, val_loader= None)
    acc= single_test(val_loader, extractor, classifier, device)
    print("Validation Acc: %f" % acc)
    acc = single_test(train_loader, extractor, classifier, device)
    print("Train Acc: %f" % acc)
