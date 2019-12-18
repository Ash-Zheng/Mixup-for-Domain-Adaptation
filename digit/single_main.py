import os
import argparse
import torch
import torch.utils.data as Data
from torchvision import  transforms, datasets

from digits.model import *
from digits.single_test import single_test
from digits.single_train_noadapt import single_train_noadapt
from digits.single_train_adapt import single_train_adapt
from utils.utils import WeightEMA

parser = argparse.ArgumentParser(description= 'Mixup Domain Adaptation')
parser.add_argument('--seed', type= int, default= 1)
parser.add_argument('--data_root', type= str, default= '../data/')
parser.add_argument('--source', type= str, default= 'mnist')
parser.add_argument('--target', type= str, default= 'mnistm')
parser.add_argument('--batchsize', type = int, default= 64)
parser.add_argument('--epochs', type= int, default= 100)
parser.add_argument('--save_model', action= 'store_true', default= False)
parser.add_argument('--snapshot', type= str, default= 'snapshot/')
parser.add_argument('--pretrain', type= str, default= 'pretrain/')
parser.add_argument('--lr', type= float, default= 0.01)
parser.add_argument('--gpu_id', type= int, default=0)
parser.add_argument('--T', type= float, default= 0.5)
parser.add_argument('--ema_decay', type= float, default= 0.999)
parser.add_argument('--num_classes', type= int, default= 10)
parser.add_argument('--log_interval', type= int, default= 100)
parser.add_argument('--net', type= str, default= 'mnist')

args = parser.parse_args()
root = args.data_root  
seed = args.seed  
snapshot = args.snapshot 
gpu_id = args.gpu_id  
source = args.source  # 源域 MNIST
target = args.target  # 目标域 MNIST_M
batchsize = args.batchsize  
epochs = args.epochs  
num_classes = args.num_classes  
pretrain = args.pretrain  
net = args.net  

if not os.path.exists(snapshot):  
    os.makedirs(snapshot)

torch.manual_seed(seed)  # 设置随机数种子

use_gpu = torch.cuda.is_available()  # 检测是否可用GPU
device = torch.device('cuda:' + str(gpu_id) if use_gpu else 'cpu')  

train_transform_mapping = {'mnist': transforms.Compose([  # 训练集transform
    transforms.ToTensor(),
    transforms.Normalize(mean= (0.1307,), std= (0.3081,))]), 
'mnistm': transforms.Compose([
    transforms.RandomCrop(28),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean= (0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))]),  
}

val_transform_mapping = {'mnist': transforms.Compose([  # 测试集transform
    transforms.ToTensor(),
    transforms.Normalize(mean= (0.1307,), std= (0.3081,))]),
'mnistm': transforms.Compose([
    transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize(mean= (0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))]),
}


train_datasets_mapping = {'mnist': datasets.MNIST(root= root, transform= train_transform_mapping['mnist']),
                    'mnistm': datasets.ImageFolder(root= root + 'MNIST_M/train/', transform= train_transform_mapping['mnistm'])}

val_datasets_mapping = {'mnist': datasets.MNIST(root= root, train= False, transform= val_transform_mapping['mnist']),
                    'mnistm': datasets.ImageFolder(root= root + 'MNIST_M/test/', transform= val_transform_mapping['mnistm'])}

train_dataloader_mapping = {'mnist': Data.DataLoader(train_datasets_mapping['mnist'], batchsize, shuffle= True, num_workers= 4, drop_last= True),  # 子进程4个，舍弃最后一组多余数据
                      'mnistm': Data.DataLoader(train_datasets_mapping['mnistm'], batchsize, shuffle= True, num_workers= 4, drop_last= True)}

val_dataloader_mapping = {'mnist': Data.DataLoader(val_datasets_mapping['mnist'], 1, num_workers= 4),
                      'mnistm': Data.DataLoader(val_datasets_mapping['mnistm'], 1, num_workers= 4)}

train_s_loader = train_dataloader_mapping[source]  # 源域loader
train_t_loader = train_dataloader_mapping[target]  # 目标域loader
val_loader = val_dataloader_mapping[target]  # 测试集loader

extractor_mapping = {'mnist': MNIST_Extractor}
classifier_mapping = {'mnist': MNIST_Classfier}

extractor = extractor_mapping[net]().to(device)  
ema_extractor = extractor_mapping[net]().to(device)
classifier = classifier_mapping[net]().to(device)  
ema_classifier = classifier_mapping[net]().to(device)

# ###################
for param in ema_extractor.parameters():
    param.detach_()  
for param in ema_classifier.parameters():
    param.detach_()

# 预训练模型
if os.path.exists(pretrain):
    try:
        extractor.load_state_dict(torch.load(pretrain + source + '_' + target + '_extractor.pth'))  # 加载预训练模型
        classifier.load_state_dict(torch.load(pretrain + source + '_' + target + '_classifier.pth'))
    except Exception as e:
        print(e)

ema_extractor_optimizer = WeightEMA(extractor, ema_extractor, device, args.ema_decay)
ema_classifier_optimizer = WeightEMA(classifier, ema_classifier, device, args.ema_decay)

# for epoch in range(epochs):  # 主循环
for epoch in range(epochs):
    single_train_adapt(train_s_loader, train_t_loader, extractor, classifier, ema_extractor_optimizer,
                       ema_classifier_optimizer, device, epoch, args, val_loader= val_loader)
    # single_train_noadapt(train_s_loader, extractor, classifier, device, epoch, args, val_loader= val_loader)
    # single_test(val_loader, extractor, classifier, device)


