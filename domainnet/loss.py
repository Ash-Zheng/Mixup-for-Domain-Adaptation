import torch
import torch.nn.functional as F

def cls_loss(pred, label):
    return -torch.mean(torch.sum(F.log_softmax(pred, dim= 1) * label, dim= 1))

def L2norm_loss(pred, label):
    pred = F.softmax(pred, dim=1)
    return torch.mean((pred - label) ** 2)