import torch
import torch.nn.functional as F

def cls_loss(pred, label):
    return -torch.mean(torch.sum(F.log_softmax(pred, dim= 1) * label, dim= 1))

def L2norm_loss(pred, label):
    pred = F.softmax(pred, dim= 1)
    return torch.mean(torch.sum((pred - label) ** 2, dim= 1))

def kl_div(pred, label):
    pred = F.softmax(pred, dim= 1)
    return torch.mean(torch.sum(label * torch.log(label / (pred + 1e-6)), dim= 1))