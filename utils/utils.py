import torch
import copy

class WeightEMA(object):

    def __init__(self, model, ema_model, device, alpha= 0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = copy.deepcopy(model).to(device)

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn = False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip((self.ema_model.parameters(), self.tmp_model.parameters())):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

def interleave_offsets(batchsize, num):
    groups = [batchsize // (num + 1)] * (num + 1)
    for x in range(batchsize - sum(groups)):
        groups[-x-1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)

    assert offsets[-1] == batchsize
    return offsets

def interleave(xy, batchsize):
    num = len(xy) - 1
    offsets = interleave_offsets(batchsize, num)
    xy = [[v[offsets[p]:offsets[p+1]] for p in range(num+1)] for v in xy]
    for i in range(1, num + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim = 1) for v in xy]

def sample_unit_vec(shape, n):
    mean = torch.zeros(shape)
    std = torch.ones(shape)
    dis = torch.distributions.Normal(mean, std)
    samples = dis.sample_n(n)
    samples = samples.view(n, -1)
    samples_norm = torch.norm(samples, 2, 1).view(n, 1)
    samples = samples/samples_norm
    return samples.view(n,*shape)

