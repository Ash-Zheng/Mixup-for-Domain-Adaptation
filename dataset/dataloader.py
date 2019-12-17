import torch.utils.data as data
import torchvision.transforms as transforms
from dataset.dataset import VisDA2019

class MultiSourceDataloader():

    def __init__(self, root, split, sources, target, batchsize, scale= 227, transform= None, target_transform= None, shuffle= False):
        self.root = root
        self.split = split
        self.sources = sources
        self.target = target
        self.batchsize = batchsize
        self.scale = scale
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle

        if self.transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((self.scale, self.scale)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.source_datasets = {}
        for source in self.sources:
            self.source_datasets[source] = VisDA2019(root= self.root, domain= source, split= self.split,
                                                     transform = self.transform, target_transform = self.target_transform)
        self.target_dataset = VisDA2019(root = self.root, domain= self.target, split= self.split,
                                        transform= self.transform, target_transform= self.target_transform)

        self.source_loaders = {}
        for source in self.sources:
            self.source_loaders[source] = data.DataLoader(self.source_datasets[source], batch_size= self.batchsize,
                                                          shuffle= self.shuffle, num_workers= 4, drop_last= True)
        self.target_loader = data.DataLoader(self.target_dataset, batch_size= self.batchsize,
                                             shuffle= self.shuffle, num_workers= 4, drop_last= True)
        self.max_iters = len(self.target_loader)

        self.source_iters = {}
        for source in self.sources:
            self.source_iters[source] = iter(self.source_loaders[source])
        self.target_iter = iter(self.target_loader)
        self.iter = 0

    def reset(self, target_dataset, shuffle, batchsize= None):
        if batchsize == None:
            batchsize = self.batchsize
        self.target_dataset = target_dataset
        self.target_loader = data.DataLoader(self.target_dataset, batch_size= batchsize,
                                             shuffle= shuffle, num_workers= 4, drop_last= True)
        for source in self.sources:
            self.source_loaders[source] = data.DataLoader(self.source_datasets[source], batch_size= batchsize,
                                                          shuffle= self.shuffle, num_workers= 4, drop_last= True)
            self.source_iters[source] = iter(self.source_loaders[source])
        self.target_iter = iter(self.target_loader)
        self.iter = 0
        self.max_iters = len(self.target_loader)

    def __iter__(self):
        return self

    def __next__(self):
        source_data = {}
        source_label = {}
        source_file = {}
        target_data = None
        target_label = None
        target_file = None
        for source in self.sources:
            try:
                source_file[source], source_data[source], source_label[source] = next(self.source_iters[source])
            except StopIteration:
                self.source_iters[source] = iter(self.source_loaders[source])
                source_file[source], source_data[source], source_label[source] = next(self.source_iters[source])

        try:
            target_file, target_data, target_label = next(self.target_iter)
        except StopIteration:
            if target_data is None or target_label is None:
                self.target_iter = iter(self.target_loader)
                target_file, target_data, target_label = next(self.target_iter)

        if self.iter > self.max_iters:
            self.iter = 0
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': source_data, 'S_label': source_label, 'S_file': source_file,
                    'T': target_data, 'T_label': target_label, 'T_file': target_file}


if __name__ == '__main__':
    VisDA_loader = MultiSourceDataloader(root='/home/cuthbert/program/dataset/VisDA2019', split= 'train',
                                         sources= ['infograph', 'sketch', 'quickdraw'], target= 'real', batchsize= 32,
                                         scale= 32)
    target_set = VisDA_loader.target_dataset
    # VisDA_loader.reset(target_set, shuffle= True)
    flag = True
    for batch_idx, data_ in enumerate(VisDA_loader):
        source_file = data_['S_file']
        source_data = data_['S']
        source_label = data_['S_label']
        target_file = data_['T_file']
        target_data = data_['T']
        target_label = data_['T_label']

        if VisDA_loader.iter > 160 and flag:
            flag = False
            VisDA_loader.reset(target_set, shuffle= True)
            print('Loader reset. Iter: %d' %(VisDA_loader.iter))

        if batch_idx % 50 == 0:
            print('batch: [%d/%d](%.2f)' %(batch_idx, VisDA_loader.max_iters, (float)(batch_idx / VisDA_loader.max_iters * 100)))

