import torch.utils.data as data
import os
import PIL.Image as Image
import re

class VisDA2019(data.Dataset):

    domains = ['infograph', 'quickdraw', 'real', 'sketch', 'clipart', 'painting']
    pattern = 'VisDA2019/(.*)'

    def __init__(self, root, domain, split= 'train', transform= None, target_transform= None, label_file= None):
        self.root = root
        self.domain = domain
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        assert self.domain in self.domains
        if label_file is None:
            self.label_file = self.root + '/' + self.domain + '_' + self.split + '.txt'
        else:
            self.label_file = label_file

        if not os.path.exists(self.label_file):
            raise FileNotFoundError('Label file is not found.')

        self.img_files = []
        self.labels = []

        with open(self.label_file) as files:
            for file in files:
                file_str = file.split()
                if len(file_str) == 2:
                    img_file, label = file_str
                    img_file = self.root + '/' + img_file
                    self.img_files.append(img_file)
                    self.labels.append(int(label))
                elif len(file_str) == 1:
                    img_file = file_str[0]
                    img_file = self.root + '/' + img_file
                    self.img_files.append(img_file)
                    self.labels.append(0)

    def __getitem__(self, index):

        img_file, target = self.img_files[index], self.labels[index]
        img = Image.open(img_file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img_file = re.search(self.pattern, img_file)[0][10:]

        return img_file, img, target

    def __len__(self):
        return len(self.labels)

class VisDA2019UnlabeledPair(data.Dataset):

    domains = ['infograph', 'quickdraw', 'real', 'sketch', 'clipart', 'painting']
    pattern = 'VisDA2019/(.*)'

    def __init__(self, root, domain, split= 'train', transform= None, target_transform= None, label_file= None):
        self.root = root
        self.domain = domain
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        assert self.domain in self.domains
        if label_file is None:
            self.label_file = self.root + '/' + self.domain + '_' + self.split + '.txt'
        else:
            self.label_file = label_file

        if not os.path.exists(self.label_file):
            raise FileNotFoundError('Label file is not found.')

        self.img_files = []

        with open(self.label_file) as files:
            for file in files:
                img_file, label = file.split()
                img_file = self.root + '/' + img_file
                self.img_files.append(img_file)

    def __getitem__(self, index):

        img_file1, img_file2 = self.img_files[index], self.img_files[index]
        img1 = Image.open(img_file1).convert('RGB')
        img2 = Image.open(img_file2).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img_file = re.search(self.pattern, img_file1)[0][10:]

        return img_file, img1, img2

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    infograph_dataset = VisDA2019(root = '/home/cuthbert/program/dataset/VisDA2019', domain= 'infograph')
    infograph_pair_dataset = VisDA2019UnlabeledPair(root= '/home/cuthbert/program/dataset/VisDA2019', domain= 'infograph')
    print(len(infograph_dataset))
    print(len(infograph_pair_dataset))

