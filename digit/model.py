import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Extractor(nn.Module):

    def __init__(self):
        super(MNIST_Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size= 5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size= 5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, input):
        if(input.shape[1] == 1):
            input = input.expand(input.shape[0], 3, input.shape[2], input.shape[3])
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 48 * 4 * 4)

        return x

class MNIST_Classfier(nn.Module):

    def __init__(self):
        super(MNIST_Classfier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        logits = F.relu(self.fc1(input))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(logits)
        logits = self.fc3(logits)

        return logits
