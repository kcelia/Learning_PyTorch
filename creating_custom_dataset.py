"""
Trying to create a non default dataset with PyTorch.
Problem : try to classify if sum of 2 numbers numbers are odd or even
(just training to create a dataset)
"""
import numpy as np
np.random.seed(1995)

import matplotlib.pyplot as plt

import torch
import torch as T
import torch.nn as nn

from torch.nn.modules import *

from tqdm import tqdm, trange
from torch.utils.data import Dataset
from torchvision import datasets, transforms


T.set_default_tensor_type('torch.FloatTensor')


class OddEvenNumbersDataset(Dataset):
    def __init__(self, train=True, dataset_size=2048):
        #on cree le dataset
        if train:
            self.x = T.tensor(np.array([
                [np.random.randint(0, 1000), np.random.randint(0, 1000)] 
                for i in range(dataset_size)
            ])).float()
            self.y = T.tensor(self.x.sum(1) % 2).float()
        else:
            self.x = T.tensor(np.array([
                [np.random.randint(0, 1000), np.random.randint(0, 1000)] 
                for i in range(dataset_size)
            ])).float()
            self.y = T.tensor(self.x.sum(1) % 2).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

class OENet(Module):
    def __init__(self):
        super(OENet, self).__init__()
        self.clf = Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.clf(x)

batch_size = 16


my_model = OENet()
optimizer = torch.optim.Adam(my_model.parameters())
loss_function = BCELoss()

train_loader = T.utils.data.DataLoader(
    OddEvenNumbersDataset(train=True),
    batch_size=batch_size,
    shuffle=True
)

test_loader = T.utils.data.DataLoader(
    OddEvenNumbersDataset(train=False),
    batch_size=batch_size,
    shuffle=True
)


nb_epochs = 10
train_history = []
test_history = []
for i in trange(nb_epochs):
    batchs_history = []
    for x, y in train_loader:
        if x.shape[0] != batch_size:
            continue
        optimizer.zero_grad()
        yhat = my_model(x.view([batch_size, 2]))
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        batchs_history.append(loss.item())
    train_history.append(np.array(batchs_history).mean())
    batchs_history = []
    for x, y in test_loader:
        if x.shape[0] != batch_size:
            continue     
        yhat = my_model(x.view([batch_size, 2]))
        loss_test = loss_function(yhat, y)
        batchs_history.append(loss.item())
    test_history.append(np.array(batchs_history).mean())


plt.title("Loss")
plt.plot(train_history, label='train')
plt.plot(test_history, label='test')
plt.legend()
plt.show()



#train accuracy
accuracy = []
for x, y in test_loader:
    if x.shape[0] != batch_size:
        continue
    yhat = my_model(x.view([batch_size, 2]))
    accuracy.append(
        ((yhat > .5).t()[0].float() == y).float().mean().item()
    )

print(np.mean(accuracy))
