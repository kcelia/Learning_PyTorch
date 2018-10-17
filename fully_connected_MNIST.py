"""
Simple FullyConnected on MNIST with PyTorch
"""
import torch
import numpy as np

import matplotlib.pyplot as plt

class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.input = (x, w) # pas la peine de sauvegarder b 
        return x @ w + b
    
    @staticmethod
    def backward(ctx, grad_output):
        x_grad = grad_output @ ctx.input[1].t() 
        w_grad = ctx.input[0].t() @ grad_output
        b_grad = grad_output.mean(dim=0)
        return x_grad, w_grad, b_grad



x = torch.randn(16, 23, requires_grad=True)
y = x @ torch.randn(23, 1) + torch.randn(16, 1) / 100
s = ((x - y) ** 2).mean()
w = torch.randn(23, 1, requires_grad=True)  
b = torch.randn(16, 1, requires_grad=True)



linear = Linear.apply #
y_pred = linear(x, w, b)
loss = (y_pred - y).pow(2).sum()
loss.backward()
w.grad

#partie 2

w = torch.nn.Parameter(torch.randn(23, 1))  
b = torch.nn.Parameter(torch.randn(16, 1))  

opt = torch.optim.Adam([w, b])

#need to set y.requires_grad to false
y = y.detach()
y.requires_grad = False

L = []
for i in range(30):
    opt.zero_grad()
    y_pred = linear(x, w, b)
    loss = (y_pred - y).pow(2).sum()
    loss.backward()
    opt.step()
    print(loss)
    L.append(loss.item())

plt.plot(L)
plt.show()

#partie 3

import torch
import torch.nn as nn

from torch.nn.modules import *

from tqdm import tqdm, trange
from torchvision import datasets, transforms


T.set_default_tensor_type('torch.FloatTensor')

batch_size = 32
nb_epochs  = 5000
nb_digits  = 10

train_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 
test_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 

#x_train = train_loader.dataset.train_data.view(-1, 784).float()
#x_test  = test_loader.dataset.test_data.view(-1,   784).float()
#y_train = train_loader.dataset.train_labels.view(-1, 1).float()
#y_test  = test_loader.dataset.test_labels.view(-1,   1).float()

class CeliaNet(Module):
    def __init__(self):
        super(CeliaNet, self).__init__()
        self.clf = Sequential(
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.clf(x)

#list(celias_model.clf[0].parameters()) #renvoie les parametres (W, b) du premier linear


celias_model = CeliaNet()
optimizer = torch.optim.Adam(celias_model.parameters())
loss_function = CrossEntropyLoss()

nb_epochs = 7
test_history = []
train_history = []
for i in trange(nb_epochs):
    batchs_history = []
    for x, y in train_loader:
        if x.shape[0] != batch_size:
            continue
        optimizer.zero_grad()
        yhat = celias_model(x.view([batch_size, 784]))
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        batchs_history.append(loss.item())
    train_history.append(np.array(batchs_history).mean())
    batchs_history = []
    for x, y in test_loader:
        if x.shape[0] != batch_size:
            continue     
        yhat = celias_model(x.view([batch_size, 784]))
        loss_test = loss_function(yhat, y)
        batchs_history.append(loss.item())
    test_history.append(np.array(batchs_history).mean())
    

accuracy = []
for x, y in test_loader:
    if x.shape[0] != batch_size:
        continue
    yhat = celias_model(x.view([batch_size, 784]))
    accuracy.append((yhat.argmax(1) == y).float().mean().item())

print(np.mean(accuracy))



plt.title("Loss MNIST")
plt.plot(train_history, label='train')#, marker="o--")
plt.plot(test_history, label='test')#, marker='r--') 
plt.legend()
plt.show()

