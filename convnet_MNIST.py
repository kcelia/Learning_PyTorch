"""
Simple convnet on MNIST with PyTorch
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch as T
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


class CeliaNet(Module):
    def __init__(self):
        super(CeliaNet, self).__init__()
        self.conv = Sequential(
            #Conv2D => (channel_in, channel_out, kernel_size)
            Conv2d(1, 5, 5),
            #channel_out = Combien de filtre on veut appliquer
            #kernel_size = kernel_w*kernel_h, ici = 5, avec kernel_w=kernel_h=5 
            #nbr_filtre = channel_in, kernel_w*kernel_h
            #nbr_param = (channel_in*kernel_w*kernel_h+1)*channel_out = (1*5*5+1)*5
            ReLU(),
            MaxPool2d(2),
            #aucun parametre
            Conv2d(5, 16, 9),
            #nbr_param = (channel_in*kernel_w*kernel_h+1)*channel_out = (5*9*9+1)*16
            ReLU(),
            Conv2d(16, 20, 4),
            #nbr_param = (channel_in*kernel_w*kernel_h+1)*channel_out = (16*4*4+1)*20
            ReLU()
        )
        self.clf = Sequential(
            Linear(20, 10),
            #channel_in, channel_out
            #channel_in = 20 (la taille du vecteur flatten = sortie de la convolution)
            #nbr_param = (channel_in+1)*channel_out
            Softmax()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        return self.clf(out)


model = CeliaNet()
optimizer = torch.optim.Adam(model.parameters())
loss_function = CrossEntropyLoss()

##inspecter le modele et verifier qu'il marche
#from torchsummary import summary
#summary(model, (1, 28, 28))


nb_epochs = 7
train_history, test_history = [], []

for i in trange(nb_epochs):
    model.train()
    batch_loss = []
    for x, y in train_loader:
        optimizer.zero_grad()
        yhat = model(x.view([x.shape[0], 1, 28, 28]))# 1: couleur
        #pour les convolution batch_size, channel_in, w, h
        #pour le linéaire Batch_size, nb_features
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    train_history.append(np.array(batch_loss).mean())
    model.eval()# utile pour le dropout, pour ne pas stocker les gradientss
    batch_loss = []
    for x, y in test_loader:
        yhat = model(x.view([x.shape[0], 1, 28, 28]))
        loss = loss_function(yhat, y)
        batch_loss.append(loss.item())
    test_history.append(np.array(batch_loss).mean())

        


plt.title("Loss MNIST")
plt.plot(train_history, label='train')#, marker="o--")
plt.plot(test_history, label='test')#, marker='r--') 
plt.legend()
plt.show()


accuracy = []
for x, y in test_loader:
    if x.shape[0] != batch_size:
        continue
    yhat = model(x.view([batch_size, 1, 28, 28]))
    accuracy.append((yhat.argmax(1) == y).float().mean().item())
print(np.mean(accuracy))




