"""
Reimplementing Highway Network on MNIST with PyTorch
https://arxiv.org/pdf/1505.00387.pdf
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
nb_epochs  = 30
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

class HighwayGated(nn.Module):
    """Définir un layer Highway:
    y = H(x,WH)· T(x,WT) + x · (1 − T(x,WT))
    possède la porte + transformation (fonction d'activation propre à elle)
    """
    def __init__(self, activation_h=nn.ReLU(), dim=20, biais=-1, desactivate_gate=False):
        """
        activation_h : fonction d'activation de la transformation
        dim : taille du layer d'entrée et de sortie (à fixer)
        biais : biais de la porte, a changer en fonctin de la profondeur
        """
        super(HighwayGated, self).__init__()

        self.desactivate_gate = desactivate_gate

        if not desactivate_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim, dim), # in_features = out_features, car meme taille dans le papier, la matrice de poids est créée en meme temps
                nn.Sigmoid()
            )
            self.gate[0].bias = nn.Parameter(abs(self.gate[0].bias) * biais) #nn.Parameter: le biais est un parametre et non n'importe quel tensor 

        self.transform = nn.Sequential(
            nn.Linear(dim, dim), # j'ajoute une couche au debut pour reduire le nombre de dimension
            activation_h
        )
   
        
    def forward(self, x):
        if self.desactivate_gate:
            return self.transform(x)
        T = self.gate(x)
        H = self.transform(x)
        return H * T + x * (1 - T)


class HighwayCelia(nn.Module):
    """Empiler des couches
    couche 1 : réduction de dimentionnalité
    couche 2 .. n - 1 : module(HighwayGated)
    couche  n : softmax
    """

    def __init__(self, in_, dim, nb_laybers=10, activation_h=nn.ReLU(), biais=-1, desactivate_gate=False):
        super(HighwayCelia, self).__init__()

        self.reduc = nn.Linear(in_, dim)

        self.highway = nn.Sequential(*[
            HighwayGated(dim=dim, activation_h=activation_h, biais=biais, desactivate_gate=desactivate_gate) 
            for i in range(nb_laybers)
        ])
        
        self.clf = nn.Sequential(
            nn.Linear(dim, 10),
            nn.Softmax()
        )
    
    def forward(self, x):
        couche1 = self.reduc(x)
        couche_n = self.highway(couche1)
        out = self.clf(couche_n)
        return out 


# Summary(HighwayCelia(784, 20), (784, ))/ (784, 20) : init, (784, ) tuple de taille 1





dim = 10
nb_laybers = [1, 3, 5, 10, 20]
biais = -4.
nb_epochs  = 1
loss_function = nn.CrossEntropyLoss()



for l in nb_laybers:
    plein, highway = [], []

    highway_model = HighwayCelia(
        784, 50, l, activation_h=nn.Tanh(), 
        biais=biais, desactivate_gate=False
    )
    highway_optimizer = T.optim.Adagrad(highway_model.parameters())#, lr=1e2, momentum=.9)


    plein_model = HighwayCelia(
        784, 71, l, activation_h=nn.Tanh(), 
        biais=biais, desactivate_gate=True
    )
    plein_optimizer = T.optim.Adagrad(plein_model.parameters())#, lr=1e2, momentum=.9)

    for i in range(nb_epochs):


        for x, y in tqdm(train_loader):

            highway_optimizer.zero_grad()
            plein_optimizer.zero_grad()

            yhat_h = highway_model(x.view([x.shape[0], -1]))
            yhat_p = plein_model(x.view([x.shape[0], -1]))

            loss_h = loss_function(yhat_h, y)
            loss_p = loss_function(yhat_p, y)

            loss_h.backward()
            loss_p.backward()

            highway_optimizer.step()
            plein_optimizer.step()

            highway.append(loss_h.item())
            plein.append(loss_p.item())


    plt.title("depth :" + str(l) + ", biais :" +str(biais))
    tranche = 10
    plein2 =  [np.mean(plein[i * tranche : i * tranche +tranche]) 
        for i in range(int(np.ceil(len(plein)/tranche)))]

    plt.plot(plein2, label="plein")
    highway2 =  [np.mean(highway[i * tranche : i * tranche +tranche]) 
        for i in range(int(np.ceil(len(highway)/tranche)))]

    plt.plot(highway2, label="highway")
    plt.legend(loc="best")
    plt.savefig("Profondeur_{}_biais_{}.png".format(l, abs(biais)))
    plt.cla() #reset figure


