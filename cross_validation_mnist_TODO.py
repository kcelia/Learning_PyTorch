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

"""

def get_fold(datax, datay, fold_id, n_folds=3, seed=1996):
    data = list(zip(datax, datay))
    random.Random(seed).shuffle(data)
    datax, datay = zip(*data)
    fold_size = len(datax) // n_folds
    return (
        np.array(datax)[fold_id * fold_size:(fold_id + 1) * fold_size],
        np.array(datay)[fold_id * fold_size:(fold_id + 1) * fold_size]
    )

def get_all_folds_except(datax, datay, fold_id, n_folds=3, seed=1996):
    X = []
    Y = []
    for i in range(n_folds):
        if fold_id != i:
            t = get_fold(datax, datay, fold_id, n_folds=n_folds, seed=seed)
            X.extend(list(t[0]))
            Y.extend(list(t[1]))
    return np.array(X), np.array(Y)

train_data   = train_loader.dataset.train_data.detach().numpy()
train_labels = train_loader.dataset.train_labels.detach().numpy()

nb_folds = 3
for i in range(nb_folds):
    pass

t, l = get_all_folds_except(train_data, train_labels, 2)

train_dataset = T.utils.data.TensorDataset(T.tensor(t).float(), T.tensor(l))
train_loader  = T.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 


"""

def save_checkpoint(state, is_best, filename='output/checkpoint.pth'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")


class Net_CO(Module):
    def __init__(self):
        super(Net_CO, self).__init__()
        self.conv = Sequential(
            Conv2d(1, 5, 5),#channel de sortie = combien de filtre on applique, kernel size = taille du filtre 5= 5X5
            ReLU(),
            MaxPool2d(2),
            Conv2d(5, 16, 9),
            ReLU(),
            Conv2d(16, 20, 4),
            ReLU()
        )
        self.clf = Sequential(
            Linear(20, 10), #20 la taille du vecteur flatten, sortie de la convolution
            Softmax()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        return self.clf(out)


model = Net_CO()
optimizer = torch.optim.Adam(model.parameters())
loss_function = CrossEntropyLoss()

##inspecter le modele et verifier qu'il marche
#from torchsummary import summary
#summary(model, (1, 28, 28))


dataset = torch.utils.data.ConcatDataset([train_loader, test_loader])


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
    save_checkpoint({
        'epoch': i,
        'state_dict': model.state_dict(),
        'current_loss': test_history[-1]
    }, test_history[-1] < test_history[-2] if len(test_history) > 1 else True)

        


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




