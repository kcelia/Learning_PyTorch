"""
Simple convnet on MNIST with PyTorch
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch as T
import torch.nn as nn

from torch.nn.modules import *
from torch.utils.data.dataset import Subset

from tqdm import tqdm, trange
from torchvision import datasets, transforms


T.set_default_tensor_type('torch.FloatTensor')

batch_size = 8
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

def save_checkpoint(state, is_best, filename='output/checkpoint.pth'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        #print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        pass #print ("=> Validation Accuracy did not improve")


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


##inspecter le modele et verifier qu'il marche
#from torchsummary import summary
#summary(model, (1, 28, 28))


# Pour avoir le meme chemin ( pour que les données soient 
# dans train_loader.tensor (qui lui est un tuple, x0, y1)
# premier split, init de base
td = train_loader.dataset.train_data
tl = train_loader.dataset.train_labels
train_dataset = T.utils.data.dataset.TensorDataset(
    td[:int(len(td) * .8)],
    tl[:int(len(td) * .8)]
)
valid_dataset = T.utils.data.dataset.TensorDataset(
    td[int(len(td) * .8):],
    tl[int(len(td) * .8):]
)


def get_train_val(train_dataset, valid_dataset):
    # pour recuperer les données
    td = train_dataset.tensors[0]
    tl = train_dataset.tensors[1]
    vd = valid_dataset.tensors[0]
    vl = valid_dataset.tensors[1]
    train_data = T.cat((vd, td))
    label_data = T.cat((vl, tl))
    tr_ld = T.utils.data.dataset.TensorDataset(
        train_data[:int(len(td) * .8)],
        label_data[:int(len(td) * .8)]
    )
    vl_ld = T.utils.data.dataset.TensorDataset(
        train_data[int(len(td) * .8):],
        label_data[int(len(td) * .8):]
    )
    return tr_ld, vl_ld
        

folds_accuracy = []
for i in range(5): #5 folds
    #on change le fold pour la cross val
    train_dataset, valid_dataset = get_train_val(train_dataset, valid_dataset)

    #on cree les loader associes aux datasets (train/valid) sample ci-dessus
    train_loader = T.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = T.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    #reset model
    model = Net_CO()
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = CrossEntropyLoss()

    #apprentissage classique
    nb_epochs = 20
    train_history, test_history = [], []
    for i in trange(nb_epochs):
        model.train()
        batch_loss = []
        for x, y in train_loader:
            x = x.float()
            optimizer.zero_grad()
            yhat = model(x.view([x.shape[0], 1, 28, 28]))
            loss = loss_function(yhat, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        train_history.append(np.array(batch_loss).mean())
        model.eval()
        batch_loss = []

        for x, y in test_loader:
            x = x.float()
            yhat = model(x.view([x.shape[0], 1, 28, 28]))
            loss = loss_function(yhat, y)
            batch_loss.append(loss.item())
        test_history.append(np.array(batch_loss).mean())
        #saving checkpoints
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
    print("accuracy fold {} : {}".format(i, np.mean(accuracy)))

    folds_accuracy.append(np.mean(accuracy))

print('==> final accuracy :', np.mean(folds_accuracy))

