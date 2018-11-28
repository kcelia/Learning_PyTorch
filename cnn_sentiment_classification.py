import torch
import torch as T

import numpy as np
import matplotlib.pyplot as plt

import logging
import torchtext.datasets as datasets
import torchtext.data as data
import torchtext
import torch.nn as nn

from tqdm import trange, tqdm


logging.basicConfig()
logger = logging.getLogger("model")
logger.setLevel(level=logging.DEBUG)


DATA_DIR="./baskiotis/"

DATASET_DIR="%s/data" % DATA_DIR
VECTORS_DIR="%s/vectors" % DATA_DIR

# text
TEXT=data.Field(lower=True,include_lengths=False,batch_first=True)
LABEL = data.Field(sequential=False, is_target=True)

# make splits for data
train, val, test = datasets.sst.SST.splits(TEXT, LABEL,root=DATASET_DIR)

# Use the vocabulary
wordemb = torchtext.vocab.GloVe("6B", dim=100, cache=VECTORS_DIR)
# Build the vocabularies
# for labels, we use special_first to False so <unk> is last
# (to discard it)
TEXT.build_vocab(train, vectors=wordemb)
LABEL.build_vocab(train, specials_first=False)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
(train, val, test), batch_size=10, device=0)

nn_embeddings = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

#dictionnaire : (key,value) key : indexe du mot, value : l'embedding du mot, un vecteur de taille 100

class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        # taille du filtre = k*c, c: taille de l'embedding
        # batch size = 10 ([10, 40, 100])
        # On prend tout le corpus et pour chaque phrase on applique une convolution
        # on obtiens une sequence de vecteurs = filtre
        # quelles sont les sous sequences qui activent le plus une phrase?
        # itos pour afficher les mots
        #channel_in = profondeur = taille de l'embedding = 100

        self.dim_red = nn.Conv1d(100, 15, 1)
        #on test avec un CNN et un FCN
        #on test avec plusieurs couches, plusieurs filtres, plusieurs kernel !la plus petite phrase fait une taille de 4
        self.cnn = nn.Sequential(
            nn.Conv1d(15, 30, 3, padding=1, dilation=1),
            nn.ReLU(), 
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(15, 30, 3, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(15, 30, 3, padding=3, dilation=3),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Linear(90, 15),
            nn.ReLU(),
            nn.Linear(15, 3),
            nn.Softmax()
        )


    def forward(self, x):
        #on test plusieur combinaisons de cnn1, cnn2, cnn3 avec les opérateurs +, *, concat 
        #out  = self.cnn(x) + self.cnn2(x) #* self.cnn1(x)
        x = self.dim_red(x)
        out = T.cat((self.cnn(x), self.cnn2(x), self.cnn3(x)), dim=1)
        out = out.max(2)[0]
        out  =  out.reshape(out.size(0), -1)
        out = self.clf(out) 
        return out


model = SentimentClassifier()
optimizer = T.optim.Adam(model.parameters()) #SGD(model.parameters(), lr=1e-2, momentum=.9)
loss_function = nn.CrossEntropyLoss()

nb_epochs = 9
train_loss, val_loss = [], []
train_acc, val_acc = [],[]

for epoch_id in tqdm(range(nb_epochs)):
    batch_history_loss = []
    batch_history_acc = []

    for x, y in train_iter: 
        if x.shape[0] < 10: #
            continue
        optimizer.zero_grad()
        x = nn_embeddings(x)
        x = x.permute(0, 2, 1)
        yhat = model(x)
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        batch_history_loss.append(loss.item())
        batch_history_acc.append((yhat.argmax(1) == y).float().mean())
    train_loss.append(np.array(batch_history_loss).mean())
    train_acc.append(np.array(batch_history_acc).mean())

    batch_history_loss = []
    batch_history_acc = []

    for x, y in val_iter:
        if x.shape[0] < 10:
            continue
        x = nn_embeddings(x)
        x = x.permute(0, 2, 1)
        yhat = model(x)
        loss = loss_function(yhat, y)
        batch_history_loss.append(loss.item())
        batch_history_acc.append((yhat.argmax(1) == y).float().mean())
    val_loss.append(np.array(batch_history_loss).mean())
    val_acc.append(np.array(batch_history_acc).mean())

print(val_acc[-1])

#meilleur score 64.848 avec Conv1d 45 channels + FC Hidden 15
plt.title("Sentiment Classification - Conv1d 45 channels + FC Hidden 15")
plt.plot(train_loss, label="loss train")
plt.plot(val_loss, label="loss test")
plt.plot(train_acc, label="acc train")
plt.plot(val_acc, label="acc test")
plt.legend()
plt.show()


train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=1024, device=0)

#visualisation des neurones des couches de convolutions
for x, y in train_iter: 
    break
x_original = x
x = nn_embeddings(x)
x = x.permute(0, 2, 1)
max_val, max_pos = model.cnn(x).max(2)
max_val.sort(0)[1][-10:].numpy()[::-1]

#visualisation de la derniere couche
#3 classes (commentaires positifs, commentaires négatifs, commentaires neutres)
for neurone in range(3):
    #pour 1 neurone on a les 10 meilleurs phrases qui l'ont le plus activé 
    phrases = model.cnn(x).max(2)[0][:,neurone].sort()[1][-10:]
    pos_dans_phrase = model.cnn(x)[phrases].max(2)[1][:,neurone]
    x_original[phrases[0]][pos_dans_phrase[0]]

    invert_dictionnary = {v: k for (k, v) in TEXT.vocab.stoi.items()}
    for i in range(10): # les 10 meilleurs phrases
        t = x_original[phrases[i]][max(pos_dans_phrase[i]-2, 0):pos_dans_phrase[i]+3+2]
        print([invert_dictionnary[i] for i in t.detach().numpy()])
    print('_____')

#Resulats
"""
['tasty', 'and', 'sweet']
['it', 'with', 'spirit']
['nuanced', 'lead', 'performances']
['lovely', ',', 'eerie']
['intimate', 'and', 'panoramic']
['elegant', ',', 'witty']
['witty', ',', 'vibrant']
['tale', 'has', 'warmth']
['brilliant', 'and', 'entertaining']
['beautiful', ',', 'entertaining']
"""

"""
['silly', 'little', 'puddle']
['mistakes', 'that', 'bad']
['laughable', '--', 'or']
['apparently', 'been', 'forced']
['off', 'so', 'bad']
['exercise', 'in', 'bad']
['stinks', 'so', 'badly']
['this', 'so', 'boring']
['above', 'the', 'stale']
['clumsy', 'and', 'convoluted']
"""

"""
['the', 'same', 'illogical']
['but', 'i', 'do']
['not', 'too', 'offensive']
['but', 'a', 'deficit']
[',', 'so', 'stilted']
['but', 'they', 'do']
['but', 'this', 'predictable']
['not', 'very', 'informative']
['this', 'so', 'boring']
['but', 'a', 'tedious']
"""


"""
['insightful', 'look', 'at']
['and', 'informative', 'documentary']
['dialogue', 'and', 'a']
['edited', '-rrb-', 'picture']
['paean', 'to', 'a']
['and', 'intimate', 'study']
['creative', 'instincts', 'are']
[',', 'aching', 'sadness']
[',', 'timeless', 'and']
['tale', 'combined', 'with']
_____
['shockingly', 'bad', 'and']
['and', 'pointless', 'french']
['burlap', 'sack', 'of']
['and', 'dreary', 'time']
['is', 'worse', ':']
['a', 'slippery', 'self-promoter']
['up', 'mired', 'in']
['so', 'badly', 'of']
[',', 'miserable', 'and']
['and', 'uninspired', '.']
_____
['and', 'ragged', 'that']
['the', 'illusion', 'of']
['or', 'common', 'decency']
['away', 'demons', 'is']
['decision', 'facing', 'jewish']
[',', 'uncouth', ',']
['opaque', 'intro', 'takes']
[',', 'wears', 'down']
['them', 'into', 'every']
['or', 'otherwise', '.']
"""



