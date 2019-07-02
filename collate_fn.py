import numpy as np

import torch 

from torch.utils.data import Dataset, DataLoader

"""
for Video application 
"""
class CustomDataset(Dataset):
    def __init__(self, len_dataset, width, length, lmin=0, lmax=2):
        """
        self.x : Generate according to a Gaussian distribution a list of 3D vector with different depths
        self.y : Randomly generate labels between [lmin, lmax]
        """
        self.x = [torch.rand((p + 1 * p, width, length)) for p in range(1, len_dataset)]
        self.y = [torch.randint(lmin, lmax, (1,1))[0] for i in range(1, len_dataset)]

    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def collate_fn(data):
    """
    This function will adjust the elements of our dataset 
    which are 3D vectors with different depths by adding at the beginning null vector

    """
    # batch contains a list of tuples of structure (sequence, target) of size 'batch size'
    maximum = max(list(map(lambda x: x[0].shape[0], data)))
    return torch.tensor([torch.cat((torch.zeros((maximum - tmp[0].shape[0], tmp[0].shape[1], tmp[0].shape[2])), 
            tmp[0])).numpy() for tmp in data]), torch.tensor(list(zip(*data))[1]).unsqueeze(1)
    

if __name__ == "__main__":
        
    dl = DataLoader(CustomDataset(len_dataset=10), batch_size=4, shuffle=True, collate_fn=collate_fn)

    for x, y  in dl:
        print(x.shape)
        print(y.shape)


