#this program is oriented toward dataloader and dataset class

import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):

    def __init__(self, x , y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return (len(self.x))
    
    def __getitem__(self, idx):
        return self.x[idx] , self.y[idx]
    
x = torch.tensor([2.0, 4.0, 6.0, 8.0])
y = torch.tensor([5.0, 9.0, 13.0, 17.0])

dataset = Dataset(x, y)
train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)


