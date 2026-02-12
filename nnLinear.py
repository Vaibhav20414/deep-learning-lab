#Function to create the most basic neuron
#Feed-forward networks are models where the data moves in just one direction
#back-propogation is just the training algo
import torch 
import torch.nn as nn   

model = nn.Linear(1, 1)

#this method in the nn module is of the form-
#nn.Linear(in_features, out_features) which represents: y = Wx + b
#if were were to put the arguments as 1, 1 this means one element in the
#weight matrix and one output which gives y = wx + b

x = torch.tensor([[2.0]])
y = model(x)

#how this model works as a linear regression model

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)
#this code defines the computation not the training 
