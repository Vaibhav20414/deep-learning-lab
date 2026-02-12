import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LinearRegressionDataset(Dataset):

    def __init__(self, x , y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return (len(self.x))
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class RegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        #instead of doing this we can use nn.Linear(1,1)

    def forward(self, x):
        #if we are using nn.Linear we can return self.linear(x)
        return (self.w*x + self.b)
    
    
#Question - How do I use the dataset and dataloader for my nn class to perform the linear regression

def train(model, dataloader, criterion, optimizer, epochs):
    # this is not useful here but, it switches the model mode to training 
    model.train()

    for epoch in range(epoch):
        total_loss = 0.0
        
        # this is for mini batch gradient
        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            #print loss every 20 epoch 
            if (epoch + 1) % 20 == 0:
               avg_loss = total_loss / len(dataloader)
               print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def main():

    torch.manual_seed(0)

    x = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    y = torch.tensor([[5.0], [9.0], [13.0], [17.0]])

    dataset = LinearRegressionDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    train(model, dataloader, criterion, optimizer, epochs=200)

    w = model.linear.weight.item()
    b = model.linear.weight.item()

    print(f"\nLearned Parameters: ")
    print(f"w = {w:.4f}, b = {b:.4f}")


if __name__ == "__main__":
    main()

