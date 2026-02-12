import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

class CSVDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

class MultipleLinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x @ self.w  + self.b
    
def train(model, criterion, train_iter, epoches, optimizer):
    model.train()

    loss_list = []

    for epoch in range(epoches):
        for batch_x, batch_y in train_iter:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        loss_list.append(loss.item())
    
    return loss_list
    
    

def test(model, criterion, test_iter):
    model.eval()

    loss_list = []

    for batch_x, batch_y in test_iter:
        with torch.no_grad():
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

        loss_list.append(loss.item())
       

    print(f"Avg loss = {sum(loss_list) / len(loss_list)}")

    return loss_list

def main():

    df = pd.read_csv("mulRegData.csv")

    x = torch.tensor(df[["x1", "x2"]].values, dtype=torch.float32)
    y = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

    full_dataset = CSVDataset(x, y)

    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len

    train_data , test_data = random_split(full_dataset, [train_len, test_len])

    train_iter = DataLoader(train_data, batch_size=2, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=test_len, shuffle=False)

    model = MultipleLinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    epoches = 200

    loss_list = train(model, criterion, train_iter, epoches, optimizer)
    test(model, criterion, test_iter)

    print(f"Parameter w = {model.w.detach().numpy()} \n b = {model.b.item()}")

    epoches_list = []

    for i in range(epoches):
        epoches_list.append(i)

    plt.plot(epoches_list, loss_list)
    plt.xlabel("Epoches")
    plt.ylabel("Losses")
    plt.title("Multiple Linear Regression")
    plt.show()

    

if __name__ == "__main__":
   main()
