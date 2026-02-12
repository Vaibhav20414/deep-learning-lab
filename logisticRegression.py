# Step 1 - Get all the modules and tools 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

#Step - 2 - Get the dataset class

class CSVDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index] , self.y[index]
    
#Step - 3 - Logistic Regression Class

class LogisticRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b
    
#Step - 4 - training and testing functions

def train(model, optimizer, epochs, criterion, train_data):
    model.train()

    for epoch in range(epochs):
        for batch_x , batch_y in train_data:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()

            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()


def test(model, criterion, test_data):
    model.eval()

    total_loss = 0

    for batch_x, batch_y in test_data:
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        
        total_loss += loss.item()

    print(f"Avg loss : {total_loss/len(test_data)}")


#Step - 5 - Main function 

def main():
    df = pd.read_csv("logistic_data.csv")

    x = torch.tensor(df["x"].values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

    full_dataset = CSVDataset(x, y)
    
    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len

    train_dataset , test_dataset = random_split(full_dataset, [train_len, test_len])

    train_iterator = DataLoader(train_dataset, batch_size= 2, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=test_len, shuffle=False)

    model = LogisticRegression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 200

    train(model, optimizer, epochs, criterion, train_iterator)
    test(model, criterion, test_iterator)

    w = model.w.item()
    b = model.b.item()

    print("\nLearned parameters:")
    print(f"w = {w:.4f}, b = {b:.4f}")

if __name__ == "__main__":
    main()



    
            

    


