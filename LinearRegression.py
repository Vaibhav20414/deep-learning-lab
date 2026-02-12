import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class CSVDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train(model, dataloader, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")

def main():
    torch.manual_seed(0)

    # -----------------------
    # Load CSV
    # -----------------------
    df = pd.read_csv("data.csv")

    x = torch.tensor(df["x"].values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

    full_dataset = CSVDataset(x, y)

    #Now split the full dataset

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # -----------------------
    # Model, loss, optimizer
    # -----------------------
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # -----------------------
    # Train and test
    # -----------------------
    train(model, train_loader, criterion, optimizer, epochs=200)
    test(model, test_loader, criterion)

    # -----------------------
    # Learned parameters
    # -----------------------
    w = model.linear.weight.item()
    b = model.linear.bias.item()

    print("\nLearned parameters:")
    print(f"w = {w:.4f}, b = {b:.4f}")

if __name__ == "__main__":
    main()
