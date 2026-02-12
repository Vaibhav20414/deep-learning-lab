import torch
import torch.nn as nn

class RegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0.0, requires_grad=True))
    
    def forward(self, x):
        return (self.w*x + self.b)
    

x = torch.tensor([2.0, 4.0, 6.0, 8.0])
y = torch.tensor([5.0, 9.0, 13.0, 17.0])


model = RegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
loss = nn.MSELoss()
epochs = 200
for epoch in range(epochs):

    y_pred = model(x)

    loss_obj = loss(y, y_pred)
    loss_obj.backward()

    optimizer.step()
    optimizer.zero_grad()

print(model.w.item())
print(model.b.item())
