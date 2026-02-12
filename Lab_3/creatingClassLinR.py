import torch

class RegressionModel:

    def __init__(self):
        
        self.w = torch.tensor(0.0, requires_grad=True)
        self.b = torch.tensor(0.0, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.w, self.b], lr=0.01)
    
    def forward(self, x):
        return self.w*x + self.b
    
    def update(self):
        self.optimizer.step()
    
    def resetGrad(self):
        self.optimizer.zero_grad()

#using this class to perform linear regression

x = torch.tensor([2.0, 4.0, 6.0, 8.0])
y = torch.tensor([5.0, 9.0, 13.0, 17.0])

model = RegressionModel()



epochs = 200

for epoch in range(epochs):
    y_pred = model.forward(x)
    loss = ((y_pred - y)**2).mean()
    loss.backward()

    model.update()

    model.resetGrad()

print(f"Final value of parameter w = {model.w.item()} and b = {model.b.item()}")


