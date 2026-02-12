import torch
import numpy

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]) 
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]) 


b = torch.tensor(1.0)
w = torch.tensor(1.0)

y_pred = w*x + b
lr = 0.000001
epochs = 5000

loss = ((y - y_pred)**2).mean()

for epoch in range(epochs):

    y_pred = w*x + b
    loss = ((y - y_pred)**2).mean()

    dldw = -2*(((y_pred - y)*x).mean())
    dldb = -2*((y_pred - y).mean())

    w -= lr*dldw
    b -= lr*dldb

    


print("Final equation: ")
print(f"{w}*x + {b}")

