import torch 

x = torch.tensor([2.0, 4.0, 6.0, 8.0])
y = torch.tensor([5.0, 9.0, 13.0, 17.0])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

optimizer = torch.optim.SGD([w,b], lr=0.01)

epochs = 200
loss_list = []
for epoch in range(epochs):
    y_pred = w*x + b
    loss = ((y_pred - y)**2).mean()
    loss.backward()
    
    # with torch.no_grad(): This line is not required because optimizer uses it internally to update
    optimizer.step()
    
    optimizer.zero_grad()
    loss_list.append(loss.item())

print(f"Final value of parameter w = {w} and b = {b}")