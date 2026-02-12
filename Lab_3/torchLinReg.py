import torch

x = torch.tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
y = torch.tensor([5.0, 9.0, 13.0, 17.0], requires_grad=True)

#req_grad = True so we can use .grad to compute the gradient
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w*x + b

#This is the forward pass, why isn't it inside of the loop?
loss = ((y - y_pred)**2).mean()

#Again this is the backward pass 
#why isn't it inside of the loop where the parameters are getting updated
loss.backward()

print(w.grad)
print(b.grad)

epochs = 200
lr = 0.01
loss_list = []

for epoch in range(epochs):
    #The addition of the these three are correction
    y_pred = w*x + b
    loss = ((y - y_pred)**2).mean()
    loss_list.append(loss)
    loss.backward()

    with torch.no_grad():
        w -= lr*(w.grad)
        b -= lr*(b.grad)

    w.grad = None
    b.grad = None 


print(w)
print(b)
    
