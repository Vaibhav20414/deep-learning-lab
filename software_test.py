import torch, time

device = "cuda"
x = torch.rand(8000, 8000, device=device)

start = time.time()
y = x @ x
torch.cuda.synchronize()
print("Time:", time.time() - start)

