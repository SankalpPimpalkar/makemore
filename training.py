import torch
import torch.nn.functional as F

# Dataset 
words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {ch:i+1 for i,ch in enumerate(chars)}
stoi['.'] = 0
itos = {i:ch for ch,i in stoi.items()} 
xs, ys = [], []

for w in words:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chrs, chrs[1:]):
        i1, i2 = stoi[ch1], stoi[ch2]
        xs.append(i1)
        ys.append(i2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# Initilization of network
G = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=G, requires_grad=True)

# Gradient descent
for k in range(10000):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W 

    # Soft Max
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(len(ys)), ys].log().mean()
    print(loss.item())

    # Backward pass
    W.grad = None
    loss.backward()

    step_sz = 0.1
    W.data += -step_sz * W.grad