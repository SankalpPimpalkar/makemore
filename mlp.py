import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {ch:i+1 for i,ch in enumerate(chars)}
stoi['.'] = 0
itos = {i:ch for ch,i in stoi.items()} 

block_sz = 3
X, Y = [], []

for w in words:
    # print(w)
    ctx = [0] * block_sz

    for ch in w + '.':
        ix = stoi[ch]
        X.append(ctx)
        Y.append(ix)
        # print("".join(itos[i] for i in ctx), '-->', itos[ix])
        ctx = ctx[1:] + [ix]
    # print()

G = torch.Generator().manual_seed(2310)

X = torch.tensor(X)
Y = torch.tensor(Y)
C = torch.randn((27,2))

W1 = torch.randn((6, 100))
B1 = torch.randn(100)
W2 = torch.randn((100, 27))
B2 = torch.randn(27)
parameters = [C, W1, W2, B1, B2]
sum([p.nelement() for p in parameters])

for p in parameters:
    p.requires_grad = True

for i in range(1000):
    ix = torch.randint(0, X.shape[0], (32,))

    # Forward Pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + B1)
    logits = h @ W2 + B2 
    loss = F.cross_entropy(logits, Y[ix])
    print(f"loss: {loss.item()}")

    step_sz = 0.1

    # Backward Pass
    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -step_sz * p.grad