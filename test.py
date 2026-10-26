import matplotlib.pyplot as plt
import torch

words = open("names.txt", "r").read().splitlines()
N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {ch:i+1 for i,ch in enumerate(chars)}
stoi['.'] = 0
itos = {i:ch for ch,i in stoi.items()} 

for w in words:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chrs, chrs[1:]):
        i1, i2 = stoi[ch1], stoi[ch2]
        N[i1, i2] += 1

G = torch.Generator().manual_seed(2147483647)
P = N.float()
P /= P.sum(1, keepdim=True)

for i in range(10):
    st = ""
    ix = 0
    while True:
        # p = torch.ones( 27) / 27.0

        # p = N[ix].float()
        # p = p / p.sum()

        p = P[ix]

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=G).item()
        st += (itos[ix])
        if ix == 0:
            break
    # print(st)

# plt.imshow(N)
# plt.colorbar()
# plt.show()


log_likelihood = 0.0
n = 0

for w in words:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chrs, chrs[1:]):
        i1, i2 = stoi[ch1], stoi[ch2]
        prob = P[i1, i2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n}")