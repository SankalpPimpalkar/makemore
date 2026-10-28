# makemore

makemore is an autoregressive character-level language modeling codebase inspired by Andrej Karpathy's deep learning tutorial series. The repository explores statistical and neural approaches to sequence generation by predicting character transitions over a vocabulary of 26 lowercase English letters and a boundary token. It implements three distinct modeling methodologies on a dataset of 32,032 names: an empirical bigram model that tabulates transition frequencies into a normalized 2D probability matrix, a single-layer linear neural network trained on one-hot bigram vectors, and a multi-layer perceptron (MLP) architecture based on Bengio et al. (2003) featuring learned character embeddings, fixed-length context windows, non-linear tanh activations, and mini-batch cross-entropy optimization using PyTorch.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Multi-Layer Perceptron Language Model (`mlp.py`)](#1-multi-layer-perceptron-language-model-mlppy)
  - [2. Bigram Probability and Evaluation Model (`test.py`)](#2-bigram-probability-and-evaluation-model-testpy)
  - [3. Neural Network Bigram Classifier (`training.py`)](#3-neural-network-bigram-classifier-trainingpy)
  - [4. Names Training Dataset (`names.txt`)](#4-names-training-dataset-namestxt)
  - [5. Environment Dependencies (`requirements.txt`)](#5-environment-dependencies-requirementstxt)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
  - [Counting Bigram Frequencies and Sampling](#counting-bigram-frequencies-and-sampling)
  - [Evaluating Bigram Negative Log-Likelihood](#evaluating-bigram-negative-log-likelihood)
  - [Training the Linear Network via Full-Batch Gradient Descent](#training-the-linear-network-via-full-batch-gradient-descent)
  - [Training the Multi-Layer Perceptron via Mini-Batch Gradient Descent](#training-the-multi-layer-perceptron-via-mini-batch-gradient-descent)
- [How It Works](#how-it-works)
- [Project Roadmap & Commit History](#project-roadmap--commit-history)

## Project Overview

Autoregressive language models predict the distribution of the next character conditioned on prior context: $P(x_t \mid x_1, x_2, \dots, x_{t-1})$. The simplest non-trivial formulation is the bigram model, which restricts context to a single preceding character ($P(x_t \mid x_{t-1})$). To capture richer phonetic sequences, higher-order models condition predictions on a sliding block of previous tokens ($P(x_t \mid x_{t-n}, \dots, x_{t-1})$). This repository demonstrates the progression of character-level modeling across three paradigms: exact bigram frequency counting, linear bigram classification, and a non-linear multi-layer perceptron with distributed character embeddings.

1. **Markov Context Conditioning**: Supports single-character bigram context ($P(x_t \mid x_{t-1})$) and sliding multi-token context windows of size $k=3$ ($P(x_t \mid x_{t-3}, x_{t-2}, x_{t-1})$).
2. **Bijective Tokenization**: Maps lowercase characters `a` through `z` and boundary token `.` into fixed integers from 0 to 26.
3. **Empirical Frequency Tabulation**: Aggregates adjacent character transition counts across the corpus into a 2D integer tensor $N \in \mathbb{Z}^{27 \times 27}$.
4. **Row-Stochastic Probability Normalization**: Normalizes raw frequencies into row probability distributions summing to 1: $P \in \mathbb{R}^{27 \times 27}$.
5. **Continuous Character Embeddings**: Projects discrete character indices into a continuous low-dimensional vector space $C \in \mathbb{R}^{27 \times 2}$.
6. **Non-Linear Multi-Layer Perceptron**: Passes concatenated context embeddings through a tanh hidden layer ($h \in \mathbb{R}^{100}$) and linear output projection.
7. **Cross-Entropy Loss Minimization**: Measures negative log-likelihood loss $\mathcal{L} = -\frac{1}{n}\sum \log P(\text{target} \mid \text{context})$ using PyTorch autograd and gradient descent.

## Key Features

- **Multiple Sequence Modeling Architectures**: Implements three modeling tiers spanning statistical and neural paradigms.
  - **Count-based Matrix Model**: Tabulates all 228,146 bigram transitions into an explicit $27 \times 27$ frequency grid.
  - **Linear Bigram Classifier**: Uses a single $27 \times 27$ weight matrix trained on one-hot encoded characters via full-batch gradient descent.
  - **Multi-Layer Perceptron (MLP)**: Uses a 3-character context window with 2D embedding lookup table, 100-neuron tanh hidden layer, and 3,481 trainable parameters.
- **Continuous Embedding Representations**: Maps 27 discrete tokens into a 2-dimensional continuous latent space $C \in \mathbb{R}^{27 \times 2}$.
- **Mini-Batch Stochastic Optimization**: Supports both full-batch gradient descent and mini-batch stochastic gradient descent with batch size 32.
- **Deterministic Tokenization System**: Encodes strings to integers and decodes integers back to strings using fixed dictionaries for `.` (index 0) and `a` through `z` (indices 1 through 26).
- **Autoregressive Multinomial Sampling**: Generates novel names token by token conditioned on previous outputs using seeded PyTorch generators.
- **Comprehensive Loss Evaluation**: Computes the exact log-likelihood, aggregate negative log-likelihood, and average per-token loss across the dataset.

## Core Components

<a id="1-multi-layer-perceptron-language-model-mlppy"></a>
### 1. Multi-Layer Perceptron Language Model (`mlp.py`)

Builds a dataset of character context sequences with block size 3 from `names.txt`, initializes a multi-layer perceptron with 2D character embeddings, a 100-unit hidden layer with `torch.tanh`, and a 27-unit linear output projection. Trains the network over 1,000 iterations using mini-batch gradient descent (batch size 32) and `F.cross_entropy`.

| Name | Implementation | Behavior |
| --- | --- | --- |
| Context Builder | `ctx = [0] * block_sz` with rolling update `ctx[1:] + [ix]` | Generates $(N, 3)$ input context tensor $X$ and $(N,)$ target tensor $Y$ |
| Embedding Table $C$ | `torch.randn((27, 2))` | Maps 27 vocabulary tokens into 2-dimensional dense embedding vectors |
| Hidden Layer ($W_1, B_1$) | `W1 = torch.randn((6, 100))`, `B1 = torch.randn(100)` | Projects concatenated $3 \times 2 = 6$ embeddings to 100 hidden dimensions |
| Tanh Activation | `h = torch.tanh(emb.view(-1, 6) @ W1 + B1)` | Applies non-linear activation across 100 hidden units |
| Output Projection ($W_2, B_2$) | `W2 = torch.randn((100, 27))`, `B2 = torch.randn(27)` | Projects hidden features into 27 unnormalized logit values |
| Mini-batch Sampler | `torch.randint(0, X.shape[0], (32,))` | Selects stochastic batches of 32 training instances per step |
| Loss Function | `F.cross_entropy(logits, Y[ix])` | Computes softmax cross-entropy loss over batch targets |
| Parameter Updates | `p.data += -step_sz * p.grad` | Updates all 5 parameter tensors in-place with step size 0.1 |

```python
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
```

<a id="2-bigram-probability-and-evaluation-model-testpy"></a>
### 2. Bigram Probability and Evaluation Model (`test.py`)

Parses `names.txt`, populates an empirical $27 \times 27$ frequency count tensor $N$, normalizes rows to form a transition probability matrix $P$, generates sequences via `torch.multinomial`, and calculates aggregate log-likelihood and average negative log-likelihood (NLL) across all 228,146 bigrams.

| Name | Implementation | Behavior |
| --- | --- | --- |
| `stoi` / `itos` | Dictionary lookups | Maps characters `.` and `a` through `z` to indices $[0, 26]$ and vice versa |
| `N` | `torch.zeros((27, 27), dtype=torch.int32)` | Accumulates co-occurrence counts for every adjacent character pair |
| `P` | `N.float() / P.sum(1, keepdim=True)` | Computes row-normalized conditional probability vectors |
| Multinomial Sampler | `torch.multinomial(p, num_samples=1, replacement=True, generator=G)` | Samples next character index autoregressively until end token `.` |
| Loss Evaluator | `log_likelihood += torch.log(prob)` | Accumulates log probabilities and derives average negative log-likelihood |

```python
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
```

<a id="3-neural-network-bigram-classifier-trainingpy"></a>
### 3. Neural Network Bigram Classifier (`training.py`)

Constructs paired input and target index tensors `xs` and `ys`, initializes a trainable parameter matrix $W \in \mathbb{R}^{27 \times 27}$ with `requires_grad=True`, and executes 10,000 steps of full-batch gradient descent to minimize negative log-likelihood loss.

| Name | Implementation | Behavior |
| --- | --- | --- |
| `xs`, `ys` | `torch.tensor(xs)`, `torch.tensor(ys)` | Encodes 228,146 bigram input-target pairs into 1D tensors |
| `W` | `torch.randn((27,27), generator=G, requires_grad=True)` | Weight matrix initialized with standard Gaussian random values |
| `xenc` | `F.one_hot(xs, num_classes=27).float()` | Encodes integer character indices as 27-dimensional one-hot vectors |
| `logits` | `xenc @ W` | Computes unnormalized log-count predictions via matrix multiplication |
| `probs` | `logits.exp() / logits.exp().sum(1, keepdim=True)` | Normalizes logits into categorical probability distributions via softmax |
| `loss` | `-probs[torch.arange(len(ys)), ys].log().mean()` | Computes mean negative log-likelihood cross-entropy loss |
| Optimizer Step | `W.data += -step_sz * W.grad` | Updates weights along negative gradient with step size 0.1 |

```python
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
```

<a id="4-names-training-dataset-namestxt"></a>
### 4. Names Training Dataset (`names.txt`)

Text corpus containing 32,032 lowercase English names, formatted with one name per line. Provides the data source for vocabulary construction, bigram frequency tabulation, and neural network training examples.

| Name | Implementation | Behavior |
| --- | --- | --- |
| Corpus Size | 32,032 lines (228,145 bytes) | Supplies all training sequences for vocabulary and models |
| Vocabulary Source | Lowercase letters `a` through `z` | Generates 26 distinct alphanumeric character tokens |
| Bigram Volume | 228,146 character transitions | Yields 228,146 input-target pairs including boundary tokens |

```text
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
evelyn
```

<a id="5-environment-dependencies-requirementstxt"></a>
### 5. Environment Dependencies (`requirements.txt`)

Specifies the pinned dependencies required to configure the Python runtime environment for tensor operations, automatic differentiation, and numerical routines.

| Name | Implementation | Behavior |
| --- | --- | --- |
| `torch` | `torch==2.14.1` | Provides tensor compute, automatic differentiation, and functional operations |
| `numpy` | `numpy==2.5.4` | Foundation library for numerical array operations |
| `torchvision` | `torchvision==0.29.1` | Auxiliary vision utilities compatible with PyTorch 2.14.1 |
| Supporting Packages | `filelock`, `fsspec`, `Jinja2`, `pillow`, `sympy`, `typing_extensions` | Provides serialization, caching, templating, symbolic math, and typing primitives |

```text
filelock==4.2.4
fsspec==2026.9.0
Jinja2==3.1.6
MarkupSafe==3.0.4
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
pillow==13.0.0
setuptools==84.2.1
sympy==1.14.0
torch==2.14.1
torchvision==0.29.1
typing_extensions==4.16.0
```

## Project Structure

```text
makemore/
├── mlp.py              # Multi-layer perceptron character language model with embedding layer
├── names.txt           # Text corpus of 32,032 names used for vocabulary and model training
├── requirements.txt    # Pinned dependency specifications for PyTorch and environment
├── test.py             # Bigram frequency counter, autoregressive sampler, and NLL evaluator
└── training.py         # Single-layer neural network bigram model trained with gradient descent
```

<a id="installation--setup"></a>
## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Git

Install Python and Git using your operating system package manager:

```bash
# macOS (Homebrew)
brew install python git

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y python3 python3-venv git

# Windows (winget)
winget install Python.Python.3.11 Git.Git
```

### Repository Setup

Clone the repository, create a virtual environment, and install dependencies:

```bash
# Clone repository
git clone https://github.com/SankalpPimpalkar/makemore.git
cd makemore

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (Command Prompt):
# .venv\Scripts\activate.bat
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

## Usage Examples

### Counting Bigram Frequencies and Sampling

Run bigram probability calculation and autoregressive sampling in Python using the logic from `test.py`:

```python
import torch

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chrs, chrs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

P = N.float()
P /= P.sum(1, keepdim=True)

G = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    st = ""
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=G).item()
        st += itos[ix]
        if ix == 0:
            break
    print(st)
```

### Evaluating Bigram Negative Log-Likelihood

Execute `test.py` to print individual bigram transition probabilities and compute the corpus-wide loss metrics:

```bash
python test.py
```

The script prints the transition probability and log-likelihood for each bigram instance, followed by aggregate values:

```text
log_likelihood=tensor(-559891.7500)
nll=tensor(559891.7500)
2.454094171524048
```

### Training the Linear Network via Full-Batch Gradient Descent

Execute `training.py` to optimize the single-layer neural network across 10,000 gradient descent iterations:

```bash
python training.py
```

The process outputs the negative log-likelihood loss at each iteration:

```text
3.758999824523926
3.7580456733703613
3.7570955753326416
3.7561497688293457
3.7552080154418945
...
```

### Training the Multi-Layer Perceptron via Mini-Batch Gradient Descent

Execute `mlp.py` to train the multi-layer perceptron with 2D character embeddings and a 100-neuron hidden layer over 1,000 mini-batches of size 32:

```bash
python mlp.py
```

The script prints the cross-entropy loss computed on each mini-batch:

```text
loss: 16.99164581298828
loss: 15.469512939453125
loss: 12.088422775268555
loss: 12.045230865478516
loss: 11.481245040893555
...
```

## How It Works

1. **Vocabulary Construction and Special Token Mapping**:
   Reads all names from `names.txt`, extracts the sorted set of unique characters, maps the boundary marker `.` to index 0, and assigns lowercase letters `a` through `z` to indices 1 through 26.
   ```python
   words = open("names.txt", "r").read().splitlines()
   chars = sorted(list(set("".join(words))))
   stoi = {ch:i+1 for i,ch in enumerate(chars)}
   stoi['.'] = 0
   itos = {i:ch for ch,i in stoi.items()}
   ```

2. **Dataset Construction: Bigrams and Context Windows**:
   - For bigram models, each name is padded with `.` at start and end, extracting consecutive pairs $(c_t, c_{t+1})$.
   - For the multi-layer perceptron, a sliding context window of size $k=3$ is maintained:
   ```python
   block_sz = 3
   X, Y = [], []
   for w in words:
       ctx = [0] * block_sz
       for ch in w + '.':
           ix = stoi[ch]
           X.append(ctx)
           Y.append(ix)
           ctx = ctx[1:] + [ix]
   ```

3. **Empirical Probability Normalization (Statistical Baseline)**:
   Populates the frequency tensor $N \in \mathbb{Z}^{27 \times 27}$ and normalizes each row by its sum to obtain conditional probabilities $P_{i,j} = \frac{N_{i,j}}{\sum_{k=0}^{26} N_{i,k}}$.
   ```python
   P = N.float()
   P /= P.sum(1, keepdim=True)
   ```

4. **Linear Bigram Forward Pass and Softmax**:
   Converts input character indices into one-hot vectors $x_{\text{enc}} \in \mathbb{R}^{228146 \times 27}$, computes logits via linear transformation $z = x_{\text{enc}} W$, and calculates normalized probabilities using softmax:
   $$P(y = j \mid x) = \frac{\exp(z_j)}{\sum_{k=0}^{26} \exp(z_k)}$$
   ```python
   xenc = F.one_hot(xs, num_classes=27).float()
   logits = xenc @ W 
   counts = logits.exp()
   probs = counts / counts.sum(1, keepdim=True)
   loss = -probs[torch.arange(len(ys)), ys].log().mean()
   ```

5. **Multi-Layer Perceptron with Distributed Embeddings**:
   Embeds context tokens through embedding table $C \in \mathbb{R}^{27 \times 2}$, concatenates the vectors into shape $(B, 6)$, passes through a tanh hidden layer ($W_1 \in \mathbb{R}^{6 \times 100}, B_1 \in \mathbb{R}^{100}$), projects to logits ($W_2 \in \mathbb{R}^{100 \times 27}, B_2 \in \mathbb{R}^{27}$), and calculates cross-entropy loss:
   ```python
   emb = C[X[ix]]
   h = torch.tanh(emb.view(-1, 6) @ W1 + B1)
   logits = h @ W2 + B2 
   loss = F.cross_entropy(logits, Y[ix])
   ```

6. **Autograd Backpropagation and Parameter Updates**:
   Computes analytical gradients $\nabla_\theta \mathcal{L}$ for all parameters using `loss.backward()` and updates values using gradient descent:
   $$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$
   ```python
   for p in parameters:
       p.grad = None
   loss.backward()
   for p in parameters:
       p.data += -step_sz * p.grad
   ```

## Project Roadmap & Commit History

1. **Initial Repository Setup and Dataset Integration** (`72e0e40`)
   - Added `names.txt` containing 32,032 training names.
   - Added `requirements.txt` documenting environment dependencies for PyTorch and numerical libraries.
   - Added initial baseline script `main.py` with bigram count matrix generation and multinomial sampling.

2. **Metric Evaluation and Model Architectures** (`Current`)
   - Replaced initial script with `test.py` to add full-dataset negative log-likelihood calculation and transition diagnostics.
   - Introduced `training.py` implementing single-layer linear bigram modeling with one-hot input encoding, softmax normalization, and full-batch gradient descent.
   - Added `mlp.py` implementing a multi-layer perceptron language model (Bengio et al. 2003) with 2D character embedding tables, context window size 3, tanh non-linearities, and mini-batch stochastic gradient descent.
