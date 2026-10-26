# makemore

makemore is an autoregressive character-level language modeling codebase inspired by Andrej Karpathy's deep learning tutorial series. The repository explores statistical and neural approaches to sequence generation by predicting character transitions over a vocabulary of 26 lowercase English letters and a boundary token. It implements two distinct modeling methodologies on a dataset of 32,032 names: an empirical bigram model that tabulates transition frequencies into a normalized 2D probability matrix, and a single-layer neural network that optimizes negative log-likelihood loss through one-hot encodings, matrix multiplication, softmax normalization, and gradient descent backpropagation using PyTorch.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Bigram Probability and Evaluation Model (`test.py`)](#1-bigram-probability-and-evaluation-model-testpy)
  - [2. Neural Network Bigram Classifier (`training.py`)](#2-neural-network-bigram-classifier-trainingpy)
  - [3. Names Training Dataset (`names.txt`)](#3-names-training-dataset-namestxt)
  - [4. Environment Dependencies (`requirements.txt`)](#4-environment-dependencies-requirementstxt)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
  - [Counting Bigram Frequencies and Sampling](#counting-bigram-frequencies-and-sampling)
  - [Evaluating Bigram Negative Log-Likelihood](#evaluating-bigram-negative-log-likelihood)
  - [Training the Neural Network via Gradient Descent](#training-the-neural-network-via-gradient-descent)
- [How It Works](#how-it-works)
- [Project Roadmap & Commit History](#project-roadmap--commit-history)

## Project Overview

Character-level autoregressive models predict the distribution of the next character conditioned on prior context: $P(x_t \mid x_1, x_2, \dots, x_{t-1})$. The simplest non-trivial formulation is the bigram model, which applies a first-order Markov assumption where transition probabilities depend solely on the single immediately preceding character: $P(x_t \mid x_{t-1})$. This repository establishes the algorithmic mechanics of sequence modeling by contrasting exact statistical frequency counting against gradient-based neural parameter estimation on identical data.

1. **First-Order Markov Formulation**: Models character transitions under the assumption that the current character depends exclusively on the previous character.
2. **Bijective Tokenization**: Maps lowercase characters `a` through `z` and a sequence boundary token `.` into fixed integers from 0 to 26.
3. **Empirical Frequency Tabulation**: Aggregates co-occurrence counts of all adjacent character pairs across the dataset into a 2D integer tensor $N \in \mathbb{Z}^{27 \times 27}$.
4. **Row-Stochastic Probability Normalization**: Converts raw frequencies into categorical probability distributions summing to 1 across rows: $P \in \mathbb{R}^{27 \times 27}$.
5. **Autoregressive Sequence Sampling**: Generates variable-length sequences iteratively using categorical multinomial sampling until the terminating token is drawn.
6. **Negative Log-Likelihood Loss Objective**: Measures model prediction error through cross-entropy loss: $\mathcal{L} = -\frac{1}{n} \sum \log P(\text{target} \mid \text{input})$.
7. **Neural Parameter Optimization**: Formulates transition probabilities as a linear transformation $W \in \mathbb{R}^{27 \times 27}$ optimized using full-batch gradient descent.

## Key Features

- **Dual Modeling Paradigms**: Provides two distinct implementations for character-level bigram modeling.
  - **Count-based Matrix Model**: Tabulates all 228,146 bigram transitions from the dataset into an explicit 2D frequency matrix.
  - **Linear Neural Network**: Employs a single trainable linear layer without hidden activations to learn transition logits.
- **Deterministic Tokenization System**: Encodes strings to integers and decodes integers back to strings.
  - **Boundary Delimiter**: Uses `.` at index 0 to designate both start-of-word and end-of-word states.
  - **Alphabet Mapping**: Assigns lowercase English letters `a` through `z` to contiguous indices 1 through 26.
- **Autoregressive Multinomial Sampling**: Samples synthetic names token by token conditioned on previous outputs with a deterministic seed generator.
- **Comprehensive Loss Evaluation**: Computes the exact log-likelihood, aggregate negative log-likelihood, and average per-bigram loss across the entire corpus.
- **Pure PyTorch Gradient Descent**: Executes the forward pass, softmax normalization, loss computation, backpropagation, and weight updates directly without external training abstractions.

## Core Components

<a id="1-bigram-probability-and-evaluation-model-testpy"></a>
### 1. Bigram Probability and Evaluation Model (`test.py`)

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

<a id="2-neural-network-bigram-classifier-trainingpy"></a>
### 2. Neural Network Bigram Classifier (`training.py`)

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

<a id="3-names-training-dataset-namestxt"></a>
### 3. Names Training Dataset (`names.txt`)

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

<a id="4-environment-dependencies-requirementstxt"></a>
### 4. Environment Dependencies (`requirements.txt`)

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

### Training the Neural Network via Gradient Descent

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

2. **Bigram Extraction and Sequence Pairing**:
   Wraps each name with start and end tokens `.` and iterates through adjacent character pairs $(c_t, c_{t+1})$ to extract training examples.
   ```python
   for w in words:
       chrs = ["."] + list(w) + ["."]
       for ch1, ch2 in zip(chrs, chrs[1:]):
           i1, i2 = stoi[ch1], stoi[ch2]
           xs.append(i1)
           ys.append(i2)
   ```

3. **Empirical Probability Normalization**:
   Populates the frequency tensor $N \in \mathbb{Z}^{27 \times 27}$ and normalizes each row by its sum to obtain conditional probabilities $P_{i,j} = \frac{N_{i,j}}{\sum_{k=0}^{26} N_{i,k}}$.
   ```python
   P = N.float()
   P /= P.sum(1, keepdim=True)
   ```

4. **Negative Log-Likelihood Evaluation**:
   Computes the log probability for each transition and aggregates the negative log-likelihood across all $n = 228,146$ bigrams:
   $$\mathcal{L}_{\text{NLL}} = -\frac{1}{n} \sum_{i=1}^n \log P(y^{(i)} \mid x^{(i)})$$
   ```python
   prob = P[i1, i2]
   logprob = torch.log(prob)
   log_likelihood += logprob
   n += 1
   ```

5. **Neural Forward Pass and Softmax Normalization**:
   Converts input character indices into one-hot vectors $x_{\text{enc}} \in \mathbb{R}^{228146 \times 27}$, computes logits via linear transformation $z = x_{\text{enc}} W$, and calculates normalized probabilities using softmax:
   $$P(y = j \mid x) = \frac{\exp(z_j)}{\sum_{k=0}^{26} \exp(z_k)}$$
   ```python
   xenc = F.one_hot(xs, num_classes=27).float()
   logits = xenc @ W 
   counts = logits.exp()
   probs = counts / counts.sum(1, keepdim=True)
   loss = -probs[torch.arange(len(ys)), ys].log().mean()
   ```

6. **Autograd Backpropagation and Parameter Updates**:
   Computes gradients $\nabla_W \mathcal{L}$ via PyTorch autograd and performs an in-place gradient descent step with learning rate $\eta = 0.1$:
   $$W \leftarrow W - \eta \nabla_W \mathcal{L}$$
   ```python
   W.grad = None
   loss.backward()
   step_sz = 0.1
   W.data += -step_sz * W.grad
   ```

## Project Roadmap & Commit History

1. **Initial Repository Setup and Dataset Integration** (`72e0e40`)
   - Added `names.txt` containing 32,032 training names.
   - Added `requirements.txt` documenting environment dependencies for PyTorch and numerical libraries.
   - Added initial baseline script `main.py` with bigram count matrix generation and multinomial sampling.

2. **Metric Evaluation and Neural Network Implementation** (`Current`)
   - Replaced initial script with `test.py` to add full-dataset negative log-likelihood calculation and transition diagnostics.
   - Introduced `training.py` implementing single-layer neural network language modeling with one-hot input encoding, softmax normalization, and iterative gradient descent optimization.
