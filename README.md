# Impact of Activation Functions on Training Dynamics

A controlled deep learning experiment studying how different activation functions affect convergence speed, gradient flow, and final accuracy — using MNIST as the benchmark dataset.

---

## Overview

This notebook implements a **T-shaped experiment design**: activation functions are the core topic, with regularization, optimizer choice, and architecture depth serving as supporting variables. Every hyperparameter is held constant across runs — only the activation function changes — making it a clean, reproducible comparison.

**Dataset:** MNIST Handwritten Digits (70,000 images, 28×28 pixels)  
**Framework:** PyTorch  
**Optimizer:** SGD (lr=0.01) — chosen deliberately to make gradient flow differences clearly visible  
**Epochs:** 15

---

## Activation Functions Compared

| Function | Formula | Key Property |
|---|---|---|
| **Baseline** | Linear (no activation) | Lower-bound reference; linear decision boundary only |
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | Saturates → vanishing gradients |
| **Tanh** | tanh(x) | Zero-centered, still saturates at ±1 |
| **ReLU** | max(0, x) | No saturation, but can produce dead neurons |
| **LeakyReLU** | max(0.01x, x) | Fixes dead neuron problem with a small negative slope |

---

## Research Questions

1. How do different activation functions affect **convergence speed** and **final accuracy**?
2. Which activation functions suffer from the **vanishing gradient problem** in deep networks?
3. How many neurons become **"dead"** (zero-activation) with ReLU in deeper architectures?
4. How do **activation distributions** evolve across layers during training?

---

## Model Architecture

### Baseline — Logistic Regression
A single linear layer (784 → 10). No activation function, no hidden layers. Serves as the lower-bound reference.

### Deep MLP (4 Hidden Layers)
```
Input (784) → Linear(128) → Activation → Dropout(0.2)
           → Linear(128) → Activation → Dropout(0.2)
           → Linear(128) → Activation → Dropout(0.2)
           → Linear(128) → Activation
           → Output(10)
```

> **Why 4 hidden layers?** Vanishing gradients compound multiplicatively with depth. A shallow network won't expose the problem clearly.  
> **Why SGD instead of Adam?** Adam's adaptive learning rates partially mask the vanishing gradient problem. SGD with a fixed learning rate makes gradient flow differences *clearly visible*.

---

## What Is Measured

Beyond standard loss and accuracy tracking, the experiment records:

- **Per-layer gradient norms** at each epoch — reveals vanishing or exploding gradients across the network depth
- **Dead neuron counts** — neurons that output zero for all inputs in a batch, permanently locked out of learning

---

## Key Results

| Model | Final Test Accuracy | Behavior |
|---|---|---|
| Baseline | ~92% | Plateaus; limited to linear boundaries |
| ReLU | ~97% | Fast convergence, healthy gradients |
| LeakyReLU | ~97% | Same as ReLU, zero dead neurons |
| Tanh | ~95–96% | Middle ground; some saturation |
| Sigmoid | ~11% (near random) | Vanishing gradients; barely learns |

### Gradient Flow
Sigmoid's derivative is bounded at 0.25. Across 4 layers: 0.25⁴ ≈ **0.004** — early layers receive gradients ~250× smaller than the output layer. They effectively stop learning. ReLU's derivative is 1 for positive inputs, allowing gradients to flow backward without shrinking.

### Dead Neurons
A small percentage of ReLU neurons die during training (concentrated in early layers). LeakyReLU eliminates this entirely by using a slope of 0.01 for negative inputs, ensuring a non-zero gradient always flows.

---

## Conclusions

1. **Sigmoid is unsuitable for hidden layers in deep networks** — vanishing gradients prevent early layers from learning.
2. **ReLU and LeakyReLU converge fastest** and reach the highest accuracy with SGD.
3. **Dead neurons are real but manageable** for ReLU; LeakyReLU eliminates them at negligible cost.
4. **Tanh is a middle ground** — zero-centered and better than Sigmoid, but still saturates in deep or poorly initialized networks.
5. **Depth amplifies all differences** — activation function choice becomes critical as networks grow deeper.

### Design Recommendations

| Use Case | Recommended Activation |
|---|---|
| Default starting point | **ReLU** |
| Deep networks, worried about dead neurons | **LeakyReLU** or ELU |
| Output layer — binary classification | Sigmoid |
| Output layer — multi-class | Softmax |
| Avoid in hidden layers | **Sigmoid** |

---

## Limitations & Future Work

- Only MNIST was tested — more complex datasets (CIFAR-10, ImageNet) may show different relative performance
- SGD was chosen to amplify gradient differences; Adam would compress the gap significantly
- Modern activations used in transformers (GELU, Swish, Mish) were not included — a natural extension
- Batch normalization was excluded intentionally, as it largely solves vanishing gradients and would mask activation function differences

---

## Requirements

```
torch
torchvision
matplotlib
numpy
pandas
tqdm
```

Install with:
```bash
pip install torch torchvision matplotlib numpy pandas tqdm
```

---

## Running the Notebook

```bash
jupyter notebook Impact_of_Activation_Functions.ipynb
```

All experiments are reproducible — random seeds are fixed at `torch.manual_seed(42)` and `np.random.seed(42)`.
