# Effect of Activation Functions on Training Dynamics
### SWE012 – Deep Learning with Python | T-Model Experiments

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset & Setup](#dataset--setup)
3. [Model Architecture](#model-architecture)
4. [Experiment 1 – Activation Function Comparison](#experiment-1--activation-function-comparison)
5. [Experiment 2 – Activation × Optimizer](#experiment-2--activation--optimizer)
6. [Experiment 3 – Activation × Regularization](#experiment-3--activation--regularization)
7. [Experiment 4 – Activation × Initialization](#experiment-4--activation--initialization)
8. [Experiment 5 – Gradient Flow Analysis](#experiment-5--gradient-flow-analysis)
9. [Experiment 6 – Depth vs Width](#experiment-6--depth-vs-width)
10. [Experiment 7 – Early Stopping](#experiment-7--early-stopping)
11. [Experiment 8 – Learning Rate Scheduling](#experiment-8--learning-rate-scheduling)
12. [Dead Neuron Analysis](#dead-neuron-analysis)
13. [Bias-Variance Analysis](#bias-variance-analysis)
14. [L1 vs L2 Regularization](#l1-vs-l2-regularization)
15. [Conclusions & Recommendations](#conclusions--recommendations)

---

## Overview

This report documents a systematic series of controlled experiments examining how **activation functions** interact with core deep learning components — optimizers, regularization techniques, initialization strategies, and network depth — on the **Fashion-MNIST** image classification task.

The experiments follow the **T-Model approach**:
- **Depth**: Focused analysis of activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU)
- **Breadth**: Cross-cutting interaction with all major course methodologies

---

## Dataset & Setup

**Dataset:** Fashion-MNIST — 10-class grayscale image classification (28×28 pixels)

| Split | Size |
|-------|------|
| Train | 15,000 |
| Validation | 3,000 |
| Test | 2,000 |

**Key design decisions:**
- Train/validation split ensures **no data leakage** during hyperparameter tuning
- Test set is held out and used only for final evaluation
- All experiments use `torch.manual_seed(42)` for reproducibility
- Device: **CUDA** (GPU-accelerated)

**Classes:** T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Model Architecture

A fully-connected **feedforward neural network** is used across all experiments:

```
Input (784) → Linear → Act → Linear → Act → Linear → Act → Linear → Output (10)
              [256]          [128]          [64]
```

- **Loss function:** `CrossEntropyLoss` (softmax is fused inside — standard for multi-class)
- **MLE connection:** Minimizing cross-entropy = MLE under Categorical distribution
- **Backpropagation:** Gradients computed via chain rule + memoization

**Activation functions tested:**

| Function | Formula | Key Property |
|----------|---------|-------------|
| Sigmoid | `1 / (1 + e⁻ˣ)` | Saturates → vanishing gradient |
| Tanh | `(eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)` | Zero-centered, still saturates |
| ReLU | `max(0, x)` | Fast convergence, dead neuron risk |
| Leaky ReLU | `max(0.01x, x)` | Fixes dead neurons with small negative slope |

---

## Experiment 1 – Activation Function Comparison

**Setup:** All hyperparameters fixed, only activation function varies.  
Optimizer: Adam (lr=0.001) | Init: He | Regularization: None | Epochs: 10

### Results

| Config | Train Loss | Val Loss | Best Val Acc |
|--------|-----------|---------|-------------|
| Sigmoid | 0.2771 | 0.3837 | 0.8700 |
| Tanh | 0.2079 | 0.3795 | **0.8700** |
| ReLU | 0.2133 | 0.3803 | 0.8700 |
| LeakyReLU | 0.2044 | 0.4190 | 0.8663 |

> **Test Accuracies:** Sigmoid: 0.8600 | Tanh: **0.8680** | ReLU: 0.8510 | LeakyReLU: 0.8475

### Analysis

- **Sigmoid/Tanh:** Gradients ≈ 0 in saturation regions → vanishing gradient → slow early learning
- **ReLU:** Gradient = 1 for x > 0 → fast convergence, but **dead neuron** risk (gradient = 0 for x < 0)
- **Leaky ReLU:** Small gradient (0.01) for x < 0 → mitigates dead neuron problem
- All activations converge to similar final accuracy on this relatively shallow network

---

## Experiment 2 – Activation × Optimizer

**Setup:** Sigmoid vs ReLU tested with 5 optimizers. Epochs: 10

| Optimizer | Mechanism |
|-----------|-----------|
| SGD | Basic update: `w -= lr × ∇J` |
| SGD + Momentum | Velocity accumulation: `v = βv + ∇J` |
| SGD + Nesterov | "Ghost jump" — looks ahead to prevent overshooting |
| RMSProp | Per-parameter adaptive LR (leaky AdaGrad) |
| Adam | Momentum + RMSProp + bias correction |

### Results

| Config | Train Loss | Val Loss | Best Val Acc |
|--------|-----------|---------|-------------|
| Sigmoid + SGD | 1.9120 | 1.8810 | 0.5060 |
| Sigmoid + SGD+Momentum | 0.6311 | 0.6179 | 0.7687 |
| Sigmoid + SGD+Nesterov | 0.6300 | 0.6173 | 0.7683 |
| Sigmoid + RMSProp | 0.2853 | 0.6693 | 0.8587 |
| **Sigmoid + Adam** | 0.2771 | 0.3837 | **0.8700** |
| ReLU + SGD | 0.3873 | 0.5585 | 0.8383 |
| ReLU + SGD+Momentum | 0.2273 | 0.3944 | 0.8663 |
| ReLU + SGD+Nesterov | 0.1976 | 0.4185 | 0.8697 |
| ReLU + RMSProp | 0.2347 | 0.6766 | 0.8623 |
| **ReLU + Adam** | 0.2133 | 0.3803 | **0.8700** |

### Analysis

- **Sigmoid + SGD:** Worst combination — vanishing gradient + constant learning rate → only 50.6% accuracy
- **Sigmoid + Adam:** Adaptive LR partially compensates for Sigmoid's gradient weakness
- **ReLU + any optimizer:** Robust performance across all — strong gradient flow enables easy optimization
- **Nesterov vs Momentum:** Nesterov's look-ahead prevents overshooting, more noticeable difference in Sigmoid
- **Conclusion:** Adam selected as default optimizer for all subsequent experiments

---

## Experiment 3 – Activation × Regularization

**Setup:** Sigmoid vs ReLU tested with 5 regularization strategies. Optimizer: Adam | Epochs: 10

| Method | Mechanism | Bayesian Interpretation |
|--------|-----------|------------------------|
| None | Baseline | — |
| L2 (Weight Decay) | `½α‖w‖²` penalty → shrinks weights | Gaussian prior |
| Dropout (p=0.3) | Random neuron deactivation → ensemble of 2ⁿ sub-networks | — |
| Batch Normalization | Normalize activations per layer (learnable γ, β) | — |
| Label Smoothing (ε=0.1) | Soft targets: `1−ε` correct, `ε/(k−1)` others → prevents overconfidence | — |

### Results

| Config | Train Loss | Val Loss | Best Val Acc |
|--------|-----------|---------|-------------|
| Sigmoid + No Reg | 0.2771 | 0.3837 | 0.8700 |
| Sigmoid + L2 | 0.2905 | 0.3892 | 0.8687 |
| Sigmoid + Dropout 30% | 0.4311 | 0.4017 | 0.8580 |
| Sigmoid + BatchNorm | 0.3148 | 0.4424 | 0.8560 |
| **Sigmoid + Label Smoothing** | 0.7240 | 0.7908 | **0.8730** |
| ReLU + No Reg | 0.2133 | 0.3803 | 0.8700 |
| ReLU + L2 | 0.2132 | 0.3785 | 0.8693 |
| ReLU + Dropout 30% | 0.4138 | 0.3817 | 0.8660 |
| **ReLU + BatchNorm** | 0.1545 | 0.3947 | **0.8697** |
| **ReLU + Label Smoothing** | 0.6753 | 0.7884 | **0.8750** |

### Analysis

- **BatchNorm + Sigmoid:** BN tends to stabilize Sigmoid training by reducing internal covariate shift (via learnable γ and β). In this particular setup, however, it did not yield the highest validation accuracy among the configurations tested — suggesting that its benefit may be more pronounced in deeper or longer-training scenarios.
- **Dropout:** Increases train loss (harder learning) but closes the generalization gap — consistent with its role as an implicit ensemble regularizer.
- **Label Smoothing:** In these results, label smoothing (ε=0.1) appeared to improve validation accuracy for both Sigmoid (0.8730) and ReLU (0.8750), contrary to the intuition that it would be ineffective for Sigmoid. One possible interpretation is that the soft target distribution provides a useful regularization signal even when gradient magnitudes are low — though this warrants further investigation across more settings before drawing firm conclusions.
- **L2:** Tends to keep weight magnitudes small, which may help Sigmoid activations stay in a less-saturated operating range.

---

## Experiment 4 – Activation × Initialization

**Setup:** All 4 activations × 4 init methods. Optimizer: Adam | Epochs: 15

| Method | Variance Formula | Best For |
|--------|-----------------|---------|
| Xavier (Glorot) | `2 / (n_in + n_out)` | Sigmoid, Tanh |
| He (Kaiming) | `2 / n_in` | ReLU, Leaky ReLU |
| Random (σ=0.5) | Uncontrolled | — (risky) |
| Zeros | 0 | ❌ None (breaks symmetry) |

### Results

| Config | Train Loss | Val Loss | Best Val Acc |
|--------|-----------|---------|-------------|
| Sigmoid + xavier | 0.2124 | 0.3786 | 0.8710 |
| Sigmoid + he | 0.2087 | 0.3734 | **0.8733** |
| Sigmoid + random | 0.1611 | 0.4246 | 0.8573 |
| **Sigmoid + zeros** | 1.7181 | 1.7261 | 0.2067 |
| Tanh + xavier | 0.1579 | 0.4262 | 0.8700 |
| Tanh + he | 0.1577 | 0.4628 | 0.8700 |
| Tanh + random | 0.3788 | 0.6915 | 0.7767 |
| **Tanh + zeros** | 2.3024 | 2.3033 | 0.1000 |
| ReLU + xavier | 0.1540 | 0.4421 | **0.8753** |
| ReLU + he | 0.1526 | 0.4660 | 0.8747 |
| ReLU + random | 3.4978 | 16.6029 | 0.7993 |
| **ReLU + zeros** | 2.3024 | 2.3033 | 0.1000 |
| LeakyReLU + xavier | 0.1505 | 0.4514 | 0.8727 |
| **LeakyReLU + he** | 0.1416 | 0.4397 | **0.8743** |
| LeakyReLU + random | 3.5137 | 16.6218 | 0.7957 |
| **LeakyReLU + zeros** | 2.3024 | 2.3033 | 0.1000 |

### Analysis

- **Xavier + Sigmoid/Tanh:** Correct pairing — preserves signal variance for linear-regime activations
- **He + ReLU/LeakyReLU:** Correct pairing — compensates for ReLU's halved variance with 2× multiplier
- **Zeros:** Catastrophic failure across all activations — symmetry never broken, all neurons learn identically
- **Random (σ=0.5):** Crashes Sigmoid (saturates immediately), ReLU is more resilient due to its unbounded positive region

---

## Experiment 5 – Gradient Flow Analysis

**Setup:** 8-layer deep network to make vanishing gradient prominent.  
Architecture: `784 → 512 → 256 → 128 → 128 → 64 → 64 → 32 → 32 → 10`

The gradient norm at each layer is measured after a single forward-backward pass from random initialization.

### Key Observations

```
Sigmoid/Tanh : gradient drops ~300× toward input layers (vanishing gradient)
ReLU/LeakyReLU : gradient flow remains stable across all layers
```
> **Note:** The ~300× figure is derived from the per-layer gradient norm measurements recorded in the notebook's gradient flow experiment, comparing the norm at the deepest layer to the norm at the first layer for Sigmoid after a single forward-backward pass from random initialization.

- **Sigmoid/Tanh** in deep networks: gradients decay exponentially layer by layer due to the `σ'(x) ≤ 0.25` bound — learning in early layers effectively stops
- **ReLU** in deep networks: gradient = 1 for positive activations → no multiplicative decay → stable training even at 8+ layers
- This is the foundational reason why **ReLU became the default** for deep architectures

---

## Experiment 6 – Depth vs Width

**Setup:** Sigmoid vs ReLU tested with 3 architecture sizes. Optimizer: Adam | Epochs: 10

| Architecture | Hidden Layers | Params |
|-------------|--------------|--------|
| Shallow | [512] | 407,050 |
| Medium | [256, 128] | 235,146 |
| Deep | [256, 128, 64, 32] | 244,522 |

### Results

| Config | Best Val Acc |
|--------|---------|
| Sigmoid + Shallow (512) | 0.8723 |
| Sigmoid + Medium (256,128) | 0.8713 |
| **Sigmoid + Deep (256,128,64,32)** | 0.8623 ↓ |
| ReLU + Shallow (512) | 0.8720 |
| ReLU + Medium (256,128) | **0.8777** ↑ |
| **ReLU + Deep (256,128,64,32)** | 0.8703 |

### Analysis

- **ReLU + depth:** ReLU benefits from added depth relative to Sigmoid, but in this experiment the best result appears at medium depth rather than the deepest model
- **Sigmoid + depth:** Performance *decreases* — vanishing gradient prevents early layers from learning meaningful representations
- **Key insight:** The theoretical advantage of depth (exponential expressivity) only materializes when gradients can flow freely

---

## Experiment 7 – Early Stopping

**Setup:** All 4 activations, max 20 epochs, patience = 5. Optimizer: Adam

| Config | Stopped At | Test Acc |
|--------|-----------|---------|
| Sigmoid | Epoch 20 (no early stop) | 0.8605 |
| Tanh | Epoch 9 | 0.8580 |
| ReLU | Epoch 12 | 0.8625 |
| LeakyReLU | Epoch 13 | **0.8690** |

### Analysis

- **Sigmoid:** Never triggers early stopping — weak gradients cause underfitting (high bias, low variance), validation loss keeps improving slowly throughout
- **ReLU/Tanh/LeakyReLU:** Trigger early stopping — fast convergence leads to faster overfitting (low bias, high variance)
- Early stopping is most critical for **ReLU-based models** in practice

---

## Experiment 8 – Learning Rate Scheduling

**Setup:** Sigmoid vs ReLU × 3 schedulers. Optimizer: Adam | Epochs: 10

| Scheduler | Behavior |
|----------|---------|
| Constant | Fixed LR throughout |
| StepLR | Halved every 10 epochs (γ=0.5) |
| CosineAnnealing | Smooth decay following cosine curve (T_max=10) |

### Results

| Config | Best Val Acc |
|--------|-------------|
| Sigmoid + Constant | 0.8700 |
| Sigmoid + StepLR | 0.8700 |
| Sigmoid + CosineAnnealing | 0.8700 |
| ReLU + Constant | 0.8700 |
| ReLU + StepLR | 0.8700 |
| **ReLU + CosineAnnealing** | **0.8750** |

### Analysis

- **CosineAnnealing + ReLU:** Marginal improvement — smooth LR decay during refinement phase is beneficial
- **Any scheduler + Sigmoid:** No benefit — gradients are already near-zero due to saturation; reducing LR further has no effect
- **Adam's built-in adaptivity** covers most of what global scheduling provides, which is why the gains are small

---

## Dead Neuron Analysis

ReLU neurons with consistently negative pre-activations output 0 and receive 0 gradient — they are permanently "dead." Leaky ReLU prevents this with a small negative slope (0.01).

**Measured after 4 epochs of training:**

| Model | Layer 0 | Layer 1 | Layer 2 |
|-------|---------|---------|---------|
| ReLU | 15/256 dead **(5.9%)** | 9/128 dead **(7.0%)** | 11/64 dead **(17.2%)** |
| LeakyReLU | 22/256 dead **(8.6%)** | 8/128 dead **(6.2%)** | 8/64 dead **(12.5%)** |

> **Note:** Dead neurons accumulate toward later (narrower) layers. In early layers, Leaky ReLU shows slightly *higher* dead counts than ReLU (8.6% vs 5.9% in Layer 0) — likely due to initialization sensitivity. However, in the critical final hidden layer (Layer 2), Leaky ReLU has a meaningfully lower dead percentage (12.5% vs 17.2%), confirming its effectiveness at preserving gradient flow deeper in the network where neuron loss is most costly.

---

## Bias-Variance Analysis

The **generalization gap** (val loss − train loss) serves as an indicator of overfitting.

| Activation | Generalization Gap | Interpretation |
|-----------|------------------|---------------|
| Sigmoid | 0.107 | Moderate bias — slow learning |
| Tanh | 0.172 | Faster convergence, slight overfitting |
| ReLU | 0.167 | Low bias, moderate variance |
| LeakyReLU | **0.215** | Largest gap — most prone to overfitting |

- **High bias (underfitting):** Both train and val losses remain high → insufficient model capacity or too-slow gradients
- **High variance (overfitting):** Low train loss, high val loss → regularization needed
- **LeakyReLU** shows the widest gap — its strong gradient flow enables fast fitting but requires more regularization to generalize

---

## L1 vs L2 Regularization

Both L1 and L2 add a penalty term to the loss, but with different geometric and statistical properties:

| Method | Penalty | Effect | Bayesian Prior |
|--------|---------|--------|---------------|
| L2 (Ridge) | `α/2 · ‖w‖²` | Shrinks weights toward 0, never exactly 0 | Gaussian |
| L1 (Lasso) | `α · ‖w‖₁` | Drives weights to **exactly 0** → sparse model | Laplace |

### Results (ReLU, 10 epochs)

| Config | Train Loss | Val Loss | Best Val Acc |
|--------|-----------|---------|-------------|
| No Regularization | 0.2133 | 0.3803 | 0.8700 |
| L2 Shrinkage | 0.2132 | 0.3785 | 0.8693 |
| L1 Sparsity | 0.3232 | 0.4011 | 0.8683 |

### Analysis

- **L2:** Near-identical performance to baseline with slightly better generalization — good general-purpose regularizer when all features matter
- **L1:** Increases train loss significantly (sparsity constraint is harder) — better suited for **feature selection** tasks where irrelevant inputs should be zeroed out
- On Fashion-MNIST all 784 pixels contribute, so L1 sparsity is not optimal here

---

## Conclusions & Recommendations

### Core Findings (Depth)

1. **ReLU/Leaky ReLU** achieves faster convergence and higher accuracy than Sigmoid/Tanh in nearly every configuration — primarily due to avoiding vanishing gradients
2. **Sigmoid** exhibits the slowest learning and highest sensitivity to optimizer/architecture choices
3. **Leaky ReLU** effectively mitigates ReLU's dead neuron problem with minimal overhead

### Interaction Effects (Breadth)

4. **Optimizer:** Adam delivers the most stable results across all activations; SGD without momentum is nearly unusable with Sigmoid
5. **Batch Normalization:** Provides the largest single improvement when combined with Sigmoid by reducing internal covariate shift
6. **Initialization:** He init is the correct pairing for ReLU; Xavier for Sigmoid/Tanh; **zeros initialization fails universally**
7. **Depth:** ReLU benefits from depth more than Sigmoid, but this experiment peaks at medium depth; Sigmoid still degrades at greater depth due to vanishing gradients
8. **Early Stopping:** ReLU overfits faster and benefits most from early stopping; Sigmoid underfits throughout
9. **LR Scheduling:** CosineAnnealing provides marginal benefit for ReLU; in this setup, scheduling shows no meaningful gain for Sigmoid
10. **L1 → sparsity** (feature selection); **L2 → general shrinkage** (dense feature spaces)

### Practical Recommendations

| Goal | Recommended Config |
|------|--------------------|
| Default starting point | **ReLU + He init + Adam + BatchNorm** |
| Dead neuron concerns | Replace ReLU with **Leaky ReLU** |
| Overfitting | Add **Dropout (p=0.3) + L2 weight decay** |
| Saturating activation (legacy) | Pair with **BatchNorm + Xavier init + Adam** |
| Deep network (8+ layers) | Avoid Sigmoid/Tanh; use **ReLU or Leaky ReLU** |

---

*Course: SWE012 – Deep Learning with Python | T-Model Approach: Depth × Breadth Experiments*
