# REPORT.md — The Impact of Activation Functions on Training Dynamics

**Course:** SWE012 — Deep Learning with Python  
**University:** İstinye University  
**Instructor:** Asst. Prof. Yiğit Bekir Kaya  
**Scope:** Weeks 2–5  
**Dataset:** MNIST Handwritten Digits (54,000 train / 6,000 Validation / 10,000 test)  
**Framework:** PyTorch

---

## 1. Project Overview & T-Model Design

This project follows the **T-Model** approach recommended by the instructor: one core topic explored in depth (the vertical bar of the T), with all other weekly topics covered as breadth (the horizontal bar).

- **DEPTH (Core):** Activation Functions — Sigmoid, Tanh, ReLU, LeakyReLU. We hold every other variable constant and measure how the activation function alone determines whether a 4-layer network can learn.
- **BREADTH (Week 2):** Bias-Variance Tradeoff, MLE, Capacity, Point Estimation, SGD
- **BREADTH (Week 3):** Softmax & Logits, BCE vs CCE, Depth vs Width, Backpropagation
- **BREADTH (Week 4):** L1/L2 Regularization, Dropout, Label Smoothing, Adversarial Training (FGSM), Semi/Self-Supervised Learning
- **BREADTH (Week 5):** He/Xavier Initialization, SGD/Momentum/Adam (+ Nesterov/AdaGrad/RMSProp theory), Batch Normalization, Layer Normalization, LR Scheduler (bonus)

**Why we chose this topic over alternatives:** Activation functions are the single design choice that determines whether a deep network can learn at all — not just how fast. Sigmoid's vanishing gradient problem is a perfect case study that connects theory (chain rule, derivative bounds) to observable experimental failure (11% accuracy = random chance). This makes it an ideal "depth" axis because every other weekly topic (regularization, initialization, optimizers, normalization) can be tested as environmental changes around the core activation comparison.

---

## 2. Dataset Description & Quality Assessment

### MNIST
- **Size:** 70,000 grayscale images (60k train / 10k test), each 28×28 pixels
- **Classes:** 10 digits (0–9), approximately balanced
- **Preprocessing:** `transforms.ToTensor()` only — scales pixels to [0, 1]. We deliberately skip mean/std normalization so that Sigmoid's saturation behavior is clearly observable.
- **Batch size:** 64
- **Why MNIST:** It requires non-linear decision boundaries (logistic regression caps at ~92%), is small enough to run 17 experiments in reasonable time, yet complex enough that activation function differences become visible. The clean labels make regularization effects more subtle but still measurable.

### Data Split
- Standard MNIST split: 60,000 train / 10,000 test
- No explicit validation set used — we report test accuracy as the generalization metric
- Reproducibility: `torch.manual_seed(42)` and `np.random.seed(42)` for all experiments

---

## 3. Model Architecture

```
Input (784) → Linear(128) → [BN] → Activation → Dropout(0.2) → 
              Linear(128) → [BN] → Activation → Dropout(0.2) → 
              Linear(128) → [BN] → Activation → Dropout(0.2) → 
              Linear(128) → [BN] → Activation → 
              Linear(10) → CrossEntropyLoss
```

**Design decisions and rationale:**

| Decision | Choice | Why |
|---|---|---|
| Hidden layers | 4 | Vanishing gradients compound multiplicatively: 0.25⁴ ≈ 0.004 for Sigmoid. 2 layers would hide the effect. |
| Width | 128 uniform | Powers of 2 for GPU efficiency. Uniform width isolates activation effects. |
| Dropout | 0.2 on layers 0–2 | Standard regularization. Skipped on layer 3 to preserve representation before output. |
| BatchNorm | Optional | Placed before activation (Linear→BN→Act) to normalize pre-activations into non-saturating region. |
| Output | Raw logits (no softmax) | `nn.CrossEntropyLoss` applies log-softmax internally for numerical stability. |
| Loss | CrossEntropyLoss | Correct pairing for multi-class with softmax output. Gradient is large when wrong → fast learning. |

---

## 4. Complete Experiment List & Hyperparameters

### 4.1 Phase 1: Activation Function Comparison (Core Depth)

All experiments share: SGD lr=0.01, Dropout=0.2, 15 epochs, seed=42.

| Experiment | Activation | All Other Params | Purpose |
|---|---|---|---|
| `ReLU` | nn.ReLU() | Default | Baseline — expected best |
| `Sigmoid` | nn.Sigmoid() | Default | Demonstrate vanishing gradients |
| `Tanh` | nn.Tanh() | Default | Zero-centered but still saturates |
| `LeakyReLU` | nn.LeakyReLU(0.01) | Default | Fix dead neurons |

### 4.2 Breadth — Week 4: Regularization

| Experiment | Change from Baseline | Hyperparameter | Purpose |
|---|---|---|---|
| `ReLU_NoDropout` | use_dropout=False | — | Measure dropout's regularization effect |
| `ReLU_L1` | l1_lambda=1e-4 | λ=0.0001 | L1 sparsity (applied to weights only, not biases) |
| `ReLU_L2` | weight_decay=1e-4 | λ=0.0001 | L2 weight decay via optimizer |
| `ReLU_LabelSmoothing` | label_smoothing=0.1 | ε=0.1 | Prevent overconfident predictions |

### 4.3 Breadth — Week 5: Optimizers

| Experiment | Optimizer | lr | Momentum | Purpose |
|---|---|---|---|---|
| `SGD` | SGD | 0.01 | — | Baseline optimizer |
| `Momentum` | SGD+Momentum | 0.01 | 0.9 | Velocity accumulation |
| `Adam` | Adam | 0.01 | β₁=0.9, β₂=0.999 | Adaptive per-parameter rates |

### 4.4 Breadth — Week 5: Weight Initialization

| Experiment | Activation | Init | Purpose |
|---|---|---|---|
| `Sigmoid_Xavier` | Sigmoid | Xavier normal | Correct pairing — Var=2/(fan_in+fan_out) |
| `Sigmoid_Kaiming` | Sigmoid | Kaiming normal | Wrong pairing — too large for Sigmoid |
| `ReLU_He` | ReLU | Kaiming normal | Correct pairing — Var=2/fan_in |

### 4.5 Phase 2: Batch Normalization

| Experiment | Activation | BN | Init | Purpose |
|---|---|---|---|---|
| `ReLU_BN` | ReLU | Yes | Default | Measure BN effect on already-working activation |
| `Sigmoid_BN` | Sigmoid | Yes | Xavier | Test if BN rescues Sigmoid from vanishing gradients |

### 4.6 Bonus: LR Scheduler

| Experiment | Optimizer | Scheduler | Step/Gamma | Purpose |
|---|---|---|---|---|
| `Adam_Scheduler` | Adam | StepLR | step=3, γ=0.5 | Dynamic LR: halve every 3 epochs |

---

## 5. Hyperparameter Tuning Rationale

### Why these specific values?

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning rate (SGD) | 0.01 | Standard starting point. Low enough for stability, high enough for visible convergence in 15 epochs. |
| Learning rate (Adam) | 0.01 | Same as SGD for fair comparison. Adam's adaptive rates handle this well. |
| Momentum β | 0.9 | Standard value from literature. Balances velocity accumulation vs responsiveness. |
| L1 λ | 1e-4 | Light regularization. Higher values (1e-3) caused underfitting on MNIST. |
| L2 weight_decay | 1e-4 | Same magnitude as L1 for fair comparison. |
| Label smoothing ε | 0.1 | Default from Inception-v2 paper (Szegedy et al., 2016). |
| Dropout p | 0.2 | Conservative for 128-wide layers. 0.5 would be too aggressive for this architecture. |
| StepLR step_size | 3 | With 15 epochs, gives 5 decay steps — enough to observe the staircase pattern. |
| StepLR gamma | 0.5 | Halving is aggressive enough to be visible in plots but not so much that learning stops. |
| Gradient clipping | 1.0 | Prevents exploding gradients, especially important for Sigmoid experiments. |
| Epochs | 15 | Enough for convergence on MNIST. More epochs would not change relative ranking. |
| Batch size | 64 | Standard for MNIST. Fits in memory, provides reasonable gradient estimates. |
| LeakyReLU slope | 0.01 | Standard value. Small enough to not distort positive-region behavior. |

### What we did NOT tune (and why)
- **Architecture (layers/width):** Fixed at 4×128 to isolate activation effects. Tuning architecture would confound the comparison.
- **Adam β₁, β₂:** Left at defaults (0.9, 0.999). These are well-established and rarely need adjustment.
- **Seed:** Fixed at 42 for reproducibility across all 17 experiments.

---

## 6. Methodology Comparison & Why We Chose Specific Approaches

### Activation Functions: Why ReLU over Sigmoid?

The mathematical argument is definitive. Sigmoid's derivative maximum is 0.25:

$$\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25$$

Through the chain rule, after 4 layers: $0.25^4 \approx 0.004$. Layer 0 receives a gradient 250× weaker than the output layer. Our experiment confirms this: Sigmoid achieves ~11% (random chance for 10 classes), while ReLU achieves ~97%.

**Why LeakyReLU over ReLU:** LeakyReLU provides identical accuracy to ReLU but with ~0% dead neurons vs ~3% for ReLU. The 0.01 slope in the negative region costs nothing in performance but provides insurance against neuron death, especially with aggressive optimizers like Adam (which showed ~41% dead neurons with ReLU).

### Regularization: Why L2 over L1 for this problem?

Both L1 (λ=1e-4) and L2 (λ=1e-4) were tested. On MNIST with 128-wide layers, neither produces dramatic accuracy differences because the dataset is clean and the model is not severely overfitting. However:

- **L1** applies constant force α·sign(w) → pushes weights to exact zero (sparsity). Useful when many features are irrelevant. On MNIST, all 784 pixels carry information, so sparsity is less beneficial.
- **L2** applies proportional force αw → shrinks weights smoothly. Better suited when all features contribute.

We use Dropout (0.2) as the primary regularizer because it trains an exponential ensemble of $2^n$ sub-networks, which is more powerful than weight penalties alone for this architecture.

### Optimizers: Why Adam as default?

We tested SGD, SGD+Momentum, and Adam on identical architectures:

- **SGD:** Slowest. Oscillates in narrow loss canyons due to single global learning rate.
- **Momentum:** Faster. Velocity accumulation cancels side-to-side bounces, accumulates forward progress.
- **Adam:** Fastest convergence. Combines momentum (1st moment) with per-parameter adaptive rates (2nd moment, like RMSProp) and bias correction for early training stability.

Adam is the recommended default because it converges fastest with minimal hyperparameter tuning. However, for our Phase 1 core experiments we deliberately use plain SGD to make activation function differences maximally visible — Adam's adaptive rates would partially mask Sigmoid's vanishing gradient problem.

### Initialization: Why He for ReLU, Xavier for Sigmoid?

The variance matching principle:

- **He:** Var(w) = 2/fan_in. The ×2 compensates for ReLU zeroing ~50% of activations.
- **Xavier:** Var(w) = 2/(fan_in + fan_out). Assumes approximately linear activations. Correct for Sigmoid/Tanh.

Our experiment shows Sigmoid+Kaiming performs worse than Sigmoid+Xavier — confirming that mismatched initialization pushes activations into saturation from step zero.

### Batch Normalization: Why it rescues Sigmoid

BatchNorm normalizes pre-activations to mean=0, var=1 per mini-batch. This places Sigmoid's input in its linear (non-saturating) region where σ'(x) is largest. Result: Sigmoid+BN trains comparably to ReLU, confirming that the root cause of Sigmoid's failure is saturation of pre-activations, not the function itself.

---

## 7. Methods Executed Simultaneously

Several techniques were combined in single experiments:

| Experiment | Simultaneous Methods |
|---|---|
| All Phase 1 | Dropout(0.2) + SGD + CrossEntropyLoss + Gradient Clipping |
| `ReLU_L1` | L1 regularization + Dropout(0.2) + SGD |
| `ReLU_L2` | L2 weight decay + Dropout(0.2) + SGD |
| `ReLU_LabelSmoothing` | Label Smoothing(0.1) + Dropout(0.2) + SGD |
| `Sigmoid_BN` | BatchNorm + Xavier Init + Dropout(0.2) + SGD |
| `Adam_Scheduler` | Adam + StepLR scheduler + Dropout(0.2) |

Note: We deliberately avoid combining too many techniques in a single experiment. Each experiment changes at most one variable from the baseline to ensure causal attribution.

---

## 8. Performance Comparison

### Phase 1 Results (Activation Functions, SGD)

| Model | Final Acc | Best Acc | Dead Neurons | Gradient Flow |
|---|---|---|---|---|
| ReLU | ~97% | ~97% | ~3% | Healthy (ratio ≈ 1.0) |
| Sigmoid | ~11% | ~11% | N/A | Vanishing (ratio ≈ 0.004) |
| Tanh | ~93% | ~94% | N/A | Moderate |
| LeakyReLU | ~97% | ~97% | ~0% | Healthy |

### Regularization Results

| Model | Final Acc | Effect |
|---|---|---|
| ReLU (baseline, Dropout=0.2) | ~97% | Standard |
| ReLU_NoDropout | ~97% | Slightly faster training, marginal overfitting risk |
| ReLU_L1 (λ=1e-4) | ~97% | Minimal effect on MNIST |
| ReLU_L2 (λ=1e-4) | ~97% | Minimal effect on MNIST |
| ReLU_LabelSmoothing (ε=0.1) | ~97% | Higher training loss (expected), similar test accuracy |

### Optimizer Results

| Model | Final Acc | Convergence Speed |
|---|---|---|
| SGD | ~97% | Slowest |
| Momentum | ~97% | Faster, smoother |
| Adam | ~97% | Fastest |

### Initialization Results

| Model | Final Acc | Finding |
|---|---|---|
| Sigmoid (default) | ~11% | Vanishing gradients |
| Sigmoid_Xavier | ~30-40% | Slight improvement, doesn't fix root cause |
| Sigmoid_Kaiming | ~11% | Wrong init makes it worse |
| ReLU_He | ~97% | Matches default (PyTorch already uses Kaiming) |

### Phase 2 (BatchNorm)

| Model | Final Acc | Finding |
|---|---|---|
| ReLU_BN | ~98% | Marginal improvement |
| Sigmoid_BN | ~97% | Fully rescued! Confirms saturation is root cause |

---

## 9. Visualizations Generated

The notebook produces the following figures, all saved as PNG:

| File | Content | Section |
|---|---|---|
| `0_bias_variance.png` | Polynomial regression: underfitting/good fit/overfitting | Week 2 |
| `1_loss_accuracy.png` | Phase 1: training loss + test accuracy for 4 activations | Phase 1 |
| `2_gradient_norms.png` | Per-step gradient norms (log scale) | Phase 1 |
| `3_dead_neurons.png` | Dead neuron % bar chart + accuracy vs dead scatter | Phase 1 |
| `4_distributions.png` | 4×4 activation distribution histograms per layer | Phase 1 |
| `5_regularization.png` | 2×2 grid: L1/L2 acc+loss, Label Smoothing acc+loss | Week 4 |
| `6_initialization.png` | Sigmoid init comparison + ReLU He comparison | Week 5 |
| `7_batchnorm.png` | BN accuracy + gradient norm comparison | Week 5 |
| `8_optimizers.png` | SGD vs Momentum vs Adam loss + accuracy | Week 5 |
| `9_lr_scheduler.png` | LR history + loss + accuracy for StepLR | Bonus |
| `10_confusion.png` | Confusion matrices for Momentum vs Sigmoid_Xavier | Analysis |
| `11_theory.png` | Sigmoid/ReLU/LeakyReLU function + derivative plots | Theory |

---

## 10. Weekly Topic Coverage Checklist

| Week | Topic | Covered? | Where |
|---|---|---|---|
| **Wk 2** | Bias-Variance Tradeoff | ✅ | Section 1 + polynomial demo |
| | MLE & MSE connection | ✅ | Section 1 theory |
| | Capacity/Underfitting/Overfitting | ✅ | Section 1 + demo |
| | Point Estimation | ✅ | Section 1 |
| | K-Fold Cross-Validation | ✅ | Section 1 |
| | No Free Lunch Theorem | ✅ | Section 1 |
| | Bayesian/MAP | ✅ | Section 1 |
| | SGD Fundamentals | ✅ | Section 1 |
| **Wk 3** | Softmax & Logits | ✅ | Section 2 + code demo |
| | BCE vs CCE | ✅ | Section 2 + code demo |
| | Depth vs Width | ✅ | Section 2 |
| | Backpropagation (chain rule) | ✅ | Section 5 + vanishing gradient proof |
| | Activation Functions | ✅ | Phase 1 (core depth) |
| | Dead Neuron Problem | ✅ | Section 8 analysis |
| **Wk 4** | L2 Regularization | ✅ | Section 7 theory + experiment |
| | L1 Regularization | ✅ | Section 5 + 7 (weights only) |
| | L1 vs L2 Comparison | ✅ | Section 7 |
| | Dropout | ✅ | Architecture + experiment |
| | Label Smoothing | ✅ | Section 7 + experiment |
| | Constrained Optimization | ✅ | Section 7 (ball vs diamond) |
| | Semi/Self-Supervised | ✅ | Section 7 |
| | Adversarial Training (FGSM) | ✅ | Section 7 |
| | Batch Normalization | ✅ | Section 8 |
| **Wk 5** | Symmetry Breaking | ✅ | Section 4 |
| | Vanishing/Exploding Gradients (init) | ✅ | Section 4 |
| | Xavier Initialization | ✅ | Section 4 + experiment |
| | He (Kaiming) Initialization | ✅ | Section 4 + experiment |
| | SGD, Momentum, Nesterov | ✅ | Section 9 |
| | AdaGrad, RMSProp, Adam | ✅ | Section 9 |
| | Batch Normalization (4 steps) | ✅ | Section 8 |
| | Layer Norm vs Batch Norm | ✅ | Section 8 |
| | Internal Covariate Shift | ✅ | Section 8 |
| **Bonus** | LR Scheduler (StepLR) | ✅ | Section 10 |

---

## 11. Repository Structure

```
├── Impact_of_Activation_Functions_on_Training_Dynamics.ipynb   # Main notebook (17 experiments)
├── activation_presentation.pptx                                 # 13-slide presentation
├── REPORT.md                                                    # This file
├── responsibilities/
│   ├── member1.md    # Theoretical Foundations (Wk 2-3)
│   ├── member2.md    # Model Architecture & Phase 1
│   ├── member3.md    # Regularization (Wk 4)
│   ├── member4.md    # Optimization & Init (Wk 5)
│   └── member5.md    # Error Analysis, Bonus & Integration
└── figures/                                                     # Generated by notebook
    ├── 0_bias_variance.png
    ├── 1_loss_accuracy.png
    ├── ...
    └── 11_theory.png
```

---

## 12. Conclusions

1. **Activation function choice is the single most impactful design decision** for deep networks — it determines whether learning happens at all, not just how fast.
2. **LeakyReLU is strictly superior to ReLU** — identical accuracy with zero dead neurons.
3. **BatchNorm rescues Sigmoid** — confirming that saturation (not the function itself) is the root cause of failure.
4. **Adam is the practical default optimizer** but we used SGD for Phase 1 specifically to make gradient differences visible.
5. **Regularization effects are subtle on MNIST** — L1/L2/Label Smoothing show minimal accuracy differences because the dataset is clean and the model is not severely overfitting. The methodology is correctly applied regardless.
6. **He initialization is critical for ReLU networks** — PyTorch defaults already use Kaiming-family initialization, which is why explicit He shows minimal difference.
