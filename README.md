# Effect of Activation Functions on Training Dynamics

**Course:** SWE012 – Deep Learning with Python  
**Instructor:** Asst. Prof. Yigit Bekir Kaya  
**University:** Istinye University, Department of Computer Engineering

## About

This project investigates how the choice of activation function (Sigmoid, Tanh, ReLU, Leaky ReLU) affects training behavior in feedforward neural networks. We use a controlled experiment design where all other variables are held constant and only the activation function changes.

We follow a **T-Model** structure:
- **Depth:** Detailed comparison of 4 activation functions on the same task
- **Breadth:** Each activation function is tested against different course topics — optimizers, regularization, weight initialization, network depth, early stopping, and learning rate scheduling

## Dataset

Fashion-MNIST (10-class grayscale image classification, 28x28 pixels). We use a 15,000-sample training subset with 3,000 validation and 2,000 test samples.

## Experiments

| # | Experiment | What it tests |
|---|-----------|---------------|
| 1 | Activation Comparison | Convergence speed and accuracy across 4 activations |
| 2 | Activation x Optimizer | SGD, Momentum, Nesterov, RMSProp, Adam |
| 3 | Activation x Regularization | L2, Dropout, BatchNorm, Label Smoothing |
| 4 | Activation x Initialization | Xavier, He, Random, Zeros |
| 5 | Gradient Flow | Gradient norms across layers in an 8-layer network |
| 6 | Depth vs Width | Shallow, Medium, Deep architectures |
| 7 | Early Stopping | Patience-based training termination |
| 8 | LR Scheduling | Constant, StepLR, Cosine Annealing |

Additional analyses: dead neuron counts, bias-variance (generalization gap), L1 vs L2 weight distributions.

## Key Findings

- ReLU and Leaky ReLU converge faster and work well with any optimizer or initialization
- Sigmoid + SGD is the worst combination (~50% accuracy due to vanishing gradients)
- Zeros initialization fails for all activations (symmetry problem)
- BatchNorm partially rescues Sigmoid by reducing internal covariate shift
- Deeper networks help ReLU but hurt Sigmoid — vanishing gradients compound per layer
- Leaky ReLU reduces dead neuron percentage compared to standard ReLU

## Repo Structure

```
├── ipynb-code/
│   └── T_Model_Deneyleri_Orijinal.ipynb   # All experiments
├── figures/                                 # Output plots from notebook
├── responsibilities/                        # Per-student contribution files
├── REPORT_fixed (1).md                      # Detailed technical report
├── requirements.txt                         # Python dependencies
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
```

Open `ipynb-code/T_Model_Deneyleri_Orijinal.ipynb` in Google Colab or Jupyter, select a GPU runtime, and run all cells.

## References

- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
- Fashion-MNIST — Zalando Research
