# Person 1 — Week 2: ML Foundations + Week 3: Theoretical Framework

## Responsibility Summary

I was responsible for the **theoretical backbone** of the project: all Week 2 (ML Foundations) content and the theoretical portions of Week 3 (Softmax, Cross-Entropy, Backpropagation). My work provides the mathematical grounding that every experiment builds upon.

**Notebook sections:** 1 (Week 2 theory + polynomial demo), 2 (Week 3 theory + softmax/CE demo)  
**Presentation slides:** 2 (T-Model), 3 (Week 2), 4 (Week 3)

---

## Week 2 Contributions

### Bias-Variance Tradeoff

I wrote the theory and created the polynomial regression demo producing `0_bias_variance.png`. The fundamental decomposition $\text{MSE} = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$ shows that increasing capacity reduces bias but increases variance. The demo visualizes this: degree 1 underfits, degree 4 generalizes, degree 15 overfits.

### MLE and MSE Connection

I documented that under Gaussian noise $p(y|x) = \mathcal{N}(y; \hat{y}, \sigma^2)$, maximizing log-likelihood reduces to minimizing $\sum\|\hat{y}-y\|^2$. The formula $\theta_{\text{ML}} = \arg\max_\theta \sum_i \log p(x^{(i)}; \theta)$ and its properties (consistency, efficiency, asymptotic normality) are covered in the theory section.

### Bayesian Perspective

$p(\theta|\text{data}) \propto p(\text{data}|\theta) \cdot p(\theta)$. Taking $-\log$ yields loss + regularizer. I showed: MAP + Gaussian prior $\Leftrightarrow$ L2 weight decay, MAP + Laplace prior $\Leftrightarrow$ L1 regularization.

### Additional Week 2 Topics

- **Point Estimation:** $\hat{\theta} = g(x^{(1)},...,x^{(m)})$ is a random variable. Bias = $E[\hat{\theta}]-\theta$, SE = $\sigma/\sqrt{m}$.
- **Capacity/Underfitting/Overfitting:** Demonstrated with polynomial degrees.
- **K-Fold Cross-Validation:** Each sample tested exactly once; essential for small datasets.
- **No Free Lunch Theorem:** No universal algorithm — must design for our distribution.
- **Hyperparameters & Validation:** Train/Val/Test split, test set used exactly once.
- **SGD:** $\theta \leftarrow \theta - \varepsilon \nabla_\theta J(\theta)$. Minibatch noise helps escape local minima.

---

## Week 3 Contributions

### Softmax & Logits

$\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$. I wrote the demo code verifying manual computation matches PyTorch. Never apply softmax before `nn.CrossEntropyLoss` — it handles log-softmax internally.

### Cross-Entropy (BCE vs CCE)

BCE: $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ → Sigmoid. CCE: $-\sum_i y_i \log\hat{y}_i$ → Softmax. Key property: gradient is large when prediction is wrong → fast learning. I wrote the manual vs PyTorch verification demo.

### Depth vs Width

Each layer folds input space → $2^L$ regions with linear parameters. Residual connections: $\text{output} = x + \text{sublayer}(x)$.

### Backpropagation Chain Rule

$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(L)}} \cdot \prod_k \sigma'(z^{(k)}) \cdot W^{(k+1)}$. This formula directly explains vanishing gradients: Sigmoid $\sigma'_{\max} = 0.25$, so after 4 layers $\approx 0.004$.
