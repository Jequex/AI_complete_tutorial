# Topic 11.3: Loss Functions & Analytical Derivatives

---

## 1. Executive Summary & The 12-Year-Old Analogy

A neural network produces raw predictions, but how does it know how bad its mistakes are? 
The **loss function** $\mathcal{L}(y, \hat{y})$ is the mathematical objective function that quantifies the discrepancy between the network's output and the ground truth.

Crucially, in backpropagation, the loss function supplies the **initial error signal** that kicks off the chain rule:

$$\delta^{[L]} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}}$$

Choosing the right loss function is essential. For instance, pairing **Mean Squared Error (MSE)** with a Sigmoid classifier causes severe gradient saturation, whereas pairing **Cross-Entropy** with Sigmoid or Softmax produces a remarkably elegant and robust error gradient:

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}}$$

```
                 LOSS FUNCTIONS & INITIAL ERROR GRADIENTS
                 
   Regression (Continuous y):                Classification (Discrete y):
      Mean Squared Error (MSE)                   Categorical Cross-Entropy (Softmax)
      
          Loss: (1/2)||y - ŷ||²                      Loss: - Σ y_k ln(p_k)
          Grad: (ŷ - y)                              Grad: (p - y)
          
      y ^            .---.                       y ^            .---.
        │          .'     '.                       │          .'     '.
      L │         /         \                    L │         /         \
        │        /     ●     \                     │        /     ●     \
        │       /   Global    \                    │       /   Saddle    \
        │      /    Minimum    \                   │      /    Points     \
        └─────┴─────────────────> ŷ                └─────┴─────────────────> z
```

---

### The 12-Year-Old Analogy: The Scoring Referee and the Penalty Box

Imagine playing a high-stakes trivia game with two very different referees:

1. **The Mild Tape-Measure Referee (MSE):**
   - You guess a number: *"The Eiffel Tower is 400 meters tall!"* The real answer is 330 meters.
   - The referee pulls out a tape measure, measures the distance you missed by ($400 - 330 = 70$), squares it ($70^2 = 4,900$), and hands you that fine.
   - *Good for:* Distance, house prices, temperature, speed—any continuous number!

2. **The High-Stakes Truth Detector (Cross-Entropy):**
   - Now you are answering a True/False or Multiple Choice question: *"Is this a dog?"*
   - You must state your confidence: *"I am 99.9% sure it is a CAT!"* But the truth is a DOG!
   - Because you were **confidently completely wrong**, the Cross-Entropy referee sends you straight to the penalty box with a near-infinite fine ($-\ln(0.001) \approx 6.91$, as $p \to 0$, loss $\to \infty$).
   - If you guess correctly with high confidence ($p = 0.999$), your fine is virtually zero ($-\ln(0.999) \approx 0.001$).
   - *The Magic of Cross-Entropy:* The gradient message sent back to the network is simply: **(What you guessed) minus (What it actually was)**!

3. **The Unfair Match Referee (Focal Loss):**
   - Suppose 99 out of 100 questions are super easy ("Is water wet?"), and only 1 question is hard ("What is the capital of Burkina Faso?").
   - A regular referee gives points for all 99 easy questions, so the student never bothers studying the hard one.
   - **Focal Loss** adds a multiplier $(1 - p)^\gamma$: if you already get an answer 99% right, your loss is crushed down to almost zero, forcing you to spend 100% of your energy studying the rare, difficult questions!

---

## 2. Mathematical Symbol Decoder

| Symbol | Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathbf{y} \in \{0, 1\}^K$ or $\mathbb{R}^K$ | Vector | Ground truth target (scalar for regression, one-hot vector for classification) |
| $\hat{\mathbf{y}}$ or $\mathbf{p} \in \mathbb{R}^K$ | Vector | Model predictions / predicted class probabilities |
| $\mathbf{z} \in \mathbb{R}^K$ | Vector | Unnormalized raw outputs (logits) before final activation |
| $\mathcal{L}$ | $\mathbb{R}_{\ge 0}$ | Scalar loss / empirical risk for a single sample or mini-batch |
| $N \in \mathbb{Z}_{\ge 1}$ | Integer | Number of observations in the mini-batch |
| $K \in \mathbb{Z}_{\ge 1}$ | Integer | Number of distinct target classes |
| $\gamma \in \mathbb{R}_{\ge 0}$ | Scalar | Focusing hyperparameter in Focal Loss (typically $\gamma = 2.0$) |
| $\alpha_t \in [0, 1]$ | Scalar | Class weighting balance parameter in Focal Loss |

---

## 3. Mean Squared Error (MSE)

For regression tasks where target $y \in \mathbb{R}$:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{2N} \sum_{i=1}^N \|y^{(i)} - \hat{y}^{(i)}\|^2$$

*(Note: The factor $\frac{1}{2}$ is mathematically standard to cleanly cancel the exponent 2 during differentiation).*

### 3.1 Gradient Derivation
Differentiating with respect to the prediction $\hat{y}_j$:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_j} = \frac{\partial}{\partial \hat{y}_j} \left[ \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 \right] = \frac{1}{2N} \cdot 2(\hat{y}_j - y_j) = \boxed{\frac{1}{N}(\hat{y}_j - y_j)}$$

### 3.2 Why MSE Fails for Classification
Suppose we use MSE for binary classification with a Sigmoid output $\hat{y} = \sigma(z)$:

$$\mathcal{L} = \frac{1}{2}(\sigma(z) - y)^2$$

Differentiating with respect to pre-activation logit $z$:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = (\hat{y} - y) \cdot \sigma(z)(1 - \sigma(z))$$

Notice the catastrophic flaw: if the target is $y = 1$ but the model is confidently wrong ($z = -10 \implies \hat{y} \approx 0$):
$$\sigma(z)(1 - \sigma(z)) \approx 0 \implies \frac{\partial \mathcal{L}}{\partial z} \approx (0 - 1) \cdot 0 = 0!$$

The gradient **vanishes precisely when the model is most wrong**! This causes severe learning plateaus. Cross-Entropy fixes this completely.

---

## 4. Binary Cross-Entropy (BCE)

For binary classification $y \in \{0, 1\}$ with predicted probability $\hat{y} = \sigma(z) \in (0, 1)$:

$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \right]$$

### 4.1 The Beautiful Cancellation: BCE + Sigmoid
Let us compute the gradient of BCE directly with respect to the **logit** $z$:

By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

1. Derivative of BCE w.r.t. $\hat{y}$:
   $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right) = -\frac{y(1 - \hat{y}) - \hat{y}(1 - y)}{\hat{y}(1 - \hat{y})} = -\frac{y - \hat{y}}{\hat{y}(1 - \hat{y})} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

2. Derivative of Sigmoid w.r.t. $z$:
   $$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

3. Multiplying together:
   $$\frac{\partial \mathcal{L}}{\partial z} = \left( \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right) \cdot \hat{y}(1 - \hat{y}) = \boxed{\hat{y} - y}$$

The problematic denominator $\hat{y}(1 - \hat{y})$ **cancels out completely**! When the network is completely wrong ($y=1, \hat{y}=0$), the error gradient is $\hat{y} - y = 0 - 1 = -1$, delivering maximum corrective force without any vanishing gradient.

---

## 5. Multi-Class Categorical Cross-Entropy with Softmax

For multi-class classification among $K$ mutually exclusive categories:
- True label is a one-hot vector $\mathbf{y} \in \{0, 1\}^K$ with $\sum_k y_k = 1$.
- Softmax maps unconstrained logits $\mathbf{z} \in \mathbb{R}^K$ to a probability simplex:

$$\hat{y}_k = p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

The Categorical Cross-Entropy loss is:
$$\mathcal{L}_{\text{CCE}}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{k=1}^K y_k \ln \hat{y}_k$$

### 5.1 Analytical Gradient Derivation w.r.t. Logits $z_i$
We want to evaluate $\frac{\partial \mathcal{L}}{\partial z_i}$:

$$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{k=1}^K \frac{y_k}{\hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial z_i}$$

The Jacobian of Softmax has two cases:
- Case $k = i$: $\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1 - \hat{y}_i)$
- Case $k \ne i$: $\frac{\partial \hat{y}_k}{\partial z_i} = -\hat{y}_k \hat{y}_i$

Substitute both cases into the sum:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\left[ \frac{y_i}{\hat{y}_i} \hat{y}_i(1 - \hat{y}_i) + \sum_{k \ne i} \frac{y_k}{\hat{y}_k} (-\hat{y}_k \hat{y}_i) \right]$$
$$= -\left[ y_i (1 - \hat{y}_i) - \hat{y}_i \sum_{k \ne i} y_k \right]$$
$$= -y_i + y_i \hat{y}_i + \hat{y}_i \sum_{k \ne i} y_k = -y_i + \hat{y}_i \underbrace{\left( y_i + \sum_{k \ne i} y_k \right)}_{= \sum_{all} y_k = 1} = \boxed{\hat{y}_i - y_i}$$

In vector notation:
$$\boxed{\nabla_{\mathbf{z}} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}}$$

---

## 6. Focal Loss for Extreme Class Imbalance

In object detection (e.g. RetinaNet, Lin et al., 2017) and fraud detection, 99.9% of samples are easy negatives. Even though each easy negative produces a tiny loss, their sum overwhelms the rare positive signals.

**Focal Loss** adds a modulating factor $(1 - p_t)^\gamma$:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$

Where:
$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$

- When an example is **easy** ($p_t \to 1$): $(1 - p_t)^\gamma \to 0$, suppressing the loss by orders of magnitude (e.g. For $p_t = 0.99$ and $\gamma = 2$, $(0.01)^2 = 0.0001$).
- When an example is **hard** ($p_t \le 0.5$): $(1 - p_t)^\gamma \approx 1$, preserving the full error gradient.

---

## 7. Loss Landscape Geometry: Convexity vs Non-Convexity

| Property | Linear Models (Linear/Logistic Regression) | Deep Neural Networks (MLPs, CNNs, Transformers) |
| :--- | :--- | :--- |
| **Hessian $\nabla^2 \mathcal{L}$** | Positive semi-definite everywhere ($\mathbf{v}^T H \mathbf{v} \ge 0$) | Indefinite with mixture of positive and negative eigenvalues |
| **Landscape Geometry** | Globally convex "bowl" | High-dimensional non-convex landscape with millions of parameters |
| **Local Minima** | Every local minimum is a global minimum | Innumerable local minima (most have nearly identical low loss) |
| **Saddle Points** | Rare / nonexistent | Highly pervasive; index of saddle point dictates escape direction |
| **Plateaus & Ravines** | None | Common in saturated regions; addressed via Momentum & Adam |

In deep networks, local minima are rarely the obstacle; **saddle points** surrounded by flat plateaus in thousands of dimensions represent the primary optimization challenge.

---

## 8. Key Takeaways & Summary
- MSE is appropriate for continuous regression, with gradient $\frac{1}{N}(\hat{y} - y)$.
- MSE should never be used with Sigmoid for classification because its gradient vanishes when the model is most wrong.
- Binary Cross-Entropy with Sigmoid and Categorical Cross-Entropy with Softmax both yield the clean, robust error gradient $\hat{\mathbf{y}} - \mathbf{y}$.
- Focal Loss modulates Cross-Entropy with $(1 - p_t)^\gamma$ to overcome extreme class imbalance.
- Deep neural loss landscapes are non-convex, dominated by saddle points rather than bad local minima.
