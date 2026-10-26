# Topic 8.2: Logistic Regression & Binary Classification

---

## 1. Executive Summary & The 12-Year-Old Analogy

While Linear Regression predicts continuous quantities (such as prices, temperatures, or inventory counts), the vast majority of practical artificial intelligence tasks require **classification**: answering discrete categorical questions.
- Is this email spam or legitimate?
- Is this patient tumor malignant or benign?
- Will this user click on this search result or scroll past it?

**Logistic Regression** bridges linear modeling and probability theory. By transforming a linear combination of input features through the non-linear **Sigmoid Activation Function**, logistic regression outputs a calibrated probability:
$$P(y = 1 \mid x) \in [0, 1]$$

```
 Probability P(y=1)
      1.0 +------------------------------************ (Pass / True)
          |                         ****
          |                      ***
      0.5 |-------------------*---------------------- (Decision Boundary: z = 0)
          |                ***
          |            ****
      0.0 +************------------------------------ (Fail / False)
          +-------------------+---------------------+---> Linear Logit (z = w^T x + b)
                            z = 0
```

---

### The 12-Year-Old Analogy: The Exam Pass/Fail Dimmer Switch

Imagine your school friends are preparing for a difficult mathematics exam. You want to build a model that predicts whether a student will **Pass** ($y=1$) or **Fail** ($y=0$) based on how many hours they spent studying ($x$).

- **Why Not Use a Straight Line?**  
  If you try drawing a straight line through the data, strange absurdities happen:
  - If someone studies 60 hours, the straight line predicts a "150% probability" of passing.
  - If someone studies 0 hours, it predicts a "-30% probability".
  - But probabilities in the real world can never be less than 0% or more than 100%!
- **The S-Shaped Dimmer Switch (The Sigmoid):**  
  Instead of a rigid wooden ruler, imagine a smooth S-shaped silicone ramp (the Sigmoid curve).
  - When study hours are extremely low, the curve hugs the floor at $0.001$.
  - When study hours are massive, the curve hugs the ceiling at $0.999$.
  - In the middle, it smoothly sweeps upward, crossing exactly $0.5$ (a 50/50 toss-up) at the tipping point.
- **Odds & Log-Odds (The Betting Slip):**  
  If a basketball team has a 75% chance of winning ($p = 0.75$), their probability of losing is 25% ($1 - p = 0.25$). The **odds** of winning are $\frac{0.75}{0.25} = 3\text{ to }1$. The **log-odds** (or *logit*) is the natural logarithm of those odds: $\ln(3) \approx 1.098$. Taking the log stretches probabilities from $[0, 1]$ across the entire number line from $-\infty$ to $+\infty$. This allows our linear equation $w^T x + b$ to operate freely!
- **Binary Cross-Entropy (The Penalty for Arrogance):**  
  How do we judge our model? If a student passes ($y=1$) and our model predicted 90% confidence, we give it a tiny pat on the back. But if the student passes and our model arrogantly proclaimed a "1% chance of passing", we slap it with a massive numerical penalty. That penalty is **Binary Cross-Entropy Loss**!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x \in \mathbb{R}^d$ | Vector space | Input feature vector representing an instance |
| $y \in \{0, 1\}$ | Discrete set | Binary ground-truth target label |
| $w \in \mathbb{R}^d, b \in \mathbb{R}$ | Parameter space | Weight vector and scalar bias term |
| $z = w^T x + b \in \mathbb{R}$ | Scalar field | Linear log-odds score (the *logit*) |
| $\sigma(z) \in (0, 1)$ | Unit interval | Standard logistic sigmoid activation function |
| $\hat{y} = \sigma(z)$ | Probability | Predicted posterior probability $P(y = 1 \mid x; w, b)$ |
| $\text{Odds}(p) = \frac{p}{1-p}$ | $\mathbb{R}_{>0}$ | Ratio of event probability to non-event probability |
| $\text{logit}(p) = \ln\left(\frac{p}{1-p}\right)$ | $\mathbb{R}$ | Inverse of the sigmoid function (maps $(0, 1) \to (-\infty, \infty)$) |
| $\mathcal{L}_{\text{BCE}}(w, b)$ | $\mathbb{R}_{\ge 0}$ | Empirical Binary Cross-Entropy (Negative Log-Likelihood) loss |
| $\nabla_w \mathcal{L} \in \mathbb{R}^d$ | Vector field | Gradient of the BCE loss with respect to weight vector $w$ |

---

## 3. The Sigmoid Function & Log-Odds Formulation

### 3.1 The Standard Logistic Sigmoid

The logistic sigmoid function $\sigma: \mathbb{R} \to (0, 1)$ is defined as:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

Key mathematical properties:
1. **Asymptotes:** $\lim_{z \to -\infty} \sigma(z) = 0$ and $\lim_{z \to +\infty} \sigma(z) = 1$.
2. **Symmetry:** $\sigma(-z) = 1 - \sigma(z)$ around the point $(0, 0.5)$.
3. **First Derivative Self-Identity:** The derivative of $\sigma(z)$ can be expressed entirely in terms of the function itself:
$$\begin{aligned}
\frac{d}{dz} \sigma(z) &= \frac{d}{dz} (1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} (-e^{-z}) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} = \left(\frac{1}{1 + e^{-z}}\right) \left(\frac{e^{-z}}{1 + e^{-z}}\right) \\
&= \left(\frac{1}{1 + e^{-z}}\right) \left(1 - \frac{1}{1 + e^{-z}}\right) \\
&= \sigma(z)(1 - \sigma(z))
\end{aligned}$$

This elegant identity makes computing gradients computationally fast and mathematically clean!

---

### 3.2 The Log-Odds (Logit) Connection

Let $p = P(y=1 \mid x) = \sigma(z)$. What does the linear expression $z = w^T x + b$ actually mean?

Let us solve for $z$ in terms of $p$:
$$p = \frac{1}{1 + e^{-z}} \implies 1 + e^{-z} = \frac{1}{p} \implies e^{-z} = \frac{1 - p}{p} \implies e^z = \frac{p}{1 - p}$$

Taking the natural logarithm of both sides:
$$\ln\left(\frac{p}{1 - p}\right) = z = w^T x + b$$

The quantity $\frac{p}{1-p}$ is the **odds** of the event occurring. Its natural logarithm is called the **logit**.  
Thus, **logistic regression is a generalized linear model where the log-odds of the positive class is modeled as a linear combination of the features.**

---

### 3.3 Linear Decision Boundary

We classify an instance as positive ($y=1$) whenever the predicted probability exceeds a chosen decision threshold (typically $\tau = 0.5$):
$$\hat{y} \ge 0.5 \iff \sigma(w^T x + b) \ge 0.5 \iff w^T x + b \ge 0$$

The boundary dividing the two classes is the set of points where the model is completely uncertain ($P(y=1 \mid x) = 0.5$):
$$\mathcal{H} = \{x \in \mathbb{R}^d : w^T x + b = 0\}$$

This defines a **flat affine hyperplane** in $\mathbb{R}^d$:
- The weight vector $w$ is the normal vector orthogonal to the decision boundary.
- The signed perpendicular distance from any point $x$ to the decision boundary is given by:
$$\text{dist}(x, \mathcal{H}) = \frac{w^T x + b}{\|w\|_2}$$

---

## 4. Maximum Likelihood Estimation & Binary Cross-Entropy

### 4.1 Bernoulli Likelihood Formulation

For each training pair $(x^{(i)}, y^{(i)})$, the true label $y^{(i)} \in \{0, 1\}$ follows a Bernoulli trial where the success probability is parameterized by our model $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$:
$$P(y^{(i)} \mid x^{(i)}; w, b) = \left(\hat{y}^{(i)}\right)^{y^{(i)}} \left(1 - \hat{y}^{(i)}\right)^{1 - y^{(i)}}$$

Assuming the $N$ observations are identically and independently distributed (i.i.d.), the total likelihood over the training dataset is the product of individual probabilities:
$$L(w, b) = \prod_{i=1}^N \left(\hat{y}^{(i)}\right)^{y^{(i)}} \left(1 - \hat{y}^{(i)}\right)^{1 - y^{(i)}}$$

---

### 4.2 Negative Log-Likelihood (Binary Cross-Entropy)

Multiplying thousands of small probabilities causes severe numerical floating-point underflow. We take the natural log to convert products into sums:
$$\ln L(w, b) = \sum_{i=1}^N \left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln\left(1 - \hat{y}^{(i)}\right) \right]$$

To formulate this as a minimization problem, we define the **Binary Cross-Entropy (BCE) Loss** as the negative average log-likelihood:
$$\mathcal{L}(w, b) = -\frac{1}{N} \ln L(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln\left(1 - \hat{y}^{(i)}\right) \right]$$

Behavior of the loss:
- If $y^{(i)} = 1$: Loss is $-\ln(\hat{y}^{(i)})$. As $\hat{y} \to 1$, loss $\to 0$. As $\hat{y} \to 0$, loss $\to +\infty$.
- If $y^{(i)} = 0$: Loss is $-\ln(1 - \hat{y}^{(i)})$. As $\hat{y} \to 0$, loss $\to 0$. As $\hat{y} \to 1$, loss $\to +\infty$.

---

## 5. Derivation of the Gradient: The Miracle of Logistic Regression

We now compute the gradient of the BCE loss with respect to the parameter vector $w$.

For a single sample $i$, let $\ell^{(i)} = -\left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$, where:
$$\hat{y}^{(i)} = \sigma(z^{(i)}), \quad \text{and} \quad z^{(i)} = w^T x^{(i)} + b$$

Applying the multivariable **Chain Rule**:
$$\frac{\partial \ell^{(i)}}{\partial w_j} = \frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}$$

Let us compute each factor individually:

1. **Derivative of Loss w.r.t. Prediction:**
$$\frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} = -\left[ \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right] = -\left[ \frac{y^{(i)}(1 - \hat{y}^{(i)}) - (1 - y^{(i)})\hat{y}^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})} \right] = -\frac{y^{(i)} - \hat{y}^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)}} = \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)}}$$

2. **Derivative of Sigmoid w.r.t. Logit:**
$$\frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \sigma(z^{(i)})(1 - \sigma(z^{(i)})) = \hat{y}^{(i)}(1 - \hat{y}^{(i)})$$

3. **Derivative of Logit w.r.t. Weight:**
$$\frac{\partial z^{(i)}}{\partial w_j} = x_j^{(i)}$$

Now, multiply all three factors together:
$$\frac{\partial \ell^{(i)}}{\partial w_j} = \left(\frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})}\right) \cdot \left(\hat{y}^{(i)}(1 - \hat{y}^{(i)})\right) \cdot x_j^{(i)}$$

Notice the denominator cancels out entirely:
$$\frac{\partial \ell^{(i)}}{\partial w_j} = \left(\hat{y}^{(i)} - y^{(i)}\right) x_j^{(i)}$$

Averaging across all $N$ samples in the dataset:
$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N \left(\hat{y}^{(i)} - y^{(i)}\right) x_j^{(i)}$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N \left(\hat{y}^{(i)} - y^{(i)}\right)$$

In compact matrix-vector notation:
$$\nabla_w \mathcal{L} = \frac{1}{N} X^T (\hat{y} - y)$$

$\blacksquare$

> **The Mathematical Miracle:**  
> The gradient vector $\nabla_w \mathcal{L} = \frac{1}{N} X^T (\hat{y} - y)$ is **identical in functional structure** to the gradient of Mean Squared Error in Linear Regression! The only difference is that $\hat{y}$ is generated by passing the linear logits through the non-linear sigmoid function $\sigma(Xw)$.

---

## 6. Gradient Descent Optimization Algorithm

Because the sigmoid non-linearity prevents an analytical closed-form solution (we cannot solve $\nabla_w \mathcal{L} = 0$ with algebra), we use iterative gradient descent:

$$w_{t+1} = w_t - \frac{\eta}{N} X^T (\sigma(Xw_t) - y)$$

### Convexity of Logistic Regression
Is logistic regression guaranteed to find the globally optimal weights?  
Yes! The Hessian matrix of the BCE loss is:
$$H = \nabla^2 \mathcal{L}(w) = \frac{1}{N} X^T D X$$
where $D$ is an $N \times N$ diagonal matrix with diagonal elements $D_{ii} = \hat{y}^{(i)}(1 - \hat{y}^{(i)}) > 0$.

For any non-zero vector $v \in \mathbb{R}^d$:
$$v^T H v = \frac{1}{N} v^T X^T D X v = \frac{1}{N} (Xv)^T D (Xv) = \frac{1}{N} \sum_{i=1}^N D_{ii} (Xv)_i^2 \ge 0$$

Because $D_{ii} > 0$ for all valid probabilities, the Hessian is **positive semi-definite** everywhere ($H \succeq 0$). The Binary Cross-Entropy loss surface is strictly **convex**, guaranteeing that gradient descent will never get trapped in a suboptimal local minimum!
