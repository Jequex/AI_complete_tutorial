# Topic 01: Exercises & Solutions — Linear Probing & Representation Geometry

## Exercise 1: Gradient Derivation & Convexity of Linear Probes

Consider a binary classification linear probe with parameters $\mathbf{w} \in \mathbb{R}^d$ and $b \in \mathbb{R}$ trained on frozen intermediate representations $\mathbf{h} \in \mathbb{R}^d$ with binary labels $y \in \{0, 1\}$.
The objective function with $L_2$ regularization parameter $\lambda > 0$ is:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln \sigma(z^{(i)}) + (1 - y^{(i)}) \ln(1 - \sigma(z^{(i)})) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

where $z^{(i)} = \mathbf{w}^\top \mathbf{h}^{(i)} + b$ and $\sigma(z) = \frac{1}{1 + e^{-z}}$.

1. Derive the partial derivatives $\frac{\partial \mathcal{L}}{\partial \mathbf{w}}$ and $\frac{\partial \mathcal{L}}{\partial b}$.
2. Compute the Hessian matrix $\mathbf{H} = \nabla^2_{\mathbf{w}} \mathcal{L}$ and prove that the objective is strictly convex for any $\lambda > 0$.
3. Why does strict convexity make linear probes uniquely advantageous over non-linear probes (e.g., multi-layer perceptron probes) when assessing representation geometry?

### Solution

#### 1. Derivation of Gradients
Recall that for the logistic sigmoid function $\sigma(z)$:
$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

Let $p^{(i)} = \sigma(z^{(i)})$. The derivative of the cross-entropy term $\ell^{(i)}$ with respect to $z^{(i)}$ is:
$$\frac{\partial \ell^{(i)}}{\partial z^{(i)}} = -\left[ \frac{y^{(i)}}{p^{(i)}} - \frac{1 - y^{(i)}}{1 - p^{(i)}} \right] \cdot p^{(i)}(1 - p^{(i)}) = -(y^{(i)} - p^{(i)}) = p^{(i)} - y^{(i)}$$

Applying the chain rule $\frac{\partial z^{(i)}}{\partial \mathbf{w}} = \mathbf{h}^{(i)}$ and $\frac{\partial z^{(i)}}{\partial b} = 1$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N \left(p^{(i)} - y^{(i)}\right) \mathbf{h}^{(i)} + \lambda \mathbf{w}$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N \left(p^{(i)} - y^{(i)}\right)$$

#### 2. Hessian and Strict Convexity
Taking the second derivative with respect to $\mathbf{w}$:
$$\mathbf{H} = \nabla^2_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \frac{\partial p^{(i)}}{\partial \mathbf{w}} (\mathbf{h}^{(i)})^\top + \lambda \mathbf{I}$$
Since $\frac{\partial p^{(i)}}{\partial \mathbf{w}} = p^{(i)}(1 - p^{(i)}) \mathbf{h}^{(i)}$, we have:
$$\mathbf{H} = \frac{1}{N} \sum_{i=1}^N p^{(i)}(1 - p^{(i)}) \mathbf{h}^{(i)} (\mathbf{h}^{(i)})^\top + \lambda \mathbf{I}$$

Notice that for any $z^{(i)} \in \mathbb{R}$, $0 < p^{(i)} < 1$, so $p^{(i)}(1 - p^{(i)}) > 0$.
For any non-zero test vector $\mathbf{u} \in \mathbb{R}^d \setminus \{\mathbf{0}\}$:
$$\mathbf{u}^\top \mathbf{H} \mathbf{u} = \frac{1}{N} \sum_{i=1}^N \underbrace{p^{(i)}(1 - p^{(i)})}_{> 0} \underbrace{(\mathbf{u}^\top \mathbf{h}^{(i)})^2}_{\ge 0} + \lambda \underbrace{\|\mathbf{u}\|_2^2}_{> 0} \ge \lambda \|\mathbf{u}\|_2^2 > 0$$

Since $\mathbf{u}^\top \mathbf{H} \mathbf{u} > 0$ for all $\mathbf{u} \neq \mathbf{0}$, $\mathbf{H}$ is strictly positive definite ($\mathbf{H} \succ 0$). Therefore, the objective function is **strictly convex**.

#### 3. Methodological Advantage
Strict convexity guarantees that there are no local minima or saddle points; gradient descent will converge to the **unique global minimum** $\mathbf{w}^*$. If an MLP probe were used instead, non-convex optimization could yield erratic results dependent on random initialization. Furthermore, an MLP probe might compute complex non-linear functions that the base model itself does not use. A strictly convex linear probe ensures that any discovered predictability reflects **linearly accessible information natively existing in $\mathbf{h}$**.

---

## Exercise 2: Signal-to-Noise Ratio (SNR) and Alignment in Residual Streams

Suppose an internal binary concept $c \in \{-1, +1\}$ is represented along a unit direction $\mathbf{v}_c \in \mathbb{R}^d$ ($\|\mathbf{v}_c\| = 1$) with signal strength $s > 0$. The residual stream activation is modeled as:
$$\mathbf{h} = s \cdot c \cdot \mathbf{v}_c + \boldsymbol{\epsilon}$$
where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_d)$ represents orthogonal background computation and other unrelated concepts.

1. Compute the expected value and variance of the 1D projection $z = \mathbf{v}_c^\top \mathbf{h}$ conditioned on $c = +1$ and $c = -1$.
2. Write down the theoretical classification error rate $\mathcal{E}$ of an optimal linear classifier that thresholds $z$ at $0$, expressed in terms of the standard Gaussian CDF $\Phi(\cdot)$.
3. If $s = 2.0$ and $\sigma = 1.0$, calculate the theoretical probe accuracy. What happens to the error rate as the signal is amplified across Transformer layers?

### Solution

#### 1. Expected Value and Variance
Computing the projection $z$:
$$z = \mathbf{v}_c^\top \mathbf{h} = s \cdot c (\mathbf{v}_c^\top \mathbf{v}_c) + \mathbf{v}_c^\top \boldsymbol{\epsilon} = s \cdot c + \mathbf{v}_c^\top \boldsymbol{\epsilon}$$
Since $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_d)$, the linear combination $\mathbf{v}_c^\top \boldsymbol{\epsilon}$ is Gaussian with:
$$\mathbb{E}[\mathbf{v}_c^\top \boldsymbol{\epsilon}] = 0$$
$$\text{Var}(\mathbf{v}_c^\top \boldsymbol{\epsilon}) = \mathbf{v}_c^\top (\sigma^2 \mathbf{I}) \mathbf{v}_c = \sigma^2 \|\mathbf{v}_c\|_2^2 = \sigma^2$$
Therefore:
$$\mathbb{E}[z \mid c = +1] = +s, \quad \mathbb{E}[z \mid c = -1] = -s, \quad \text{Var}(z) = \sigma^2$$

#### 2. Error Rate Formulation
The decision rule classifies $\hat{c} = +1$ if $z > 0$, and $\hat{c} = -1$ if $z \le 0$.
Conditioned on $c = +1$, an error occurs if $z \le 0$:
$$P(\text{Error} \mid c = +1) = P(s + \mathbf{v}_c^\top \boldsymbol{\epsilon} \le 0) = P\left(\frac{\mathbf{v}_c^\top \boldsymbol{\epsilon}}{\sigma} \le -\frac{s}{\sigma}\right) = \Phi\left(-\frac{s}{\sigma}\right)$$
By symmetry, $P(\text{Error} \mid c = -1) = \Phi(-s/\sigma)$.
Thus, the total error rate is:
$$\mathcal{E} = \Phi\left(-\frac{s}{\sigma}\right)$$

#### 3. Quantitative Calculation
For $s = 2.0$ and $\sigma = 1.0$:
$$\frac{s}{\sigma} = 2.0 \implies \mathcal{E} = \Phi(-2.0) \approx 0.0228 \quad (2.28\%)$$
The theoretical accuracy is:
$$\text{Accuracy} = 1 - \mathcal{E} = 1 - 0.0228 = 97.72\%$$
As the sequence progresses through the Transformer, attention and MLP layers amplify the concept signal along $\mathbf{v}_c$ (increasing $s$), causing the error rate $\Phi(-s/\sigma)$ to decay exponentially toward zero.

---

## Exercise 3: Selectivity & Memorization Bounds under High Dimensions

A researcher evaluates a linear probe on an intermediate layer activation with dimension $d = 4096$ using a small dataset of $N = 256$ examples. The probe achieves $\text{Acc}_{\text{task}} = 98.4\%$.

1. Explain why high probe accuracy with $N < d$ could be entirely a mathematical artifact of high dimensionality rather than evidence of a true representation.
2. The researcher implements Hewitt & Liang's control task: assigning random binary labels to the 256 examples and re-fitting the probe with the same regularization $\lambda$. The control probe achieves $\text{Acc}_{\text{control}} = 96.1\%$. Compute the selectivity and explain what this reveals.
3. What two concrete remedies should the researcher apply to rigorously validate the finding?

### Solution

#### 1. High-Dimensional Dimensionality Artifact
When the number of parameters $d$ exceeds the number of training samples $N$ ($d > N$), any set of $N$ points in general position in $\mathbb{R}^d$ is linearly separable for **any arbitrary binary labeling** (Cover's Theorem on the capacity of hyperplanes). Therefore, a linear model with $d = 4096$ and $N = 256$ can achieve nearly $100\%$ training accuracy simply by memorizing noise.

#### 2. Selectivity Calculation & Diagnosis
$$\text{Selectivity} = \text{Acc}_{\text{task}} - \text{Acc}_{\text{control}} = 98.4\% - 96.1\% = 2.3\% \quad (0.023)$$
A selectivity of only $2.3\%$ reveals that the probe's high performance is almost completely due to its capacity to memorize the small dataset, not because the representation natively encodes the concept. The experiment fails the selectivity test.

#### 3. Concrete Remedies
1. **Increase Sample Size & Cross-Validation**: Collect at least $N \gg d$ samples (e.g., $N \ge 10,000$) and evaluate exclusively on an unseen test set.
2. **Dimension Reduction & Strong Regularization**: Project the activations onto a lower-dimensional subspace (e.g., via PCA or tight bottleneck rank $k \ll d$) and tune the $L_2$ regularization $\lambda$ until the control task accuracy drops to chance ($50\%$), then re-evaluate the real task accuracy.
