# Topic 6.5: Exercises & Solutions — Mutual Information & Information Bottleneck

---

## 📝 Practice Exercises

### Exercise 1: Mutual Information of a Binary Symmetric Channel (BSC)
**Difficulty:** Introductory to Intermediate (Age 12+)

A communication channel transmits a binary digit $X \in \{0, 1\}$.
Due to noise, the bit flips with crossover probability $p \in (0, 0.5)$:
$$P(Y = 1 \mid X = 0) = p, \quad P(Y = 0 \mid X = 0) = 1 - p$$
$$P(Y = 0 \mid X = 1) = p, \quad P(Y = 1 \mid X = 1) = 1 - p$$
The transmitter sends $0$ and $1$ with equal probability: $P(X = 0) = P(X = 1) = 0.50$.

**Tasks:**
1. Compute the marginal entropy $H(X)$ in bits.
2. Compute the conditional entropy $H(Y \mid X = 0)$ and $H(Y \mid X = 1)$.
3. Compute the overall conditional entropy $H(Y \mid X)$.
4. Compute the Mutual Information $I(X; Y) = H(Y) - H(Y \mid X)$ in terms of the binary entropy function $H_b(p)$.
5. If $p = 0.10$ (10% bit flip error rate), what is $I(X; Y)$? How much information successfully crosses the channel?

---

### Exercise 2: The Data Processing Inequality
**Difficulty:** Intermediate (Mathematical Proof)

The Data Processing Inequality (DPI) is one of the most foundational laws in AI:
> *"No clever post-processing algorithm can ever create information that was not already present in the data."*

Formally, let $X \to Y \to Z$ form a Markov chain, meaning $P(Z \mid Y, X) = P(Z \mid Y)$ (conditioned on $Y$, $Z$ is independent of $X$).

**Task:**
Using the chain rule of mutual information $I(X; Y, Z) = I(X; Z) + I(X; Y \mid Z) = I(X; Y) + I(X; Z \mid Y)$, prove that:
$$I(X; Y) \ge I(X; Z)$$
Explain why this guarantees that an AI model cannot magically increase mutual information about an object beyond what the camera sensor captured.

---

### Exercise 3: Feature Selection via Mutual Information
**Difficulty:** Applied (Machine Learning Engineering)

A machine learning engineer is building a credit card fraud detection model with label $Y \in \{\text{Fraud}, \text{Legitimate}\}$ with $H(Y) = 0.80\text{ bits}$.
She measures two potential features:
- **Feature A (Transaction Amount):** $H(Y \mid X_A) = 0.25\text{ bits}$.
- **Feature B (Customer Zip Code):** $H(Y \mid X_B) = 0.70\text{ bits}$.

**Tasks:**
1. Compute $I(X_A; Y)$ and $I(X_B; Y)$.
2. Which feature should the engineer prioritize, and why?
3. What percentage of the label uncertainty does Feature A resolve?

---

### Exercise 4: Tishby's Information Bottleneck Optimal Solution
**Difficulty:** Advanced (Calculus of Variations / Optimization)

In Naftali Tishby's Information Bottleneck, we minimize the functional:
$$\mathcal{L}_{\text{IB}} = I(X; T) - \beta I(T; Y)$$
subject to the normalization constraint $\sum_{t} P(t \mid x) = 1$ for all $x$.

**Tasks:**
1. Express $I(X; T)$ and $I(T; Y)$ using conditional and marginal distributions.
2. Explain intuitively what happens when $\beta \to 0$ versus when $\beta \to \infty$.
3. Describe the state of the internal representation $T$ in both extreme limits.

---

## 💡 Complete Step-by-Step Solutions

---

### Solution 1: Binary Symmetric Channel (BSC)

#### Step 1: Marginal Entropy $H(X)$
Since $X$ is a fair coin ($P(0) = P(1) = 0.50$):
$$H(X) = -0.50 \log_2(0.50) - 0.50 \log_2(0.50) = 1.0\text{ bit}$$

#### Step 2: Conditional Entropies
Given $X = 0$, $Y$ follows distribution $(1-p, p)$:
$$H(Y \mid X = 0) = - (1-p) \log_2(1-p) - p \log_2(p) = H_b(p)$$
Given $X = 1$, $Y$ follows distribution $(p, 1-p)$:
$$H(Y \mid X = 1) = - p \log_2(p) - (1-p) \log_2(1-p) = H_b(p)$$

#### Step 3: Total Conditional Entropy
$$H(Y \mid X) = P(X=0) H(Y \mid X=0) + P(X=1) H(Y \mid X=1) = 0.5 H_b(p) + 0.5 H_b(p) = H_b(p)$$

#### Step 4: Marginal Distribution of $Y$ & Mutual Information
By symmetry, $P(Y = 0) = P(X=0)(1-p) + P(X=1)p = 0.5(1-p) + 0.5p = 0.50$.
Thus $H(Y) = 1.0\text{ bit}$.
The Mutual Information is:
$$I(X; Y) = H(Y) - H(Y \mid X) = 1 - H_b(p)\text{ bits}$$

#### Step 5: Numerical Evaluation for $p = 0.10$
$$H_b(0.10) = -0.10 \log_2(0.10) - 0.90 \log_2(0.90) \approx 0.3322 + 0.1368 = 0.4690\text{ bits}$$
$$I(X; Y) = 1.0000 - 0.4690 = 0.5310\text{ bits}$$
With a $10\%$ error rate, each transmitted bit successfully conveys $0.531$ bits of reliable information.

---

### Solution 2: The Data Processing Inequality

#### Proof:
Expand $I(X; Y, Z)$ using the Chain Rule of Mutual Information in two ways:
$$I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y)$$
$$I(X; Y, Z) = I(X; Z) + I(X; Y \mid Z)$$
Equating both expressions:
$$I(X; Y) + I(X; Z \mid Y) = I(X; Z) + I(X; Y \mid Z)$$

Because $X \to Y \to Z$ forms a Markov chain, $X$ and $Z$ are conditionally independent given $Y$.
Therefore, conditional mutual information vanishes:
$$I(X; Z \mid Y) = 0$$
This simplifies the equality to:
$$I(X; Y) + 0 = I(X; Z) + I(X; Y \mid Z) \implies I(X; Y) = I(X; Z) + I(X; Y \mid Z)$$
Since mutual information is always non-negative ($I(X; Y \mid Z) \ge 0$):
$$I(X; Y) \ge I(X; Z) \quad \blacksquare$$

#### Significance for Artificial Intelligence:
If an original photograph $X$ is fed into a neural network with layers $L_1 \to L_2 \to \dots \to L_k$, the Data Processing Inequality guarantees:
$$I(X; \text{Raw Pixels}) \ge I(X; L_1) \ge I(X; L_2) \ge \dots \ge I(X; L_k)$$
Information about the raw pixels is permanently lost or discarded at each layer. A neural network cannot manufacture information; it can only extract and isolate the signal relevant to the target!

---

### Solution 3: Feature Selection via Mutual Information

#### Step 1: Compute Mutual Information
Recall $I(X; Y) = H(Y) - H(Y \mid X)$:
- **Feature A:** $I(X_A; Y) = 0.80 - 0.25 = 0.55\text{ bits}$
- **Feature B:** $I(X_B; Y) = 0.80 - 0.70 = 0.10\text{ bits}$

#### Step 2: Prioritization
The engineer should prioritize **Feature A**.
It shares $0.55$ bits of mutual information with the fraud label, which is $5.5\times$ more informative than Feature B ($0.10$ bits).

#### Step 3: Percentage of Uncertainty Resolved
$$\frac{I(X_A; Y)}{H(Y)} \times 100\% = \frac{0.55}{0.80} \times 100\% = 68.75\%$$
Feature A alone eliminates nearly $69\%$ of all uncertainty about fraud!

---

### Solution 4: Tishby's Information Bottleneck Limits

#### 1. Formulations:
$$I(X; T) = \sum_{x, t} P(x, t) \log_2 \left(\frac{P(t \mid x)}{P(t)}\right)$$
$$I(T; Y) = \sum_{t, y} P(t, y) \log_2 \left(\frac{P(y \mid t)}{P(y)}\right)$$

#### 2. Extreme Limit $\beta \to 0$ (Maximum Compression):
When $\beta \to 0$, the objective becomes:
$$\min_\theta I(X; T)$$
The network cares only about compressing the input and ignores task performance entirely.
The optimal solution is a constant trivial representation: $P(t \mid x) = P(t)$, meaning $T$ is completely independent of $X$.
Here $I(X; T) = 0$ and $I(T; Y) = 0$. The network collapses into a blank slate.

#### 3. Extreme Limit $\beta \to \infty$ (Maximum Preservation):
When $\beta \to \infty$, the objective becomes:
$$\max_\theta I(T; Y)$$
The network prioritizes memorizing everything that might help predict $Y$, ignoring compression.
The optimal representation simply preserves the entire input: $T = X$, giving $I(X; T) = H(X)$ and $I(T; Y) = I(X; Y)$.
This leads to severe overfitting, as the network memorizes background noise and idiosyncrasies of the training set.

An intermediate, optimal $\beta^*$ produces the Goldilocks zone: optimal compression yielding maximum generalization on unseen test data!

