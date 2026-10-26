# Topic 6.3: Exercises & Solutions — Cross-Entropy & Perplexity

---

## 📝 Practice Exercises

### Exercise 1: Cross-Entropy vs. Entropy on a 3-Class Prediction
**Difficulty:** Introductory (Age 12+)

An AI image classifier is predicting whether an animal is a `[Dog, Cat, Bird]`.
The ground-truth reality for this category of animal follows true distribution $P$:
$$P(\text{Dog}) = 0.50, \quad P(\text{Cat}) = 0.25, \quad P(\text{Bird}) = 0.25$$

Two rival AI models produce the following prediction probabilities:
- **Model 1 ($Q_1$):** $Q_1 = [0.50, 0.25, 0.25]$ (perfect prediction)
- **Model 2 ($Q_2$):** $Q_2 = [0.25, 0.50, 0.25]$ (confused dog for cat)

**Tasks:**
1. Compute the true entropy $H(P)$ in bits.
2. Compute the cross-entropy $H(P, Q_1)$ in bits.
3. Compute the cross-entropy $H(P, Q_2)$ in bits.
4. Show that $H(P, Q_2) > H(P)$, and compute the extra bits wasted by Model 2.

---

### Exercise 2: Perplexity of an LLM on a Test Sentence
**Difficulty:** Intermediate (Applied NLP)

A mini language model generates a 4-token sequence: `"AI changes the world"`.
The conditional probabilities predicted by the model for each token at each step are:
- $P(\text{"AI"} \mid \text{start}) = 0.50$
- $P(\text{"changes"} \mid \text{"AI"}) = 0.25$
- $P(\text{"the"} \mid \text{"AI changes"}) = 0.50$
- $P(\text{"world"} \mid \text{"AI changes the"}) = 0.125$

**Tasks:**
1. Compute the Negative Log-Likelihood (NLL) loss for each individual token using $\log_2$.
2. Compute the average cross-entropy loss $\bar{\mathcal{L}}$ over the entire 4-token sequence in bits.
3. Compute the Perplexity ($\text{PPL} = 2^{\bar{\mathcal{L}}}$) of the model on this sentence.
4. Interpret the resulting perplexity: what does this number physically mean in terms of guessing choices?

---

### Exercise 3: Complete Calculus Derivation: Softmax Cross-Entropy Gradient
**Difficulty:** Advanced (Deep Learning Calculus)

In deep learning backpropagation, neural networks compute raw logits $\mathbf{z} = (z_1, z_2, \dots, z_K)$ and output probabilities via Softmax:
$$Q_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
The cross-entropy loss for a single training example with one-hot ground-truth label $\mathbf{y} = (y_1, \dots, y_K)$ (where $y_c = 1$ for correct class $c$ and $0$ otherwise) is:
$$\mathcal{L} = -\sum_{k=1}^K y_k \ln Q_k = -\ln Q_c$$

**Task:**
Using the chain rule of calculus, derive the partial derivative of the loss with respect to an arbitrary logit $z_i$:
$$\frac{\partial \mathcal{L}}{\partial z_i} = Q_i - y_i$$
Explain why this elegant result makes training neural networks with softmax cross-entropy so numerically efficient.

---

### Exercise 4: Analytical Limits of Softmax Temperature
**Difficulty:** Intermediate (Limits & Analysis)

Given logits $\mathbf{z} = (z_1, \dots, z_K)$ with a unique maximum $z_{\max} > z_j$ for all $j \ne \max$, analyze the temperature-scaled softmax:
$$Q_i(T) = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

**Tasks:**
1. Prove that $\lim_{T \to \infty} Q_i(T) = \frac{1}{K}$ for all $i \in \{1, \dots, K\}$.
2. Prove that $\lim_{T \to 0^+} Q_i(T) = \begin{cases} 1 & \text{if } i = \max \\ 0 & \text{if } i \ne \max \end{cases}$.

---

## 💡 Complete Step-by-Step Solutions

---

### Solution 1: Cross-Entropy vs. Entropy

#### Step 1: Compute True Entropy $H(P)$
$$H(P) = - \sum_{x} P(x) \log_2 P(x)$$
$$H(P) = - \left[ 0.50 \log_2(0.50) + 0.25 \log_2(0.25) + 0.25 \log_2(0.25) \right]$$
Since $\log_2(0.50) = -1$ and $\log_2(0.25) = -2$:
$$H(P) = - \left[ 0.50(-1) + 0.25(-2) + 0.25(-2) \right] = -[-0.50 - 0.50 - 0.50] = 1.50\text{ bits}$$

#### Step 2: Cross-Entropy of Model 1 ($Q_1$)
Because $Q_1 = P$:
$$H(P, Q_1) = -\sum P(x) \log_2 Q_1(x) = H(P) = 1.50\text{ bits}$$

#### Step 3: Cross-Entropy of Model 2 ($Q_2$)
$$H(P, Q_2) = - \left[ P(\text{Dog}) \log_2 Q_2(\text{Dog}) + P(\text{Cat}) \log_2 Q_2(\text{Cat}) + P(\text{Bird}) \log_2 Q_2(\text{Bird}) \right]$$
$$H(P, Q_2) = - \left[ 0.50 \log_2(0.25) + 0.25 \log_2(0.50) + 0.25 \log_2(0.25) \right]$$
$$H(P, Q_2) = - \left[ 0.50(-2) + 0.25(-1) + 0.25(-2) \right] = -[-1.00 - 0.25 - 0.50] = 1.75\text{ bits}$$

#### Step 4: Difference (Excess Bits / KL Divergence)
$$H(P, Q_2) - H(P) = 1.75 - 1.50 = 0.25\text{ bits}$$
Model 2 wastes an extra $0.25$ bits per prediction because its beliefs fail to align with ground truth.

---

### Solution 2: Perplexity of an LLM

#### Step 1: Token Losses in Bits ($-\log_2 P$)
1. Token 1 (`"AI"`): $-\log_2(0.50) = 1.0\text{ bit}$
2. Token 2 (`"changes"`): $-\log_2(0.25) = 2.0\text{ bits}$
3. Token 3 (`"the"`): $-\log_2(0.50) = 1.0\text{ bit}$
4. Token 4 (`"world"`): $-\log_2(0.125) = -\log_2(2^{-3}) = 3.0\text{ bits}$

#### Step 2: Average Cross-Entropy Loss
$$\bar{\mathcal{L}} = \frac{1 + 2 + 1 + 3}{4} = \frac{7}{4} = 1.75\text{ bits / token}$$

#### Step 3: Perplexity
$$\text{PPL} = 2^{\bar{\mathcal{L}}} = 2^{1.75} = 2^{1} \times 2^{0.75} \approx 2 \times 1.68179 = 3.3636$$

#### Step 4: Interpretation
A perplexity of $3.36$ means that across this sentence, the model was predicting next tokens with the same uncertainty as choosing uniformly from roughly **$3.36$ equally likely words** at each step.

---

### Solution 4: Softmax Cross-Entropy Gradient Derivation

#### Step 1: Express the Loss
$$\mathcal{L} = -\sum_{k=1}^K y_k \ln Q_k$$
Using the chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^K \frac{\partial \mathcal{L}}{\partial Q_k} \frac{\partial Q_k}{\partial z_i}$$
Since $\frac{\partial \mathcal{L}}{\partial Q_k} = -\frac{y_k}{Q_k}$:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{k=1}^K \frac{y_k}{Q_k} \frac{\partial Q_k}{\partial z_i}$$

#### Step 2: Differentiate Softmax $\frac{\partial Q_k}{\partial z_i}$
Recall $Q_k = \frac{e^{z_k}}{S}$ where $S = \sum_{j=1}^K e^{z_j}$.
Using the quotient rule:
- **Case 1: $k = i$**
  $$\frac{\partial Q_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = \frac{e^{z_i}}{S} \left( 1 - \frac{e^{z_i}}{S} \right) = Q_i (1 - Q_i)$$
- **Case 2: $k \ne i$**
  $$\frac{\partial Q_k}{\partial z_i} = \frac{0 \cdot S - e^{z_k} e^{z_i}}{S^2} = -\frac{e^{z_k}}{S} \frac{e^{z_i}}{S} = -Q_k Q_i$$

#### Step 3: Combine Cases
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{y_i}{Q_i} \left[ Q_i(1 - Q_i) \right] - \sum_{k \ne i} \frac{y_k}{Q_k} \left[ -Q_k Q_i \right]$$
Cancel out $Q_i$ and $Q_k$:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i (1 - Q_i) + \sum_{k \ne i} y_k Q_i = -y_i + y_i Q_i + Q_i \sum_{k \ne i} y_k$$
Combine the terms with $Q_i$:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + Q_i \left( y_i + \sum_{k \ne i} y_k \right) = -y_i + Q_i \left( \sum_{k=1}^K y_k \right)$$
Since ground-truth labels sum to $1$ ($\sum_{k=1}^K y_k = 1$):
$$\mathbf{\frac{\partial \mathcal{L}}{\partial z_i} = Q_i - y_i \quad \blacksquare}$$

#### Significance:
The gradient is simply **(Predicted Probability - True Label)**!
If the model predicted $Q_i = 0.80$ on a false class ($y_i = 0$), the gradient is $+0.80$, pushing the logit down.
If the model predicted $Q_i = 0.70$ on the true class ($y_i = 1$), the gradient is $0.70 - 1.0 = -0.30$, pushing the logit up!
No expensive matrix inversions or second derivatives are needed.

---

### Solution 4: Analytical Limits of Softmax Temperature

#### 1. High Temperature Limit ($T \to \infty$):
As $T \to \infty$, the exponent $\frac{z_i}{T} \to 0$.
Thus:
$$\lim_{T \to \infty} \exp\left(\frac{z_i}{T}\right) = e^0 = 1$$
Therefore:
$$\lim_{T \to \infty} Q_i(T) = \frac{1}{\sum_{j=1}^K 1} = \frac{1}{K} \quad \blacksquare$$

#### 2. Low Temperature Limit ($T \to 0^+$):
Divide numerator and denominator by $\exp\left(\frac{z_{\max}}{T}\right)$:
$$Q_i(T) = \frac{\exp\left( \frac{z_i - z_{\max}}{T} \right)}{1 + \sum_{j \ne \max} \exp\left( \frac{z_j - z_{\max}}{T} \right)}$$
Since $z_{\max} > z_j$ for all $j \ne \max$, the difference $z_j - z_{\max} < 0$.
As $T \to 0^+$, $\frac{z_j - z_{\max}}{T} \to -\infty$, which implies:
$$\lim_{T \to 0^+} \exp\left( \frac{z_j - z_{\max}}{T} \right) = 0$$

- For $i = \max$:
  $$Q_{\max}(T) \to \frac{\exp(0)}{1 + 0} = \frac{1}{1} = 1$$
- For $i \ne \max$:
  $$Q_i(T) \to \frac{0}{1 + 0} = 0 \quad \blacksquare$$

