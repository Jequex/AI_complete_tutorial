# Topic 6.1: Exercises & Solutions — Entropy & Information

---

## 📝 Practice Exercises

### Exercise 1: Surprisal and Units of Information
**Difficulty:** Introductory (Age 12+)

An AI agent receives three weather reports from three different weather sensors:
1. **Sensor A (Rain in the Sahara Desert):** The probability of rain today is $P(\text{Rain}) = \frac{1}{64}$.
2. **Sensor B (Cloudy in Seattle):** The probability of cloudy skies is $P(\text{Cloudy}) = \frac{1}{2}$.
3. **Sensor C (Sun in Cairo):** The probability of sunny skies is $P(\text{Sun}) = 1.0$.

**Tasks:**
1. Calculate the self-information (surprisal) $I(x) = -\log_2 P(x)$ in **bits** for each sensor report.
2. Convert the surprisal of Sensor A from **bits** to **nats** (natural log base $e$) using the conversion factor $\ln(2) \approx 0.69315$.
3. Which sensor provides the most information? Why does an event with probability $1.0$ convey zero information?

---

### Exercise 2: Analytical Maximization of the Binary Entropy Function
**Difficulty:** Intermediate (Calculus Derivation)

The binary entropy function describes the uncertainty of any two-outcome Bernoulli trial with parameter $p \in (0, 1)$:
$$H_b(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$

**Tasks:**
1. Express $H_b(p)$ using the natural logarithm $\ln$ using the change-of-base rule.
2. Differentiate $H_b(p)$ with respect to $p$ to find $\frac{dH_b}{dp}$.
3. Set $\frac{dH_b}{dp} = 0$ and solve for the critical point $p^*$.
4. Evaluate the second derivative $\frac{d^2H_b}{dp^2}$ at $p^*$ to prove that this critical point is a global maximum.
5. What is the maximum possible entropy of a binary system in bits?

---

### Exercise 3: Derivation of the Maximum Entropy Distribution (Discrete Alphabet)
**Difficulty:** Advanced (Lagrange Multipliers)

Suppose a discrete random variable $X$ can take on $K$ distinct outcomes $\{x_1, x_2, \dots, x_K\}$ with probabilities $p_i = P(X = x_i)$.
We wish to find the probability distribution $\mathbf{p} = (p_1, \dots, p_K)$ that maximizes Shannon entropy:
$$H(X) = -\sum_{i=1}^K p_i \ln p_i$$
subject to the probability normalization constraint:
$$\sum_{i=1}^K p_i = 1, \quad p_i \ge 0$$

**Tasks:**
1. Formulate the Lagrangian function $\mathcal{L}(p_1, \dots, p_K, \lambda)$ using Lagrange multiplier $\lambda$.
2. Take the partial derivative $\frac{\partial \mathcal{L}}{\partial p_i}$ for an arbitrary state $i$ and set it to zero.
3. Solve for each $p_i$ in terms of $\lambda$.
4. Use the normalization constraint $\sum_{i=1}^K p_i = 1$ to solve for the constant $\lambda$ and establish that $p_i = \frac{1}{K}$ for all $i$.
5. Compute the resulting maximum entropy $H_{\max}(X)$ in bits.

---

### Exercise 4: Constructing an Optimal Huffman Code by Hand
**Difficulty:** Applied & Algorithmic

Consider a specialized LLM vocabulary consisting of only 5 tokens $\{A, B, C, D, E\}$ with the following probabilities:
- $P(A) = 0.40$
- $P(B) = 0.20$
- $P(C) = 0.15$
- $P(D) = 0.15$
- $P(E) = 0.10$

**Tasks:**
1. Calculate the Shannon entropy $H(X)$ of this token distribution in bits to 4 decimal places.
2. Construct the Huffman coding tree step-by-step by repeatedly merging the two least probable nodes.
3. Assign binary prefix codewords (`0` for left branch, `1` for right branch) to each token.
4. Calculate the expected codeword length $L = \sum_{i=1}^5 P(x_i) \cdot \text{length}(c_i)$.
5. Verify Shannon's Source Coding Theorem:
   $$H(X) \le L < H(X) + 1$$
6. Compute the coding efficiency $\eta = \frac{H(X)}{L} \times 100\%$.

---

## 💡 Complete Step-by-Step Solutions

---

### Solution 1: Surprisal and Units of Information

#### Step 1: Compute Surprisal in Bits
Recall the definition of self-information (surprisal):
$$I(x) = -\log_2 P(x) = \log_2 \left(\frac{1}{P(x)}\right)$$

- **Sensor A (Sahara Rain):**
  $$P(\text{Rain}) = \frac{1}{64} = 2^{-6}$$
  $$I(\text{Rain}) = -\log_2(2^{-6}) = -(-6) = 6\text{ bits}$$

- **Sensor B (Seattle Cloudy):**
  $$P(\text{Cloudy}) = \frac{1}{2} = 2^{-1}$$
  $$I(\text{Cloudy}) = -\log_2(2^{-1}) = -(-1) = 1\text{ bit}$$

- **Sensor C (Cairo Sun):**
  $$P(\text{Sun}) = 1.0$$
  $$I(\text{Sun}) = -\log_2(1.0) = 0\text{ bits}$$

#### Step 2: Convert Bits to Nats
By the change-of-base formula for logarithms:
$$\log_2(z) = \frac{\ln(z)}{\ln(2)} \implies \ln(z) = \log_2(z) \cdot \ln(2)$$
Therefore:
$$I_{\text{nats}}(x) = I_{\text{bits}}(x) \times \ln(2) \approx 6 \times 0.693147 = 4.1589\text{ nats}$$

#### Step 3: Interpretation
- **Sensor A** provides the most information ($6\text{ bits}$). Because rain in the Sahara is rare ($1$ in $64$), hearing that it is raining eliminates $6$ consecutive doublings of uncertainty.
- **Sensor C** conveys $0\text{ bits}$ of information because sunny weather in Cairo is completely certain ($P=1.0$). If you already know something with $100\%$ certainty before a message arrives, the message contains zero surprise, alters no beliefs, and imparts zero new knowledge.

---

### Solution 2: Maximization of the Binary Entropy Function

#### Step 1: Change of Base
Recall that $\log_2(u) = \frac{\ln(u)}{\ln 2}$. Thus:
$$H_b(p) = -\frac{1}{\ln 2} \Big[ p \ln p + (1-p) \ln(1-p) \Big]$$

#### Step 2: First Derivative
Using the product rule on $p \ln p$:
$$\frac{d}{dp}[p \ln p] = 1 \cdot \ln p + p \cdot \frac{1}{p} = \ln p + 1$$
Using the product rule and chain rule on $(1-p)\ln(1-p)$:
$$\frac{d}{dp}[(1-p)\ln(1-p)] = (-1) \cdot \ln(1-p) + (1-p) \cdot \left(\frac{-1}{1-p}\right) = -\ln(1-p) - 1$$
Combining both terms:
$$\frac{d}{dp}\Big[ p \ln p + (1-p)\ln(1-p) \Big] = (\ln p + 1) - (\ln(1-p) + 1) = \ln p - \ln(1-p) = \ln\left(\frac{p}{1-p}\right)$$
Multiplying by $-\frac{1}{\ln 2}$:
$$\frac{dH_b}{dp} = -\frac{1}{\ln 2} \ln\left(\frac{p}{1-p}\right) = -\log_2\left(\frac{p}{1-p}\right)$$

#### Step 3: Solve for Critical Point
Setting the derivative equal to zero:
$$-\log_2\left(\frac{p}{1-p}\right) = 0 \iff \frac{p}{1-p} = 2^0 = 1$$
$$p = 1 - p \implies 2p = 1 \implies p^* = \frac{1}{2}$$

#### Step 4: Second Derivative Test
Differentiating $\frac{dH_b}{dp} = -\frac{1}{\ln 2} (\ln p - \ln(1-p))$:
$$\frac{d^2H_b}{dp^2} = -\frac{1}{\ln 2} \left[ \frac{1}{p} - \frac{-1}{1-p} \right] = -\frac{1}{\ln 2} \left[ \frac{1}{p} + \frac{1}{1-p} \right] = -\frac{1}{\ln 2} \frac{1}{p(1-p)}$$
Evaluating at $p^* = \frac{1}{2}$:
$$\left.\frac{d^2H_b}{dp^2}\right|_{p = 0.5} = -\frac{1}{\ln 2} \frac{1}{(0.5)(0.5)} = -\frac{4}{\ln 2} \approx -5.77 < 0$$
Since the second derivative is strictly negative everywhere on $p \in (0, 1)$, the binary entropy function is strictly concave, and $p^* = \frac{1}{2}$ is the unique global maximum.

#### Step 5: Maximum Entropy Value
$$H_b(0.5) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = -0.5(-1) - 0.5(-1) = 0.5 + 0.5 = 1.0\text{ bit}$$
A completely fair coin flip represents exactly 1 bit of Shannon entropy.

---

### Solution 3: Maximum Entropy Distribution for Discrete Alphabet

#### Step 1: Formulate the Lagrangian
We maximize $H(X) = -\sum_{i=1}^K p_i \ln p_i$ subject to $\sum_{i=1}^K p_i - 1 = 0$.
The Lagrangian is:
$$\mathcal{L}(p_1, \dots, p_K, \lambda) = -\sum_{i=1}^K p_i \ln p_i - \lambda \left( \sum_{i=1}^K p_i - 1 \right)$$

#### Step 2: Differentiate with Respect to $p_i$
$$\frac{\partial \mathcal{L}}{\partial p_i} = -\left( \ln p_i + p_i \cdot \frac{1}{p_i} \right) - \lambda = -(\ln p_i + 1) - \lambda$$
Setting to zero:
$$-(\ln p_i + 1) - \lambda = 0 \implies \ln p_i = -(1 + \lambda)$$

#### Step 3: Solve for $p_i$
Exponentiating both sides:
$$p_i = e^{-(1 + \lambda)}$$
Notice that the right-hand side is constant for every $i \in \{1, 2, \dots, K\}$. This implies that all probabilities must be identical:
$$p_1 = p_2 = \dots = p_K = c$$

#### Step 4: Apply Normalization Constraint
$$\sum_{i=1}^K p_i = \sum_{i=1}^K c = K \cdot c = 1 \implies c = \frac{1}{K}$$
Hence, the uniform distribution $p_i^* = \frac{1}{K}$ maximizes entropy.

#### Step 5: Maximum Entropy Value
Substituting $p_i = \frac{1}{K}$ back into Shannon's formula:
$$H_{\max}(X) = -\sum_{i=1}^K \frac{1}{K} \log_2\left(\frac{1}{K}\right) = -K \left( \frac{1}{K} (-\log_2 K) \right) = \log_2 K\text{ bits}$$

---

### Solution 4: Constructing an Optimal Huffman Code

#### Step 1: Compute Shannon Entropy
$$H(X) = -\sum_{i=1}^5 P(x_i) \log_2 P(x_i)$$
- For $A$ ($0.40$): $-0.40 \log_2(0.40) \approx -0.40 \times (-1.3219) = 0.5288$
- For $B$ ($0.20$): $-0.20 \log_2(0.20) \approx -0.20 \times (-2.3219) = 0.4644$
- For $C$ ($0.15$): $-0.15 \log_2(0.15) \approx -0.15 \times (-2.7370) = 0.4105$
- For $D$ ($0.15$): $-0.15 \log_2(0.15) \approx -0.15 \times (-2.7370) = 0.4105$
- For $E$ ($0.10$): $-0.10 \log_2(0.10) \approx -0.10 \times (-3.3219) = 0.3322$

Summing all values:
$$H(X) = 0.5288 + 0.4644 + 0.4105 + 0.4105 + 0.3322 = 2.1464\text{ bits}$$

#### Step 2: Huffman Tree Construction
Initial forest of nodes:
- `(A: 0.40)`, `(B: 0.20)`, `(C: 0.15)`, `(D: 0.15)`, `(E: 0.10)`

- **Merge 1:** The two lowest nodes are $D (0.15)$ and $E (0.10)$.
  Combine them into internal node $N_1$ with weight $0.15 + 0.10 = 0.25$.
  Remaining pool: `(A: 0.40)`, `(N1: 0.25)`, `(B: 0.20)`, `(C: 0.15)`.

- **Merge 2:** The two lowest nodes are $C (0.15)$ and $B (0.20)$.
  Combine them into internal node $N_2$ with weight $0.15 + 0.20 = 0.35$.
  Remaining pool: `(A: 0.40)`, `(N2: 0.35)`, `(N1: 0.25)`.

- **Merge 3:** The two lowest nodes are $N_1 (0.25)$ and $N_2 (0.35)$.
  Combine them into internal node $N_3$ with weight $0.25 + 0.35 = 0.60$.
  Remaining pool: `(A: 0.40)`, `(N3: 0.60)`.

- **Merge 4:** Combine $A (0.40)$ and $N_3 (0.60)$ to create the root with weight $1.00$.

#### Step 3: Codeword Assignment
Traversing from root (assigning `0` to left, `1` to right):
- Root $\to$ Left: $A \implies \text{Codeword} = \mathbf{0}$ (length 1)
- Root $\to$ Right: $N_3$ (prefix `1`):
  - $N_3 \to$ Left: $N_1$ (prefix `10`):
    - $N_1 \to$ Left: $D \implies \text{Codeword} = \mathbf{100}$ (length 3)
    - $N_1 \to$ Right: $E \implies \text{Codeword} = \mathbf{101}$ (length 3)
  - $N_3 \to$ Right: $N_2$ (prefix `11`):
    - $N_2 \to$ Left: $C \implies \text{Codeword} = \mathbf{110}$ (length 3)
    - $N_2 \to$ Right: $B \implies \text{Codeword} = \mathbf{111}$ (length 3)

Summary Table:
| Token | Probability $P(x)$ | Codeword | Length $l_i$ |
| :---: | :---: | :---: | :---: |
| **A** | 0.40 | `0` | 1 |
| **B** | 0.20 | `111` | 3 |
| **C** | 0.15 | `110` | 3 |
| **D** | 0.15 | `100` | 3 |
| **E** | 0.10 | `101` | 3 |

#### Step 4: Compute Average Codeword Length $L$
$$L = \sum_{i=1}^5 P(x_i) \cdot l_i = (0.40 \times 1) + (0.20 \times 3) + (0.15 \times 3) + (0.15 \times 3) + (0.10 \times 3)$$
$$L = 0.40 + 0.60 + 0.45 + 0.45 + 0.30 = 2.20\text{ bits}$$

#### Step 5: Verify Shannon's Bound
$$H(X) = 2.1464 \le L = 2.20 < H(X) + 1 = 3.1464$$
The Huffman average length $L = 2.20$ bits strictly obeys Shannon's Source Coding Theorem!

#### Step 6: Compute Coding Efficiency
$$\eta = \frac{H(X)}{L} \times 100\% = \frac{2.1464}{2.2000} \times 100\% \approx 97.56\%$$
The Huffman code achieves over $97.5\%$ optimal theoretical compaction without losing a single bit of information.

