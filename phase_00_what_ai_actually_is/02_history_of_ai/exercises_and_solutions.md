# Exercises & Detailed Solutions: Topic 0.2

Welcome to the problem set for **Topic 0.2: History of AI — From Alan Turing to Modern Reasoning**.

---

## 📝 Problem Set

### Problem 1: Manual Synthesis of a Perceptron Decision Boundary (NAND Gate)
The logical **NAND (NOT-AND)** gate is the fundamental universal building block of computer architecture. Its truth table is:

| $x_1$ | $x_2$ | Target $y$ |
| :---: | :---: | :---: |
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

- **Part A:** Is the NAND gate linearly separable? Explain geometrically.
- **Part B:** A perceptron classifies an input as $1$ if $w_1 x_1 + w_2 x_2 + b \geq 0$, and $0$ if $w_1 x_1 + w_2 x_2 + b < 0$. Write down the system of 4 linear inequalities corresponding to the NAND truth table.
- **Part C:** Find an explicit set of integer values for $w_1$, $w_2$, and $b$ that satisfy all 4 inequalities. Verify your solution for every input pair.

---

### Problem 2: The Minsky-Papert XOR Proof (Algebraic Derivation)
In the lecture, we introduced Minsky and Papert's 1969 proof that a single perceptron cannot compute XOR.
The truth table for XOR is:
- $(0, 0) \to 0$
- $(1, 0) \to 1$
- $(0, 1) \to 1$
- $(1, 1) \to 0$

Assume there exists a set of real numbers $w_1, w_2, b \in \mathbb{R}$ such that:
$$f(x_1, x_2) = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b \geq 0 \\ 0 & \text{if } w_1 x_1 + w_2 x_2 + b < 0 \end{cases}$$
correctly computes XOR.

Reconstruct the full algebraic proof step-by-step to show that this assumption leads to a mathematical contradiction.

---

### Problem 3: The Pendulum of AI Paradigms (Comparative Analysis)
Consider two historical approaches to building a medical diagnosis system:
- **System A (1982 Expert System):** Built with 2,000 hand-written `IF-THEN` rules by expert physicians.
- **System B (2024 Multimodal Foundation Model):** A 70-billion parameter neural network trained on millions of clinical case files and medical scans.

Compare the two systems across:
1. Handling of unexpected or corrupted input data.
2. Transparency / Interpretability of the reasoning process.
3. Scalability to new diseases or medical discoveries.

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Manual Synthesis of a Perceptron (NAND Gate)

#### Part A: Geometric Linear Separability
Yes! If we plot the four points on a 2D Cartesian plane:
- $(0, 0)$ is labeled $+1$
- $(0, 1)$ is labeled $+1$
- $(1, 0)$ is labeled $+1$
- $(1, 1)$ is labeled $0$

Only a single point $(1, 1)$ is $0$. We can easily draw a straight line separating $(1, 1)$ from the other three points (for example, the line $x_1 + x_2 = 1.5$). Therefore, NAND is linearly separable.

---

#### Part B: The 4 Inequalities
Using the decision rule $z = w_1 x_1 + w_2 x_2 + b$:

1. For input $(0, 0) \to y = 1$:
   $$w_1(0) + w_2(0) + b \geq 0 \implies \mathbf{b \geq 0}$$

2. For input $(0, 1) \to y = 1$:
   $$w_1(0) + w_2(1) + b \geq 0 \implies \mathbf{w_2 + b \geq 0}$$

3. For input $(1, 0) \to y = 1$:
   $$w_1(1) + w_2(0) + b \geq 0 \implies \mathbf{w_1 + b \geq 0}$$

4. For input $(1, 1) \to y = 0$:
   $$w_1(1) + w_2(1) + b < 0 \implies \mathbf{w_1 + w_2 + b < 0}$$

---

#### Part C: Explicit Integer Parameter Values & Verification
By symmetry, we can set $w_1 = w_2 = w$.
From inequality (4):
$$2w + b < 0 \implies 2w < -b$$
From inequality (1):
$$b \geq 0$$
Let us pick:
$$b = 2$$
Then $2w < -2 \implies w < -1$. Let us pick:
$$w_1 = -2, \quad w_2 = -2, \quad b = 2$$

Now, let us verify all 4 cases:
1. **Input $(0, 0)$:**
   $$z = (-2)(0) + (-2)(0) + 2 = 0 + 0 + 2 = +2$$
   Since $z = +2 \geq 0$, $\hat{y} = 1$. **(Correct! Target = 1)**

2. **Input $(0, 1)$:**
   $$z = (-2)(0) + (-2)(1) + 2 = 0 - 2 + 2 = 0$$
   Since $z = 0 \geq 0$, $\hat{y} = 1$. **(Correct! Target = 1)**

3. **Input $(1, 0)$:**
   $$z = (-2)(1) + (-2)(0) + 2 = -2 + 0 + 2 = 0$$
   Since $z = 0 \geq 0$, $\hat{y} = 1$. **(Correct! Target = 1)**

4. **Input $(1, 1)$:**
   $$z = (-2)(1) + (-2)(1) + 2 = -2 - 2 + 2 = -2$$
   Since $z = -2 < 0$, $\hat{y} = 0$. **(Correct! Target = 0)**

All 4 conditions are satisfied! A single perceptron can easily compute NAND.

---

### Solution 2: The Minsky-Papert XOR Proof

Let us write out the four inequalities required for XOR:

$$\begin{aligned}
\text{(Eq. 1 for } (0,0) \to 0\text{):} \quad & b < 0 \\
\text{(Eq. 2 for } (1,0) \to 1\text{):} \quad & w_1 + b \geq 0 \\
\text{(Eq. 3 for } (0,1) \to 1\text{):} \quad & w_2 + b \geq 0 \\
\text{(Eq. 4 for } (1,1) \to 0\text{):} \quad & w_1 + w_2 + b < 0
\end{aligned}$$

#### Step 1: Sum Eq. 2 and Eq. 3
Add Eq. 2 and Eq. 3 together:
$$(w_1 + b) + (w_2 + b) \geq 0 + 0$$
$$w_1 + w_2 + 2b \geq 0$$

Group the terms to highlight Eq. 4:
$$(w_1 + w_2 + b) + b \geq 0 \quad \text{--- (Inequality A)}$$

#### Step 2: Analyze the Signs from Eq. 4 and Eq. 1
- Eq. 4 explicitly asserts:
  $$(w_1 + w_2 + b) < 0 \quad \text{(a strictly negative number)}$$
- Eq. 1 explicitly asserts:
  $$b < 0 \quad \text{(a strictly negative number)}$$

#### Step 3: Compute the Sum of Two Negative Numbers
The sum of two strictly negative numbers must be strictly negative:
$$\underbrace{(w_1 + w_2 + b)}_{< 0} + \underbrace{b}_{< 0} < 0 \quad \text{--- (Inequality B)}$$

#### Step 4: Identify the Contradiction
- Inequality A states that $(w_1 + w_2 + b) + b \geq 0$.
- Inequality B states that $(w_1 + w_2 + b) + b < 0$.

No real quantity can be both non-negative ($\geq 0$) and strictly negative ($< 0$) simultaneously.
Thus, our initial premise that a single linear threshold unit could compute XOR is false.
$$\therefore \text{XOR is not linearly separable.} \quad \blacksquare$$

---

### Solution 3: Comparative Analysis of AI Paradigms

| Evaluation Dimension | System A: 1982 Expert System (Symbolic) | System B: 2024 Foundation Model (Statistical/Deep) |
| :--- | :--- | :--- |
| **1. Corrupted / Noisy Inputs** | **Extremely Brittle:** A single misspelled symptom or missing blood test causes an `IF` condition to evaluate to `False`, aborting the rule chain or returning no diagnosis. | **Highly Robust:** Continuous embeddings map misspelled words or noisy sensor data into nearby semantic vector space, degrading gracefully. |
| **2. Interpretability** | **Glass Box (100% Explainable):** Can trace the exact firing order: *"Rule 42 fired because Fever > 101, which triggered Rule 108..."* | **Black Box / Emergent:** Reasoning is distributed across billions of parameters. Requires advanced mechanistic probing, attention analysis, or Chain-of-Thought prompting to audit. |
| **3. Scalability** | **Linear Bottleneck:** Adding a newly discovered disease requires manually revising thousands of interlocking rules to prevent logical contradictions. | **Autonomous Ingestion:** Can absorb thousands of medical research papers directly through pre-training or Retrieval-Augmented Generation (RAG). |

