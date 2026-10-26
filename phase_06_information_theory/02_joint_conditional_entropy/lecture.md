# Topic 6.2: Joint & Conditional Entropy — How Context Resolves Uncertainty

---

## 🧒 1. Intuitive Analogy: The Umbrella Detective & Context Clues

Imagine you are standing inside an office building with no windows. You want to guess what someone walking through the front lobby is holding.
Is it an **umbrella**, or a **coffee cup**?
Right now, you are completely in the dark. Your uncertainty is high.

Now imagine a security guard tells you one single clue:
> *"It is pouring rain outside!"*

Suddenly, your uncertainty collapses! You can say with $99\%$ confidence: *"They are holding an umbrella!"*

In artificial intelligence and information theory:
- **Joint Entropy $H(X, Y)$** is the total uncertainty about **both things together** (the weather outside AND the object in hand).
- **Conditional Entropy $H(Y \mid X)$** is the uncertainty that remains about $Y$ (the object in hand) **after you already know $X$** (the weather).

### The Golden Rule of Context: "Information Never Hurts"
In information theory, observing a related clue $X$ can **never increase** your uncertainty about $Y$ on average:
$$H(Y \mid X) \le H(Y)$$
This is why modern language models (like GPT-4 and Claude) become drastically more confident and accurate as you give them more context: the conditional entropy of the next token drops dramatically when conditioned on previous sentences!

---

## 🔣 2. Mathematical Symbol Decoder Table

Before diving into the formulas, let us define each mathematical symbol clearly:

| Symbol | Mathematical Name | Plain English Translation |
| :--- | :--- | :--- |
| $X, Y$ | Discrete Random Variables | Two different random events (e.g., $X = \text{Previous Word}$, $Y = \text{Next Word}$). |
| $P(x, y)$ | Joint Probability Mass Function | The probability that $X = x$ and $Y = y$ happen simultaneously. |
| $P(y \mid x)$ | Conditional Probability | The probability that $Y = y$ given that we already know $X = x$. |
| $H(X, Y)$ | Joint Shannon Entropy | The total uncertainty (in bits) when observing the pair $(X, Y)$ simultaneously. |
| $H(Y \mid X)$ | Conditional Shannon Entropy | The remaining uncertainty in $Y$ after the value of $X$ is revealed. |
| $H(X)$ | Marginal Entropy | The standalone uncertainty of variable $X$ ignoring everything else. |
| $\sum_{x} \sum_{y}$ | Double Summation | Loop over every possible outcome of $X$, and for each, loop over every possible outcome of $Y$. |

---

## 📐 3. Formal Definitions & Mathematical Formulations

### Definition 3.1: Joint Entropy $H(X, Y)$
The joint entropy of two discrete random variables $(X, Y)$ distributed according to $P(x, y)$ is defined as:
$$H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log_2 P(x, y)$$

Just like single-variable entropy, joint entropy measures the average number of yes/no questions needed to determine the state of both variables together.

---

### Definition 3.2: Conditional Entropy $H(Y \mid X)$
The conditional entropy of $Y$ given $X$ is the expected value (weighted average) of the entropy of the conditional distribution $P(Y \mid X = x)$:
$$H(Y \mid X) = \sum_{x \in \mathcal{X}} P(x) H(Y \mid X = x)$$

Expanding $H(Y \mid X = x) = -\sum_{y \in \mathcal{Y}} P(y \mid x) \log_2 P(y \mid x)$:
$$H(Y \mid X) = -\sum_{x \in \mathcal{X}} P(x) \sum_{y \in \mathcal{Y}} P(y \mid x) \log_2 P(y \mid x)$$

Using the fundamental probability rule $P(x) P(y \mid x) = P(x, y)$:
$$\mathbf{H(Y \mid X) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log_2 P(y \mid x)}$$

---

## 🔬 4. The Chain Rule of Entropy: Complete Derivation

The Chain Rule states that the total uncertainty of a pair equals the uncertainty of the first variable plus the remaining uncertainty of the second given the first:
$$H(X, Y) = H(X) + H(Y \mid X)$$

### Step-by-Step Proof:
Recall that conditional probability is defined as:
$$P(x, y) = P(x) P(y \mid x) \implies \log_2 P(x, y) = \log_2 P(x) + \log_2 P(y \mid x)$$

Now substitute this into the definition of joint entropy:
$$H(X, Y) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(x, y)$$
$$H(X, Y) = -\sum_{x} \sum_{y} P(x, y) \left[ \log_2 P(x) + \log_2 P(y \mid x) \right]$$

Split the double summation into two distinct terms:
$$H(X, Y) = \underbrace{-\sum_{x} \sum_{y} P(x, y) \log_2 P(x)}_{\text{Term 1}} + \underbrace{\left( -\sum_{x} \sum_{y} P(x, y) \log_2 P(y \mid x) \right)}_{\text{Term 2}}$$

Let us simplify **Term 1**:
Notice that $\log_2 P(x)$ does not depend on $y$, so we can factor it out of the inner sum over $y$:
$$-\sum_{x} \left( \sum_{y} P(x, y) \right) \log_2 P(x)$$
By marginal probability, summing over all possible states of $y$ yields the marginal distribution: $\sum_{y} P(x, y) = P(x)$.
Therefore:
$$\text{Term 1} = -\sum_{x} P(x) \log_2 P(x) = H(X)$$

Now look at **Term 2**:
$$\text{Term 2} = -\sum_{x} \sum_{y} P(x, y) \log_2 P(y \mid x) = H(Y \mid X)$$

Adding Term 1 and Term 2 together:
$$H(X, Y) = H(X) + H(Y \mid X) \quad \blacksquare$$

By symmetry of variables, we also have:
$$H(X, Y) = H(Y) + H(X \mid Y)$$

---

## 📉 5. Fundamental Theorem: Conditioning Never Increases Entropy

### Theorem:
For any discrete random variables $X$ and $Y$:
$$H(Y \mid X) \le H(Y)$$
with equality if and only if $X$ and $Y$ are statistically independent ($P(x, y) = P(x)P(y)$).

### Proof (via Jensen's Inequality & Concavity of $\log$):
Consider the difference between marginal entropy and conditional entropy:
$$H(Y) - H(Y \mid X) = -\sum_{y} P(y) \log_2 P(y) - \left( -\sum_{x, y} P(x, y) \log_2 P(y \mid x) \right)$$

Express $P(y) = \sum_{x} P(x, y)$ in the first sum:
$$H(Y) - H(Y \mid X) = -\sum_{x, y} P(x, y) \log_2 P(y) + \sum_{x, y} P(x, y) \log_2 P(y \mid x)$$
$$H(Y) - H(Y \mid X) = \sum_{x, y} P(x, y) \log_2 \left( \frac{P(y \mid x)}{P(y)} \right)$$

Using $P(y \mid x) = \frac{P(x, y)}{P(x)}$:
$$H(Y) - H(Y \mid X) = \sum_{x, y} P(x, y) \log_2 \left( \frac{P(x, y)}{P(x) P(y)} \right)$$

This expression is the **Kullback-Leibler Divergence** $D_{\text{KL}}(P(x, y) \parallel P(x)P(y))$.
Using the fundamental log-inequality $\ln(u) \le u - 1$ (with equality only when $u = 1$):
$$-\sum_{x, y} P(x, y) \ln \left( \frac{P(x) P(y)}{P(x, y)} \right) \ge -\sum_{x, y} P(x, y) \left( \frac{P(x) P(y)}{P(x, y)} - 1 \right)$$
$$= -\left( \sum_{x, y} P(x) P(y) - \sum_{x, y} P(x, y) \right) = -(1 - 1) = 0$$

Multiplying by $\frac{1}{\ln 2} > 0$:
$$H(Y) - H(Y \mid X) \ge 0 \implies H(Y \mid X) \le H(Y) \quad \blacksquare$$

---

## 🤖 6. AI Application: Why Transformers Need Context (Autoregressive Factorization)

Every generative language model (GPT-4, LLaMA, Claude) models the joint probability of a sequence of $T$ tokens $\mathbf{w} = (w_1, w_2, \dots, w_T)$ using the probabilistic chain rule:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

Taking the negative logarithm on both sides and taking expectations gives the **Chain Rule of Entropy for Sequences**:
$$H(W_1, W_2, \dots, W_T) = H(W_1) + H(W_2 \mid W_1) + H(W_3 \mid W_1, W_2) + \dots + H(W_T \mid W_1, \dots, W_{T-1})$$

### What this means:
- **Unigram (Zero Context):** Guessing word $w_t$ in isolation has high entropy ($\approx 10\text{ bits}$).
- **Bigram (1 Word Context):** Guessing word $w_t$ after seeing the previous word drops entropy to $\approx 6\text{ bits}$.
- **Full Transformer (8,000+ Tokens Context):** Guessing the next word given thousands of preceding tokens drops conditional entropy to **less than 1.5 bits**!

Context literally squeezes the uncertainty out of the universe.

