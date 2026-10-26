# Topic 6.2: Exercises & Solutions — Joint & Conditional Entropy

---

## 📝 Practice Exercises

### Exercise 1: Joint & Conditional Entropy from a Contingency Table
**Difficulty:** Introductory (Age 12+)

A self-driving delivery robot tracks two variables on its delivery route:
- Weather $X \in \{\text{Sunny}, \text{Rainy}\}$
- Traffic Delay $Y \in \{\text{On Time}, \text{Delayed}\}$

Over 100 delivery trips, the joint probability table $P(X = x, Y = y)$ is:

| | $Y = \text{On Time}$ | $Y = \text{Delayed}$ |
| :--- | :---: | :---: |
| **$X = \text{Sunny}$** | $0.40$ | $0.10$ |
| **$X = \text{Rainy}$** | $0.10$ | $0.40$ |

**Tasks:**
1. Compute the marginal probabilities $P(X)$ and $P(Y)$.
2. Calculate the marginal entropies $H(X)$ and $H(Y)$ in bits.
3. Compute the joint entropy $H(X, Y)$ in bits.
4. Calculate the conditional entropy $H(Y \mid X)$ directly using conditional probabilities.
5. Verify the Chain Rule: $H(X, Y) = H(X) + H(Y \mid X)$.
6. Does knowing the weather reduce uncertainty about whether the robot is delayed? Quantify the uncertainty reduction $H(Y) - H(Y \mid X)$.

---

### Exercise 2: The Chain Rule for Three Variables
**Difficulty:** Intermediate (Algebraic Derivation)

In sequence modeling and deep autoregressive models, we often deal with chains of three or more tokens $(X, Y, Z)$.

**Task:**
Using the two-variable chain rule $H(A, B) = H(A) + H(B \mid A)$, formally prove the three-variable chain rule:
$$H(X, Y, Z) = H(X) + H(Y \mid X) + H(Z \mid X, Y)$$
Explain what each term represents in the context of an autoregressive Large Language Model generating a 3-word sentence.

---

### Exercise 3: Statistical Independence and Entropy Equality
**Difficulty:** Intermediate (Mathematical Proof)

Prove that $H(Y \mid X) = H(Y)$ if and only if $X$ and $Y$ are statistically independent ($P(x, y) = P(x)P(y)$ for all $x, y$).

---

### Exercise 4: First-Order Markov Chains in Language Modeling
**Difficulty:** Applied (NLP & Information Theory)

Consider a sequence of random variables $X_1, X_2, \dots, X_T$ forming a first-order Markov chain, meaning the probability of each token depends only on the immediately preceding token:
$$P(X_t \mid X_{t-1}, X_{t-2}, \dots, X_1) = P(X_t \mid X_{t-1})$$

**Tasks:**
1. Simplify the joint entropy of the sequence $H(X_1, X_2, \dots, X_T)$ using the Markov property.
2. If the Markov chain is stationary with entropy rate $\mathcal{H} = \lim_{T \to \infty} \frac{1}{T} H(X_1, \dots, X_T)$, express $\mathcal{H}$ in terms of single-step conditional entropy.

---

## 💡 Complete Step-by-Step Solutions

---

### Solution 1: Joint & Conditional Entropy from a Contingency Table

#### Step 1: Marginal Probabilities
Sum across columns to get $P(X)$:
- $P(X = \text{Sunny}) = 0.40 + 0.10 = 0.50$
- $P(X = \text{Rainy}) = 0.10 + 0.40 = 0.50$

Sum across rows to get $P(Y)$:
- $P(Y = \text{On Time}) = 0.40 + 0.10 = 0.50$
- $P(Y = \text{Delayed}) = 0.10 + 0.40 = 0.50$

#### Step 2: Marginal Entropies
Both $X$ and $Y$ are uniform binary variables:
$$H(X) = -0.50 \log_2(0.50) - 0.50 \log_2(0.50) = 1.0\text{ bit}$$
$$H(Y) = -0.50 \log_2(0.50) - 0.50 \log_2(0.50) = 1.0\text{ bit}$$

#### Step 3: Joint Entropy $H(X, Y)$
$$H(X, Y) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(x, y)$$
Notice the four cells have values $0.40, 0.10, 0.10, 0.40$:
- For $P = 0.40$: $-0.40 \log_2(0.40) \approx -0.40 \times (-1.32193) = 0.52877\text{ bits}$
- For $P = 0.10$: $-0.10 \log_2(0.10) \approx -0.10 \times (-3.32193) = 0.33219\text{ bits}$

$$H(X, Y) = 2 \times 0.52877 + 2 \times 0.33219 = 1.05754 + 0.66438 = 1.7219\text{ bits}$$

#### Step 4: Conditional Entropy $H(Y \mid X)$
Compute conditional distributions:
- Given $X = \text{Sunny}$:
  $P(\text{On Time} \mid \text{Sunny}) = \frac{0.40}{0.50} = 0.80$
  $P(\text{Delayed} \mid \text{Sunny}) = \frac{0.10}{0.50} = 0.20$
  $H(Y \mid X = \text{Sunny}) = -0.80 \log_2(0.80) - 0.20 \log_2(0.20) \approx 0.2575 + 0.4644 = 0.7219\text{ bits}$

- Given $X = \text{Rainy}$:
  $P(\text{On Time} \mid \text{Rainy}) = \frac{0.10}{0.50} = 0.20$
  $P(\text{Delayed} \mid \text{Rainy}) = \frac{0.40}{0.50} = 0.80$
  $H(Y \mid X = \text{Rainy}) = -0.20 \log_2(0.20) - 0.80 \log_2(0.80) \approx 0.7219\text{ bits}$

Now take the weighted average over $P(X)$:
$$H(Y \mid X) = P(\text{Sunny}) H(Y \mid \text{Sunny}) + P(\text{Rainy}) H(Y \mid \text{Rainy})$$
$$H(Y \mid X) = 0.50(0.7219) + 0.50(0.7219) = 0.7219\text{ bits}$$

#### Step 5: Verify the Chain Rule
$$H(X) + H(Y \mid X) = 1.0000 + 0.7219 = 1.7219\text{ bits} = H(X, Y)$$
The Chain Rule holds exactly!

#### Step 6: Uncertainty Reduction
Uncertainty before knowing weather: $H(Y) = 1.0000\text{ bit}$.
Uncertainty after knowing weather: $H(Y \mid X) = 0.7219\text{ bits}$.
$$\Delta H = H(Y) - H(Y \mid X) = 1.0000 - 0.7219 = 0.2781\text{ bits}$$
Knowing the weather reduces delay uncertainty by over $27.8\%$!

---

### Solution 2: The Chain Rule for Three Variables

#### Derivation:
Group the first two variables together as a compound variable $W = (X, Y)$.
By the standard two-variable Chain Rule:
$$H(X, Y, Z) = H(W, Z) = H(W) + H(Z \mid W)$$
Substitute $W = (X, Y)$:
$$H(X, Y, Z) = H(X, Y) + H(Z \mid X, Y)$$
Now apply the two-variable Chain Rule to $H(X, Y)$:
$$H(X, Y) = H(X) + H(Y \mid X)$$
Substitute this back:
$$H(X, Y, Z) = H(X) + H(Y \mid X) + H(Z \mid X, Y) \quad \blacksquare$$

#### LLM Interpretation:
- $H(X)$: Uncertainty in generating the first word from scratch (e.g., `"The"`).
- $H(Y \mid X)$: Uncertainty in generating the second word given the first (e.g., `"The cat"`).
- $H(Z \mid X, Y)$: Uncertainty in generating the third word given both prior words (e.g., `"The cat sat"`).

---

### Solution 3: Statistical Independence and Entropy Equality

#### Proof:
Recall that conditional entropy is:
$$H(Y \mid X) = -\sum_{x, y} P(x, y) \log_2 P(y \mid x)$$

If $X$ and $Y$ are statistically independent, then by definition:
$$P(y \mid x) = \frac{P(x, y)}{P(x)} = \frac{P(x)P(y)}{P(x)} = P(y)$$
Substitute $P(y \mid x) = P(y)$ into the conditional entropy expression:
$$H(Y \mid X) = -\sum_{x, y} P(x, y) \log_2 P(y)$$
Since $\log_2 P(y)$ does not depend on $x$, we can factor it out of the summation over $x$:
$$H(Y \mid X) = -\sum_{y} \log_2 P(y) \left( \sum_{x} P(x, y) \right)$$
Using the marginal property $\sum_{x} P(x, y) = P(y)$:
$$H(Y \mid X) = -\sum_{y} P(y) \log_2 P(y) = H(Y)$$
Conversely, if $H(Y \mid X) = H(Y)$, then $H(Y) - H(Y \mid X) = I(X; Y) = D_{\text{KL}}(P(x,y) \parallel P(x)P(y)) = 0$, which strictly requires $P(x,y) = P(x)P(y)$ everywhere. $\blacksquare$

---

### Solution 4: First-Order Markov Chains in Language Modeling

#### Step 1: Sequential Expansion
Using the general sequence chain rule:
$$H(X_1, X_2, \dots, X_T) = H(X_1) + \sum_{t=2}^T H(X_t \mid X_{t-1}, X_{t-2}, \dots, X_1)$$
By the Markov property, $X_t$ is conditionally independent of all history prior to $X_{t-1}$:
$$H(X_t \mid X_{t-1}, X_{t-2}, \dots, X_1) = H(X_t \mid X_{t-1})$$
Therefore:
$$H(X_1, X_2, \dots, X_T) = H(X_1) + \sum_{t=2}^T H(X_t \mid X_{t-1})$$

#### Step 2: Entropy Rate $\mathcal{H}$
If the process is stationary, $H(X_t \mid X_{t-1})$ is constant for all $t \ge 2$, equal to $H(X_2 \mid X_1)$.
Thus:
$$\lim_{T \to \infty} \frac{1}{T} \left( H(X_1) + (T - 1) H(X_2 \mid X_1) \right) = H(X_2 \mid X_1)$$
The long-run entropy rate of a stationary first-order Markov chain is simply the single-step conditional entropy $H(X_2 \mid X_1)$!

