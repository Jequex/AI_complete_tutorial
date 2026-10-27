# Topic 1.2: Exercises & Solutions — AI Software Engineering, Tooling & Reproducibility

Welcome to the problem set for **Topic 1.2: AI Software Engineering, Tooling, and Reproducibility**!

Test your understanding of numerical stability, algorithmic profiling, and reproducible experiment design. Work through each exercise before opening the collapsible solution.

---

## 📝 Problem Set

### Problem 1: Unit Testing Numerical Invariants & Safe Softmax

#### The Context
The standard mathematical definition of the Softmax function for an input logit vector $\mathbf{z} \in \mathbb{R}^K$ is:
$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

In 64-bit IEEE floating-point arithmetic, if $z_i \ge 710$, $e^{z_i}$ overflows to `inf`, leading to $\frac{\infty}{\infty} = \text{NaN}$ (Not a Number), which destroys neural network weights.
The standard numerical stabilization trick subtracts the maximum logit $m = \max_k z_k$:
$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i - m}}{\sum_{j=1}^K e^{z_j - m}}$$

#### Tasks
1. Prove mathematically why subtracting an arbitrary constant $c$ from every logit does not alter the output probability distribution.
2. Calculate by hand the safe softmax probabilities for logits $\mathbf{z} = [1000.0, 1001.0, 1002.0]$.
3. Write a `unittest.TestCase` suite verifying sum-to-1 and stability on extreme logits.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Numerical Stability & Safe Softmax

#### Part 1: Mathematical Proof of Invariance to Constant Shift
Let $c$ be any real scalar constant. Replace $z_i$ with $z_i - c$:
$$\frac{e^{z_i - c}}{\sum_{j=1}^K e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_{j=1}^K \left( e^{z_j} \cdot e^{-c} \right)} = \frac{e^{z_i} \cdot e^{-c}}{e^{-c} \cdot \sum_{j=1}^K e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
The factor $e^{-c}$ cancels out in both numerator and denominator! Choosing $c = \max_k z_k$ guarantees that the largest exponent is $e^0 = 1.0$, completely preventing numerical overflow!

#### Part 2: Step-by-Step Hand Calculation
Given logits $\mathbf{z} = [1000.0, 1001.0, 1002.0]$:
1. Find maximum: $m = 1002.0$.
2. Shift logits:
   $$z_1 - m = 1000.0 - 1002.0 = -2.0$$
   $$z_2 - m = 1001.0 - 1002.0 = -1.0$$
   $$z_3 - m = 1002.0 - 1002.0 = 0.0$$
3. Exponentiate:
   $$e^{-2} \approx 0.1353$$
   $$e^{-1} \approx 0.3679$$
   $$e^0 = 1.0000$$
4. Compute denominator:
   $$\text{Sum} = 0.1353 + 0.3679 + 1.0000 = \mathbf{1.5032}$$
5. Normalize probabilities:
   $$p_1 = \frac{0.1353}{1.5032} \approx \mathbf{0.0900} \quad (9.0\%)$$
   $$p_2 = \frac{0.3679}{1.5032} \approx \mathbf{0.2447} \quad (24.5\%)$$
   $$p_3 = \frac{1.0000}{1.5032} \approx \mathbf{0.6653} \quad (66.5\%)$$
   Check: $0.0900 + 0.2447 + 0.6653 = 1.0000$ ✅

#### Part 3: Python Implementation & Unit Tests
```python
import math
import unittest

def safe_softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    shifted_exp = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(shifted_exp)
    return [e / sum_exp for e in shifted_exp]

class TestSafeSoftmax(unittest.TestCase):
    def test_extreme_overflow(self):
        logits = [1000.0, 1001.0, 1002.0]
        probs = safe_softmax(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertAlmostEqual(probs[2], 0.6652, places=3)
```
</details>

---

### Problem 2: Baseline Random Loss Invariant Check

You are training a Transformer language model with a vocabulary size of $V = 32,000$ tokens.

1. State the exact formula for the expected initial cross-entropy loss $\mathcal{L}_{\text{random}}$ assuming a uniform random prediction over the vocabulary.
2. Calculate the exact numerical baseline value using natural logs.
3. If your training script initializes the model and prints `Epoch 0 Loss: 1.45`, what does this invariant tell you about your implementation?

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Baseline Random Loss Invariant Check

#### Step 1: Theoretical Formula
$$\mathcal{L}_{\text{random}} = -\ln\left(\frac{1}{V}\right) = \ln(V)$$

#### Step 2: Numerical Calculation
$$V = 32,000$$
$$\mathcal{L}_{\text{random}} = \ln(32,000) \approx \mathbf{10.3735}$$

#### Step 3: Analysis
If the initial loss is **$1.45$** instead of **$\sim 10.37$**, there is a severe bug in the data pipeline or evaluation setup:
- A loss of $1.45$ corresponds to an effective vocabulary of $e^{1.45} \approx 4.26$ tokens!
- Probable bugs: Data leakage (labels accidentally passed as inputs), masking bug (padding tokens counted as correct predictions), or computing loss on an empty slice.
- Testing this invariant in unit tests prevents deploying broken pipelines!
</details>

---

### Problem 3: Algorithmic Profiling & Asymptotic Complexity

A researcher has two implementations of a dataset deduplication module for $N$ text documents:
- **Implementation A:** Nested list iteration: for each item, check `if item not in seen_list: seen_list.append(item)`.
- **Implementation B:** Hash set lookup: `seen_set = set(documents)`.

1. What is the time complexity in Big-O notation of Implementation A vs. Implementation B?
2. If $N = 100,000$ and checking an element in a list takes $1\mu\text{s}$, estimate the runtime of Implementation A vs. Implementation B.

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: Algorithmic Profiling & Asymptotic Complexity

#### Step 1: Big-O Asymptotic Complexity
- **Implementation A (List search):** Checking `item in seen_list` takes $\mathcal{O}(k)$ time where $k$ is the current length of the list. Summing over $N$ items gives:
  $$\sum_{k=1}^N \mathcal{O}(k) = \mathcal{O}(N^2)$$
- **Implementation B (Hash set):** Hash set lookup and insertion takes $\mathcal{O}(1)$ average time. Over $N$ items, this takes:
  $$\mathcal{O}(N)$$

#### Step 2: Estimated Runtimes for $N = 100,000$
- **Implementation A:** $\frac{N(N+1)}{2} \approx \frac{10^{10}}{2} = 5 \times 10^9$ operations.
  $$\text{Time} \approx 5 \times 10^9 \times 10^{-6}\text{ s} = 5,000\text{ seconds} \approx \mathbf{1.39\text{ hours}!}$$
- **Implementation B:** $100,000$ operations.
  $$\text{Time} \approx 100,000 \times 10^{-6}\text{ s} = \mathbf{0.1\text{ seconds}!}$$
- *Conclusion:* Switching from a list to a hash set speeds up execution by over **50,000x**!
</details>
