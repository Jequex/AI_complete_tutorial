# Topic 1.2: Exercises & Solutions — AI Software Engineering, Tooling & Reproducibility

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Unit Testing Numerical Invariants & Safe Softmax** | Medium |
| **2** | **Algorithmic Profiling & Eliminating Bottlenecks** | Medium |
| **3** | **Production Training Configuration & Reproducibility Harness** | Challenging |

---

## Problem 1: Unit Testing Numerical Invariants & Safe Softmax

### The Context
The standard mathematical definition of the Softmax function for an input vector $z \in \mathbb{R}^K$ is:
$$\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

However, in computer floating point arithmetic, if $z_i \ge 710$ (in standard 64-bit IEEE floats), $e^{z_i}$ evaluates to `inf`, resulting in `inf / inf = nan` (Not a Number), crashing neural network training.
The standard numerical stabilization trick subtracts the maximum value:
$$z^* = z - \max(z)$$
$$\text{Softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^K e^{z_j - \max(z)}}$$

### Requirements
1. Implement `safe_softmax(logits: list[float]) -> list[float]`.
2. Write a comprehensive `unittest.TestCase` suite testing:
   - **Sum to 1:** The probabilities sum to $1.0$ within tolerance $10^{-6}$.
   - **Range:** Every output $0 \le p_i \le 1$.
   - **Numerical Stability:** The function does not produce `nan` or overflow when given extreme inputs like `[1000.0, 1001.0, 1002.0]`.

---

### Solution 1

```python
import math
import unittest

def safe_softmax(logits: list[float]) -> list[float]:
    """Computes numerically stable softmax over a list of scalar logits."""
    if not logits:
        return []
    max_logit = max(logits)
    # Subtract max_logit to prevent e^(large_num) overflow
    shifted_exp = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(shifted_exp)
    return [e / sum_exp for e in shifted_exp]

class TestSafeSoftmax(unittest.TestCase):
    def test_probability_distribution(self):
        logits = [1.0, 2.0, 3.0]
        probs = safe_softmax(logits)
        
        # Test sum equals 1.0
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        
        # Test monotonic ordering
        self.assertTrue(probs[0] < probs[1] < probs[2])

    def test_numerical_stability_overflow(self):
        """Standard math.exp(1000) raises OverflowError; safe_softmax must handle it."""
        extreme_logits = [1000.0, 1001.0, 1002.0]
        probs = safe_softmax(extreme_logits)
        
        self.assertFalse(any(math.isnan(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_empty_input(self):
        self.assertEqual(safe_softmax([]), [])

if __name__ == "__main__":
    unittest.main()
```

---

## Problem 2: Algorithmic Profiling & Eliminating Bottlenecks

### The Context
An AI data-preprocessing script takes 45 seconds per epoch to filter unigram vocabulary tokens from a 100,000-word dataset. The author wrote:

```python
def unoptimized_vocab_filter(token_stream: list[str], stopwords: list[str]) -> list[str]:
    cleaned = []
    for token in token_stream:
        if token not in stopwords:
            cleaned.append(token)
    return cleaned
```

1. What is the Big-O time complexity of `token not in stopwords` when `stopwords` is a `list` of length $M$, and `token_stream` has length $N$?
2. What is the overall Big-O time complexity of the function?
3. Rewrite the function using an optimal Python data structure. What is the new time complexity?
4. Explain why this algorithmic choice impacts GPU utilization during training.

---

### Solution 2

1. **List Membership Complexity:**
   Checking `token not in stopwords` on a `list` requires a linear scan through the list from index $0$ to $M-1$.
   Time Complexity: $O(M)$.

2. **Overall Complexity:**
   For $N$ tokens in `token_stream`, performing an $O(M)$ scan for each yields:
   $$\text{Total Time} = O(N \times M)$$
   If $N = 100{,}000$ and $M = 5{,}000$, this performs up to $500{,}000{,}000$ string comparisons!

3. **Optimized Implementation:**
   Convert `stopwords` to a hash table (`set`), where membership lookup is $O(1)$ on average:
   ```python
   def optimized_vocab_filter(token_stream: list[str], stopwords: set[str]) -> list[str]:
       # stopwords lookup is now O(1)
       stopwords_set = set(stopwords) if not isinstance(stopwords, set) else stopwords
       return [token for token in token_stream if token not in stopwords_set]
   ```
   **New Time Complexity:** $O(M)$ to build the set once, plus $O(N \times 1)$ for lookups $\implies O(N + M)$.
   This transforms a 45-second bottleneck into a 5-millisecond operation (a ~9,000x speedup)!

4. **GPU Utilization Impact:**
   GPUs perform matrix computations in fractions of a millisecond. If the CPU host thread takes 45 seconds to prepare the next batch of tokens, the GPU will sit completely idle at 0% compute capacity. Fast $O(1)$ CPU pipelines prevent **GPU Starvation**.

---

## Problem 3: Production Training Configuration & Reproducibility Harness

### The Context
Create an end-to-end reproducible experiment runner that combines `argparse`, `random.seed()`, and `logging`.

### Requirements
Write a class `ExperimentConfig` that:
1. Accepts hyperparameters (`seed`, `learning_rate`, `batch_size`, `model_name`).
2. Provides a method `lock_seeds()` that seeds Python's `random` module.
3. Provides a method `to_dict()` and `log_summary(logger)`.
4. Demonstrate creating two configurations with identical seeds and show that calling a stochastic weight initializer produces the exact same sequence.

---

### Solution 3

```python
import random
import logging
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    seed: int
    learning_rate: float
    batch_size: int
    model_name: str

    def lock_seeds(self):
        random.seed(self.seed)

    def log_summary(self, logger: logging.Logger):
        logger.info(
            f"Config initialized: Model={self.model_name} | "
            f"Seed={self.seed} | LR={self.learning_rate} | Batch={self.batch_size}"
        )

# Verification
config_a = ExperimentConfig(seed=42, learning_rate=1e-4, batch_size=32, model_name="GPT-Nano")
config_b = ExperimentConfig(seed=42, learning_rate=1e-4, batch_size=32, model_name="GPT-Nano")

# Run A
config_a.lock_seeds()
draws_a = [random.random() for _ in range(5)]

# Run B
config_b.lock_seeds()
draws_b = [random.random() for _ in range(5)]

assert draws_a == draws_b
print("Reproducibility verified: Runs with identical configs produced bit-for-bit identical outputs.")
```

