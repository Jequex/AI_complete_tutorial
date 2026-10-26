# Topic 4.1: Exercises & Solutions — Probability Fundamentals & Bayes' Theorem

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Kolmogorov Axiom Proofs & 3-Set Inclusion-Exclusion** | Medium |
| **2** | **Bayesian Sequential Updating: The Two-Test Miracle** | Medium |
| **3** | **Naive Bayes Math & Laplace Smoothing from Scratch** | Challenging |

---

## Problem 1: Kolmogorov Axiom Proofs & 3-Set Inclusion-Exclusion

### The Context
In computer vision and NLP, we frequently calculate the probability that at least one of several detector modules fires. The Inclusion-Exclusion principle extends Kolmogorov's additivity axiom to overlapping sets.

### Part A: Derived Properties from Axioms
Using only Kolmogorov's three axioms:
1. $P(A) \ge 0$
2. $P(\Omega) = 1.0$
3. $P(\bigcup A_i) = \sum P(A_i)$ for disjoint sets

Prove that:
1. $P(\emptyset) = 0$
2. $P(A^c) = 1 - P(A)$
3. If $A \subseteq B$, then $P(A) \le P(B)$ (Monotonicity)

### Part B: Three-Set Inclusion-Exclusion
Derive the formula for $P(A \cup B \cup C)$ in terms of individual probabilities, pairwise intersections, and the triple intersection.

---

### Solution 1

#### Part A: Axiomatic Proofs
1. **Proof that $P(\emptyset) = 0$:**
   Notice that $\Omega$ and $\emptyset$ are disjoint ($\Omega \cap \emptyset = \emptyset$), and their union is $\Omega \cup \emptyset = \Omega$.
   By Axiom 3 (Additivity for disjoint sets):
   $$P(\Omega \cup \emptyset) = P(\Omega) + P(\emptyset)$$
   $$P(\Omega) = P(\Omega) + P(\emptyset)$$
   Subtracting $P(\Omega)$ (which is $1$ by Axiom 2) from both sides:
   $$P(\emptyset) = 0 \quad \blacksquare$$

2. **Proof that $P(A^c) = 1 - P(A)$:**
   By definition of complement, $A$ and $A^c$ are mutually disjoint ($A \cap A^c = \emptyset$), and their union is the entire sample space ($A \cup A^c = \Omega$).
   Applying Axiom 3:
   $$P(A \cup A^c) = P(A) + P(A^c)$$
   $$P(\Omega) = P(A) + P(A^c)$$
   Since $P(\Omega) = 1$ (Axiom 2):
   $$1 = P(A) + P(A^c) \implies P(A^c) = 1 - P(A) \quad \blacksquare$$

3. **Proof of Monotonicity ($A \subseteq B \implies P(A) \le P(B)$):**
   If $A \subseteq B$, we can decompose $B$ into two disjoint pieces: $A$, and the elements of $B$ outside $A$ ($B \setminus A$):
   $$B = A \cup (B \setminus A) \quad \text{where } A \cap (B \setminus A) = \emptyset$$
   By Axiom 3:
   $$P(B) = P(A) + P(B \setminus A)$$
   By Axiom 1 (Non-negativity), $P(B \setminus A) \ge 0$. Therefore:
   $$P(B) \ge P(A) \quad \blacksquare$$

#### Part B: 3-Set Inclusion-Exclusion
Treat $(A \cup B)$ as a single event $D$. Then $A \cup B \cup C = D \cup C$:
$$P(D \cup C) = P(D) + P(C) - P(D \cap C)$$

Expanding $P(D) = P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
Next, expand $D \cap C = (A \cup B) \cap C = (A \cap C) \cup (B \cap C)$ using the distributive law:
$$P(D \cap C) = P((A \cap C) \cup (B \cap C)) = P(A \cap C) + P(B \cap C) - P((A \cap C) \cap (B \cap C))$$
Notice that $(A \cap C) \cap (B \cap C) = A \cap B \cap C$.
Substituting back into the full equation:
$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C) \quad \blacksquare$$

---

## Problem 2: Bayesian Sequential Updating: The Two-Test Miracle

### The Context
In the lecture, we observed that for a disease with prevalence $P(D) = 0.0001$, a test with 99% sensitivity and 1% false positive rate yielded an initial posterior of only:
$$P(D \mid \text{Test}_1 = +) \approx 0.98\%$$

Now, suppose the patient takes a **second, completely independent test** (using a different chemical reaction or laboratory), and it **also comes back positive (+)**!

### Mathematical Task
1. What becomes the **new prior** probability for the second test?
2. Compute the new updated posterior probability $P(D \mid \text{Test}_1 = +, \text{Test}_2 = +)$.
3. Explain why the probability leaps dramatically from under 1% to over 50%.

---

### Solution 2

#### 1. The New Prior
In Bayesian inference, **today's posterior becomes tomorrow's prior**.
After the first test, our updated belief is:
$$P(\text{Disease}) = P(D \mid \text{Test}_1 = +) \approx 0.009804$$
$$P(\text{Healthy}) = 1 - 0.009804 = 0.990196$$

#### 2. Calculating the Second Posterior
For independent tests:
- $P(\text{Test}_2 = + \mid \text{Disease}) = 0.99$
- $P(\text{Test}_2 = + \mid \text{Healthy}) = 0.01$

Applying Bayes' Theorem:
$$\text{Numerator} = P(\text{Test}_2 = + \mid D) \times P(D) = 0.99 \times 0.009804 = 0.009706$$

$$\text{Evidence } P(\text{Test}_2 = +) = (0.99 \times 0.009804) + (0.01 \times 0.990196)$$
$$P(\text{Evidence}) = 0.009706 + 0.009902 = 0.019608$$

$$\text{Updated Posterior} = \frac{0.009706}{0.019608} \approx \mathbf{0.495 \approx 49.5\%}$$
*(If a third independent positive test is administered, the posterior surges above $99\%$!)*

#### 3. Why It Leaps:
The first test weeded out 99% of the massive healthy population. The new prior (almost 1%) is $100\times$ higher than the original background prevalence ($0.01\%$). With a strong prior, the second test's evidence decisively shifts the balance of probability!

---

## Problem 3: Naive Bayes Spam Classification by Hand

### The Context
A spam filter evaluates the message: **"buy cash prize"**.
Given training statistics:
- Priors: $P(\text{Spam}) = 0.60$, $P(\text{Ham}) = 0.40$
- Vocabulary size $V = 100$ words
- Word counts in training set:

| Word | Count in Spam ($N_{\text{spam}}=500$) | Count in Ham ($N_{\text{ham}}=1000$) |
| :--- | :---: | :---: |
| "buy" | 40 | 10 |
| "cash" | 25 | 2 |
| "prize" | 35 | 1 |

Using **Laplace (+1) Smoothing**:
$$P(w \mid C) = \frac{\text{Count}(w, C) + 1}{\sum_{w'} \text{Count}(w', C) + V}$$

1. Compute smoothed likelihoods $P(\text{buy} \mid \text{Spam})$ and $P(\text{buy} \mid \text{Ham})$.
2. Compute the unnormalized posterior score for Spam and Ham.
3. Compute the final normalized probability $P(\text{Spam} \mid \text{"buy cash prize"})$.

---

### Solution 3

#### 1. Smoothed Likelihoods
- Total spam words with smoothing: $500 + 100 = 600$
- Total ham words with smoothing: $1000 + 100 = 1100$

**For Spam ($/ 600$):**
- $P(\text{buy} \mid \text{Spam}) = \frac{40 + 1}{600} = \frac{41}{600} \approx 0.06833$
- $P(\text{cash} \mid \text{Spam}) = \frac{25 + 1}{600} = \frac{26}{600} \approx 0.04333$
- $P(\text{prize} \mid \text{Spam}) = \frac{35 + 1}{600} = \frac{36}{600} \approx 0.06000$

**For Ham ($/ 1100$):**
- $P(\text{buy} \mid \text{Ham}) = \frac{10 + 1}{1100} = \frac{11}{1100} = 0.01000$
- $P(\text{cash} \mid \text{Ham}) = \frac{2 + 1}{1100} = \frac{3}{1100} \approx 0.002727$
- $P(\text{prize} \mid \text{Ham}) = \frac{1 + 1}{1100} = \frac{2}{1100} \approx 0.001818$

#### 2. Unnormalized Posterior Scores
**Spam Score:**
$$\text{Score}_{\text{spam}} = P(\text{Spam}) \times P(\text{buy} \mid \text{Spam}) \times P(\text{cash} \mid \text{Spam}) \times P(\text{prize} \mid \text{Spam})$$
$$\text{Score}_{\text{spam}} = 0.60 \times 0.06833 \times 0.04333 \times 0.06000 \approx 1.0660 \times 10^{-4}$$

**Ham Score:**
$$\text{Score}_{\text{ham}} = P(\text{Ham}) \times P(\text{buy} \mid \text{Ham}) \times P(\text{cash} \mid \text{Ham}) \times P(\text{prize} \mid \text{Ham})$$
$$\text{Score}_{\text{ham}} = 0.40 \times 0.01000 \times 0.002727 \times 0.001818 \approx 1.9833 \times 10^{-8}$$

#### 3. Normalized Probability
$$\text{Total Evidence} = (1.0660 \times 10^{-4}) + (1.9833 \times 10^{-8}) \approx 1.0662 \times 10^{-4}$$
$$P(\text{Spam} \mid \text{"buy cash prize"}) = \frac{1.0660 \times 10^{-4}}{1.0662 \times 10^{-4}} \approx \mathbf{99.98\%}$$
*Conclusion: The classifier marks the email as SPAM with near-total certainty!*

