# Topic 4.1: Exercises & Solutions — Probability Fundamentals & Bayes' Theorem

Welcome to the problem set for **Topic 4.1: Probability Fundamentals, Axioms & Bayes' Theorem**.

Grab pencil and paper. Work through each problem before expanding the collapsible solution panels.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Kolmogorov Axiom Proofs & 3-Set Inclusion-Exclusion** | Medium | Verifying consistent probability bounds in multi-label classifiers |
| **2** | **Bayesian Sequential Updating: The Two-Test Miracle** | Medium | Sequential evidence updating in robotics, medical AI, and active learning |
| **3** | **Naive Bayes Math & Laplace Smoothing from Scratch** | Challenging | Spam classification and zero-frequency edge cases in NLP |

---

## Problem 1: Kolmogorov Axiom Proofs & 3-Set Inclusion-Exclusion

### 🧒 The Context: Maya's Marble Game Rules
Using only Kolmogorov's three foundational axioms:
1. $P(A) \ge 0$ (Non-negativity)
2. $P(\Omega) = 1.0$ (Unitarity)
3. $P(\bigcup A_i) = \sum P(A_i)$ for mutually disjoint sets ($A_i \cap A_j = \emptyset$)

### Tasks:
1. Prove from the axioms that $P(\emptyset) = 0$.
2. Prove that $P(A^c) = 1 - P(A)$.
3. Prove Monotonicity: If $A \subseteq B$, then $P(A) \le P(B)$.
4. Derive the 3-Set Inclusion-Exclusion formula for $P(A \cup B \cup C)$.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Proof that $P(\emptyset) = 0$
$\Omega$ and $\emptyset$ are disjoint ($\Omega \cap \emptyset = \emptyset$), and their union is $\Omega \cup \emptyset = \Omega$.
By Axiom 3:
$$P(\Omega \cup \emptyset) = P(\Omega) + P(\emptyset)$$
$$P(\Omega) = P(\Omega) + P(\emptyset)$$
Subtracting $P(\Omega)$ (which is $1.0$ by Axiom 2) from both sides:
$$P(\emptyset) = \mathbf{0} \quad \blacksquare$$

#### Step 2: Proof that $P(A^c) = 1 - P(A)$
By definition of the complement, $A$ and $A^c$ are mutually disjoint ($A \cap A^c = \emptyset$), and their union covers the entire sample space ($A \cup A^c = \Omega$).
By Axiom 3:
$$P(A \cup A^c) = P(A) + P(A^c)$$
$$P(\Omega) = P(A) + P(A^c)$$
By Axiom 2, $P(\Omega) = 1.0$:
$$1.0 = P(A) + P(A^c) \implies \mathbf{P(A^c) = 1 - P(A)} \quad \blacksquare$$

#### Step 3: Proof of Monotonicity ($A \subseteq B \implies P(A) \le P(B)$)
If $A$ is a subset of $B$, we can decompose $B$ into two non-overlapping pieces: $A$, and the elements of $B$ outside $A$ ($B \setminus A$):
$$B = A \cup (B \setminus A) \quad \text{where } A \cap (B \setminus A) = \emptyset$$
By Axiom 3:
$$P(B) = P(A) + P(B \setminus A)$$
By Axiom 1, probabilities must be non-negative: $P(B \setminus A) \ge 0$.
Therefore:
$$P(B) \ge P(A) \implies \mathbf{P(A) \le P(B)} \quad \blacksquare$$

#### Step 4: Three-Set Inclusion-Exclusion Formula
For two sets: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
For three sets, let $D = B \cup C$:
$$P(A \cup D) = P(A) + P(D) - P(A \cap D)$$
Substituting $P(D) = P(B \cup C) = P(B) + P(C) - P(B \cap C)$ and using the distributive law $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$:
$$\mathbf{P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)}$$
- 🧒 **What the math is actually doing:** We add individual areas, subtract the double-counted overlaps, and add back the triple-overlap that was subtracted one time too many!
</details>

---

## Problem 2: Bayesian Sequential Updating: The Two-Test Miracle

### 🧒 The Context: Barnaby's Second Medical Test
In our lecture, Barnaby the puppy tested positive on a $99\%$-accurate test for a rare disease ($0.1\%$ prevalence).
Despite the $99\%$ accuracy, his true chance of having the disease was only **$9.0\%$** due to the overwhelming number of healthy puppies.
Maya is worried, so she immediately orders an **independent second test** from a different medical laboratory.
The second test *also* comes back **Positive** ($\text{Test}_2^+$)!

### Tasks:
1. Using the Posterior from Test 1 ($P(\text{Sick}) = 0.09016$) as the new **Prior** for Test 2, calculate the updated probability $P(\text{Sick} \mid \text{Test}_1^+, \text{Test}_2^+)$.
2. Show that two independent positive tests transform a $9\%$ suspicion into an overwhelming confirmation!

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Set Up the New Prior from Test 1
- New Prior: $P(\text{Sick}) = 0.09016$
- New Prior Complement: $P(\text{Healthy}) = 1 - 0.09016 = 0.90984$

The test specifications remain identical:
- $P(\text{Test}_2^+ \mid \text{Sick}) = 0.99$
- $P(\text{Test}_2^+ \mid \text{Healthy}) = 0.01$

#### Step 2: Calculate New Total Probability of Positive Result $P(\text{Test}_2^+)$
$$P(\text{Test}_2^+) = P(\text{Test}_2^+ \mid \text{Sick}) P(\text{Sick}) + P(\text{Test}_2^+ \mid \text{Healthy}) P(\text{Healthy})$$
$$P(\text{Test}_2^+) = (0.99 \times 0.09016) + (0.01 \times 0.90984)$$
$$P(\text{Test}_2^+) = 0.089258 + 0.009098 = \mathbf{0.098356}$$

#### Step 3: Apply Bayes' Theorem
$$P(\text{Sick} \mid \text{Test}_1^+, \text{Test}_2^+) = \frac{P(\text{Test}_2^+ \mid \text{Sick}) P(\text{Sick})}{P(\text{Test}_2^+)}$$
$$P(\text{Sick} \mid \text{Test}_1^+, \text{Test}_2^+) = \frac{0.089258}{0.098356} \approx \mathbf{0.9075} \approx \mathbf{90.8\%}!$$

#### Step 4: The Takeaway
- Starting Prior: $0.1\%$
- After Test 1: $9.0\%$
- After Test 2: $\mathbf{90.8\%}$!
Two independent tests slashed the probability of a false alarm from $91\%$ down to under $10\%$.
This is **Sequential Bayesian Updating**—the mathematical foundation of Kalman filters in robotics and belief updates in AI agents!
</details>

---

## Problem 3: Naive Bayes Math & Laplace Smoothing from Scratch

### 🧒 The Context: Timmy's Spam Filter
Timmy is training a Naive Bayes spam classifier on a small dataset of emails:
- Total Emails: $10$ ($4$ Spam, $6$ Ham)
- The word *"lottery"* appears in $3$ Spam emails, and $0$ Ham emails.
- The word *"lunch"* appears in $0$ Spam emails, and $4$ Ham emails.

### Tasks:
1. Compute the Class Priors $P(\text{Spam})$ and $P(\text{Ham})$.
2. If a new email arrives containing *"lottery"* and *"lunch"*, compute raw Naive Bayes probabilities without smoothing. Why does the classifier crash with a divide-by-zero or zero-probability failure?
3. Apply **Laplace Smoothing** ($\alpha = 1$, vocabulary size $|V| = 2$) and compute the smoothed posterior probability $P(\text{Spam} \mid \text{"lottery", "lunch"})$.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Class Priors
- $P(\text{Spam}) = \frac{4}{10} = \mathbf{0.40}$
- $P(\text{Ham}) = \frac{6}{10} = \mathbf{0.60}$

#### Step 2: The Zero-Probability Disaster
Without smoothing:
- $P(\text{"lunch"} \mid \text{Spam}) = \frac{0}{4} = 0.0$
- $P(\text{"lottery"} \mid \text{Ham}) = \frac{0}{6} = 0.0$

When evaluating $P(\text{Spam} \mid \text{"lottery", "lunch"})$:
$$P(\text{Spam}) \times P(\text{"lottery"} \mid \text{Spam}) \times P(\text{"lunch"} \mid \text{Spam}) = 0.40 \times \frac{3}{4} \times 0.0 = \mathbf{0.0}$$
When evaluating Ham:
$$P(\text{Ham}) \times P(\text{"lottery"} \mid \text{Ham}) \times P(\text{"lunch"} \mid \text{Ham}) = 0.60 \times 0.0 \times \frac{4}{6} = \mathbf{0.0}$$
Both scores are $0.0$, making classification mathematically undefined ($\frac{0}{0}$ crash)! A single unseen word vetoes all other evidence.

#### Step 3: Laplace Smoothing ($\alpha = 1, |V| = 2$)
Formula: $P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\text{total words in } c + \alpha |V|}$.
Here for word presence:
- **For Spam (Denominator = $4 + 1 \times 2 = 6$):**
  $$P(\text{"lottery"} \mid \text{Spam}) = \frac{3 + 1}{4 + 2} = \frac{4}{6} = \frac{2}{3}$$
  $$P(\text{"lunch"} \mid \text{Spam}) = \frac{0 + 1}{4 + 2} = \frac{1}{6}$$
  $$\text{Score}(\text{Spam}) = 0.40 \times \frac{2}{3} \times \frac{1}{6} = 0.40 \times \frac{2}{18} = \frac{0.80}{18} \approx \mathbf{0.0444}$$

- **For Ham (Denominator = $6 + 1 \times 2 = 8$):**
  $$P(\text{"lottery"} \mid \text{Ham}) = \frac{0 + 1}{6 + 2} = \frac{1}{8}$$
  $$P(\text{"lunch"} \mid \text{Ham}) = \frac{4 + 1}{6 + 2} = \frac{5}{8}$$
  $$\text{Score}(\text{Ham}) = 0.60 \times \frac{1}{8} \times \frac{5}{8} = 0.60 \times \frac{5}{64} = \frac{3.0}{64} \approx \mathbf{0.0469}$$

#### Step 4: Normalized Posterior Probability
$$\text{Total} = 0.0444 + 0.0469 = 0.0913$$
$$P(\text{Spam} \mid \text{Email}) = \frac{0.0444}{0.0913} \approx \mathbf{48.6\%}$$
$$P(\text{Ham} \mid \text{Email}) = \frac{0.0469}{0.0913} \approx \mathbf{51.4\%}$$

Laplace smoothing saved the model from crashing and correctly produced a balanced ~50-50 decision!
</details>
