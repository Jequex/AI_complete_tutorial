# Topic 4.1: Probability Fundamentals, Axioms & Bayes' Theorem

Welcome to **Topic 4.1: Probability Fundamentals, Axioms & Bayes' Theorem**. In this lecture, we build the mathematical machinery of uncertainty from first principles. In artificial intelligence, models almost never operate with 100% certainty. Instead, an AI observes noisy clues from the environment, computes probabilities, and updates its beliefs using **Bayes' Theorem**.

---

## 1. Symbol Decoder Table

Before diving into the equations, review the standard mathematical notation:

| Symbol | Mathematical Name | Plain-English Meaning | Child-Friendly Analogy |
| :---: | :--- | :--- | :--- |
| $\Omega$ | Sample Space | The set of **all possible** outcomes | A giant jar containing every marble that could ever be drawn |
| $\omega$ | Elementary Outcome | A single specific result | One single marble pulled from the jar |
| $E, A, B$ | Event | A subset of outcomes ($E \subseteq \Omega$) | Drawing any blue marble (or any marble with stripes) |
| $P(A)$ | Probability of $A$ | A number between $0$ and $1$ measuring chance | The percentage of the jar filled with blue marbles |
| $\cap$ | Intersection ("AND") | Both event $A$ **and** event $B$ happen | A marble that is both blue **and** has stripes |
| $\cup$ | Union ("OR") | Either event $A$ **or** event $B$ (or both) happens | A marble that is blue **or** has stripes |
| $A^c$ | Complement ("NOT") | Event $A$ does **not** happen | Any marble that is **not** blue |
| $P(A \mid B)$ | Conditional Probability | Probability of $A$ **given that** $B$ occurred | Looking only at the striped marbles, what fraction are blue? |
| $\theta$ | Parameter / Hypothesis | What the AI is trying to infer (class, weight) | The hidden identity of the mystery suspect |
| $D$ | Data / Evidence | Observed sensory input (pixels, words, symptoms)| The muddy footprints found at the crime scene |

---

## 2. Sample Spaces, Events, and Kolmogorov's Axioms

### The 12-Year-Old Mental Model: The Game Spinner
Imagine a giant circular game spinner divided into colored sections.
- The **Sample Space ($\Omega$)** is the entire circle (all possible angles where the needle can stop).
- An **Event ($A$)** is a specific slice of the circle (e.g., landing on "Red").
- The **Probability ($P(A)$)** is the fraction of the circle covered by that slice.

```
       ┌───────────────────────────────┐
       │     Sample Space: Ω (Area = 1)│
       │                               │
       │     ┌─────────┐   ┌─────────┐ │
       │     │ Event A │   │ Event B │ │
       │     │ (Blue)  │   │(Striped)│ │
       │     └────┬────┘   └────┬────┘ │
       │          └──────┬──────┘      │
       │                 ▼             │
       │            Intersection       │
       │              (A ∩ B)          │
       └───────────────────────────────┘
```

In 1933, the mathematician Andrey Kolmogorov established that all probability theory rests upon three simple rules:

### Kolmogorov's Three Axioms
1. **Axiom 1 (Non-negativity):** For every event $A$, the probability is a non-negative real number:
   $$P(A) \ge 0$$
   *(You cannot have a -20% chance of rain!)*

2. **Axiom 2 (Unitarity / Total Probability):** The probability of the entire sample space is exactly $1$:
   $$P(\Omega) = 1.0$$
   *(Something in the universe is 100% guaranteed to happen!)*

3. **Axiom 3 (Countable Additivity):** If events $A_1, A_2, A_3, \dots$ are mutually disjoint (meaning no two can happen simultaneously, $A_i \cap A_j = \emptyset$), then the probability of any of them occurring is the direct sum:
   $$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

### Crucial Derived Properties
From these three axioms, all standard probability formulas naturally emerge:
- **Complement Rule:** $P(A^c) = 1 - P(A)$
- **Impossible Event:** $P(\emptyset) = 0$
- **Inclusion-Exclusion (Union of overlapping events):**
  $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
  *(We subtract $P(A \cap B)$ so we don't count the overlapping region twice!)*

---

## 3. Conditional Probability & Independence

### What Happens When You Learn New Information?
Before you look out the window, you might think the chance of carrying an umbrella is $P(\text{Umbrella}) = 0.20$.
Once you look out the window and observe dark stormy clouds ($B$), your sample space **shrinks**. You throw away all sunny scenarios and only examine stormy worlds:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad (\text{provided } P(B) > 0)$$

### The Multiplication Rule
Rearranging the conditional probability definition gives the joint probability of two events occurring:
$$P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$$

### Statistical Independence
Two events $A$ and $B$ are **statistically independent** if knowing that $B$ happened gives you **zero** new information about $A$:
$$P(A \mid B) = P(A) \iff P(A \cap B) = P(A) \times P(B)$$

> [!WARNING] Common Beginner Mistake: Mutually Exclusive vs. Independent
> - **Mutually Exclusive** means the events **cannot happen together** ($A \cap B = \emptyset \implies P(A \cap B) = 0$). For example: rolling a 2 and rolling a 5 on a single die.
> - **Independent** means one happening **does not affect the probability** of the other ($P(A \cap B) = P(A)P(B) > 0$).
> If two events with non-zero probabilities are mutually exclusive, they are *strongly dependent*, because knowing $A$ occurred guarantees $B$ did *not* occur!

---

## 4. The Law of Total Probability

Suppose the entire world $\Omega$ is partitioned into $k$ non-overlapping slices $B_1, B_2, \dots, B_k$ (like slices of a pizza). Any event $A$ overlaps with these slices:

$$P(A) = \sum_{i=1}^k P(A \cap B_i) = \sum_{i=1}^k P(A \mid B_i) P(B_i)$$

```
┌────────────────────────────────────────┐
│  Slice B₁   │  Slice B₂   │  Slice B₃  │
│             │   ┌─────┐   │            │
│             │   │  A  │   │            │
│             │   └─────┘   │            │
└────────────────────────────────────────┘
```

---

## 5. Bayes' Theorem: The Engine of Belief Updating

Thomas Bayes formulated the most celebrated theorem in machine learning. It tells us how an AI system updates its beliefs about a hypothesis ($\theta$) after observing sensory evidence ($D$).

### The Mathematical Formula
Equating $P(\theta \cap D) = P(D \cap \theta)$:
$$P(D \mid \theta) P(\theta) = P(\theta \mid D) P(D)$$

Dividing both sides by $P(D)$ gives **Bayes' Rule**:

$$\underbrace{P(\theta \mid D)}_{\text{Posterior}} = \frac{\overbrace{P(D \mid \theta)}^{\text{Likelihood}} \times \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(D)}_{\text{Evidence / Marginal Likelihood}}}$$

Using the Law of Total Probability to expand the denominator:
$$P(\theta_j \mid D) = \frac{P(D \mid \theta_j) P(\theta_j)}{\sum_{i=1}^k P(D \mid \theta_i) P(\theta_i)}$$

### Deconstructing the Four Pillars of Bayes' Rule:
1. **Prior Probability $P(\theta)$:** What you believed about hypothesis $\theta$ *before* observing any new evidence.
2. **Likelihood $P(D \mid \theta)$:** If hypothesis $\theta$ were true, what is the probability that we would observe this specific evidence $D$?
3. **Evidence $P(D)$:** The total probability of observing data $D$ across all possible hypotheses (acts as a normalizing constant so probabilities sum to 1).
4. **Posterior Probability $P(\theta \mid D)$:** Your updated, refined belief in hypothesis $\theta$ *after* taking the evidence into account.

---

## 6. The False Positive Paradox (Why 99% Accurate Tests Fool People)

To see why AI models must understand Bayesian probabilities, consider the famous **Medical Test Paradox**.

### The Scenario
A rare disease affects $1$ out of every $10{,}000$ people in the population:
- **Prior probability of disease:** $P(\text{Disease}) = 0.0001$
- **Prior probability of healthy:** $P(\text{Healthy}) = 0.9999$

A laboratory invents a rapid diagnostic test with:
- **True Positive Rate (Sensitivity):** $P(\text{Positive} \mid \text{Disease}) = 0.99$ (99% accurate if you are sick)
- **False Positive Rate:** $P(\text{Positive} \mid \text{Healthy}) = 0.01$ (1% false alarm rate on healthy people)

You take the test, and the result comes back **Positive (+)**!
**What is the probability that you actually have the disease?**
*Most people (and even many doctors) answer "99%!". The true answer is about 1%!*

### Step-by-Step Bayesian Calculation:
1. **Numerator (Likelihood $\times$ Prior):**
   $$P(\text{Positive} \mid \text{Disease}) \times P(\text{Disease}) = 0.99 \times 0.0001 = 0.000099$$

2. **Denominator (Total Evidence $P(\text{Positive})$):**
   Using the Law of Total Probability:
   $$P(\text{Positive}) = P(\text{Pos} \mid \text{Disease}) P(\text{Disease}) + P(\text{Pos} \mid \text{Healthy}) P(\text{Healthy})$$
   $$P(\text{Positive}) = (0.99 \times 0.0001) + (0.01 \times 0.9999)$$
   $$P(\text{Positive}) = 0.000099 + 0.009999 = 0.010098$$

3. **Posterior Probability $P(\text{Disease} \mid \text{Positive})$:**
   $$P(\text{Disease} \mid \text{Positive}) = \frac{0.000099}{0.010098} \approx 0.009804 \approx \mathbf{0.98\%}$$

### Why Does This Happen?
Out of $1{,}000{,}000$ people:
- $100$ people actually have the disease. The test detects $\approx 99$ of them.
- $999{,}900$ people are completely healthy. The 1% false positive rate flags $\approx 9{,}999$ healthy people!
- Total positive results: $99 + 9{,}999 = 10{,}098$.
- Of those positive results, only $99$ are genuinely sick: $\frac{99}{10{,}098} \approx 0.98\%$!

*Takeaway for AI:* When detecting rare events (fraud detection, cyber attacks, rare defects), even a 99% accurate model will produce far more false alarms than true positives unless the **prior probability** is explicitly factored in via Bayes' Rule!

