# Topic 03: Exercises & Solutions — Speculative Decoding

---

### Exercise 1: Step-by-Step Numerical Verification of Rejection Sampling

Consider a vocabulary of four tokens $\mathcal{V} = \{w_1, w_2, w_3, w_4\}$.
At a given step, the draft model $M_q$ and target model $M_p$ output the following probability distributions:
- Target distribution $p = [0.10, 0.50, 0.30, 0.10]$
- Draft distribution  $q = [0.40, 0.30, 0.10, 0.20]$

**Questions:**
1. Compute the acceptance probability $\alpha(w_i) = \min\left(1, \frac{p(w_i)}{q(w_i)}\right)$ for each of the four tokens.
2. Calculate the overall acceptance probability $\beta = \sum_{i} q(w_i) \alpha(w_i)$.
3. Compute the Total Variation distance $D_{\text{TV}}(p, q)$ and verify that $\beta = 1 - D_{\text{TV}}(p, q)$.
4. Construct the residual recovery distribution $p'(w_i)$.
5. Calculate the total probability of emitting each token $w_i$ under speculative decoding:
   $$\mathbb{P}_{\text{spec}}(w_i) = q(w_i) \alpha(w_i) + (1 - \beta) p'(w_i)$$
   and verify that $\mathbb{P}_{\text{spec}}(w_i) = p(w_i)$ exactly.

---

#### Solution to Exercise 1

**1. Token Acceptance Probabilities:**
- $w_1$: $\alpha(w_1) = \min(1, 0.10 / 0.40) = 0.25$
- $w_2$: $\alpha(w_2) = \min(1, 0.50 / 0.30) = 1.00$
- $w_3$: $\alpha(w_3) = \min(1, 0.30 / 0.10) = 1.00$
- $w_4$: $\alpha(w_4) = \min(1, 0.10 / 0.20) = 0.50$

**2. Overall Acceptance Rate $\beta$:**
$$\beta = \sum_{i=1}^4 q(w_i) \alpha(w_i) = (0.40)(0.25) + (0.30)(1.00) + (0.10)(1.00) + (0.20)(0.50)$$
$$\beta = 0.10 + 0.30 + 0.10 + 0.10 = \mathbf{0.60}$$

**3. Total Variation Verification:**
$$D_{\text{TV}}(p, q) = \frac{1}{2} \sum_{i=1}^4 |p(w_i) - q(w_i)|$$
$$= \frac{1}{2} \left[ |0.10 - 0.40| + |0.50 - 0.30| + |0.30 - 0.10| + |0.10 - 0.20| \right]$$
$$= \frac{1}{2} [0.30 + 0.20 + 0.20 + 0.10] = \frac{0.80}{2} = \mathbf{0.40}$$
Notice:
$$1 - D_{\text{TV}}(p, q) = 1 - 0.40 = \mathbf{0.60} = \beta \quad \checkmark$$

**4. Residual Recovery Distribution $p'$:**
The unnormalized residual values are $(p(w_i) - q(w_i))^+ = \max(0, p(w_i) - q(w_i))$:
- $w_1$: $\max(0, 0.10 - 0.40) = 0.0$
- $w_2$: $\max(0, 0.50 - 0.30) = 0.20$
- $w_3$: $\max(0, 0.30 - 0.10) = 0.20$
- $w_4$: $\max(0, 0.10 - 0.20) = 0.0$
Normalizing denominator: $Z = 0.0 + 0.20 + 0.20 + 0.0 = 0.40 = 1 - \beta$.
Therefore:
- $p'(w_1) = 0.0$
- $p'(w_2) = \frac{0.20}{0.40} = 0.50$
- $p'(w_3) = \frac{0.20}{0.40} = 0.50$
- $p'(w_4) = 0.0$

**5. Total Emitted Distribution Verification:**
$$\mathbb{P}_{\text{spec}}(w_i) = q(w_i) \alpha(w_i) + (1 - \beta) p'(w_i)$$
- $w_1$: $(0.40)(0.25) + (0.40)(0.0) = 0.10 + 0.0 = \mathbf{0.10} = p(w_1)$
- $w_2$: $(0.30)(1.00) + (0.40)(0.50) = 0.30 + 0.20 = \mathbf{0.50} = p(w_2)$
- $w_3$: $(0.10)(1.00) + (0.40)(0.50) = 0.10 + 0.20 = \mathbf{0.30} = p(w_3)$
- $w_4$: $(0.20)(0.50) + (0.40)(0.0) = 0.10 + 0.0 = \mathbf{0.10} = p(w_4)$

The emitted distribution is identically equal to target model $p = [0.10, 0.50, 0.30, 0.10]$! $\blacksquare$

---

### Exercise 2: Analytical Optimization of Speculative Lookahead Horizon $\gamma^*$

Let $\beta \in (0, 1)$ be the constant token acceptance probability and $r = c_{\text{draft}} / c_{\text{target}}$ be the computational cost ratio of one draft pass to one target pass.
The wall-clock speedup is modeled as:
$$S(\gamma) = \frac{1 - \beta^{\gamma+1}}{(1 - \beta)(1 + \gamma r)}$$

**Questions:**
1. For $\beta = 0.75$ and $r = 0.06$, compute the speedup $S(\gamma)$ for $\gamma \in \{1, 2, 3, 4, 5, 6, 7, 8\}$.
2. Identify the optimal lookahead $\gamma^*$ that maximizes speedup.
3. Explain why increasing $\gamma$ past $\gamma^*$ results in declining speedup.

---

#### Solution to Exercise 2

**1. Table of Speedups for $\beta = 0.75, r = 0.06$:**
Using $S(\gamma) = \frac{1 - 0.75^{\gamma+1}}{0.25 \cdot (1 + 0.06 \gamma)}$:

- $\gamma = 1$: $\mathbb{E}[N] = \frac{1 - 0.75^2}{0.25} = \frac{1 - 0.5625}{0.25} = 1.750$; Cost = $1 + 0.06 = 1.06$; $S(1) = 1.750 / 1.06 = \mathbf{1.651\times}$
- $\gamma = 2$: $\mathbb{E}[N] = \frac{1 - 0.75^3}{0.25} = 2.3125$; Cost = $1 + 0.12 = 1.12$; $S(2) = 2.3125 / 1.12 = \mathbf{2.065\times}$
- $\gamma = 3$: $\mathbb{E}[N] = \frac{1 - 0.75^4}{0.25} = 2.7344$; Cost = $1 + 0.18 = 1.18$; $S(3) = 2.7344 / 1.18 = \mathbf{2.317\times}$
- $\gamma = 4$: $\mathbb{E}[N] = \frac{1 - 0.75^5}{0.25} = 3.0508$; Cost = $1 + 0.24 = 1.24$; $S(4) = 3.0508 / 1.24 = \mathbf{2.460\times}$
- $\gamma = 5$: $\mathbb{E}[N] = \frac{1 - 0.75^6}{0.25} = 3.2881$; Cost = $1 + 0.30 = 1.30$; $S(5) = 3.2881 / 1.30 = \mathbf{2.529\times}$
- $\gamma = 6$: $\mathbb{E}[N] = \frac{1 - 0.75^7}{0.25} = 3.4661$; Cost = $1 + 0.36 = 1.36$; $S(6) = 3.4661 / 1.36 = \mathbf{2.549\times}$
- $\gamma = 7$: $\mathbb{E}[N] = \frac{1 - 0.75^8}{0.25} = 3.5996$; Cost = $1 + 0.42 = 1.42$; $S(7) = 3.5996 / 1.42 = \mathbf{2.535\times}$
- $\gamma = 8$: $\mathbb{E}[N] = \frac{1 - 0.75^9}{0.25} = 3.6997$; Cost = $1 + 0.48 = 1.48$; $S(8) = 3.6997 / 1.48 = \mathbf{2.500\times}$

**2. Optimal Lookahead Horizon:**
The speedup peaks at $\gamma^* = 6$ with an acceleration of $\mathbf{2.549\times}$.

**3. Why Speedup Declines for $\gamma > 6$:**
Because $\beta < 1$, the probability of accepting the $k$-th token decays exponentially as $\beta^k$ (for $k=7$, probability of reaching step 7 is $0.75^7 \approx 0.133$).
However, the draft model computation cost increases linearly ($r \cdot \gamma$). Beyond $\gamma^*$, the marginal cost of computing an additional draft token exceeds the diminishing expected token return $\beta^{\gamma+1}$, reducing net generation efficiency.
