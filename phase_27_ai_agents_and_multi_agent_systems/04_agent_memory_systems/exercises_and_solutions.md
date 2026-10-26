# Topic 04: Exercises & Solutions — Agent Memory Systems

---

### Exercise 1: Multi-Factor Memory Scoring & Ranking

An agent memory system uses the composite scoring objective:
$$S(m, q, t) = w_r \cdot R(m, t) + w_i \cdot I(m) + w_s \cdot \text{Sim}(m, q)$$
with weights:
$$w_r = 0.25, \quad w_i = 0.35, \quad w_s = 0.40$$
and exponential recency decay rate $\lambda = 0.05$:
$$R(m, t) = \exp(-\lambda (t - t_m))$$

At current time $t = 50$, the agent evaluates three candidate memories against user query $q$:

| Memory ID | Content | Created / Accessed $t_m$ | Importance $I(m)$ | Cosine Sim $\text{Sim}(m, q)$ |
| :--- | :--- | :--- | :--- | :--- |
| **$m_1$** | *"User's preferred programming language is Rust"* | $t_1 = 10$ | $0.90$ | $0.85$ |
| **$m_2$** | *"Current compiler error: unresolved import `tokio`"* | $t_2 = 48$ | $0.60$ | $0.40$ |
| **$m_3$** | *"System log: Garbage collection freed 42MB"* | $t_3 = 49$ | $0.10$ | $0.05$ |

**Question:**
1. Compute the recency score $R(m_k, 50)$ for each memory.
2. Compute the total composite score $S(m_k, q, 50)$ for each memory.
3. Which memory is ranked #1 for retrieval into working context?

---

#### Solution to Exercise 1

**1. Recency Scores $R(m_k, 50) = \exp(-0.05 \cdot (50 - t_k))$:**
- For $m_1$: $\Delta t_1 = 50 - 10 = 40$.
  $$R(m_1, 50) = \exp(-0.05 \times 40) = \exp(-2.0) \approx \mathbf{0.1353}$$
- For $m_2$: $\Delta t_2 = 50 - 48 = 2$.
  $$R(m_2, 50) = \exp(-0.05 \times 2) = \exp(-0.10) \approx \mathbf{0.9048}$$
- For $m_3$: $\Delta t_3 = 50 - 49 = 1$.
  $$R(m_3, 50) = \exp(-0.05 \times 1) = \exp(-0.05) \approx \mathbf{0.9512}$$

**2. Composite Scores $S = 0.25 R + 0.35 I + 0.40 \text{Sim}$:**
- **For $m_1$:**
  $$S(m_1) = 0.25(0.1353) + 0.35(0.90) + 0.40(0.85) = 0.0338 + 0.3150 + 0.3400 = \mathbf{0.6888}$$
- **For $m_2$:**
  $$S(m_2) = 0.25(0.9048) + 0.35(0.60) + 0.40(0.40) = 0.2262 + 0.2100 + 0.1600 = \mathbf{0.5962}$$
- **For $m_3$:**
  $$S(m_3) = 0.25(0.9512) + 0.35(0.10) + 0.40(0.05) = 0.2378 + 0.0350 + 0.0200 = \mathbf{0.2928}$$

**3. Ranking:**
$$S(m_1) = 0.6888 > S(m_2) = 0.5962 > S(m_3) = 0.2928$$
**Memory $m_1$ is ranked #1.**
Despite being created 40 steps ago ($\text{Recency} \approx 0.135$), its high intrinsic importance ($0.90$) and strong semantic relevance ($0.85$) overpower the fresher, but less relevant memories.

---

### Exercise 2: Deriving the Memory Half-Life under Exponential Decay

Let the recency decay function be $R(m, t) = \exp(-\lambda \Delta t)$ where $\Delta t = t - t_m \ge 0$.

**Question:**
1. Derive the formula for the memory half-life $T_{1/2}$ (the number of elapsed steps required for recency to drop to exactly $0.5$).
2. Suppose an engineer wants memories to experience a half-life of exactly $T_{1/2} = 25$ conversation turns. Calculate the required decay parameter $\lambda$ to 4 decimal places.

---

#### Solution to Exercise 2

**1. Derivation of Half-Life:**
Set $R(m, t) = 0.5$:
$$\exp(-\lambda T_{1/2}) = 0.5 = \frac{1}{2}$$
Take the natural logarithm of both sides:
$$-\lambda T_{1/2} = \ln\left(\frac{1}{2}\right) = -\ln(2)$$
Solving for $T_{1/2}$:
$$\boxed{T_{1/2} = \frac{\ln(2)}{\lambda} \approx \frac{0.69315}{\lambda}}$$

**2. Calculating $\lambda$ for $T_{1/2} = 25$:**
$$\lambda = \frac{\ln(2)}{T_{1/2}} = \frac{0.69315}{25} \approx \mathbf{0.0277}$$
Setting $\lambda = 0.0277$ guarantees that a memory retains $50\%$ recency after 25 turns, $25\%$ after 50 turns, and $12.5\%$ after 75 turns.
