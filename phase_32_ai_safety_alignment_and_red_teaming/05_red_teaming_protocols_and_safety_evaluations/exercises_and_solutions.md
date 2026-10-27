# Topic 05: Exercises & Solutions — Red Teaming Protocols & Safety Evaluations

---

### Exercise 1: Computing ASR, FRR, and Harmonic Safety Score

An AI safety team benchmarks three candidate safety guardrail pipelines ($M_1, M_2, M_3$) against an adversarial evaluation dataset containing $N_{\text{adv}} = 1000$ adversarial attack attempts, and a benign evaluation dataset containing $N_{\text{benign}} = 2000$ standard user requests.

The empirical test results are recorded below:
- **Guardrail $M_1$ (Permissive)**: Blocks 600 attacks; incorrectly refuses 20 benign requests.
- **Guardrail $M_2$ (Balanced)**: Blocks 920 attacks; incorrectly refuses 100 benign requests.
- **Guardrail $M_3$ (Over-Defensive)**: Blocks 990 attacks; incorrectly refuses 900 benign requests.

**Questions:**
1. Compute the Attack Success Rate (ASR) and False Refusal Rate (FRR) for all three systems.
2. Calculate the Harmonic Safety Score $H_{\text{safety}} = 2 \frac{(1 - \text{ASR})(1 - \text{FRR})}{(1 - \text{ASR}) + (1 - \text{FRR})}$ for each pipeline.
3. Which system achieves the optimal Pareto balance according to the harmonic safety objective?

---

#### Solution to Exercise 1

**1. Compute ASR and FRR:**
- **System $M_1$**:
  - Successful attacks = $1000 - 600 = 400$.
  - $\text{ASR}_1 = \frac{400}{1000} = \mathbf{0.4000}$ ($40.0\%$).
  - $\text{FRR}_1 = \frac{20}{2000} = \mathbf{0.0100}$ ($1.0\%$).
- **System $M_2$**:
  - Successful attacks = $1000 - 920 = 80$.
  - $\text{ASR}_2 = \frac{80}{1000} = \mathbf{0.0800}$ ($8.0\%$).
  - $\text{FRR}_2 = \frac{100}{2000} = \mathbf{0.0500}$ ($5.0\%$).
- **System $M_3$**:
  - Successful attacks = $1000 - 990 = 10$.
  - $\text{ASR}_3 = \frac{10}{1000} = \mathbf{0.0100}$ ($1.0\%$).
  - $\text{FRR}_3 = \frac{900}{2000} = \mathbf{0.4500}$ ($45.0\%$).

**2. Calculate Harmonic Safety Score $H_{\text{safety}}$:**
- For $M_1$:
  - Robustness $R_1 = 1 - 0.40 = 0.60$; Helpfulness $H_1 = 1 - 0.01 = 0.99$.
  - $H_{\text{safety}, 1} = 2 \cdot \frac{0.60 \times 0.99}{0.60 + 0.99} = \frac{1.188}{1.59} \approx \mathbf{0.7472}$.
- For $M_2$:
  - Robustness $R_2 = 1 - 0.08 = 0.92$; Helpfulness $H_2 = 1 - 0.05 = 0.95$.
  - $H_{\text{safety}, 2} = 2 \cdot \frac{0.92 \times 0.95}{0.92 + 0.95} = \frac{1.748}{1.87} \approx \mathbf{0.9348}$.
- For $M_3$:
  - Robustness $R_3 = 1 - 0.01 = 0.99$; Helpfulness $H_3 = 1 - 0.45 = 0.55$.
  - $H_{\text{safety}, 3} = 2 \cdot \frac{0.99 \times 0.55}{0.99 + 0.55} = \frac{1.089}{1.54} \approx \mathbf{0.7071}$.

**3. Conclusion:**
Pipeline **$M_2$** achieves the highest Harmonic Safety Score ($0.9348$), effectively suppressing $92\%$ of adversarial attacks while preserving a $95\%$ acceptance rate on benign user queries. System $M_3$, despite blocking $99\%$ of attacks, cripples real-world utility with a catastrophic $45\%$ false refusal rate.

