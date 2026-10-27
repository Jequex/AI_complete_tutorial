# Topic 04: Exercises & Solutions — LLM-as-a-Judge & Preference Benchmarks

---

### Exercise 1: Bradley-Terry Win Expectations & Elo Updates

In an LLM preference evaluation arena, two models are paired:
- Model $A$ (e.g., Frontier Model): Rating $R_A = 1650$
- Model $B$ (e.g., Open-Source Base): Rating $R_B = 1500$
- Elo update parameter: $K = 32$

**Questions:**
1. Compute the expected win probability $E_A = \mathbb{P}(A \succ B)$ and $E_B = \mathbb{P}(B \succ A)$.
2. Case 1: Model $A$ wins as expected ($S_A = 1, S_B = 0$). Compute the new ratings $R_A'$ and $R_B'$.
3. Case 2: Model $B$ scores an upset victory ($S_A = 0, S_B = 1$). Compute the new ratings $R_A'$ and $R_B'$.
4. Verify that total rating points are conserved in both cases ($R_A' + R_B' = R_A + R_B$).

---

#### Solution to Exercise 1

**1. Expected Scores:**
Rating difference:
$$R_B - R_A = 1500 - 1650 = -150$$
Expected score of Model $A$:
$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} = \frac{1}{1 + 10^{-150/400}} = \frac{1}{1 + 10^{-0.375}}$$
Note $10^{-0.375} \approx 0.4217$:
$$E_A = \frac{1}{1 + 0.4217} = \frac{1}{1.4217} \approx \mathbf{0.7034 \text{ (or } 70.34\%)}$$
Expected score of Model $B$:
$$E_B = 1 - E_A = 1 - 0.7034 = \mathbf{0.2966 \text{ (or } 29.66\%)}$$

**2. Case 1: Model $A$ Wins ($S_A = 1, S_B = 0$):**
$$R_A' = R_A + K(S_A - E_A) = 1650 + 32(1 - 0.7034) = 1650 + 32(0.2966) = 1650 + 9.49 = \mathbf{1659.49}$$
$$R_B' = R_B + K(S_B - E_B) = 1500 + 32(0 - 0.2966) = 1500 - 9.49 = \mathbf{1490.51}$$

**3. Case 2: Model $B$ Pulls an Upset ($S_A = 0, S_B = 1$):**
$$R_A' = 1650 + 32(0 - 0.7034) = 1650 - 32(0.7034) = 1650 - 22.51 = \mathbf{1627.49}$$
$$R_B' = 1500 + 32(1 - 0.2966) = 1500 + 32(0.7034) = 1500 + 22.51 = \mathbf{1522.51}$$
*Notice*: An upset generates a much larger rating transfer ($\Delta = 22.51$) than an expected victory ($\Delta = 9.49$).

**4. Conservation Check:**
- Initial: $1650 + 1500 = 3150$.
- Case 1: $1659.49 + 1490.51 = \mathbf{3150.00}$.
- Case 2: $1627.49 + 1522.51 = \mathbf{3150.00}$.
Total Elo points are strictly conserved.

---

### Exercise 2: Position Bias and Bidirectional Swapping De-biasing

An automated LLM judge evaluates 100 questions comparing Model $A$ against Model $B$.
Unknown to the benchmark creator, Model $A$ and Model $B$ are identical in capability ($\text{True Win Probability} = 0.50$), but the judge suffers from a severe **Position 1 primacy bias**:
- When evaluating `[Response A, Response B]`, the judge picks Position 1 with probability $0.65$ and Position 2 with probability $0.35$.

**Questions:**
1. If the benchmark creator naively always places Model $A$ in Position 1 and Model $B$ in Position 2 for all 100 questions, what will be the observed win rate of Model $A$? What false conclusion will be published?
2. If the creator employs bidirectional evaluation (evaluating each question twice: once with $(A, B)$ and once with $(B, A)$):
   - What is the probability that Model $A$ wins both orientations?
   - What is the probability that Model $B$ wins both orientations?
   - What proportion of matches will be resolved as ties?
   - What is the debiased effective win rate of Model $A$ versus Model $B$?

---

#### Solution to Exercise 2

**1. Unidirectional Evaluation:**
- Model $A$ is placed in Position 1 for every match.
- Observed win rate of Model $A$: $\mathbf{65.0\%}$.
- **False Conclusion**: The author falsely claims that Model $A$ decisively beats Model $B$ with a 65-35 win rate, even though the models are identical!

**2. Bidirectional Swapping:**
Let $J_1$ be the judge's pick in Pass 1 ($A$ in Pos 1, $B$ in Pos 2).
Let $J_2$ be the judge's pick in Pass 2 ($B$ in Pos 1, $A$ in Pos 2).

Because the judge favors Position 1 with $p = 0.65$:
- In Pass 1: $\mathbb{P}(J_1 = A) = 0.65$, $\mathbb{P}(J_1 = B) = 0.35$.
- In Pass 2: $\mathbb{P}(J_2 = B) = 0.65$, $\mathbb{P}(J_2 = A) = 0.35$.

Assuming independent passes:
- **Model $A$ Wins**: Requires winning both passes ($J_1 = A$ and $J_2 = A$):
  $$\mathbb{P}(\text{Final } A) = \mathbb{P}(J_1 = A) \times \mathbb{P}(J_2 = A) = 0.65 \times 0.35 = \mathbf{0.2275 \text{ (22.75\%)}}$$
- **Model $B$ Wins**: Requires winning both passes ($J_1 = B$ and $J_2 = B$):
  $$\mathbb{P}(\text{Final } B) = \mathbb{P}(J_1 = B) \times \mathbb{P}(J_2 = B) = 0.35 \times 0.65 = \mathbf{0.2275 \text{ (22.75\%)}}$$
- **Tie**: All inconsistent passes where the judge flipped preference with position:
  $$\mathbb{P}(\text{Tie}) = 1 - (0.2275 + 0.2275) = 1 - 0.4550 = \mathbf{0.5450 \text{ (54.50\%)}}$$

- **Debiased Win Ratio**:
  $$\text{Effective Ratio} = \frac{\mathbb{P}(\text{Final } A)}{\mathbb{P}(\text{Final } A) + \mathbb{P}(\text{Final } B)} = \frac{0.2275}{0.2275 + 0.2275} = \mathbf{50.0\%}$$
  Position swapping completely neutralizes the $65\%$ order bias and reveals the exact ground truth ($50/50$ parity).

---

### Exercise 3: Bradley-Terry Maximum Likelihood Estimation

In a mini tournament with 3 models ($M_1, M_2, M_3$), the observed win counts are:
- $M_1$ vs $M_2$: $M_1$ wins 30 times, $M_2$ wins 10 times ($W_{12} = 30, W_{21} = 10$).
- $M_1$ vs $M_3$: $M_1$ wins 40 times, $M_3$ wins 10 times ($W_{13} = 40, W_{31} = 10$).
- $M_2$ vs $M_3$: $M_2$ wins 20 times, $M_3$ wins 20 times ($W_{23} = 20, W_{32} = 20$).

**Questions:**
1. Write the log-likelihood function $\ln \mathcal{L}(\pi_1, \pi_2, \pi_3)$ of the Bradley-Terry model.
2. Fix baseline gauge $\pi_3 = 1.0$. Write the gradient equations $\frac{\partial \ln \mathcal{L}}{\partial \pi_1} = 0$ and $\frac{\partial \ln \mathcal{L}}{\partial \pi_2} = 0$.
3. Compute the relative skill strengths $\pi_1, \pi_2$ and convert them to Elo ratings setting $R_3 = 1000$.

---

#### Solution to Exercise 3

**1. Log-Likelihood:**
$$\ln \mathcal{L} = W_{12}\ln\left(\frac{\pi_1}{\pi_1+\pi_2}\right) + W_{21}\ln\left(\frac{\pi_2}{\pi_1+\pi_2}\right) + W_{13}\ln\left(\frac{\pi_1}{\pi_1+\pi_3}\right) + W_{31}\ln\left(\frac{\pi_3}{\pi_1+\pi_3}\right) + W_{23}\ln\left(\frac{\pi_2}{\pi_2+\pi_3}\right) + W_{32}\ln\left(\frac{\pi_3}{\pi_2+\pi_3}\right)$$

Substituting match counts:
$$\ln \mathcal{L} = 30\ln\pi_1 + 10\ln\pi_2 - 40\ln(\pi_1+\pi_2) + 40\ln\pi_1 + 10\ln\pi_3 - 50\ln(\pi_1+\pi_3) + 20\ln\pi_2 + 20\ln\pi_3 - 40\ln(\pi_2+\pi_3)$$
$$= 70\ln\pi_1 + 30\ln\pi_2 + 30\ln\pi_3 - 40\ln(\pi_1+\pi_2) - 50\ln(\pi_1+\pi_3) - 40\ln(\pi_2+\pi_3)$$

**2. Gradient Conditions (setting $\pi_3 = 1.0$):**
Total wins for Model 1 is $W_1 = 30 + 40 = 70$:
$$\frac{\partial \ln \mathcal{L}}{\partial \pi_1} = \frac{70}{\pi_1} - \frac{40}{\pi_1 + \pi_2} - \frac{50}{\pi_1 + 1} = 0$$

Total wins for Model 2 is $W_2 = 10 + 20 = 30$:
$$\frac{\partial \ln \mathcal{L}}{\partial \pi_2} = \frac{30}{\pi_2} - \frac{40}{\pi_1 + \pi_2} - \frac{40}{\pi_2 + 1} = 0$$

**3. Solving for $\pi_1, \pi_2$ and Elo Ratings:**
Notice that $M_2$ and $M_3$ tied $20\text{--}20$, so by symmetry $\pi_2 \approx \pi_3 = 1.0$.
Check with $\pi_2 = 1.0$:
$$\frac{\partial \ln \mathcal{L}}{\partial \pi_2} = \frac{30}{1.0} - \frac{40}{\pi_1 + 1.0} - \frac{40}{1.0 + 1.0} = 30 - \frac{40}{\pi_1 + 1} - 20 = 10 - \frac{40}{\pi_1 + 1} = 0 \implies \pi_1 + 1 = 4 \implies \mathbf{\pi_1 = 3.0}$$
Check with Model 1 equation:
$$\frac{70}{3.0} - \frac{40}{3.0 + 1.0} - \frac{50}{3.0 + 1.0} = 23.33 - 10 - 12.5 = 23.33 - 22.5 \approx 0 \quad (\text{Exact balance is } \pi_1 \approx 3.0)$$

Thus:
- $\pi_1 \approx 3.00, \quad \pi_2 = 1.00, \quad \pi_3 = 1.00$

Converting to Elo scale ($R_i = R_3 + 400 \log_{10}(\pi_i / \pi_3)$):
- $R_3 = \mathbf{1000}$
- $R_2 = 1000 + 400 \log_{10}(1.0) = \mathbf{1000}$
- $R_1 = 1000 + 400 \log_{10}(3.0) = 1000 + 400(0.4771) = 1000 + 190.85 = \mathbf{1191}$

Model 1 sits $\approx 191$ Elo points ahead of Models 2 and 3.
