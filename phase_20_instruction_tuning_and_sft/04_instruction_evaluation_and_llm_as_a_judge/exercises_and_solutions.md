# Topic 20.4: Exercises & Solutions — Instruction Evaluation & LLM-as-a-Judge

---

### Problem 1: Bidirectional Consensus Win-Rate Calculation (Easy)

An evaluation team tests two language models, Model Alpha ($M_\alpha$) and Model Beta ($M_\beta$), on 100 benchmark prompts using an automated LLM judge.
Due to position bias, the judge has a strong preference for whichever response is presented first (in Position A).

The results of the two evaluation passes are:
- **Pass 1 ($M_\alpha$ in Pos A, $M_\beta$ in Pos B):**
  - Judge votes $A$ ($M_\alpha$ wins): 70 matches
  - Judge votes $B$ ($M_\beta$ wins): 20 matches
  - Judge votes Tie: 10 matches
- **Pass 2 ($M_\beta$ in Pos A, $M_\alpha$ in Pos B):**
  - Judge votes $A$ ($M_\beta$ wins): 60 matches
  - Judge votes $B$ ($M_\alpha$ wins): 25 matches
  - Judge votes Tie: 15 matches

Assume the match-level joint distribution between Pass 1 and Pass 2 across the 100 prompts is:
- Prompt set where $M_\alpha$ won both passes: 20 prompts
- Prompt set where $M_\beta$ won both passes: 15 prompts
- All remaining 65 prompts had position contradictions (slot A won both times) or individual ties.

1. Compute the naive (unidirectional) win rate for $M_\alpha$ from Pass 1 alone (treating ties as $0.5$ win).
2. Compute the debiased bidirectional win rate for $M_\alpha$ under consensus scoring.
3. Compare the two results and state how many percentage points of false advantage were eliminated.

<details>
<summary>View Complete Solution</summary>

**Step 1: Naive Win Rate (Pass 1):**
In Pass 1, $M_\alpha$ had 70 wins and 10 ties out of 100 matches:
$$\text{WinRate}_{\text{naive}}(M_\alpha) = \frac{\text{Wins} + 0.5 \times \text{Ties}}{N} = \frac{70 + 0.5 \times 10}{100} = \frac{75}{100} = 75.0\%$$
A naive analysis would claim $M_\alpha$ overwhelmingly crushed $M_\beta$ with a $75\%$ win rate!

**Step 2: Debiased Bidirectional Win Rate:**
Under bidirectional consensus:
- $M_\alpha$ is declared the winner on a prompt if and only if $M_\alpha$ won in both orientations:
  $$\text{Wins}_{\text{consensus}}(M_\alpha) = 20$$
- $M_\beta$ is declared the winner on a prompt if and only if $M_\beta$ won in both orientations:
  $$\text{Wins}_{\text{consensus}}(M_\beta) = 15$$
- The remaining $100 - (20 + 15) = 65$ prompts are classified as ties/contradictions:
  $$\text{Ties}_{\text{consensus}} = 65$$

The debiased win rate is:
$$\text{WinRate}_{\text{debiased}}(M_\alpha) = \frac{20 + 0.5 \times 65}{100} = \frac{20 + 32.5}{100} = \frac{52.5}{100} = 52.5\%$$

**Step 3: Comparison:**
- Naive win rate: $75.0\%$
- Debiased win rate: $52.5\%$
- False advantage eliminated: $75.0\% - 52.5\% = 22.5$ percentage points!
The apparent blowout was an artifact of position bias. In reality, the two models are closely matched ($52.5\%$ vs $47.5\%$).
</details>

---

### Problem 2: Bradley-Terry Maximum Likelihood Gradient Derivation (Medium)

In the Bradley-Terry preference model, the probability that model $i$ is preferred over model $j$ is:
$$P(i \succ j) = \sigma(r_i - r_j) = \frac{1}{1 + e^{-(r_i - r_j)}}$$
where $\mathbf{r} = (r_1, \dots, r_K)$ are latent skill parameters.

Suppose $N_{ij}$ matches were played between model $i$ and model $j$, and model $i$ won $W_{ij}$ times (assume no ties for simplicity, so $W_{ji} = N_{ij} - W_{ij}$).
The log-likelihood of the tournament outcome is:
$$\ell(\mathbf{r}) = \sum_{i < j} \left[ W_{ij} \log P(i \succ j) + W_{ji} \log P(j \succ i) \right]$$

1. Derive the analytical gradient $\frac{\partial \ell}{\partial r_i}$.
2. Interpret the gradient expression in terms of "observed wins" vs "expected wins".
3. Show that at the optimum $\mathbf{r}^*$, total observed wins for each model equals its total expected wins.

<details>
<summary>View Complete Solution</summary>

**Step 1: Gradient Derivation:**
Let $p_{ij} = P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}}$.
Notice that $p_{ji} = 1 - p_{ij}$.
Recall the derivative of the logistic sigmoid $\sigma(u)$ with respect to $u$:
$$\frac{d}{du} \log \sigma(u) = 1 - \sigma(u)$$
$$\frac{d}{du} \log(1 - \sigma(u)) = -\sigma(u)$$

For a fixed pair $(i, j)$ with $i < j$:
$$\ell_{ij} = W_{ij} \log \sigma(r_i - r_j) + W_{ji} \log(1 - \sigma(r_i - r_j))$$
Differentiating with respect to $r_i$:
$$\frac{\partial \ell_{ij}}{\partial r_i} = W_{ij} (1 - p_{ij}) - W_{ji} p_{ij} = W_{ij} - (W_{ij} + W_{ji}) p_{ij} = W_{ij} - N_{ij} p_{ij}$$

Now summing across all opponents $j \ne i$:
$$\boxed{\frac{\partial \ell}{\partial r_i} = \sum_{j \ne i} \left( W_{ij} - N_{ij} P(i \succ j) \right)}$$

**Step 2: Interpretation:**
- $W_{ij}$ is the **observed number of wins** model $i$ scored against model $j$.
- $N_{ij} P(i \succ j)$ is the **expected number of wins** model $i$ should score against model $j$ according to the current skill estimates $\mathbf{r}$.
- If model $i$ won more matches than expected ($W_{ij} > N_{ij} p_{ij}$), the gradient is positive, driving $r_i$ higher.

**Step 3: Optimality Condition:**
Setting $\nabla_{\mathbf{r}} \ell = \mathbf{0}$:
$$\forall i, \quad \sum_{j \ne i} W_{ij} = \sum_{j \ne i} N_{ij} P^*(i \succ j)$$
$$\text{Total Observed Wins of Model } i = \text{Total Expected Wins of Model } i$$
The maximum likelihood solution aligns the theoretical probabilities with empirical tournament win tallies.
</details>

---

### Problem 3: Verbosity Bias & Length Penalty Function Design (Hard)

A benchmark designer notes that on open-ended creative writing prompts, an LLM judge awards win rates according to perceived utility:
$$U(y) = Q(y) + \gamma \cdot \ln(L(y))$$
where $Q(y)$ is the true linguistic quality, $L(y)$ is response word count, and $\gamma = 0.5$ is the verbosity bias coefficient.

Consider two models:
- **Model 1 (Concise):** $Q_1 = 3.0$, produces outputs of length $L_1 = 150$ words.
- **Model 2 (Word Salad Flooder):** $Q_2 = 2.4$, produces padded outputs of length $L_2 = 600$ words.

1. Compute the perceived utilities $U_1$ and $U_2$ by the biased judge (use $\ln(150) \approx 5.01$ and $\ln(600) \approx 6.40$).
2. Assuming pairwise probability is modeled by the standard logistic function $P(\text{Judge prefers } 2 \succ 1) = \frac{1}{1 + e^{-(U_2 - U_1)}}$, compute the unadjusted win rate of the inferior, wordy model.
3. Formulate an explicit length correction penalty $\Pi(L_1, L_2)$ that restores the win probability to reflect true quality $Q_1$ and $Q_2$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Utility Calculations:**
- For Model 1:
  $$U_1 = Q_1 + \gamma \ln(L_1) = 3.0 + 0.5 \times 5.01 = 3.0 + 2.505 = 5.505$$
- For Model 2:
  $$U_2 = Q_2 + \gamma \ln(L_2) = 2.4 + 0.5 \times 6.40 = 2.4 + 3.200 = 5.600$$
Even though Model 2 has significantly worse quality ($2.4 < 3.0$), its perceived utility is higher ($5.600 > 5.505$)!

**Step 2: Unadjusted Win Probability:**
The utility difference is:
$$\Delta U = U_2 - U_1 = 5.600 - 5.505 = +0.095$$
The probability the judge chooses the inferior Model 2 is:
$$P(2 \succ 1) = \frac{1}{1 + e^{-0.095}} = \frac{1}{1 + 0.9094} = \frac{1}{1.9094} \approx 52.37\%$$
Model 2 wins the benchmark despite being inferior.

**Step 3: Length Penalty Design:**
To recover true qualities $Q_1$ and $Q_2$, the scoring engine must subtract the length term:
$$\Delta U_{\text{true}} = Q_2 - Q_1 = 2.4 - 3.0 = -0.60$$
We define the length adjustment penalty:
$$\Pi(L_1, L_2) = \gamma \left( \ln(L_2) - \ln(L_1) \right) = \gamma \ln\left(\frac{L_2}{L_1}\right)$$
The corrected utility difference is:
$$\Delta U_{\text{corrected}} = (U_2 - U_1) - \Pi(L_1, L_2) = 0.095 - 0.5 \times (6.40 - 5.01) = 0.095 - 0.695 = -0.60$$
Under corrected utility:
$$P_{\text{corrected}}(2 \succ 1) = \frac{1}{1 + e^{-(-0.60)}} = \frac{1}{1 + e^{0.60}} = \frac{1}{1 + 1.822} \approx 35.43\%$$
Model 1 (the high-quality concise model) now correctly wins $100\% - 35.43\% = 64.57\%$ of matches.
</details>
