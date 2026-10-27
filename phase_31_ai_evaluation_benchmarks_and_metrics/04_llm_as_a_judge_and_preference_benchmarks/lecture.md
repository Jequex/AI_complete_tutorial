# Topic 04: LLM-as-a-Judge & Preference Benchmarks

## 1. Introduction & 12-Year-Old Intuition

Imagine you organize a school science fair with 100 students submitting science projects. You cannot hire 50 university professors to grade every project—it is too slow and too expensive. 

Instead, you ask the high school science teacher (or a smart senior student) to act as a **judge**.
You have two choices for how the judge grades:
1. **Single-Project Scoring**: Give the project a grade from 1 to 10.
   - *The Problem*: When the judge grades Project #1 at 9:00 AM, they are in a great mood and give it an 8. By 4:00 PM, after looking at 40 posters, they are tired and give an identical project a 5. Numerical scores drift wildly.
2. **Pairwise Comparison (Head-to-Head Duel)**: Put two projects side-by-side on a table and ask: *"Which of these two projects is better?"*
   - Humans and AI judges are vastly better at pairwise comparison than assigning absolute numbers!

However, your judge has subconscious biases:
- **Position Bias (Order Effect)**: The judge picks the project placed on the left side $65\%$ of the time, simply because they read left-to-right!
- **Verbosity Bias**: If Student A submits a concise, accurate 1-page paper, while Student B submits a 20-page poster filled with decorative graphs and bold words, the judge falls for the fluff and picks Student B.
- **Self-Enhancement Bias**: If the judge wrote one of the starter templates, they favor any student who copied their style.

To build trustworthy AI benchmarks like **LMSYS Chatbot Arena** or **MT-Bench**, we must:
1. Use **Position-Swapping**: We test Project A on the left and Project B on the right. Then we swap them and test Project B on the left and Project A on the right. If the judge changes their mind, we call it a tie!
2. Use the **Bradley-Terry & Elo Rating System**: Just like competitive chess or multiplayer video games, we calculate dynamic skill ratings from thousands of pairwise matches.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain | Description & Engineering Significance |
| :--- | :--- | :--- | :--- |
| $\mathcal{M}_A, \mathcal{M}_B$ | Competitor Models | Model Entities | The two LLMs competing head-to-head on a given user prompt. |
| $\mathcal{J}$ | Judge LLM | LLM Entity | The evaluator model (e.g. GPT-4o, Claude 3.5 Sonnet) providing judgments. |
| $S_A$ | Actual Match Outcome | $\{1.0, 0.5, 0.0\}$ | $1.0$ if Model A wins, $0.5$ if tie, $0.0$ if Model A loses. |
| $R_A, R_B$ | Elo Skill Ratings | $\mathbb{R}$ (typically $1000\text{--}2000$) | Numerical latent skill of models on the Elo scale. |
| $E_A$ | Expected Score | $[0, 1]$ | $\mathbb{P}(\mathcal{M}_A \succ \mathcal{M}_B)$ predicted by the Bradley-Terry model. |
| $K$ | Elo $K$-Factor | Positive Real ($K \in [8, 32]$) | Step size scaling rate updates after each evaluated match. |
| $\pi_A, \pi_B$ | Bradley-Terry Latent Strength | $\mathbb{R}^+$ | Positive latent skill where $R_A = 400 \log_{10} \pi_A + \text{const}$. |
| $\mathbf{W}$ | Win-Rate Matrix | $\mathbb{R}^{M \times M}$ | Empirical matrix where $W_{ij}$ is the win-rate of Model $i$ against Model $j$. |
| $\Delta_{\text{pos}}$ | Position Bias Delta | $\mathbb{R}$ | Discrepancy in win-rate when placed in Position 1 vs Position 2. |
| $\lambda_{\text{len}}$ | Verbosity Correlation | $[-1, 1]$ | Pearson correlation between token length difference and judge preference. |

---

## 3. Evaluation Paradigms: Single-Answer vs. Pairwise

```
                       LLM EVALUATION PARADIGMS
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
   SINGLE-ANSWER GRADING                               PAIRWISE PREFERENCE
   • Evaluates 1 model response in isolation           • Evaluates Model A vs Model B side-by-side
   • Score: [1 to 5] or [1 to 10]                      • Output: A > B, B > A, or Tie (A = B)
   • Prompt: "Grade this response on clarity..."       • Matches human preference ranking
   • Severe scale compression (clumping at 4-5)        • Transitive consistency via Elo / Bradley-Terry
   • Susceptible to prompt drift and calibration shifts • Higher discriminative sensitivity
```

---

## 4. The Three Systematic Biases of LLM Judges

Empirical studies on MT-Bench and Chatbot Arena (Zheng et al., NeurIPS 2023) identified three pervasive systematic biases in LLM judges:

### 4.1 Position Bias (Order Effect)
LLM judges exhibit a strong primacy (or recency) effect. Given two identical responses:
$$\mathbb{P}(\text{Judge chooses Position 1}) \ne 0.50$$
State-of-the-art models frequently exhibit up to a $60\%\text{--}70\%$ preference for whichever model is placed in Response 1 (or Response 2 depending on prompt formatting).

### 4.2 Verbosity Bias
LLM judges exhibit a heavy heuristic preference for length:
$$\mathbb{P}(\text{Win} \mid \text{Length}(A) > \text{Length}(B)) \gg 0.50$$
Even when a concise answer is completely accurate, an answer packed with decorative bullet points, greetings, and repetitive elaborations is awarded a higher grade.

### 4.3 Self-Enhancement Bias
When an LLM (e.g. Model Family $X$) acts as judge over anonymized outputs:
$$\mathbb{P}(\mathcal{J}_X \text{ prefers } \mathcal{M}_X) > \mathbb{P}(\mathcal{J}_Y \text{ prefers } \mathcal{M}_X)$$
Judges recognize the token stylistic fingerprint, sentence cadence, and formatting idiosyncrasies of their own weights.

---

## 5. Position-Swapping De-Biasing

To eliminate first-order position bias, every match must be evaluated **bidirectionally**:

1. **Pass 1**: Present prompt with `[Response A = Model 1, Response B = Model 2]`. 
   Let Judge decision be $J_1 \in \{\text{Win}_A, \text{Win}_B, \text{Tie}\}$.
2. **Pass 2**: Present prompt with `[Response A = Model 2, Response B = Model 1]`. 
   Let Judge decision be $J_2 \in \{\text{Win}_A, \text{Win}_B, \text{Tie}\}$.

### 5.1 Resolution Policy
$$\boxed{\text{Final Outcome}(1 \text{ vs } 2) = \begin{cases} 
\text{Model 1 Wins} & \text{if } J_1 = \text{Win}_A \text{ and } J_2 = \text{Win}_B \\
\text{Model 2 Wins} & \text{if } J_1 = \text{Win}_B \text{ and } J_2 = \text{Win}_A \\
\text{Tie} & \text{otherwise (inconsistent / order-flipped)}
\end{cases}}$$

This simple protocol neutralizes position bias and drops random noise by converting inconsistent order flips directly into ties.

---

## 6. The Bradley-Terry Rating Model

The **Bradley-Terry Model** (Bradley & Terry, 1952) posits that each model $i$ possesses an unobserved positive latent quality parameter $\pi_i \in \mathbb{R}^+$. The probability that Model $A$ beats Model $B$ is:
$$\mathbb{P}(\mathcal{M}_A \succ \mathcal{M}_B) = \frac{\pi_A}{\pi_A + \pi_B}$$

### 6.1 Mapping to the Elo Rating Scale
Defining Elo rating $R_A$ via the logarithmic change of variables:
$$R_A = 400 \log_{10} \pi_A \iff \pi_A = 10^{R_A / 400}$$

Substituting into the Bradley-Terry equation yields the **Elo Expected Score Formula**:
$$\boxed{E_A = \mathbb{P}(\mathcal{M}_A \succ \mathcal{M}_B) = \frac{10^{R_A / 400}}{10^{R_A / 400} + 10^{R_B / 400}} = \frac{1}{1 + 10^{(R_B - R_A)/400}}}$$

By symmetry:
$$E_B = 1 - E_A = \frac{1}{1 + 10^{(R_A - R_B)/400}}$$

### 6.2 Key Elo Properties
- If $R_A = R_B$, $E_A = \frac{1}{1 + 1} = 0.50$ (even match).
- If $R_A - R_B = +400$, $E_A = \frac{1}{1 + 10^{-1}} = \frac{1}{1.1} \approx \mathbf{90.9\%}$.
- If $R_A - R_B = +200$, $E_A = \frac{1}{1 + 10^{-0.5}} = \frac{1}{1 + 0.3162} \approx \mathbf{75.9\%}$.

---

## 7. Online Elo Update Mechanics & Convergence

When Model $A$ plays Model $B$, the empirical result $S_A$ is observed:
$$S_A = \begin{cases} 1.0 & \text{if Model } A \text{ wins} \\ 0.5 & \text{if tie} \\ 0.0 & \text{if Model } B \text{ wins} \end{cases}$$

The rating update rule is:
$$\boxed{R_A' = R_A + K \cdot \left(S_A - E_A\right)}$$
$$\boxed{R_B' = R_B + K \cdot \left(S_B - E_B\right) = R_B - K \cdot \left(S_A - E_A\right)}$$

where $K$ is the learning rate / step size. Notice that $R_A' + R_B' = R_A + R_B$; **Elo updates are zero-sum**.

```
Expected Score E_A
  1.0 ┌───────────────────────────────────────┐
      │                                . · * *│
      │                          . · '        │
      │                     . · '             │
  0.5 │─────────────────· * ─────────────────│ Even match (R_A = R_B)
      │             · '                       │
      │        · '                            │
  0.0 └* · '──────────────────────────────────┴─▶ Rating Difference
     -400               0                   +400   (R_A - R_B)
```

### 7.2 Maximum Likelihood Estimation (Offline Bradley-Terry)
In production benchmark leaderboards (e.g. Chatbot Arena), ratings are computed via offline **Maximum Likelihood Estimation (MLE)** across the entire match history rather than order-dependent online Elo updates.
Given $W_{ij}$ observed wins of model $i$ over model $j$, the log-likelihood is:
$$\ln \mathcal{L}(\boldsymbol{\pi}) = \sum_{i < j} \left[ W_{ij} \ln \left(\frac{\pi_i}{\pi_i + \pi_j}\right) + W_{ji} \ln \left(\frac{\pi_j}{\pi_i + \pi_j}\right) \right]$$
Optimizing $\nabla_{\boldsymbol{\pi}} \ln \mathcal{L} = \mathbf{0}$ yields the unique, order-invariant global rating vector.
