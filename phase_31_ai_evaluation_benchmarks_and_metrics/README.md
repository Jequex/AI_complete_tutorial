# Phase 31: AI Evaluation, Benchmarks & Metrics

Welcome to **Phase 31: AI Evaluation, Benchmarks & Metrics** of the Complete Artificial Intelligence & Machine Learning Engineering Curriculum.

In real-world applied AI and state-of-the-art research, **a model is only as credible as the statistical rigor of its evaluation**. Leaderboards are saturated with claims of breakthrough performance—models boasting $84.7\%$ accuracy over an $84.1\%$ baseline, or claiming superiority in chatbot arenas. Yet without proper hypothesis testing, calibration analysis, order-bias neutralization, and hallucination auditing, such metrics frequently represent nothing more than **statistical noise, benchmark contamination, and Goodhart's Law in action**.

Phase 31 bridges the gap between heuristic benchmarking and rigorous scientific evaluation. We deconstruct the full modern evaluation suite: from foundational statistical tests and probability calibration to NLP sequence metrics, LLM-as-a-Judge tournament systems, and adversarial factuality stress-testing.

---

## Master Architectural Roadmap

```
                       PHASE 31: AI EVALUATION, BENCHMARKS & METRICS
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
  [01] Traditional ML Evaluation                                  [02] Calibration & Thresholding
• Paired Student's t-Test & Dietterich 5x2-fold                 • Confusion Matrix & Operating Curves
• Wilcoxon Signed-Rank Test (Non-Parametric)                    • ROC-AUC & Mann-Whitney U Equivalence
• Resampling Permutation Tests (Exact Sign Flips)               • Precision-Recall AUC under Imbalance
• Percentile & Empirical Basic Bootstrap CIs                    • Murphy's Brier Score 3-Way Decomposition
• Family-Wise Error Rate (FWER) & Bonferroni                    • Expected Calibration Error (ECE) & MCE
• Benjamini-Hochberg False Discovery Rate (FDR)                 • Temperature Scaling Logit Rectification
             │                                                               │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
                             [03] NLP & Language Model Metrics
                          • Cross-Entropy Loss to Perplexity (PPL)
                          • Branching Factor Information Theory
                          • BLEU-N Modified Clipped Precision & BP
                          • ROUGE-1/2 & 2D Dynamic Programming LCS
                          • SQuAD String Normalization Pipeline
                          • Exact Match (EM) vs Token-Level F1
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
  [04] LLM-as-a-Judge & Preferences                               [05] Hallucination & Robustness
• Single-Answer vs Pairwise Preferences                         • Atomic Proposition Decomposition
• Judge Biases: Order, Verbosity, Self-Enhancement              • Knowledge Triple Grounding (s, r, o)
• Bidirectional Swapping De-Biasing Protocol                    • FActScore & Claim Hallucination Rate
• Bradley-Terry Elo Win Probability Function                    • Document-Level Hallucination Indicator
• Zero-Sum Dynamic Rating Updates                               • QWERTY Typos & Distractor Perturbations
• Offline Maximum Likelihood Estimation (MM Solver)             • Goodhart's Law: Regressive, Extremal, Causal
• Empirical Head-to-Head Win-Rate Matrices                      • Benchmark Contamination via N-Gram Overlap
```

---

## Mathematical Pillars & Core Derivations

### 1. Student's Paired $t$-Test & Non-Parametric Resampling
For paired differences $d_i = X_{A, i} - X_{B, i}$ across $n$ evaluation folds:
$$\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i, \qquad s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2}$$
$$\boxed{t = \frac{\bar{d}}{s_d / \sqrt{n}} \sim \text{Student-}t(\nu = n - 1)}$$

Under Monte Carlo permutation testing with $B$ randomized sign-flips $s^{(b)} \in \{-1, +1\}^n$:
$$\boxed{p_{\text{perm}} = \frac{1 + \sum_{b=1}^B \mathbb{I}(|T^{(b)}| \ge |T_{\text{obs}}|)}{B + 1}}$$

Multiple testing discovery control via Benjamini-Hochberg (BH) for sorted $p_{(1)} \le \dots \le p_{(m)}$:
$$\boxed{k^* = \max \left\{ k \;\middle|\; p_{(k)} \le \frac{k}{m} Q \right\} \implies \text{Reject } H_{0, (1)}, \dots, H_{0, (k^*)}}$$

### 2. Discrimination, Calibration & Murphy's Decomposition
The Area Under the ROC Curve ($\text{AUC}_{\text{ROC}}$) is mathematically equivalent to the normalized Mann-Whitney $U$-statistic:
$$\boxed{\text{AUC}_{\text{ROC}} = \frac{U}{N_+ N_-} = \frac{1}{N_+ N_-}\sum_{i \in S_+} \sum_{j \in S_-} \left( \mathbb{I}(\hat{p}_i > \hat{p}_j) + \frac{1}{2}\mathbb{I}(\hat{p}_i = \hat{p}_j) \right)}$$

Partitioning predictions into $M$ equal bins decomposes the Brier score:
$$\boxed{\text{BS} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2 = \text{Reliability} - \text{Resolution} + \text{Uncertainty}}$$
$$\text{REL} = \sum_{m=1}^M \frac{n_m}{N} (\bar{p}_m - \bar{y}_m)^2, \quad \text{RES} = \sum_{m=1}^M \frac{n_m}{N} (\bar{y}_m - \bar{y})^2, \quad \text{UNC} = \bar{y}(1 - \bar{y})$$

Temperature scaling modifies logits $z_i$ via scalar $T > 0$: $\hat{p}_i(T) = \sigma(z_i / T)$, minimizing Expected Calibration Error:
$$\boxed{\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|}$$
while leaving ROC-AUC and threshold-0.5 accuracy strictly invariant.

### 3. Language Model Perplexity & Generation Alignment
Perplexity reflects the exponential of token-level cross-entropy loss:
$$\boxed{PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(x_i \mid x_{<i})\right) = 2^{H_2(p, q)}}$$

BLEU-4 with modified clipped $n$-gram precision and brevity penalty:
$$\boxed{\text{BLEU}-4 = BP \cdot \exp\left(\sum_{n=1}^4 \frac{1}{4} \log p_n\right)}, \qquad BP = \min\left(1, \; \exp\left(1 - \frac{r}{c}\right)\right)$$

ROUGE-L Longest Common Subsequence computed via 2D Dynamic Programming in $\mathcal{O}(N \cdot M)$ time:
$$R_{\text{LCS}} = \frac{\text{LCS}(C, R)}{|R|}, \quad P_{\text{LCS}} = \frac{\text{LCS}(C, R)}{|C|}, \quad \boxed{F_{\text{LCS}} = \frac{2 R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + P_{\text{LCS}}}}$$

### 4. The Bradley-Terry Preference Model & Elo Convergence
The probability that Model $A$ is preferred over Model $B$ given ratings $R_A, R_B$:
$$\boxed{E_A = \mathbb{P}(\mathcal{M}_A \succ \mathcal{M}_B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}}$$
Online Elo update rule for outcome $S_A \in \{1.0, 0.5, 0.0\}$:
$$\boxed{R_A' = R_A + K(S_A - E_A), \qquad R_B' = R_B + K(S_B - E_B)}$$

Bidirectional position swapping neutralizes order bias:
$$\text{Final Result} = \begin{cases} \text{A Wins} & \text{if A wins both orientations} \\ \text{B Wins} & \text{if B wins both orientations} \\ \text{Tie} & \text{otherwise (inconsistent / order flip)} \end{cases}$$

### 5. Atomic Factuality & Goodhart's Law
Atomic claim extraction decomposes generated document $\mathcal{D}$ into propositions $\mathcal{C} = \{c_1, \dots, c_K\}$ verified against knowledge base $\mathcal{K}$:
$$\boxed{\text{FActScore} = \frac{\sum_{k=1}^K v(c_k \mid \mathcal{K})}{K}, \qquad H_{\text{claim}} = 1 - \text{FActScore}}$$
$$\boxed{H_{\text{doc}} = \mathbb{I}\left(\sum_{k=1}^K (1 - v(c_k \mid \mathcal{K})) > 0\right)}$$

Robustness retention under prompt perturbations:
$$\boxed{\rho_{\text{robust}} = \frac{\text{Score}(\mathcal{T}(D))}{\text{Score}(D)}}$$

---

## Phase Navigation & Directory Index

| Topic | Directory | Core Concepts | Resources |
| :--- | :--- | :--- | :--- |
| **01** | `01_traditional_ml_evaluation_and_statistical_testing/` | Paired $t$-test, Wilcoxon signed-rank, Permutation tests, Percentile/Basic Bootstrap CIs, Bonferroni, Benjamini-Hochberg FDR. | [Lecture (HTML)](01_traditional_ml_evaluation_and_statistical_testing/lecture.html) &bull; [Exercises](01_traditional_ml_evaluation_and_statistical_testing/exercises_and_solutions.html) &bull; [Experiment](01_traditional_ml_evaluation_and_statistical_testing/experiment.py) |
| **02** | `02_classification_calibration_and_thresholding/` | Confusion matrix, ROC-AUC, Mann-Whitney $U$, PR-AUC, Murphy's Brier decomposition, ECE, MCE, Temperature Scaling. | [Lecture (HTML)](02_classification_calibration_and_thresholding/lecture.html) &bull; [Exercises](02_classification_calibration_and_thresholding/exercises_and_solutions.html) &bull; [Experiment](02_classification_calibration_and_thresholding/experiment.py) |
| **03** | `03_nlp_and_language_model_metrics/` | Perplexity from cross-entropy, BLEU-4 with brevity penalty, ROUGE-1/2/L via DP LCS, SQuAD Exact Match & Token F1. | [Lecture (HTML)](03_nlp_and_language_model_metrics/lecture.html) &bull; [Exercises](03_nlp_and_language_model_metrics/exercises_and_solutions.html) &bull; [Experiment](03_nlp_and_language_model_metrics/experiment.py) |
| **04** | `04_llm_as_a_judge_and_preference_benchmarks/` | Pairwise evaluation, Position/verbosity bias, Bidirectional de-biasing, Bradley-Terry Elo, Tournament MLE convergence. | [Lecture (HTML)](04_llm_as_a_judge_and_preference_benchmarks/lecture.html) &bull; [Exercises](04_llm_as_a_judge_and_preference_benchmarks/exercises_and_solutions.html) &bull; [Experiment](04_llm_as_a_judge_and_preference_benchmarks/experiment.py) |
| **05** | `05_hallucination_factuality_and_robustness_auditing/` | Atomic fact decomposition, Knowledge triples, FActScore, Prompt perturbations, The 4 faces of Goodhart's Law, Contamination. | [Lecture (HTML)](05_hallucination_factuality_and_robustness_auditing/lecture.html) &bull; [Exercises](05_hallucination_factuality_and_robustness_auditing/exercises_and_solutions.html) &bull; [Experiment](05_hallucination_factuality_and_robustness_auditing/experiment.py) |

---

## Running the Complete Experiment Suite

All experiment scripts are written exclusively using the **Python 3 Standard Library** (zero external pip packages required). 
Run each self-verifying test from the phase root directory:

```bash
# 1. Traditional ML Evaluation & Statistical Testing
python3 01_traditional_ml_evaluation_and_statistical_testing/experiment.py

# 2. Classification Calibration & Temperature Scaling
python3 02_classification_calibration_and_thresholding/experiment.py

# 3. NLP Metrics: Perplexity, BLEU-4, ROUGE-L & Token F1
python3 03_nlp_and_language_model_metrics/experiment.py

# 4. LLM-as-a-Judge, Preference Tournaments & Bradley-Terry Elo
python3 04_llm_as_a_judge_and_preference_benchmarks/experiment.py

# 5. Hallucination Auditing, Perturbations & Contamination Scanning
python3 05_hallucination_factuality_and_robustness_auditing/experiment.py
```
