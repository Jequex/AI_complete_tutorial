# Topic 21.3: RLHF & PPO Policy Gradients

---

## 1. Executive Summary & The 12-Year-Old Analogy

With a trained Reward Model $r_\phi(x, y)$ acting as a continuous evaluator of human satisfaction, we can formulate post-training alignment as a **Reinforcement Learning (RL)** problem. Instead of optimizing next-token cross-entropy against static human demonstrations, the language model acts as an active **policy agent** exploring the combinatorial space of language generation to maximize its expected reward.

However, standard unconstrained reinforcement learning immediately triggers catastrophic failure: the model drifts into degenerate token sequences, repeats punctuation hacks, or exploits blind spots in the Reward Model. To solve this, **Reinforcement Learning from Human Feedback (RLHF)** adds a strict **Kullback-Leibler (KL) divergence penalty** relative to the initial Supervised Fine-Tuned reference model:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

```
+----------------------------------------------------------------------------------------------------+
|                                THE COMPLETE 3-STEP RLHF PIPELINE                                   |
|                                                                                                    |
|  STEP 1: PRE-TRAINING & SFT                                                                        |
|  Web Corpus ---> [Pre-training] ---> Base Model ---> [SFT on Instructions] ---> SFT Model (π_SFT) |
|                                                                                                    |
|  STEP 2: REWARD MODELING                                                                           |
|  π_SFT generates pairs (y_1, y_2) ---> Human Annotators choose y_w > y_l ---> Train RM r_φ(x, y)   |
|                                                                                                    |
|  STEP 3: REINFORCEMENT LEARNING WITH PPO                                                           |
|  Prompt x ---> Active Policy π_θ generates y ---> RM r_φ scores y                                  |
|            ---> Reference Model π_ref computes KL penalty ---> PPO clipped gradient updates π_θ   |
+----------------------------------------------------------------------------------------------------+
```

### The 12-Year-Old Analogy: The Gymnastics Coach with a Safety Bungee Cord

Imagine you are an elite gymnast practicing for the Olympics:

1. **The Competitor (The Policy Model $\pi_\theta$):**
   - You want to invent an incredible new dismount off the balance beam to get a perfect 10.0 score from the judges.
2. **The Olympic Judge (The Reward Model $r_\phi$):**
   - The judge watches your routine and flashes a score on the scoreboard.
3. **The Danger (Catastrophic Exploitation):**
   - If left completely unsupervised, you might realize that if you jump off the beam, land on a trampoline outside the gym, bounce over the fence, and land on a soft mattress, you won't fall! Technically you didn't touch the floor, but you completely destroyed the rules of gymnastics.
4. **The Coach & The Bungee Cord (The Reference Model $\pi_{\text{ref}}$ & KL Penalty $\beta$):**
   - To keep you safe, your coach attaches you to a sturdy bungee cord harness tied to the center beam.
   - The bungee cord lets you stretch, jump high, and try creative flips (improving your score).
   - But if you leap too far away from the beam into crazy, dangerous acrobatics, the bungee cord pulls you right back!
   - The **KL divergence** is that safety bungee cord. It lets the model become more helpful and polite, while preventing it from mutating into gibberish or unhinged reward-hacking loops.

---

## 2. Mathematical Symbol Decoder

| Symbol | Formal Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\pi_\theta(y \mid x)$ | Policy Distribution | Trainable language model generating token sequences given prompt $x$. |
| $\pi_{\text{ref}}(y \mid x)$ | Reference Distribution | Frozen SFT model preventing the policy from drifting into gibberish. |
| $r_\phi(x, y)$ | Scalar Function | Frozen Reward Model evaluating complete prompt-response pairs. |
| $V_\psi(s_t)$ | Value Function | Trainable Critic network predicting expected cumulative future return from state $s_t$. |
| $D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ | Divergence $\ge 0$ | Kullback-Leibler divergence between active policy and reference model. |
| $\beta > 0$ | Hyperparameter | KL penalty coefficient controlling the strength of the reference leash. |
| $s_t = (x, y_{<t})$ | State Space | Prompt tokens concatenated with generated response tokens prior to step $t$. |
| $a_t = y_t \in \mathcal{V}$ | Action Space | Token chosen from the vocabulary at time step $t$. |
| $r_t(\theta)$ | Probability Ratio | $\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$, measuring policy change. |
| $\hat{A}_t$ | Scalar Return | Estimated advantage of taking action $a_t$ in state $s_t$ over baseline. |
| $\delta_t^V$ | Scalar Residual | Temporal Difference (TD) residual $R_t + \gamma V(s_{t+1}) - V(s_t)$. |
| $\epsilon \in (0, 0.3)$ | Hyperparameter | PPO clipping threshold (typically $0.1$ or $0.2$). |

---

## 3. The 4-Model System in Memory

During RLHF fine-tuning with Proximal Policy Optimization (PPO), the training infrastructure must maintain **four distinct neural networks** in GPU memory simultaneously:

```
+----------------------------------------------------------------------------------------------------+
|                                 THE 4-MODEL MEMORY TOPOLOGY IN PPO                                 |
|                                                                                                    |
|  1. ACTOR / POLICY π_θ       [TRAINABLE]: Generates responses, receives PPO policy updates.        |
|  2. CRITIC / VALUE MODEL V_ψ [TRAINABLE]: Predicts state value V(s_t), receives MSE value updates.  |
|  3. REFERENCE MODEL π_ref    [FROZEN]:    Computes baseline log-probs for KL penalty.              |
|  4. REWARD MODEL r_φ         [FROZEN]:    Assigns final task score to completed response.          |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Memory & VRAM Calculation
Consider training an 8B parameter model using standard AdamW (16-bit weights, 32-bit optimizer states):

| Model Component | Status | Parameters | Weight Memory (FP16) | Optimizer States & Gradients | Total VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Actor ($\pi_\theta$)** | Trainable | 8 Billion | 16 GB | 48 GB (Grads + AdamW) | **64 GB** |
| **Critic ($V_\psi$)** | Trainable | 8 Billion | 16 GB | 48 GB (Grads + AdamW) | **64 GB** |
| **Reference ($\pi_{\text{ref}}$)** | Frozen | 8 Billion | 16 GB | 0 GB | **16 GB** |
| **Reward ($r_\phi$)** | Frozen | 8 Billion | 16 GB | 0 GB | **16 GB** |
| **Total VRAM** | — | **32 Billion** | **64 GB** | **96 GB** | **160 GB** |

*Note:* Even with ZeRO-3 parameter sharding, LoRA adapters, or offloading, coordinating four 8B models across distributed clusters requires massive VRAM and introduces significant GPU communication latency. This operational complexity directly inspired the invention of DPO (Topic 21.4).

---

## 4. Token-Level KL Formulation & Per-Step Rewards

The global RLHF objective optimizes expected response return minus total sequence KL divergence:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) \right]$$

In autoregressive language models, the sequence probability factorizes across tokens:
$$\log \pi(y \mid x) = \sum_{t=1}^T \log \pi(y_t \mid x, y_{<t})$$

Therefore, the exact KL divergence between the two factorized conditional distributions decomposes into a sum of per-token log-probability differences:

$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^T \left( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right) \right]$$

### 4.1 Token-Level Reward Distribution
To train the policy with temporal reinforcement learning, we construct a per-token reward signal $R_t$:

$$R_t = \begin{cases} 
-\beta \left( \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t) \right) & \text{for } 1 \le t < T \\
r_\phi(x, y) - \beta \left( \log \pi_\theta(y_T \mid s_T) - \log \pi_{\text{ref}}(y_T \mid s_T) \right) & \text{for } t = T 
\end{cases}$$

- At intermediate tokens ($t < T$), the model receives a pure KL penalty whenever it chooses tokens that deviate from the reference policy.
- At the terminal token ($t = T$, `<eos>`), the full scalar reward $r_\phi(x, y)$ is injected along with the final token's KL penalty.

---

## 5. Proximal Policy Optimization (PPO) in Language Models

In standard Policy Gradient (REINFORCE), the gradient is $\nabla_\theta J(\theta) = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$. However, taking unconstrained gradient steps on language models causes severe policy collapse: if a single parameter update moves the policy into a bad region, subsequent text generations become unrecoverable gibberish.

**PPO (Schulman et al., 2017)** solves this using a **Clipped Surrogate Objective**.

### 5.1 The Probability Ratio
Let $\pi_{\text{old}}$ denote the policy weights used to sample the current batch of rollouts. The probability ratio measures how much the updated policy $\pi_\theta$ differs from $\pi_{\text{old}}$:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t) \right)$$

- If $\pi_\theta = \pi_{\text{old}}$, then $r_t(\theta) = 1.0$.
- If $\pi_\theta$ assigns higher probability to action $a_t$, $r_t(\theta) > 1.0$.

### 5.2 The Clipped Surrogate Objective
$$\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]}$$

```
                Clipped Surrogate Loss (Positive Advantage A > 0)
         Loss
          ^
          |                        -----------------  (Clipped at 1 + ε)
          |                       /
          |                      /
          |                     /
          |                    /
          +-------------------+-----------------------------> r_t(θ)
                             1.0            1 + ε
```

- **When $\hat{A}_t > 0$ (Action was better than average):**
  We want to increase $\pi_\theta(a_t \mid s_t)$, so $r_t(\theta) > 1$. However, the clipping function bounds the objective once $r_t(\theta) > 1 + \epsilon$. The gradient becomes zero, preventing the policy from over-committing to a single rollout.
- **When $\hat{A}_t < 0$ (Action was worse than average):**
  We want to decrease $\pi_\theta(a_t \mid s_t)$, so $r_t(\theta) < 1$. The clipping function bounds the objective once $r_t(\theta) < 1 - \epsilon$, preventing overly destructive punitive updates.

---

## 6. Generalized Advantage Estimation (GAE)

To evaluate whether a token $a_t$ was better or worse than expected, we compute the **advantage** $\hat{A}_t = Q(s_t, a_t) - V(s_t)$. The value function $V_\psi(s_t)$ predicts the expected discounted return from prefix $s_t$.

The 1-step Temporal Difference (TD) error is:
$$\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$$

In language generation, discount factor $\gamma = 1.0$ (episodic text generation without temporal decay). **Generalized Advantage Estimation ($\text{GAE}(\gamma, \lambda)$)** computes an exponentially decaying sum of future TD residuals:

$$\boxed{\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V}$$

Equivalently, GAE is computed backwards recursively from $t = T-1$ down to $t = 0$:
$$\hat{A}_T = 0, \quad \hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}$$

- $\lambda = 0$: Pure 1-step TD advantage $\hat{A}_t = \delta_t^V$ (low variance, high bias).
- $\lambda = 1$: Full Monte Carlo return minus baseline (zero bias, high variance).
- $\lambda \approx 0.95$: Optimal trade-off used universally in LLM RLHF.

---

## 7. The Full PPO Training Step & Critic Loss

In each PPO iteration:
1. **Rollout Generation:** The policy $\pi_{\text{old}}$ generates completions $y \sim \pi_{\text{old}}(\cdot \mid x)$ for a batch of prompts.
2. **Scoring:** The reward model scores $r_\phi(x, y)$, and reference model calculates per-token KL penalties.
3. **Advantage Estimation:** The critic evaluates $V_\psi(s_t)$, and GAE advantages $\hat{A}_t$ are computed and normalized (zero mean, unit variance).
4. **Policy & Value Optimization (Multiple Epochs):**
   $$\mathcal{L}_{\text{total}}(\theta, \psi) = -\mathcal{L}^{\text{CLIP}}(\theta) + c_1 \mathcal{L}_{\text{value}}(\psi) - c_2 \mathcal{H}(\pi_\theta)$$
   where value loss is mean squared error against returns:
   $$\mathcal{L}_{\text{value}}(\psi) = \frac{1}{2} \left( V_\psi(s_t) - (\hat{A}_t + V_{\text{old}}(s_t)) \right)^2$$
   and $\mathcal{H}(\pi_\theta)$ is policy entropy promoting exploration.

---

## 8. Summary & Architectural Next Steps

- RLHF aligns models by treating generation as an RL policy problem guided by a Reward Model.
- A **KL divergence penalty** $\beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ acts as a mathematical safety tether, preventing catastrophic distribution drift and reward hacking.
- PPO uses a **clipped probability ratio** $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$ to ensure stable policy updates.
- **GAE** balances bias and variance when evaluating token advantages.
- However, PPO requires **four neural networks concurrently in memory** and is notoriously brittle to tune.
- In **Topic 21.4**, we will explore how **Direct Preference Optimization (DPO)** mathematically eliminates the need for both the Reward Model and Critic Model entirely!
