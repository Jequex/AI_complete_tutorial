# Topic 21.3: Exercises & Solutions — RLHF & PPO Policy Gradients

---

### Problem 1: 4-Model VRAM Arithmetic & Memory Footprint (Easy)

Consider an engineering team preparing to align a 7-billion parameter language model using standard PPO in FP16 precision (2 bytes per parameter).
The system requires four models:
1. Policy (Actor $\pi_\theta$): Trainable with AdamW optimizer (16-bit parameters, 16-bit gradients, 32-bit first and second optimizer moments).
2. Critic (Value Model $V_\psi$): Trainable with identical architecture and AdamW optimizer.
3. Reference Model ($\pi_{\text{ref}}$): Frozen, inference only (weights only).
4. Reward Model ($r_\phi$): Frozen, inference only (weights only).

1. Compute the static memory required for each of the four models individually.
2. What is the total static VRAM required across all four models?
3. How many 80 GB A100/H100 GPUs are needed solely to hold these model weights and optimizer states, ignoring KV-cache and activation memory?

<details>
<summary>View Complete Solution</summary>

**Step 1: Memory calculation per model:**
For a 7B model ($7 \times 10^9$ parameters):
- Model weights (FP16): $7 \times 10^9 \times 2\text{ bytes} = 14\text{ GB}$.
- Gradients (FP16): $7 \times 10^9 \times 2\text{ bytes} = 14\text{ GB}$.
- AdamW moments (FP32: $m$ and $v$ each take 4 bytes): $7 \times 10^9 \times 8\text{ bytes} = 56\text{ GB}$.
- Master FP32 weights (used in standard mixed-precision training): $7 \times 10^9 \times 4\text{ bytes} = 28\text{ GB}$.

Total for a trainable model (Actor or Critic):
$$\text{Memory}_{\text{Trainable}} = 14\text{ GB (weights)} + 14\text{ GB (grads)} + 56\text{ GB (Adam moments)} = 84\text{ GB}$$
(If FP32 master weights are kept, this reaches $112\text{ GB}$; with standard modern FP16-AdamW implementations, it is at least $84\text{ GB}$).

For a frozen model (Reference or Reward):
$$\text{Memory}_{\text{Frozen}} = 14\text{ GB (weights only)}$$

**Step 2: Total Static VRAM:**
$$\text{Total} = \text{Actor}(84\text{ GB}) + \text{Critic}(84\text{ GB}) + \text{Reference}(14\text{ GB}) + \text{Reward}(14\text{ GB}) = 196\text{ GB}$$

**Step 3: Hardware requirements:**
$$\frac{196\text{ GB}}{80\text{ GB/GPU}} = 2.45 \implies \text{At least 3 to 4 } 80\text{ GB GPUs}$$
Even before accounting for context activations, attention KV-caches, and rollout buffers, standard PPO requires at least 4 top-tier enterprise GPUs simply to load the four models into memory.
</details>

---

### Problem 2: Token-Level vs. Sequence-Level KL Divergence (Medium)

Let $\mathbf{y} = (y_1, y_2, \dots, y_T)$ be a sequence generated autoregressively by policy $\pi_\theta$ given prompt $x$.
$$\pi_\theta(\mathbf{y} \mid x) = \prod_{t=1}^T \pi_\theta(y_t \mid x, y_{<t}), \quad \pi_{\text{ref}}(\mathbf{y} \mid x) = \prod_{t=1}^T \pi_{\text{ref}}(y_t \mid x, y_{<t})$$

1. Prove that the total sequence-level KL divergence:
   $$D_{\text{KL}}(\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)) = \sum_{\mathbf{y}} \pi_\theta(\mathbf{y} \mid x) \log \frac{\pi_\theta(\mathbf{y} \mid x)}{\pi_{\text{ref}}(\mathbf{y} \mid x)}$$
   equals the expected sum of per-token KL divergences:
   $$\mathbb{E}_{\mathbf{y} \sim \pi_\theta} \left[ \sum_{t=1}^T \left( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right) \right]$$
2. Why is distributing the KL penalty across each token timestep $t$ superior to subtracting the entire sequence KL penalty at the final token $T$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Algebraic proof:**
By properties of logarithms on a product:
$$\log \frac{\pi_\theta(\mathbf{y} \mid x)}{\pi_{\text{ref}}(\mathbf{y} \mid x)} = \log \left( \prod_{t=1}^T \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} \right) = \sum_{t=1}^T \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$
Taking the expectation over $\mathbf{y} \sim \pi_\theta(\cdot \mid x)$:
$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \sum_{\mathbf{y}} \pi_\theta(\mathbf{y} \mid x) \sum_{t=1}^T \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$
Interchanging the summation order:
$$= \sum_{t=1}^T \mathbb{E}_{\mathbf{y}_{<t} \sim \pi_\theta} \left[ \sum_{y_t} \pi_\theta(y_t \mid x, y_{<t}) \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} \right]$$
$$= \mathbb{E}_{\mathbf{y} \sim \pi_\theta} \left[ \sum_{t=1}^T \left( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right) \right] \quad \blacksquare$$

**Step 2: Benefit of per-token reward distribution:**
If the KL penalty is applied solely as a lump-sum deduction at the terminal token $T$, the reinforcement learning critic $V_\psi(s_t)$ has difficulty with **temporal credit assignment**: it cannot determine which specific token in a 500-token paragraph was responsible for drifting from the reference distribution. Distributing the KL penalty per-token provides immediate feedback at step $t$, instantly penalizing off-distribution tokens right when they are generated.
</details>

---

### Problem 3: PPO Probability Ratio Clipping Analysis (Medium)

Recall the clipped surrogate objective:
$$\mathcal{L}^{\text{CLIP}}(\theta) = \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right)$$
where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$ and $\epsilon = 0.20$.

1. Suppose $\hat{A}_t = +4.0$ (highly advantageous action) and the current update makes the action $1.5\times$ more likely ($r_t = 1.50$).
   - Compute the unclipped term $r_t \hat{A}_t$.
   - Compute the clipped term $\text{clip}(r_t, 0.8, 1.2) \hat{A}_t$.
   - Which term is selected by the $\min$ operator, and what is the gradient $\frac{\partial \mathcal{L}}{\partial r_t}$?
2. Suppose $\hat{A}_t = -3.0$ (bad action) and the update makes the action much less likely ($r_t = 0.60$).
   - Compute the unclipped term $r_t \hat{A}_t$.
   - Compute the clipped term $\text{clip}(r_t, 0.8, 1.2) \hat{A}_t$.
   - Which term is selected by the $\min$ operator, and what is the gradient $\frac{\partial \mathcal{L}}{\partial r_t}$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Case $\hat{A}_t = +4.0, r_t = 1.50$:**
- Unclipped term: $r_t \hat{A}_t = 1.50 \times 4.0 = 6.00$.
- Clipped term: $\text{clip}(1.50, 0.8, 1.2) \times 4.0 = 1.20 \times 4.0 = 4.80$.
- Selection:
  $$\min(6.00, 4.80) = 4.80 \quad (\text{the clipped term is selected!})$$
- Gradient: Because the objective is clamped at $1.20 \times 4.0$, $\frac{\partial \mathcal{L}}{\partial r_t} = 0$.
The clipping prevents the policy from taking an excessively aggressive gradient step on this sample.

**Step 2: Case $\hat{A}_t = -3.0, r_t = 0.60$:**
- Unclipped term: $r_t \hat{A}_t = 0.60 \times (-3.0) = -1.80$.
- Clipped term: $\text{clip}(0.60, 0.8, 1.2) \times (-3.0) = 0.80 \times (-3.0) = -2.40$.
- Selection:
  $$\min(-1.80, -2.40) = -2.40 \quad (\text{the clipped term is selected!})$$
- Gradient: Because the objective is clamped at $0.80 \times (-3.0)$, $\frac{\partial \mathcal{L}}{\partial r_t} = 0$.
Again, the gradient vanishes, preventing an excessively punitive gradient update from pushing the probability towards zero too fast.
</details>

---

### Problem 4: Hand Calculation of GAE Advantage on a 4-Token Trajectory (Hard)

Consider a 4-token response trajectory $(t = 0, 1, 2, 3)$ with:
- Discount factor $\gamma = 1.0$
- GAE decay parameter $\lambda = 0.90$
- Critic value estimates: $V = [1.2, 1.5, 1.8, 2.0]$
- Terminal state value after completion: $V(s_4) = 0.0$
- Per-step rewards (including KL and terminal reward): $R = [0.0, -0.1, 0.0, +3.0]$

1. Compute the TD error $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$ for each timestep $t \in \{0, 1, 2, 3\}$.
2. Using the backward recursion $\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$ with $\hat{A}_4 = 0$, compute the GAE advantages $\hat{A}_3, \hat{A}_2, \hat{A}_1, \hat{A}_0$.
3. Which token receives the highest positive advantage?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute TD errors $\delta_t$:**
- For $t = 0$: $\delta_0 = R_0 + V_1 - V_0 = 0.0 + 1.5 - 1.2 = +0.30$
- For $t = 1$: $\delta_1 = R_1 + V_2 - V_1 = -0.1 + 1.8 - 1.5 = +0.20$
- For $t = 2$: $\delta_2 = R_2 + V_3 - V_2 = 0.0 + 2.0 - 1.8 = +0.20$
- For $t = 3$ (terminal token): $\delta_3 = R_3 + V_4 - V_3 = 3.0 + 0.0 - 2.0 = +1.00$

**Step 2: Backward recursion for GAE ($\gamma \lambda = 1.0 \times 0.90 = 0.90$):**
- For $t = 3$:
  $$\hat{A}_3 = \delta_3 + 0.90 \times \hat{A}_4 = 1.00 + 0.0 = 1.000$$
- For $t = 2$:
  $$\hat{A}_2 = \delta_2 + 0.90 \times \hat{A}_3 = 0.20 + 0.90 \times 1.00 = 0.20 + 0.900 = 1.100$$
- For $t = 1$:
  $$\hat{A}_1 = \delta_1 + 0.90 \times \hat{A}_2 = 0.20 + 0.90 \times 1.10 = 0.20 + 0.990 = 1.190$$
- For $t = 0$:
  $$\hat{A}_0 = \delta_0 + 0.90 \times \hat{A}_1 = 0.30 + 0.90 \times 1.19 = 0.30 + 1.071 = 1.371$$

**Step 3: Result Analysis:**
The advantages across the sequence are:
$$\hat{\mathbf{A}} = [1.371, 1.190, 1.100, 1.000]$$
The earliest token ($t = 0$) receives the highest advantage ($1.371$). Because each step had a positive TD error ($\delta_t > 0$) culminating in the strong terminal reward ($+3.0$), the benefits accumulate backwards, crediting the initial decisions that steered the response toward a successful outcome.
</details>
