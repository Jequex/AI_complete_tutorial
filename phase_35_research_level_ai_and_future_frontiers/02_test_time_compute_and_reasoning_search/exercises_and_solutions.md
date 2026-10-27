# Topic 02: Exercises & Solutions — Test-Time Compute and Reasoning Search

## Exercise 1: Combinatorial Pruning: PRM vs. ORM Best-of-$N$ Search

Consider a mathematical deduction problem that requires a sequence of $T = 5$ logical steps.
- At each step $t \in \{1, \dots, T\}$, the model generates $b = 4$ candidate deductions.
- The probability that any single deduction is logically valid is $p = 0.7$, independently across steps.
- A proof is only correct if **all** $T$ steps are valid.

1. Calculate the total size of the full reasoning tree (number of terminal leaves) and the probability $P_{\text{correct}}$ that a randomly sampled full trajectory is entirely sound.
2. Under an **Outcome Reward Model (ORM)** with Best-of-$N$ sampling, calculate the minimum number of full trajectories $N$ that must be sampled to have a $99\%$ probability of generating at least one completely correct solution.
3. Under a **Process Reward Model (PRM)** that filters out invalid deductions at each step with $100\%$ precision, calculate the expected number of candidate nodes generated across all $T$ levels. Compare the computational efficiency.

---

### Solution 1:

#### 1. Full Tree Size & Trajectory Success Probability
- Total terminal leaves in a tree of depth $T = 5$ with branching factor $b = 4$:
  $$N_{\text{leaves}} = b^T = 4^5 = 1,024 \text{ leaves}$$
- Since each step has probability $p = 0.7$ of being sound, the probability that an entire 5-step chain has no errors is:
  $$P_{\text{correct}} = p^T = 0.7^5 = 0.16807 \quad (\approx 16.8\%)$$

#### 2. ORM Best-of-$N$ Sample Requirement
To achieve $P(\text{at least 1 success}) \ge 0.99$:
$$1 - (1 - P_{\text{correct}})^N \ge 0.99 \implies (1 - 0.16807)^N \le 0.01$$
$$N \ln(0.83193) \le \ln(0.01) \implies N \ge \frac{\ln(0.01)}{\ln(0.83193)} = \frac{-4.60517}{-0.18399} \approx 25.03$$
Rounding up, the ORM requires $N = 26$ full trajectories.
Since each trajectory requires generating $T = 5$ steps, the total steps generated are:
$$\text{Steps}_{\text{ORM}} = 26 \times 5 = 130 \text{ steps}$$

#### 3. PRM Step-Level Pruning
With step-level PRM verification:
- Level 0: 1 root.
- Level 1: Generates $b = 4$ candidates. An expected $b \cdot p = 4 \times 0.7 = 2.8$ survive.
- Level 2: The $2.8$ survivors generate $2.8 \times 4 = 11.2$ candidates. $11.2 \times 0.7 = 7.84$ survive.
- Level $k$: Number of active nodes is $(b \cdot p)^k = (2.8)^k$.
Total nodes evaluated across all 5 levels:
$$\text{Nodes}_{\text{PRM}} = \sum_{k=1}^5 4 \times (2.8)^{k-1} = 4 \times \frac{2.8^5 - 1}{2.8 - 1} = 4 \times \frac{172.1 - 1}{1.8} \approx 380 \text{ total evaluated candidates}$$
Crucially, **100% of the surviving paths at depth 5 are verified solutions** (generating an expected $2.8^5 \approx 172$ valid solutions simultaneously), whereas the ORM repeatedly evaluates failed branches that were already ruined at Step 1.

---

## Exercise 2: PUCT Upper Confidence Bound & Asymptotic Exploration

Consider the PUCT selection rule used in MCTS:

$$a^* = \arg\max_{a \in \mathcal{A}} \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

where $P(s, a) > 0$ for all $a \in \mathcal{A}$, $Q(s, a) \in [0, 1]$, and $c_{\text{puct}} > 0$.

Suppose an action $a_{\text{unvisited}}$ has never been selected ($N(s, a_{\text{unvisited}}) = 0$), while another action $a_{\text{best}}$ has been visited $N(s, a_{\text{best}})$ times with $Q(s, a_{\text{best}}) = 1.0$.

1. Derive the minimum parent visit count $N(s)$ after which $a_{\text{unvisited}}$ is guaranteed to be selected over $a_{\text{best}}$.
2. Explain the significance of this mathematical guarantee for preventing reasoning search from getting permanently trapped in deceptive local minima.

---

### Solution 2:

#### 1. Derivation of the Selection Threshold
The score for the unvisited action $a_u$ with $N(s, a_u) = 0$ is:
$$\text{Score}(a_u) = Q(s, a_u) + c_{\text{puct}} P(s, a_u) \frac{\sqrt{N(s)}}{1 + 0} = 0 + c_{\text{puct}} P(s, a_u) \sqrt{N(s)}$$
The score for the current best action $a_b$ is upper-bounded by:
$$\text{Score}(a_b) = Q(s, a_b) + c_{\text{puct}} P(s, a_b) \frac{\sqrt{N(s)}}{1 + N(s, a_b)} \le 1.0 + c_{\text{puct}} P(s, a_b) \frac{\sqrt{N(s)}}{1 + N(s, a_b)}$$
$a_u$ is selected whenever $\text{Score}(a_u) > \text{Score}(a_b)$. A sufficient condition is:
$$c_{\text{puct}} P(s, a_u) \sqrt{N(s)} > 1.0 + c_{\text{puct}} P(s, a_b) \frac{\sqrt{N(s)}}{1 + N(s, a_b)}$$
As $N(s, a_b) \to \infty$, the exploration bonus of $a_b$ vanishes: $\frac{\sqrt{N(s)}}{1 + N(s, a_b)} \to 0$ (since $N(s, a_b) \approx N(s)$).
Thus, the condition simplifies asymptotically to:
$$c_{\text{puct}} P(s, a_u) \sqrt{N(s)} > 1.0 \implies \sqrt{N(s)} > \frac{1}{c_{\text{puct}} P(s, a_u)} \implies N(s) > \left( \frac{1}{c_{\text{puct}} P(s, a_u)} \right)^2$$

#### 2. Significance for Reasoning Search
In multi-step reasoning, heuristic priors $P(s, a)$ or early evaluations can be **deceptive** (e.g. an arithmetic operation that looks attractive because it matches the target number early, but burns necessary digits).
Because $\sqrt{N(s)}$ grows without bound, the exploration bonus of any neglected step will eventually exceed any finite $Q$-value. This mathematical property guarantees that **no candidate reasoning branch can be permanently starved of exploration**, ensuring that counter-intuitive sub-fractional solutions (like $8 / (3 - 8/3) = 24$) will inevitably be evaluated.

---

## Exercise 3: STaR (Self-Taught Reasoner) Policy Gradient

In the Self-Taught Reasoner (STaR) framework, a model generates a rationale $\mathbf{z} = (z_1, \dots, z_T)$ to solve question $\mathbf{x}$ and arrive at final answer $\hat{y}$.
The objective maximizes expected reward under binary outcome verification $\mathbb{I}(\hat{y} = y^*)$:

$$J(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \mathbb{E}_{\mathbf{z}, \hat{y} \sim P_\theta(\cdot \mid \mathbf{x})} \left[ \mathbb{I}(\hat{y} = y^*) \right] \right]$$

1. Derive the REINFORCE policy gradient $\nabla_\theta J(\theta)$ with rationale filtering.
2. Explain how test-time MCTS search can be used to generate synthetic training data for fine-tuning $P_\theta$ to internalize multi-step reasoning without search at deployment.

---

### Solution 3:

#### 1. Policy Gradient Derivation
Using the log-derivative trick $\nabla_\theta P_\theta(\mathbf{z}, \hat{y} \mid \mathbf{x}) = P_\theta(\mathbf{z}, \hat{y} \mid \mathbf{x}) \nabla_\theta \ln P_\theta(\mathbf{z}, \hat{y} \mid \mathbf{x})$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \sum_{\mathbf{z}, \hat{y}} P_\theta(\mathbf{z}, \hat{y} \mid \mathbf{x}) \nabla_\theta \ln P_\theta(\mathbf{z}, \hat{y} \mid \mathbf{x}) \cdot \mathbb{I}(\hat{y} = y^*) \right]$$

Factoring the trajectory autoregressively $P_\theta(\mathbf{z}, \hat{y} \mid \mathbf{x}) = \prod_{t=1}^{T+1} P_\theta(z_t \mid \mathbf{x}, z_{<t})$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathbf{x}} \left[ \mathbb{E}_{\mathbf{z}, \hat{y} \sim P_\theta} \left[ \left( \sum_{t=1}^{T+1} \nabla_\theta \ln P_\theta(z_t \mid \mathbf{x}, z_{<t}) \right) \cdot \mathbb{I}(\hat{y} = y^*) \right] \right]$$

Under **rationale filtering**, all trajectories where $\hat{y} \neq y^*$ have reward 0 and contribute 0 gradient. The gradient update reduces strictly to standard Supervised Fine-Tuning (SFT) over the subset of **verified rationales**:

$$\nabla_\theta J(\theta) \propto \sum_{(\mathbf{x}, \mathbf{z}^*) \in \mathcal{D}_{\text{verified}}} \sum_{t=1}^{T+1} \nabla_\theta \ln P_\theta(z_t^* \mid \mathbf{x}, z_{<t}^*)$$

#### 2. Distilling Search into Model Weights
Running MCTS at test time is computationally intensive. In the frontier reasoning paradigm:
1. An offline teacher model runs heavy MCTS (e.g., 2,000 simulations per problem with PRM verification) over millions of hard math/code tasks.
2. Only the successfully discovered, PRM-verified winning paths $\mathbf{z}^*$ are retained.
3. The base model is fine-tuned on these winning rationales.
Through this process, the model **internalizes the search heuristic**, learning to generate high-probability reasoning chains that avoid traps during standard single-pass greedy decoding.
