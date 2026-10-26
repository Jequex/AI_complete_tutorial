# Topic 04: The Policy Gradient Theorem & REINFORCE

## 1. Introduction & 12-Year-Old Intuition

Imagine you are learning how to shoot free throws in basketball.

In **value-based learning** (like Q-learning), before every single shot, you pull out a thick notebook, write down your exact elbow angle, distance, and wind speed, calculate the numerical value of 50 different micro-adjustments, and pick the one with the highest calculated score. In high-speed continuous motion, that's far too slow and clunky!

In **policy-based learning**, your brain adjusts its muscles directly. You shoot the ball.
- If the ball swishes smoothly through the net (+10 reward), your brain tells your motor cortex: *"Whatever muscle sequence I just fired, do more of that next time!"*
- If the ball clangs off the backboard (-5 reward), your brain tells your muscles: *"Whatever sequence I just fired, do less of that next time!"*

This is the essence of the **Policy Gradient Theorem**:
We don't need to assign numbers to every hypothetical alternative. We sample actual actions, observe how well the outcome turned out, and adjust the policy parameters in the direction of the log-probability gradient scaled by the reward.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Domain | Meaning |
| :--- | :--- | :--- | :--- |
| $\theta$ | Policy Parameters | Vector $\mathbb{R}^D$ | Weights determining action probabilities $\pi_\theta(a \mid s)$ (e.g. neural network weights). |
| $J(\theta)$ | Performance Objective | Scalar $\mathbb{R}$ | Expected total discounted return under policy $\pi_\theta$. |
| $\tau$ | Trajectory Rollout | Sequence | $(s_0, a_0, r_1, s_1, \dots, s_T)$ sampled from environment under $\pi_\theta$. |
| $P(\tau; \theta)$ | Trajectory Likelihood | $[0, 1]$ | Probability density of observing trajectory $\tau$ given policy $\pi_\theta$. |
| $\nabla_\theta \log \pi_\theta(a \mid s)$ | Score Function | Vector $\mathbb{R}^D$ | Direction in parameter space that increases the probability of taking action $a$ in state $s$. |
| $G_t$ | Return-to-Go | Scalar $\mathbb{R}$ | $\sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}$: actual rewards accumulated after time step $t$. |
| $b(s)$ | Baseline Function | Scalar $\mathbb{R}$ | State-dependent reference (e.g. $V(s)$) subtracted to reduce variance without introducing bias. |
| $A^\pi(s, a)$ | Advantage Function | Scalar $\mathbb{R}$ | $Q^\pi(s, a) - V^\pi(s)$: how much better action $a$ is compared to the average policy action in state $s$. |

---

## 3. The Objective Function & Parameterization

In policy gradient methods, we directly parameterize a stochastic policy:
$$\pi_\theta(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s; \theta)$$

For discrete action spaces, this is typically parameterized by a neural network with softmax logits:
$$\pi_\theta(a \mid s) = \frac{\exp(z_\theta(s)_a)}{\sum_{a' \in \mathcal{A}} \exp(z_\theta(s)_{a'})}$$

The scalar objective we maximize is the expected total trajectory return:
$$J(\theta) \triangleq \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau; \theta) R(\tau) d\tau$$
where $R(\tau) = \sum_{t=0}^{T-1} \gamma^t R_{t+1}$.

---

## 4. Full Step-by-Step Derivation of the Policy Gradient Theorem

We wish to compute the gradient $\nabla_\theta J(\theta)$ so that we can optimize policy parameters via gradient ascent:
$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

### Step 1: Differentiate the Expectation Integral
$$\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau; \theta) R(\tau) d\tau = \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau$$

### Step 2: The Log-Derivative Trick (Likelihood Ratio)
Notice that for any positive differentiable function $f(\theta)$:
$$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)} \iff \nabla_\theta f(\theta) = f(\theta) \nabla_\theta \log f(\theta)$$
Applying this identity to $P(\tau; \theta)$:
$$\nabla_\theta J(\theta) = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]$$

### Step 3: Expanding the Trajectory Log-Likelihood
By the Markov property, the trajectory probability is:
$$P(\tau; \theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t)$$
Taking the natural logarithm:
$$\log P(\tau; \theta) = \log \rho_0(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

### Step 4: The Environmental Dynamics Vanish
Now differentiate with respect to policy parameters $\theta$:
$$\nabla_\theta \log P(\tau; \theta) = \nabla_\theta \log \rho_0(s_0) + \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \nabla_\theta \log \mathcal{P}(s_{t+1} \mid s_t, a_t)$$
The initial state distribution $\rho_0(s_0)$ and the physical transition kernel $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ are properties of the environment—**they do not depend on $\theta$**!
Therefore, their gradients are identically zero:
$$\nabla_\theta \log \rho_0(s_0) = \mathbf{0}, \quad \nabla_\theta \log \mathcal{P}(s_{t+1} \mid s_t, a_t) = \mathbf{0}$$
Yielding the crucial result:
$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

### Step 5: Causality and Reward-to-Go
Substituting back into the expectation:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( \sum_{t'=0}^{T-1} \gamma^{t'} R_{t'+1} \right) \right]$$
By causality, an action taken at time step $t$ cannot retroactively affect rewards earned in the past ($t' < t$). Conditioned on history up to $t$, $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t \mid s_t) R_{t'+1}] = \mathbf{0}$ for all $t' < t$.
Dropping past rewards yields the **Reward-to-Go Policy Gradient Theorem**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$
where $G_t \triangleq \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}$.

---

## 5. The Baseline Subtraction Theorem

The raw REINFORCE policy gradient estimator has notoriously high variance. We reduce variance by subtracting a state-dependent baseline $b(s)$ from the return.

### Theorem: Zero Bias from State Baseline
For any function $b(s)$ that does not depend on action $a$:
$$\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)} \left[ \nabla_\theta \log \pi_\theta(a \mid s) b(s) \right] = \mathbf{0}$$

### Proof:
$$\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) b(s) = b(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}$$
$$= b(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) = b(s) \nabla_\theta \left( \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \right)$$
Because $\pi_\theta(\cdot \mid s)$ is a valid probability distribution, $\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) = 1$ identically for all $\theta$:
$$= b(s) \nabla_\theta (1) = b(s) \cdot \mathbf{0} = \mathbf{0} \quad \blacksquare$$

Hence, subtracting $b(s_t)$ produces an **identically unbiased gradient**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) (G_t - b(s_t)) \right]$$

Setting $b(s) = V^\pi(s)$ transforms the scalar weight into the **Advantage Function**:
$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) \approx G_t - V(s_t)$$
If an action achieves an outcome superior to the baseline expectation ($G_t > V(s_t)$), its probability is boosted; if inferior ($G_t < V(s_t)$), its probability is suppressed.
