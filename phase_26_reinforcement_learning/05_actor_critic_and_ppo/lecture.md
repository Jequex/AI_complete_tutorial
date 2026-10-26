# Topic 05: Actor-Critic Architectures & Proximal Policy Optimization (PPO)

## 1. Introduction & 12-Year-Old Intuition

Imagine an aspiring young actor auditioning on Broadway, accompanied by an experienced theatre director sitting in the front row:

- **The Actor ($\pi_\theta$)**: The performer on stage trying out different lines, expressions, postures, and dance moves.
- **The Critic ($V_\phi$)**: The seasoned director watching every move. Instead of waiting until opening night to see if tickets sell out, the director whispers immediate feedback after each line: *"That dramatic pause was brilliant—keep that!"* or *"That posture was stiff—relax your shoulders."*

In early policy gradient methods (REINFORCE), the actor had no critic; they had to perform the whole three-hour play from start to finish before finding out whether the audience clapped or threw rotten tomatoes.

In **Actor-Critic**, the Critic provides immediate, low-variance critique at every single line.

And what about **Proximal Policy Optimization (PPO)**?
Imagine that after hearing one compliment from the director, the actor decides to do five wild backflips and smash through a wall! That sudden, radical overreaction could ruin the entire audition.
PPO places **training wheels (a clipping clamp)** on the actor's updates:
*"You may improve your style, but you are not allowed to change your habits by more than 20% in a single rehearsal."*

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Meaning |
| :--- | :--- | :--- | :--- |
| $\pi_\theta(a \mid s)$ | Actor Network | Policy | Probability distribution over actions parameterized by weights $\theta$. |
| $V_\phi(s)$ | Critic Network | Value Function | Expected state return parameterized by weights $\phi$. |
| $\delta_t^V$ | TD Residual | Scalar $\mathbb{R}$ | 1-step Bellman error: $R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$. |
| $\lambda$ | GAE Decay Parameter | $\lambda \in [0, 1]$ | Tradeoff parameter balancing bias (low $\lambda$) against variance (high $\lambda$). |
| $\hat{A}_t^{\text{GAE}}$ | Generalized Advantage | Scalar $\mathbb{R}$ | Exponentially decayed sum of future TD residuals: $\sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$. |
| $r_t(\theta)$ | Probability Ratio | $\mathbb{R}_{\ge 0}$ | Importance weight $\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$. |
| $\epsilon$ | PPO Clip Parameter | Scalar (e.g. $0.2$) | Permissible policy deviation boundary $[1 - \epsilon, 1 + \epsilon]$. |
| $\mathcal{L}^{\text{CLIP}}$ | Clipped Surrogate Objective | Scalar $\mathbb{R}$ | Pessimistic lower bound objective preventing destructive policy collapse. |

---

## 3. The Actor-Critic Architecture

The fundamental variance limitation of REINFORCE arises from Monte Carlo returns $G_t = \sum_k \gamma^k R_{t+k+1}$.
Actor-Critic replaces $G_t$ with a bootstrapped advantage evaluated by the Critic:
$$A(s_t, a_t) \approx R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t) = \delta_t^V$$

```
                               Actor-Critic Cycle
                               
                                Environment
                                 ▲        │
                 Action a_t      │        │ Next state s_{t+1},
               ┌─────────────────┘        │ Reward r_{t+1}
               │                          ▼
          ┌─────────┐               ┌────────────┐
          │  Actor  │               │   Critic   │
          │ π_θ(a|s)│               │   V_ϕ(s)   │
          └────▲────┘               └─────┬──────┘
               │                          │
               │    Advantage A(s, a)     │
               └──────────────────────────┘
```

The Actor and Critic update synchronously:
- **Actor update**: $\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t \mid s_t) \delta_t^V$
- **Critic update**: $\phi \leftarrow \phi + \alpha_\phi \delta_t^V \nabla_\phi V_\phi(s_t)$

---

## 4. Generalized Advantage Estimation (GAE)

John Schulman et al. (2015) introduced Generalized Advantage Estimation to construct a flexible spectrum between high-bias 1-step TD and high-variance Monte Carlo.

Define the $k$-step advantage estimators:
$$\hat{A}_t^{(1)} \triangleq \delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
$$\hat{A}_t^{(2)} \triangleq \delta_t^V + \gamma \delta_{t+1}^V = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2}) - V(S_t)$$
$$\hat{A}_t^{(k)} \triangleq \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \gamma^k V(S_{t+k}) - V(S_t)$$

The GAE($\gamma, \lambda$) estimator is the exponentially weighted average of all $k$-step estimators:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

### Boundary Behaviors:
- **$\lambda = 0$**: $\hat{A}_t = \delta_t^V$ (1-step TD). Minimal variance, but maximal bias if $V$ is inaccurate.
- **$\lambda = 1$**: $\hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t) = G_t - V(S_t)$ (Empirical Monte Carlo). Zero bias, but maximal variance.
- **$\lambda \in [0.90, 0.98]$**: Ideal empirical sweet-spot.

### Efficient Backward Recursion:
For a trajectory of length $T$:
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}$$
computed in $O(T)$ time by traversing backward from $t = T-1$ down to $0$.

---

## 5. Trust Region Policy Optimization (TRPO) & PPO

### 5.1 Why Standard Policy Gradients Collapse
In standard supervised learning, if a model takes a bad gradient step, it receives the same training data on the next epoch and can correct itself.
In Reinforcement Learning, **the policy generates its own future training data**. If a policy update is too large, the agent falls into an unrecoverable degenerate regime (e.g., spinning in circles or jumping off a cliff), never collecting informative data again.

TRPO (Schulman et al. 2015) enforced a hard constraint on the policy change:
$$\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \parallel \pi_\theta(\cdot \mid s_t)) \right] \le \delta$$
While mathematically principled, TRPO requires calculating the second-order Fisher Information Matrix and solving conjugate gradients, making it computationally expensive and difficult to scale.

### 5.2 Proximal Policy Optimization (PPO, Schulman et al. 2017)
PPO retains TRPO's stability using only standard first-order stochastic gradient descent through the **Clipped Surrogate Objective**.

Define the probability ratio:
$$r_t(\theta) \triangleq \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
Note that $r_t(\theta_{\text{old}}) = 1$.

The clipped objective is:
$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.2$).

```
         Case 1: Advantage A > 0 (Good action)           Case 2: Advantage A < 0 (Bad action)
           
           L^CLIP                                          L^CLIP
             ▲                                               ▲
             │          / Unclipped: r * A                   │
             │         /                                     │                 / Unclipped: r * A
   (1+ε) A ──┼────────/────── Clipped Flat                   │                /
             │       /                                       │               /
             │      /                                        │              /
           A ┼─────/                                  (1-ε) A┼─────────────/───── Clipped Flat
             │    /                                          │            /
             │   /                                         A ┼───────────/
             └──┴────┴───────► r_t                           └──┴────┴───┴───────► r_t
               1   1+ε                                         1-ε   1
```

### 5.3 Detailed Breakdown of the Clipping Mechanism:
1. **Positive Advantage ($\hat{A}_t > 0$)**: The action performed better than expected.
   - We want to increase its probability ($r_t > 1$).
   - However, when $r_t(\theta) > 1 + \epsilon$, the term $\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t = (1+\epsilon)\hat{A}_t$ is smaller than $r_t \hat{A}_t$.
   - The $\min$ operator selects the clipped value! The gradient becomes zero, **stopping further probability inflation**.
2. **Negative Advantage ($\hat{A}_t < 0$)**: The action performed worse than expected.
   - We want to decrease its probability ($r_t < 1$).
   - However, when $r_t(\theta) < 1 - \epsilon$, $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t = (1-\epsilon)\hat{A}_t$ is more negative (smaller) than $r_t \hat{A}_t$.
   - The $\min$ operator selects the clipped ceiling! The gradient becomes zero, **preventing overzealous policy destruction**.

### 5.4 The Complete PPO Joint Objective
In practice, neural architectures optimize the actor and critic jointly with an entropy bonus $\mathcal{H}(\pi_\theta)$ to encourage exploration:
$$\mathcal{L}^{\text{PPO}}(\theta, \phi) = \hat{\mathbb{E}}_t \left[ \mathcal{L}_t^{\text{CLIP}}(\theta) - c_1 \left( V_\phi(s_t) - V_t^{\text{targ}} \right)^2 + c_2 \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right]$$
where $c_1 \approx 0.5$ and $c_2 \approx 0.01$.
Because PPO prevents disastrous updates, we can safely perform **multiple epochs of mini-batch SGD** on the same batch of collected experience, dramatically boosting sample efficiency.
