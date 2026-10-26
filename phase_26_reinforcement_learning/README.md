# Phase 26: Reinforcement Learning

Welcome to **Phase 26: Reinforcement Learning** of the Complete Artificial Intelligence & Machine Learning Engineering Curriculum.

Reinforcement Learning (RL) formalizes the mathematics of goal-directed learning and decision-making under uncertainty. Unlike Supervised Learning (which learns from static, ground-truth labeled datasets) or Unsupervised Learning (which uncovers latent data distributions), Reinforcement Learning agents learn through active, closed-loop trial and error in an environment, optimizing long-term cumulative reward.

---

## Master Architectural Roadmap

```
                                  PHASE 26: REINFORCEMENT LEARNING
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
      [01] Formal Foundations                                           [02] Dynamic Programming
   • MDP 5-Tuple ⟨S, A, P, R, γ⟩                                     • Bellman Expectation Equations
   • The Markov Property                                             • Bellman Optimality Equations
   • Trajectory Distributions                                        • Banach Fixed-Point Theorem
   • Discounted Return G_t                                           • γ-Contraction Mapping Proof (L_∞)
   • Value Functions V^π(s), Q^π(s, a)                               • Policy Iteration & Value Iteration
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                               [03] Temporal Difference & DQN
                            • Model-Free Learning & Bootstrapping
                            • Monte Carlo vs TD(0) Error δ_t
                            • On-Policy SARSA vs Off-Policy Q-Learning
                            • Deep Q-Networks (DQN, Mnih et al. 2015)
                            • Replay Buffer D & Target Network Q_{θ⁻}
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
      [04] Policy Gradients & REINFORCE                                 [05] Actor-Critic & PPO
   • Value-Based vs Policy-Based Paradigm                            • 2-Network Topology: Actor π_θ, Critic V_ϕ
   • Parameterized Stochastic Policies π_θ(a|s)                      • 1-Step TD Advantage vs GAE(γ, λ)
   • Policy Gradient Theorem Derivation                              • Bias-Variance Spectrum Analysis
   • Likelihood Ratio / Log-Derivative Trick                         • Trust Region Policy Optimization (TRPO)
   • REINFORCE Algorithm & Baseline Subtraction                      • PPO Clipped Surrogate Objective L^CLIP
```

---

## Mathematical Pillars & Core Derivations

### 1. The Markov Decision Process (MDP)
A discrete-time Markov Decision Process is defined by the 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

The **Markov Property** guarantees memorylessness:
$$\mathbb{P}(S_{t+1}=s_{t+1}, R_{t+1}=r_{t+1} \mid S_t=s_t, A_t=a_t, \dots, S_0=s_0, A_0=a_0) = \mathbb{P}(S_{t+1}=s_{t+1}, R_{t+1}=r_{t+1} \mid S_t=s_t, A_t=a_t)$$

The **Discounted Return** accumulates future rewards:
$$G_t \triangleq \sum_{k=0}^\infty \gamma^k R_{t+k+1}, \quad \gamma \in [0, 1)$$

### 2. Bellman Equations & Contraction Mapping
For any policy $\pi$, the **Bellman Expectation Equation** is:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$

The **Bellman Optimality Operator** $\mathcal{T}^*$ satisfies:
$$(\mathcal{T}^* V)(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s') \right]$$

**Theorem ($\gamma$-Contraction Mapping under $\ell_\infty$):**
$$\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty$$
By the **Banach Fixed-Point Theorem**, $V^*$ exists, is unique, and can be computed via value iteration $V_{k+1} = \mathcal{T}^* V_k$ with linear convergence rate $\gamma$.

### 3. Model-Free Temporal Difference & Q-Learning
Bootstraps directly from observed transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ without knowing environment dynamics:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

**Deep Q-Networks (DQN):** Combines neural function approximation with an Experience Replay Buffer $\mathcal{D}$ and periodic Target Network $\theta^-$ to eliminate temporal correlation and non-stationary targets:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

### 4. The Policy Gradient Theorem
For objective $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$, the exact gradient is:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( G_t - b(s_t) \right) \right]$$
where any state baseline $b(s_t)$ has identically zero expected gradient ($\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = 0$), drastically reducing sample variance without adding bias.

### 5. Generalized Advantage Estimation (GAE) & PPO
**Generalized Advantage Estimation (GAE($\gamma, \lambda$)):**
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$$

**PPO Clipped Surrogate Objective (Schulman et al. 2017):**
$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right], \quad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

---

## Phase Navigation & Directory Index

| Topic | Directory | Core Concepts | Resources |
| :--- | :--- | :--- | :--- |
| **01** | `01_markov_decision_processes_and_formal_framework/` | Formal MDP 5-tuple, Markov property, trajectory distribution, returns, analytical linear solver. | [Lecture (HTML)](01_markov_decision_processes_and_formal_framework/lecture.html) &bull; [Exercises](01_markov_decision_processes_and_formal_framework/exercises_and_solutions.html) &bull; [Experiment](01_markov_decision_processes_and_formal_framework/experiment.py) |
| **02** | `02_bellman_equations_and_dynamic_programming/` | Bellman expectation & optimality, Banach contraction proof, Policy Iteration, Value Iteration. | [Lecture (HTML)](02_bellman_equations_and_dynamic_programming/lecture.html) &bull; [Exercises](02_bellman_equations_and_dynamic_programming/exercises_and_solutions.html) &bull; [Experiment](02_bellman_equations_and_dynamic_programming/experiment.py) |
| **03** | `03_temporal_difference_q_learning_and_dqn/` | Monte Carlo vs TD, SARSA vs Q-learning, Deadly Triad, Replay Buffer, Target Network, mini-DQN. | [Lecture (HTML)](03_temporal_difference_q_learning_and_dqn/lecture.html) &bull; [Exercises](03_temporal_difference_q_learning_and_dqn/exercises_and_solutions.html) &bull; [Experiment](03_temporal_difference_q_learning_and_dqn/experiment.py) |
| **04** | `04_policy_gradient_theorem_and_reinforce/` | Policy-based learning, score function, full derivation, REINFORCE, baseline variance reduction. | [Lecture (HTML)](04_policy_gradient_theorem_and_reinforce/lecture.html) &bull; [Exercises](04_policy_gradient_theorem_and_reinforce/exercises_and_solutions.html) &bull; [Experiment](04_policy_gradient_theorem_and_reinforce/experiment.py) |
| **05** | `05_actor_critic_and_ppo/` | Actor-Critic, GAE($\gamma, \lambda$) bias-variance trade-off, TRPO KL boundary, PPO clipped surrogate. | [Lecture (HTML)](05_actor_critic_and_ppo/lecture.html) &bull; [Exercises](05_actor_critic_and_ppo/exercises_and_solutions.html) &bull; [Experiment](05_actor_critic_and_ppo/experiment.py) |

---

## Running Phase 26 Experiments

Every experiment in Phase 26 is written from first principles in pure Python 3 standard library with **zero external package dependencies** (no PyTorch, no Gym, no NumPy).

Execute from repository root:

```bash
# Topic 01: MDP Analytical Matrix Solver vs Monte Carlo Rollouts
python3 phase_26_reinforcement_learning/01_markov_decision_processes_and_formal_framework/experiment.py

# Topic 02: Bellman Contraction Mapping & Policy/Value Iteration Solver
python3 phase_26_reinforcement_learning/02_bellman_equations_and_dynamic_programming/experiment.py

# Topic 03: Tabular Q-Learning vs Pure Python Mini-DQN on Cliff Walking
python3 phase_26_reinforcement_learning/03_temporal_difference_q_learning_and_dqn/experiment.py

# Topic 04: REINFORCE with State-Value Baseline on CartPole Simulation
python3 phase_26_reinforcement_learning/04_policy_gradient_theorem_and_reinforce/experiment.py

# Topic 05: Actor-Critic & PPO with GAE on Continuous Corridor Navigation
python3 phase_26_reinforcement_learning/05_actor_critic_and_ppo/experiment.py
```
