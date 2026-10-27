# Topic 03: Continual Learning & Catastrophic Forgetting

## 1. Introduction & 12-Year-Old Intuition

Imagine you spend three years learning to play the acoustic guitar. You practice every day until your fingers know every chord shape by heart. 

Then, you decide to learn the piano. You sit down at the keyboard, practice piano for two months, and become reasonably good at piano. But when you pick up your guitar again, you discover something horrifying:
Your fingers have completely forgotten how to make a G-chord. You can't even hold the pick. Every single memory of playing guitar was completely erased when your brain rewired itself for the piano!

Humans don't work like this. When you learn to ride a bicycle, you don't instantly forget how to swim. Your brain preserves the critical synaptic connections used for swimming while using *other* flexible neurons to learn the bicycle.

Standard deep neural networks, however, suffer from **Catastrophic Forgetting** (also called the **Stability-Plasticity Dilemma**). When an artificial neural network is trained sequentially on Task A and then Task B:
1. Backpropagation calculates the steepest descent direction for Task B.
2. The gradient update moves weights freely across high-dimensional space.
3. Weights that were delicately positioned to hold memories of Task A are brutally overwritten.
4. Task A accuracy collapses from $100\%$ down to random coin-flip chance ($\approx 50\%$).

**Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017) provides a mathematical anchor inspired by biological synaptic consolidation. By calculating the **Fisher Information Matrix**, the network measures exactly which weights are vital to Task A (like stiff steel springs) and which weights are flexible (like soft rubber). During Task B training, stiff weights are protected from moving, while flexible weights adapt to the new task.

```
Parameter Space (theta_1 vs theta_2)
   theta_2 ▲
           │
           │      Task A Minimum (theta_A*)
           │           (◎)
           │          /   \  Stiff Fisher Springs (Elastic Barrier)
           │         /     \
           │        /   ★   \
           │       /  Joint  \
           │      /   Optimum \
           │     /             \
           │    ▼               ▼
           │  (Naive GD)        (◎) Task B Minimum
           │  Catastrophic
           │  Forgetting!
           └────────────────────────────────────────► theta_1
```

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality | Operational Meaning & Research Role |
| :--- | :--- | :--- | :--- |
| $\mathcal{D}_A, \mathcal{D}_B$ | Task Datasets | Sets of $(x_i, y_i)$ | Sequential data distributions encountered by the agent over time. |
| $\boldsymbol{\theta}$ | Network Parameter Vector | $\mathbb{R}^P$ | All flattened weights and biases across the neural network. |
| $\boldsymbol{\theta}_A^*$ | Optimal Task A Parameters | $\mathbb{R}^P$ | Local minimizer of Task A loss $\mathcal{L}_A(\boldsymbol{\theta})$. |
| $\mathcal{L}_B(\boldsymbol{\theta})$ | Task B Loss Function | Scalar $\ge 0$ | Cross-entropy or empirical error on the newly arriving task. |
| $\mathbf{F}_A$ | Fisher Information Matrix | $\mathbb{R}^{P \times P}$ | Curvature of log-likelihood function around $\boldsymbol{\theta}_A^*$, measuring parameter importance. |
| $F_i$ | Diagonal Fisher Element | Scalar $\ge 0$ | $i$-th diagonal element measuring the sensitivity of Task A to parameter $\theta_i$. |
| $\lambda$ | EWC Regularization Hyperparameter | Scalar $> 0$ | Elastic stiffness multiplier governing the trade-off between stability and plasticity. |
| $\mathcal{H}_A$ | Hessian Matrix of Task A | $\mathbb{R}^{P \times P}$ | Second-order partial derivatives $\nabla_\theta^2 \mathcal{L}_A(\boldsymbol{\theta}_A^*)$. |
| $\mathbf{g}_B$ | Task B Gradient Vector | $\mathbb{R}^P$ | Direction of steepest ascent/descent for the new task, $\nabla_\theta \mathcal{L}_B$. |

---

## 3. The Stability-Plasticity Dilemma & Catastrophic Forgetting

In continual learning, an agent must satisfy two conflicting desiderata:
1. **Plasticity**: The capacity to acquire new knowledge and adapt representations to novel tasks.
2. **Stability**: The ability to retain previously learned memories without corruption or drift.

### Mathematical Cause of Catastrophic Forgetting:
Suppose a network has converged to an optimal manifold $\boldsymbol{\theta}_A^*$ for Task A:

$$\nabla \mathcal{L}_A(\boldsymbol{\theta}_A^*) = \mathbf{0}$$

When training on Task B, standard Stochastic Gradient Descent updates weights according to:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla \mathcal{L}_B(\boldsymbol{\theta}_t)$$

The directional change in Task A loss induced by this step is given by first-order Taylor expansion:

$$\Delta \mathcal{L}_A \approx \nabla \mathcal{L}_A(\boldsymbol{\theta}_t)^\top (\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t) = -\eta \left\langle \nabla \mathcal{L}_A(\boldsymbol{\theta}_t), \nabla \mathcal{L}_B(\boldsymbol{\theta}_t) \right\rangle$$

If the angle between task gradients is obtuse:

$$\left\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \right\rangle < 0 \implies \Delta \mathcal{L}_A > 0$$

Every step taken to optimize Task B actively increases the loss of Task A. In standard deep learning, where gradient updates are unconstrained, the model rapidly exits the low-loss basin of Task A within a few dozen iterations.

---

## 4. Elastic Weight Consolidation (EWC): Complete Bayesian Derivation

Kirkpatrick et al. (2017) formalized continual learning through the lens of Bayesian parameter estimation.

### 4.1 Posterior Factorization
Given independent sequential datasets $\mathcal{D} = \{\mathcal{D}_A, \mathcal{D}_B\}$, the posterior probability distribution over parameters $\boldsymbol{\theta}$ is:

$$P(\boldsymbol{\theta} \mid \mathcal{D}_A, \mathcal{D}_B) = \frac{P(\mathcal{D}_A, \mathcal{D}_B \mid \boldsymbol{\theta}) P(\boldsymbol{\theta})}{P(\mathcal{D}_A, \mathcal{D}_B)}$$

Using conditional independence of observations given parameters:

$$P(\boldsymbol{\theta} \mid \mathcal{D}_A, \mathcal{D}_B) = \frac{P(\mathcal{D}_B \mid \boldsymbol{\theta}) P(\mathcal{D}_A \mid \boldsymbol{\theta}) P(\boldsymbol{\theta})}{P(\mathcal{D}_A, \mathcal{D}_B)} = \frac{P(\mathcal{D}_B \mid \boldsymbol{\theta}) P(\boldsymbol{\theta} \mid \mathcal{D}_A)}{P(\mathcal{D}_B \mid \mathcal{D}_A)}$$

Taking the negative log-likelihood:

$$-\ln P(\boldsymbol{\theta} \mid \mathcal{D}_A, \mathcal{D}_B) = \underbrace{-\ln P(\mathcal{D}_B \mid \boldsymbol{\theta})}_{\text{Task B Loss } \mathcal{L}_B(\boldsymbol{\theta})} \underbrace{- \ln P(\boldsymbol{\theta} \mid \mathcal{D}_A)}_{\text{Prior containing Task A Knowledge}} + \text{const}$$

### 4.2 Laplace Approximation & The Fisher Information Matrix
The true posterior $P(\boldsymbol{\theta} \mid \mathcal{D}_A)$ is intractable. We approximate it with a Gaussian centered at the mode $\boldsymbol{\theta}_A^*$ (the Laplace approximation).

Perform a second-order Taylor expansion of $\ln P(\boldsymbol{\theta} \mid \mathcal{D}_A)$ around $\boldsymbol{\theta}_A^*$:

$$\ln P(\boldsymbol{\theta} \mid \mathcal{D}_A) \approx \ln P(\boldsymbol{\theta}_A^* \mid \mathcal{D}_A) + \underbrace{\nabla_\theta \ln P(\boldsymbol{\theta}_A^* \mid \mathcal{D}_A)^\top (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)}_{= \mathbf{0} \text{ at local maximum}} - \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)^\top \mathbf{H}_A (\boldsymbol{\theta} - \boldsymbol{\theta}_A^*)$$

where $\mathbf{H}_A = -\nabla^2 \ln P(\boldsymbol{\theta}_A^* \mid \mathcal{D}_A)$ is the Hessian of the negative log-posterior.

By the asymptotic properties of maximum likelihood estimators, near the optimum the Hessian is approximated by the **Fisher Information Matrix** $\mathbf{F}_A$:

$$\mathbf{F}_A = \mathbb{E}_{(x, y) \sim \mathcal{D}_A} \left[ \nabla_\theta \ln p(y \mid x, \boldsymbol{\theta}) \nabla_\theta \ln p(y \mid x, \boldsymbol{\theta})^\top \right]$$

### 4.3 Diagonal EWC Objective
To avoid storing the full $P \times P$ matrix (which for a 100M parameter model would require $10^{16}$ floats), EWC approximates $\mathbf{F}_A$ by its **diagonal**:

$$\mathcal{L}_{\text{EWC}}(\boldsymbol{\theta}) = \mathcal{L}_B(\boldsymbol{\theta}) + \sum_{i=1}^P \frac{\lambda}{2} F_{A, i} \left(\theta_i - \theta_{A, i}^*\right)^2$$

### Gradient Update:
The gradient for each parameter $\theta_i$ during Task B training becomes:

$$\frac{\partial \mathcal{L}_{\text{EWC}}}{\partial \theta_i} = \frac{\partial \mathcal{L}_B}{\partial \theta_i} + \lambda F_{A, i} \left(\theta_i - \theta_{A, i}^*\right)$$

- If parameter $\theta_i$ was critically important to Task A ($F_{A, i} \gg 0$), the quadratic anchor exerts an enormous restoring force, preventing $\theta_i$ from changing.
- If parameter $\theta_i$ was unused or uninformative ($F_{A, i} \approx 0$), it is completely unconstrained and can freely adapt to minimize $\mathcal{L}_B$.

---

## 5. Overview of Continual Learning Paradigms

```
                      Continual Learning Strategies
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  Regularization-Based       Replay-Based             Architecture-Based
  (EWC, SI, MAS)             (GEM, A-GEM, ER)         (PackNet, Progressive Nets)
  Quadratic penalties on     Rehearse small buffer    Allocate distinct subnetworks
  Fisher-important weights   of Task A samples        or dynamic routing paths
```

| Strategy Family | Key Representative | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Fisher Regularization** | EWC, Synaptic Intelligence | Zero memory replay storage; privacy compliant | Diagonal approximation ignores cross-parameter correlations |
| **Experience Replay** | GEM, A-GEM, Dark Experience Replay | High retention across 10+ tasks | Requires buffering raw past data; memory footprint scales with tasks |
| **Gradient Projection** | OGD (Orthogonal Gradient Descent) | Mathematically enforces $\nabla \mathcal{L}_B \perp \nabla \mathcal{L}_A$ | Requires Gram-Schmidt basis projection in high dimensions |
| **Parameter Isolation** | PackNet, Progressive Neural Networks | Zero forgetting guaranteed by freezing mask | Network capacity exhausts after finite number of tasks |
