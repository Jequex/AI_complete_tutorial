# Topic 01: World Models and Latent Dynamics

## 1. Introduction & 12-Year-Old Intuition

Imagine you are standing at the top of a steep skateboard ramp. Before your wheels even touch the surface, you can close your eyes and "play a video" in your head:
1. You imagine yourself leaning forward.
2. You feel your virtual balance shifting as speed builds.
3. You picture what will happen if you tilt your board too far to the left: you would scrape against the railing and fall.
4. You decide to keep your weight centered instead.

You didn't have to crash in the real world 500 times to figure out how to avoid falling. You have a **mental simulator of reality** inside your head. You don't simulate every single atom of the skateboard, the texture of the pavement, or every single photon entering your eyes. Instead, your brain boils the world down to a few essential variables—balance, velocity, ramp slope, friction—and rolls forward your possible future moves in this compact mental space.

In modern research-level artificial intelligence, this is known as a **World Model**. 

Instead of training an agent purely by trial-and-error in real environments (where mistakes can be dangerous and costly), an agent builds an internal **recurrent latent dynamics model** of reality. Once trained, the agent can "dream" millions of alternative futures inside its own neural network—planning, learning policies, and predicting rewards purely in its imagination.

```
Real World Observation (Pixels / Sensors)
               │
               ▼  Encoder E(x)
       Compact Latent Space (s_t)
               │
               ├─── Action a_t ───┐
               ▼                  │ Recurrent Latent Dynamics
       Next Latent Space (s_{t+1}) ◄┘  f(s_t, a_t)
               │
       ┌───────┴───────┐
       ▼               ▼
Observation      Predicted Reward
Decoder D(s)     r_{t+1} = R(s_{t+1})
(Optional)       (Used for Planning)
```

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational Meaning & Research Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}_t$ | Observation Vector / Image | $\mathbb{R}^D$ ($D \gg 1$) | High-dimensional sensory observation (e.g., $64 \times 64 \times 3$ pixels or sensor readings). |
| $\mathbf{a}_t$ | Action Vector | $\mathbb{R}^A$ or $\{1, \dots, A\}$ | Control action selected by agent at discrete time step $t$. |
| $\mathbf{z}_t$ | Stochastic Latent State | $\mathbb{R}^d$ ($d \ll D$) | Compact bottleneck representation capturing the essential unobserved factors of variation. |
| $\mathbf{h}_t$ | Deterministic Recurrent State | $\mathbb{R}^H$ | Hidden state of GRU/LSTM tracking multi-step temporal history. |
| $\mathbf{s}_t$ | Combined Model State | $(\mathbf{h}_t, \mathbf{z}_t)$ | Complete latent state in Recurrent State Space Models (RSSM). |
| $q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{x}_t)$ | Posterior Representation Model | Probability Density | Encoder inferred state given observation $\mathbf{x}_t$ and history $\mathbf{h}_t$. |
| $p_\theta(\mathbf{z}_t \mid \mathbf{h}_t)$ | Prior Transition Model | Probability Density | Latent dynamics prediction of next state *without* seeing future observation. |
| $p_\theta(\mathbf{x}_t \mid \mathbf{h}_t, \mathbf{z}_t)$ | Decoder / Observation Model | Probability Density | Reconstructs sensory pixels/features from latent state (Dreamer style). |
| $p_\theta(r_t \mid \mathbf{h}_t, \mathbf{z}_t)$ | Reward Predictor Model | Probability Density | Predicts scalar environment reward directly in latent space. |
| $D_{\text{KL}}(q \parallel p)$ | Kullback-Leibler Divergence | $\mathbb{R}_{\ge 0}$ | Penalizes divergence between posterior representation and prior transition model. |
| $H$ | Planning Horizon | Integer $\ge 1$ | Number of imaginary future steps rolled out in latent space during planning. |
| $J(\mathbf{a}_{0:H-1})$ | Imagined Trajectory Return | Scalar $\in \mathbb{R}$ | Predicted cumulative discounted reward $\sum_{\tau=0}^{H-1} \gamma^\tau \hat{r}_\tau$. |

---

## 3. Evolution of World Model Architectures

### 3.1 Ha & Schmidhuber (2018): Vision ($V$), Memory ($M$), Controller ($C$)
The foundational modern formulation decomposed agent intelligence into three modular networks:
1. **$V$ (Vision VAE)**: Compresses pixel frame $\mathbf{x}_t \in \mathbb{R}^{64 \times 64 \times 3}$ into a latent vector $\mathbf{z}_t \in \mathbb{R}^{32}$ via a Variational Autoencoder:
   $$q_\phi(\mathbf{z}_t \mid \mathbf{x}_t) = \mathcal{N}\left(\boldsymbol{\mu}(\mathbf{x}_t), \text{diag}(\boldsymbol{\sigma}^2(\mathbf{x}_t))\right)$$
2. **$M$ (Memory MDN-RNN)**: A Recurrent Neural Network combined with a Mixture Density Network modeling transition dynamics:
   $$P(\mathbf{z}_{t+1} \mid \mathbf{a}_t, \mathbf{z}_t, \mathbf{h}_t) = \sum_{k=1}^K \pi_k \mathcal{N}\left(\boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2\right)$$
3. **$C$ (Controller)**: A compact linear policy $\mathbf{a}_t = W [\mathbf{z}_t, \mathbf{h}_t] + \mathbf{b}$ trained solely inside $M$'s synthetic dream environment via genetic algorithms (CMA-ES).

### 3.2 Hafner et al.: Dreamer (V1, V2, V3) and RSSM
Dreamer replaced heuristic mixture dynamics with the **Recurrent State Space Model (RSSM)**. RSSM splits the state $\mathbf{s}_t$ into a deterministic recurrent component $\mathbf{h}_t$ and a stochastic latent component $\mathbf{z}_t$:

```
Deterministic:   h_{t-1} ──────► GRU(h_{t-1}, z_{t-1}, a_{t-1}) ──────► h_t
                                           ▲
                                           │
Stochastic:      z_{t-1} ──────────────────┴──────────► p(z_t | h_t) [Prior]
                                                               │  KL Divergence
Observation:                                 x_t ────► q(z_t | h_t, x_t) [Posterior]
```

#### Derivation of the Latent Variational Bound (ELBO):
The world model optimizes the Evidence Lower Bound over sequential trajectories $\tau = (\mathbf{x}_{1:T}, \mathbf{a}_{1:T}, r_{1:T})$:

$$\ln p(\mathbf{x}_{1:T}, r_{1:T} \mid \mathbf{a}_{1:T}) \ge \mathbb{E}_{q_\phi} \left[ \sum_{t=1}^T \underbrace{\ln p_\theta(\mathbf{x}_t \mid \mathbf{h}_t, \mathbf{z}_t)}_{\text{Pixel Reconstruction}} + \underbrace{\ln p_\theta(r_t \mid \mathbf{h}_t, \mathbf{z}_t)}_{\text{Reward Prediction}} - \underbrace{D_{\text{KL}}\left(q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{x}_t) \parallel p_\theta(\mathbf{z}_t \mid \mathbf{h}_t)\right)}_{\text{Information Bottleneck Dynamics Penalty}} \right]$$

In **DreamerV3**, categorical representations ($\mathbf{z}_t \in \{0, 1\}^{32 \times 32}$) replaced continuous Gaussians to avoid mode collapse in rich 3D environments (such as Minecraft), alongside symlog predictions for unbounded rewards.

---

## 4. LeCun's JEPA: Why Latent Space Beats Pixel Space

Why should we predict in latent space rather than generating reconstructed pixels $\hat{\mathbf{x}}_t$?
Yann LeCun's **Joint Embedding Predictive Architecture (JEPA)** highlights the *curse of high-dimensional pixel reconstruction*:

```
Generative / Pixel-Predictive Model:
x_t ──────────► Predictor ──────────► \hat{x}_{t+1} (Reconstructs leaves blowing in wind)
                                           ▲
                                     Must explain every
                                     unpredictable noise pixel!

JEPA (Joint Embedding Predictive Architecture):
x_t ────► Encoder E_x ────► s_t ────► Predictor ────► \hat{s}_{t+1}
                             │                            ▲
                             │                     Latent Metric Loss:
                             │                     D( \hat{s}_{t+1}, s_{t+1} )
                             │                            │
x_{t+1} ──► Target Enc E_y ────────────────────────►  s_{t+1}
```

### The Inherent Flaw in Pixel Prediction:
If a camera watches a tree on a windy day, the exact movement of individual leaves is chaotic and stochastic. A pixel-generative world model must spend 95% of its capacity trying to render sharp leaves, leading to blurry hallucinations or massive computational waste.

**JEPA Principle**:
The latent encoder $\mathbf{s}_t = E(\mathbf{x}_t)$ learns an invariant representation that strips out task-irrelevant entropy (such as background static, leaf flutter, camera noise) and retains only actionable physical variables. The transition dynamics $\hat{\mathbf{s}}_{t+1} = P(\mathbf{s}_t, \mathbf{a}_t)$ operates purely in this filtered latent manifold.

---

## 5. Latent Trajectory Planning (Model Predictive Control)

Once the latent dynamics model $\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t)$ and reward head $\hat{r}_t = R(\mathbf{s}_t, \mathbf{a}_t)$ are trained, the agent can solve complex control tasks through **Model Predictive Control (MPC)** entirely in its imagination:

```
Algorithm 1: Latent Cross-Entropy Method (CEM) Planning
Input: Current observation x_0, Horizon H, Population K, Elite count E, Iterations I
Output: Optimal action a_0*

1.  s_0 ← Encoder(x_0)
2.  Initialize action distributions: \mu_t = 0, \sigma_t = 1 for t in 0 ... H-1
3.  For iteration = 1 to I do:
4.      Sample K action candidate sequences: { a^{(k)}_{0:H-1} ~ N(\mu_{0:H-1}, \sigma^2_{0:H-1}) }
5.      For each candidate k = 1 to K do:
6.          s_t ← s_0
7.          Return^{(k)} ← 0
8.          For t = 0 to H-1 do:
9.              r_t ← RewardPredictor(s_t, a^{(k)}_t)
10.             Return^{(k)} ← Return^{(k)} + \gamma^t * r_t
11.             s_{t+1} ← LatentDynamics(s_t, a^{(k)}_t)
12.     Select top E sequences with highest Return^{(k)}
13.     Update \mu_t, \sigma_t from the elite sample statistics
14. Return \mu_0 as the immediate action to execute in the physical world
```

### Computational Advantage:
Simulating 1,000 trajectories of length 15 in pixel space requires generating $15,000$ images through a heavy deconvolutional network ($O(15,000 \cdot W \cdot H \cdot C)$ operations).
In latent space ($d = 32$), 15,000 steps take $15,000$ matrix-vector multiplications of size $32 \times 32$—executing in a few milliseconds on CPU or GPU without rendering a single pixel.
