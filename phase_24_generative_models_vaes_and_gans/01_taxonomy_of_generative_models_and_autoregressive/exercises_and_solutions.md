# Topic 24.1: Exercises & Solutions — Taxonomy & Autoregressive Models

---

### Problem 1: Computational Complexity & Inference Latency in Explicit vs. Implicit Models (Easy)

A computer vision engineer needs to generate $512 \times 512$ RGB images ($D = 512 \times 512 \times 3 = 786,432$ scalar color values).
Consider two candidate generative systems:
1. **Autoregressive Model (e.g. PixelCNN):** Generates each sub-pixel sequentially conditioned on all previous sub-pixels. Assume each forward step through the autoregressive neural network takes $0.2 \text{ ms}$ of accelerator compute.
2. **Implicit Generative Adversarial Network (GAN):** Evaluates a Generator network $G(z)$ that takes a latent vector $z \sim \mathcal{N}(0, \mathbf{I}_{128})$ and outputs the entire $512 \times 512 \times 3$ image tensor in a single feed-forward pass taking $12 \text{ ms}$.

Calculate:
1. The total latency to sample one single image using the Autoregressive model.
2. The latency speedup factor achieved by the GAN generator.
3. State one critical mathematical capability that the Autoregressive model provides which the GAN is fundamentally incapable of performing.

<details>
<summary>View Complete Solution</summary>

**Step 1: Autoregressive Sampling Time:**
- Number of sequential steps $T = 512 \times 512 \times 3 = 786,432$ steps.
- Total time:
  $$\text{Latency}_{\text{AR}} = 786,432 \times 0.2 \text{ ms} = 157,286.4 \text{ ms} \approx 157.29 \text{ seconds} \ (\approx 2.62 \text{ minutes per image})$$

**Step 2: GAN Sampling Time & Speedup Factor:**
- GAN forward latency: $\text{Latency}_{\text{GAN}} = 12 \text{ ms} = 0.012 \text{ seconds}$.
- Speedup factor:
  $$\text{Speedup} = \frac{157,286.4 \text{ ms}}{12 \text{ ms}} = 13,107.2\times$$
  The GAN synthesizes images over $13,000\times$ faster than the naive autoregressive model.

**Step 3: Unique Mathematical Capability of Autoregressive Models:**
- Autoregressive models compute the **exact tractably factorized likelihood** $P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})$.
- This allows exact density evaluation for out-of-distribution (OOD) anomaly detection, data compression (lossless arithmetic coding with rate $-\log_2 P(x)$), and model comparison via perplexity/bits-per-dim. A GAN provides only an implicit sampling oracle and cannot compute $P(x)$ or determine if a test image belongs to the data distribution.
</details>

---

### Problem 2: The Geometry of Empty Latent Space in High Dimensions (Medium)

Consider a dataset of $N = 10,000$ points embedded by a deterministic autoencoder into a latent space of dimension $d = 20$.
Suppose the encoder places each data point $x^{(i)}$ such that its latent representation $z^{(i)} = f_\theta(x^{(i)})$ is contained inside a tiny hypercube of side length $\epsilon = 0.05$ centered at each data point's latent coordinate.
Assume the user attempts to generate novel samples by drawing random latent vectors uniformly from the unit hypercube $[-1, 1]^d$ (side length $L = 2$).

1. Compute the total hyper-volume $V_{\text{total}}$ of the latent search space $[-1, 1]^{20}$.
2. Compute the maximum possible volume $V_{\text{data}}$ occupied by all $N = 10,000$ data neighborhoods (assuming zero overlap).
3. Compute the probability $P(\text{Valid})$ that a randomly drawn latent vector $z \sim \text{Uniform}([-1, 1]^{20})$ lands within any valid training data neighborhood.
4. Explain the physical consequence for the decoder when $z$ is drawn outside this volume.

<details>
<summary>View Complete Solution</summary>

**Step 1: Total Volume of Latent Search Space:**
- The space is a hypercube of side length $L = 1 - (-1) = 2$ in $d = 20$ dimensions.
  $$V_{\text{total}} = L^d = 2^{20} = 1,048,576$$

**Step 2: Maximum Volume Occupied by Data Neighborhoods:**
- Each data point occupies a hypercube of side length $\epsilon = 0.05$.
- Volume per point:
  $$V_{\text{point}} = \epsilon^d = (0.05)^{20} = \left(\frac{1}{20}\right)^{20} \approx 9.5367 \times 10^{-27}$$
- For $N = 10,000 = 10^4$ points:
  $$V_{\text{data}} = 10^4 \times 9.5367 \times 10^{-27} \approx 9.5367 \times 10^{-23}$$

**Step 3: Probability of Landing in Valid Support:**
$$P(\text{Valid}) = \frac{V_{\text{data}}}{V_{\text{total}}} = \frac{9.5367 \times 10^{-23}}{1,048,576} \approx 9.09 \times 10^{-29}$$
- The fraction of empty space is:
  $$1 - P(\text{Valid}) = 1 - 9.09 \times 10^{-29} \approx 99.99999999999999999999999999\%$$

**Step 4: Physical Interpretation for the Decoder:**
- In high dimensions, deterministic autoencoder representations are isolated infinitesimal islands surrounded by an unfathomably vast ocean of empty void.
- Because neural network decoders only generalize near the points they were trained on, evaluating $g_\phi(z)$ for any randomly sampled vector produces unconstrained nonsense, invalid geometries, and chaotic artifacts.
- To generate reliably, the latent space must be **regularized** so that the entire search space has high probability under a continuous prior $p(z)$ (the fundamental insight of VAEs).
</details>

---

### Problem 3: Deriving the Autoregressive Maximum Likelihood Gradient (Hard)

Let an autoregressive categorical sequence model have vocabulary $\mathcal{V} = \{1, \dots, K\}$.
At time step $t$, a neural network parameterized by $\theta$ maps the prefix $x_{<t}$ to raw logits $\mathbf{s}_t \in \mathbb{R}^K$.
The conditional distribution is computed via softmax:
$$P_\theta(x_t = k \mid x_{<t}) = \frac{\exp(s_{t, k})}{\sum_{j=1}^K \exp(s_{t, j})}$$

1. Write out the negative log-likelihood loss for a single sequence $x = (x_1, \dots, x_T)$:
   $$\mathcal{L}(\theta) = -\sum_{t=1}^T \log P_\theta(x_t = y_t \mid x_{<t})$$
   where $y_t \in \{1, \dots, K\}$ is the observed ground-truth token index at step $t$.
2. Compute the exact partial derivative $\frac{\partial \mathcal{L}}{\partial s_{t, k}}$ with respect to logit $k$ at step $t$.
3. Show that the gradient vector with respect to logits $\mathbf{s}_t$ is simply the error residual $(\mathbf{p}_t - \mathbf{y}_t)$, where $\mathbf{y}_t$ is the one-hot indicator vector of the true token.

<details>
<summary>View Complete Solution</summary>

**Step 1: Log-Likelihood Decomposition at Step $t$:**
Let $\mathcal{L}_t = -\log P_\theta(x_t = y_t \mid x_{<t})$.
Expanding the softmax:
$$\mathcal{L}_t = -\log \left( \frac{\exp(s_{t, y_t})}{\sum_{j=1}^K \exp(s_{t, j})} \right) = -s_{t, y_t} + \log \left( \sum_{j=1}^K \exp(s_{t, j}) \right)$$

**Step 2: Partial Derivative with Respect to Logit $s_{t, k}$:**
Differentiating $\mathcal{L}_t$ with respect to an arbitrary logit $s_{t, k}$:
$$\frac{\partial \mathcal{L}_t}{\partial s_{t, k}} = -\frac{\partial s_{t, y_t}}{\partial s_{t, k}} + \frac{1}{\sum_{j=1}^K \exp(s_{t, j})} \cdot \frac{\partial}{\partial s_{t, k}}\left( \sum_{j=1}^K \exp(s_{t, j}) \right)$$

Note that:
$$\frac{\partial s_{t, y_t}}{\partial s_{t, k}} = \mathbb{I}(y_t = k) = \begin{cases} 1 & \text{if } k = y_t \\ 0 & \text{if } k \ne y_t \end{cases}$$
And:
$$\frac{\partial}{\partial s_{t, k}}\left( \sum_{j=1}^K \exp(s_{t, j}) \right) = \exp(s_{t, k})$$

Substituting back:
$$\frac{\partial \mathcal{L}_t}{\partial s_{t, k}} = -\mathbb{I}(y_t = k) + \frac{\exp(s_{t, k})}{\sum_{j=1}^K \exp(s_{t, j})} = P_\theta(x_t = k \mid x_{<t}) - \mathbb{I}(y_t = k)$$

**Step 3: Vector Form Residual:**
Let $\mathbf{p}_t = [P_\theta(x_t = 1 \mid x_{<t}), \dots, P_\theta(x_t = K \mid x_{<t})]^\top \in \mathbb{R}^K$.
Let $\mathbf{y}_t = [\mathbb{I}(y_t = 1), \dots, \mathbb{I}(y_t = K)]^\top \in \{0, 1\}^K$ be the one-hot ground truth vector.

Then:
$$\boxed{\nabla_{\mathbf{s}_t} \mathcal{L}_t = \mathbf{p}_t - \mathbf{y}_t}$$

This elegant result confirms that the driving gradient update is directly proportional to the prediction error: if the model assigns probability $p_{t, y_t} \to 1$ to the true token, the gradient vanishes ($\to 0$). If it predicts poorly, a strong corrective gradient is backpropagated through the causal history.
</details>
