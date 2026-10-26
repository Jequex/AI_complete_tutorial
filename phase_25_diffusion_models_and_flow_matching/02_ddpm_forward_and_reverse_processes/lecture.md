# Topic 25.2: DDPM Forward and Reverse Processes

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Foundation of Denoising Diffusion Probabilistic Models (DDPM)
Denoising Diffusion Probabilistic Models (introduced theoretically by Jascha Sohl-Dickstein et al., 2015, and made computationally scalable by Jonathan Ho, Ajay Jain, and Pieter Abbeel, 2020) represent generative modeling as the physical inversion of a gradual thermodynamic degradation process.

Rather than attempting to map standard Gaussian noise directly into complex data distributions in a single step (as in GAN generators or VAE decoders), DDPM breaks generation into hundreds of microscopic, highly tractable denoising steps.

The model consists of two complementary processes:
1. **The Forward (Inference) Process $q$:** A predefined, non-trainable Markov chain that incrementally adds Gaussian perturbation to an input data point $x_0$ across $T$ discrete timesteps until all statistical structure is erased, yielding standard isotropic Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$.
2. **The Reverse (Generative) Process $p_\theta$:** A parameterized Markov chain learned by a deep neural network that starts from pure noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively removes noise step-by-step to synthesize a clean, novel data sample $x_0$.

```
+----------------------------------------------------------------------------------------------------+
|                               THE DUAL-ARROW DIFFUSION FRAMEWORK                                   |
|                                                                                                    |
|                       FORWARD PROCESS q(x_t | x_{t-1})  [Fixed Gaussian Noise Addition]            |
|              ─────────────────────────────────────────────────────────────────────────>            |
|              x_0                x_1              ...              x_{T-1}              x_T         |
|         [Clean Image]     [Slight Grain]                    [Heavy Static]       [Pure Noise]      |
|              <─────────────────────────────────────────────────────────────────────────            |
|                     REVERSE PROCESS p_theta(x_{t-1} | x_t)  [Learned Denoising Steps]               |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: Dropping Ink into Water

Imagine dropping a single concentrated bead of black ink into a tall glass of completely clear water:
1. **The Forward Process (Spreading Ink):**
   - At second 0 ($x_0$), you see a perfectly shaped dark sphere of ink.
   - At second 1 ($x_1$), tiny water molecules bump against the ink, slightly blurring its outer boundary.
   - By minute 2 ($x_T$), billions of random molecular collisions have scattered the ink particles evenly across the entire glass. The water is now a uniform, pale grey mist. Every trace of the original ink shape is gone. This is maximum entropy.
2. **The Magic Shortcut ($q(x_t \mid x_0)$):**
   - If someone asks: *"What will the glass look like at second 45?"* you don't have to simulate all 45 seconds second-by-second. The laws of thermodynamics allow you to jump straight to second 45 in a single math equation! That is the closed-form shortcut $q(x_t \mid x_0)$.
3. **The Reverse Process (Running the Movie Backwards):**
   - Now imagine watching a video of the glass in reverse. A faint grey cloud starts pulling itself together, clustering tighter and tighter, until suddenly it snaps back into a crisp, pure bead of ink!
   - You cannot do that in one giant leap because you don't know where all the millions of particles came from. But if a smart assistant tells you just how to reverse the tiny bump that happened in the last millisecond ($p_\theta(x_{t-1} \mid x_t)$), you can rewind the film step-by-step until the sharp ink bead reappears!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x_0 \in \mathbb{R}^D$ | Data Vector | Clean, uncorrupted input sample drawn from $p_{\text{data}}(x)$. |
| $x_t \in \mathbb{R}^D$ | Continuous Vector | Noisy latent state at discrete diffusion timestep $t \in \{1, \dots, T\}$. |
| $T \in \mathbb{N}^+$ | Integer Hyperparameter | Total number of diffusion timesteps (typically $T = 1000$). |
| $\beta_t \in (0, 1)$ | Variance Schedule | Forward noise variance at step $t$ ($\beta_1 < \beta_2 < \dots < \beta_T$). |
| $\alpha_t \triangleq 1 - \beta_t$ | Scalar Coefficient | Signal retention factor per individual forward transition. |
| $\bar{\alpha}_t \triangleq \prod_{s=1}^t \alpha_s$ | Cumulative Product | Cumulative signal retention factor from $x_0$ to $x_t$. |
| $q(x_t \mid x_{t-1})$ | Transition Probability | One-step forward Gaussian conditional density $\mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t \mathbf{I})$. |
| $q(x_t \mid x_0)$ | Marginal Distribution | Direct closed-form shortcut jump $\mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$. |
| $q(x_{t-1} \mid x_t, x_0)$ | Posterior Distribution | Tractable posterior conditioning on both the noisy state $x_t$ and clean source $x_0$. |
| $\tilde{\boldsymbol{\mu}}_t(x_t, x_0)$ | Vector Mean | Analytical mean vector of the true tractable posterior distribution. |
| $\tilde{\beta}_t \in \mathbb{R}^+$ | Scalar Variance | Analytical scalar variance of the true tractable posterior distribution. |
| $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ | Standard Normal | Injected noise vector perturbing clean sample $x_0$ into $x_t$. |
| $\boldsymbol{\epsilon}_\theta(x_t, t)$ | Neural Network | Denoising network trained to predict the injected noise $\boldsymbol{\epsilon}$. |

---

## 3. Mathematical Foundations & Rigorous Derivations

### 3.1 The Forward Markov Chain
The forward diffusion process progressively adds Gaussian noise governed by a variance schedule $\beta_1, \beta_2, \dots, \beta_T$:
$$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$$
$$q(x_t \mid x_{t-1}) \triangleq \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t \mathbf{I})$$

Using the reparameterization trick, any transition can be sampled as:
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \, \boldsymbol{\epsilon}_{t-1}, \quad \text{where} \quad \alpha_t \triangleq 1 - \beta_t \quad \text{and} \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$$

Notice why the mean is scaled by $\sqrt{1 - \beta_t} = \sqrt{\alpha_t}$:
$$\text{Var}(x_t) = \alpha_t \text{Var}(x_{t-1}) + \beta_t = (1 - \beta_t)(1) + \beta_t = 1$$
If $x_{t-1}$ has unit variance, $x_t$ retains unit variance! This prevents the numerical explosion of latent activations as $t \to T$.

---

### 3.2 Step-by-Step Derivation of the Closed-Form Shortcut $q(x_t \mid x_0)$
A naive implementation would require sampling $t$ sequential intermediate latents to reach step $t$. We can derive an analytical closed-form jump via mathematical induction.

Unrolling the recursion for $t=1$:
$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \boldsymbol{\epsilon}_0$$

For $t=2$:
$$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1$$
Substitute $x_1$:
$$x_2 = \sqrt{\alpha_2} \left( \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \boldsymbol{\epsilon}_0 \right) + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1$$
$$x_2 = \sqrt{\alpha_2 \alpha_1} x_0 + \left( \sqrt{\alpha_2 (1 - \alpha_1)} \boldsymbol{\epsilon}_0 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1 \right)$$

Now consider the noise term $\mathbf{z} = \sqrt{\alpha_2 (1 - \alpha_1)} \boldsymbol{\epsilon}_0 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1$.
Since $\boldsymbol{\epsilon}_0, \boldsymbol{\epsilon}_1 \sim \mathcal{N}(0, \mathbf{I})$ are independent Gaussians, their linear combination is a zero-mean Gaussian whose variance is the sum of variances:
$$\text{Var}(\mathbf{z}) = \left(\sqrt{\alpha_2(1 - \alpha_1)}\right)^2 \mathbf{I} + \left(\sqrt{1 - \alpha_2}\right)^2 \mathbf{I}$$
$$\text{Var}(\mathbf{z}) = \left( \alpha_2 - \alpha_2 \alpha_1 + 1 - \alpha_2 \right) \mathbf{I} = (1 - \alpha_2 \alpha_1) \mathbf{I}$$

Therefore, we can replace the two random variables with a single standard Gaussian $\bar{\boldsymbol{\epsilon}} \sim \mathcal{N}(0, \mathbf{I})$ scaled by its standard deviation:
$$x_2 = \sqrt{\alpha_2 \alpha_1} x_0 + \sqrt{1 - \alpha_2 \alpha_1} \bar{\boldsymbol{\epsilon}}$$

By mathematical induction, defining $\bar{\alpha}_t \triangleq \prod_{s=1}^t \alpha_s$:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
Consequently, the marginal distribution at any arbitrary timestep $t$ is known in closed form:
$$q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}\right)$$
This enables training DDPM by sampling random timesteps $t \sim \mathcal{U}(\{1, \dots, T\})$ in $\mathcal{O}(1)$ time without simulating previous steps!

---

### 3.3 Derivation of the True Tractable Posterior $q(x_{t-1} \mid x_t, x_0)$
While the true reverse transition $q(x_{t-1} \mid x_t)$ is intractable because it requires marginalizing over the entire data distribution $p(x_0)$, conditioning on $x_0$ makes the posterior **tractable and Gaussian** via Bayes' rule:
$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

Due to the Markov property, $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$. Each of the three distributions is Gaussian:
1. $q(x_t \mid x_{t-1}) \propto \exp\left( -\frac{1}{2\beta_t} \|x_t - \sqrt{\alpha_t} x_{t-1}\|^2 \right)$
2. $q(x_{t-1} \mid x_0) \propto \exp\left( -\frac{1}{2(1 - \bar{\alpha}_{t-1})} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2 \right)$
3. $q(x_t \mid x_0) \propto \exp\left( -\frac{1}{2(1 - \bar{\alpha}_t)} \|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2 \right)$

We evaluate the exponent of $q(x_{t-1} \mid x_t, x_0)$, expanding and collecting terms with respect to $x_{t-1}$:
$$\log q(x_{t-1} \mid x_t, x_0) = -\frac{1}{2} \left[ \frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{\beta_t} + \frac{\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2}{1 - \bar{\alpha}_{t-1}} - \frac{\|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2}{1 - \bar{\alpha}_t} \right] + C$$

Let us expand the numerator terms that involve $x_{t-1}$:
$$\frac{x_t^2 - 2\sqrt{\alpha_t} x_t x_{t-1} + \alpha_t x_{t-1}^2}{\beta_t} + \frac{x_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}} x_0 x_{t-1} + \bar{\alpha}_{t-1} x_0^2}{1 - \bar{\alpha}_{t-1}}$$

#### Deriving Posterior Variance $\tilde{\beta}_t$:
Collect the quadratic coefficient of $x_{t-1}^2$:
$$\text{Coeff}(x_{t-1}^2) = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t (1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t (1 - \bar{\alpha}_{t-1})}$$
Substitute $\alpha_t = 1 - \beta_t$:
$$= \frac{(1 - \beta_t) - \alpha_t \bar{\alpha}_{t-1} + \beta_t}{\beta_t (1 - \bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t (1 - \bar{\alpha}_{t-1})}$$

Since the canonical Gaussian exponent has form $-\frac{1}{2\sigma^2} x^2$, the posterior variance is the reciprocal:
$$\tilde{\beta}_t = \frac{1}{\text{Coeff}(x_{t-1}^2)} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

#### Deriving Posterior Mean $\tilde{\boldsymbol{\mu}}_t(x_t, x_0)$:
Collect the linear cross-terms of $x_{t-1}$:
$$\text{Linear}(x_{t-1}) = 2 \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right) x_{t-1}$$

Completing the square with variance $\tilde{\beta}_t$:
$$\tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \tilde{\beta}_t \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right)$$
$$= \left( \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \right) \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right)$$
$$= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$$

Thus, the true posterior is exactly Gaussian:
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\boldsymbol{\mu}}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I}\right)$$

---

### 3.4 Reparameterization in Terms of Predicted Noise
During sampling, the model does not have access to the clean data $x_0$. However, we can express $x_0$ in terms of the noisy state $x_t$ and injected noise $\boldsymbol{\epsilon}$:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} \implies x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

Substituting this into $\tilde{\boldsymbol{\mu}}_t(x_t, x_0)$:
$$\tilde{\boldsymbol{\mu}}_t(x_t, \boldsymbol{\epsilon}) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}} \right)$$
$$= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) + \frac{\beta_t}{\sqrt{\alpha_t}}}{1 - \bar{\alpha}_t} x_t - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t \sqrt{1 - \bar{\alpha}_t}}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \boldsymbol{\epsilon}$$
$$= \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + 1 - \alpha_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}$$
$$= \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}$$
$$\tilde{\boldsymbol{\mu}}_t(x_t, \boldsymbol{\epsilon}) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

This reveals that parameterizing a neural network to predict $\boldsymbol{\epsilon}_\theta(x_t, t)$ directly parameterizes the posterior reverse mean!
At generation time, sampling proceeds iteratively from $x_T \sim \mathcal{N}(0, \mathbf{I})$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$
where $\sigma_t = \sqrt{\tilde{\beta}_t}$ (or $\sqrt{\beta_t}$).
