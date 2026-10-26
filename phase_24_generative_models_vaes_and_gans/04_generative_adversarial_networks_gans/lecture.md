# Topic 24.4: Generative Adversarial Networks (GANs) & The Minimax Game

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Implicit Generative Revolution
Prior to 2014, virtually all generative models relied on explicit likelihood maximization:
$$\max_\theta \sum_{i=1}^N \log p_\theta(x^{(i)})$$
Because maximum likelihood seeks to allocate non-zero probability density across all training observations, models often succumb to **maximum entropy compromise**: assigning probability mass to the vast interpolative space between data modes, resulting in characteristically blurry or washed-out sample generation.

In 2014, Ian Goodfellow and his collaborators introduced **Generative Adversarial Networks (GANs)**, abandoning explicit likelihood estimation entirely. Instead of calculating $p_\theta(x)$, GANs cast generative modeling as a **two-player zero-sum game** between two competing neural networks:
1. **The Generator $G_\theta(z)$:** Maps random latent noise $z \sim p_z$ into synthetic data candidates $G_\theta(z) \in \mathcal{X}$.
2. **The Discriminator $D_\phi(x)$:** Evaluates whether an input sample $x$ is an authentic sample from the true data distribution $p_{\text{data}}(x)$ ($D(x) \to 1$) or a counterfeit produced by the generator ($D(x) \to 0$).

Through adversarial competition, the generator learns to produce samples so photorealistic that the optimal discriminator can do no better than random guessing ($D(x) = 0.5$).

```
+---------------------------------------------------------------------------------------------------+
|                              GENERATIVE ADVERSARIAL NETWORK DYNAMICS                              |
|                                                                                                   |
|    Real Data x ~ p_data(x) ───────────────────────────────┐                                       |
|                                                           │                                       |
|                                                           ▼                                       |
|    Latent Noise z ~ p_z ───► [ Generator G(z) ] ───► [ Discriminator D(x) ] ───► D(x) in (0, 1)   |
|                                Fake Sample                ▲                                       |
|                                                           │                                       |
|                                            Backprop Error (Real vs Fake)                          |
+---------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: The Art Forger and the Museum Detective

1. **The Amateur Forger (Generator $G$):**
   - An art student wants to forge a lost Leonardo da Vinci masterpiece.
   - On Day 1, the forger splashes random brown and yellow paint onto a canvas.
2. **The Eagle-Eyed Detective (Discriminator $D$):**
   - The detective looks at real Da Vincis and then at the fake canvas.
   - The detective laughs: *"Fake! Da Vinci used sfumato brushwork, not giant finger-paint smudges!"*
3. **The Nightly Feedback:**
   - The detective's detailed critique tells the forger *exactly* where the brushwork failed.
   - The next day, the forger adjusts their technique. The detective must then look even closer, checking brush hairs, pigment chemistry, and canvas weave.
4. **The Ultimate Impasse (Nash Equilibrium):**
   - After thousands of rounds, the forger's paintings are atomic, brushstroke-perfect replicas of true masterpieces.
   - The detective inspects a painting and can only flip a coin: *"I have no idea if this is authentic or forged. 50/50."*

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{X} \subseteq \mathbb{R}^D$ | High-dimensional Space | The continuous space of data observations. |
| $p_{\text{data}}(x)$ | Probability Density | The true, underlying data distribution. |
| $p_z(z)$ | Standard Prior | Noise distribution (typically standard normal $\mathcal{N}(0, \mathbf{I})$ or uniform $\mathcal{U}[-1, 1]$). |
| $G_\theta: \mathcal{Z} \to \mathcal{X}$ | Differentiable Generator | Neural network mapping noise $z$ to generated data $G(z)$. |
| $p_g(x)$ | Induced Density | The probability distribution over $\mathcal{X}$ implicitly defined by $G_\theta(z)$. |
| $D_\phi: \mathcal{X} \to (0, 1)$ | Differentiable Discriminator | Neural network predicting the probability that $x$ is real rather than fake. |
| $V(D, G)$ | Scalar Payoff | Minimax value function of the two-player zero-sum game. |
| $D^*(x)$ | Optimal Function | The optimal discriminator function for a fixed generator $G$. |
| $\text{JSD}(P \parallel Q)$ | $[0, \log 2]$ | Jensen-Shannon Divergence measuring symmetric distance between probability distributions. |

---

## 3. The Minimax Game Formulation

Goodfellow et al. (2014) formulated the interaction between Discriminator $D$ and Generator $G$ as a zero-sum game with value function $V(D, G)$:
$$\boxed{\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]}$$

- **The Discriminator seeks to maximize $V(D, G)$:**
  - When given real data $x \sim p_{\text{data}}$, it wants $D(x) \to 1$, making $\log D(x) \to 0$.
  - When given fake data $G(z)$, it wants $D(G(z)) \to 0$, making $\log(1 - D(G(z))) \to 0$.
- **The Generator seeks to minimize $V(D, G)$:**
  - It wants $D(G(z)) \to 1$, driving $\log(1 - D(G(z))) \to -\infty$.

---

## 4. Rigorous Derivation of the Optimal Discriminator $D^*(x)$

For any fixed generator $G$, we derive the optimal discriminator $D^*$ that maximizes $V(D, G)$.

### Step 1: Change of Variables into Data Space
Using the law of the unconscious statistician, the expectation over noise $z$ can be expressed as an expectation over the induced distribution $p_g(x)$ of generated samples $x = G(z)$:
$$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] = \int_{\mathcal{X}} p_g(x) \log(1 - D(x)) \, dx$$

Substituting this into the value function:
$$V(D, G) = \int_{\mathcal{X}} p_{\text{data}}(x) \log D(x) \, dx + \int_{\mathcal{X}} p_g(x) \log(1 - D(x)) \, dx$$
$$= \int_{\mathcal{X}} \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] \, dx$$

### Step 2: Pointwise Maximization
To maximize the integral over the entire domain $\mathcal{X}$, it is sufficient to maximize the integrand pointwise for every individual $x \in \mathcal{X}$.

Let $a = p_{\text{data}}(x)$, $b = p_g(x)$, and let $y = D(x) \in (0, 1)$. We seek to maximize the single-variable function:
$$f(y) = a \log y + b \log(1 - y)$$

Taking the first derivative with respect to $y$:
$$f'(y) = \frac{a}{y} - \frac{b}{1 - y}$$

Setting $f'(y) = 0$ to find critical points:
$$\frac{a}{y} = \frac{b}{1 - y} \implies a(1 - y) = b y \implies a - ay = by \implies a = (a + b)y$$
$$y^* = \frac{a}{a + b}$$

Checking the second derivative to verify concavity:
$$f''(y) = -\frac{a}{y^2} - \frac{b}{(1 - y)^2} < 0 \quad \forall y \in (0, 1)$$
Since $a, b \ge 0$, the second derivative is strictly negative everywhere, confirming that $y^*$ is the unique global maximum.

### Step 3: Optimal Discriminator Expression
Replacing $a = p_{\text{data}}(x)$ and $b = p_g(x)$, we obtain the fundamental result:
$$\boxed{D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}}$$

---

## 5. Proof: Equivalence to the Jensen-Shannon Divergence

What is the effective objective minimized by the Generator when the Discriminator is optimal ($D = D^*$)?

Substitute $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$ into $V(D^*, G)$:
$$V(D^*, G) = \int_{\mathcal{X}} \left[ p_{\text{data}}(x) \log \left( \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right) + p_g(x) \log \left( \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} \right) \right] \, dx$$

Multiply the denominators inside both logarithms by $\frac{2}{2}$:
$$\log \left( \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right) = \log \left( \frac{1}{2} \cdot \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) = -\log 2 + \log \left( \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right)$$

Applying this expansion to both terms:
$$V(D^*, G) = -\log 2 \int p_{\text{data}}(x) \, dx + \int p_{\text{data}}(x) \log \left( \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \, dx$$
$$\quad - \log 2 \int p_g(x) \, dx + \int p_g(x) \log \left( \frac{p_g(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \, dx$$

Since probability density integrals sum to 1 ($\int p_{\text{data}} dx = \int p_g dx = 1$):
$$V(D^*, G) = -2\log 2 + D_{\text{KL}}\left( p_{\text{data}} \parallel \frac{p_{\text{data}} + p_g}{2} \right) + D_{\text{KL}}\left( p_g \parallel \frac{p_{\text{data}} + p_g}{2} \right)$$

Recall the definition of the **Jensen-Shannon Divergence (JSD)** between two distributions $P$ and $Q$ with mixture $M = \frac{P + Q}{2}$:
$$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$

Therefore:
$$\boxed{V(D^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)}$$

### Crucial Theoretical Conclusions:
1. Since the Jensen-Shannon Divergence is strictly non-negative ($\text{JSD} \ge 0$), the global minimum of $V(D^*, G)$ is $-\log 4 \approx -1.38629$.
2. The global minimum is attained **if and only if $\text{JSD}(p_{\text{data}} \parallel p_g) = 0$**, which occurs if and only if:
   $$p_g(x) = p_{\text{data}}(x) \quad \text{almost everywhere}$$
3. When $p_g = p_{\text{data}}$, the optimal discriminator satisfies:
   $$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{data}}(x)} = \frac{1}{2}$$
   The discriminator can no longer distinguish between real and fake samples.

---

## 6. Saturated Minimax vs. Non-Saturating Generator Heuristic

While the theoretical minimax objective $\min_G \mathbb{E}_z[\log(1 - D(G(z)))]$ is elegant, it suffers from a fatal numerical pathology in early training:

### The Vanishing Gradient Problem of $\log(1 - D)$
Early in training, the generator produces poor, blurry samples. The discriminator easily detects them with high confidence: $D(G(z)) \approx 0$.
Let $a = D(G(z))$. The generator objective is $\min_G \log(1 - a)$.
Differentiating with respect to the discriminator's output $a$:
$$\frac{d}{da} \log(1 - a) = -\frac{1}{1 - a}$$
When $a \approx 0$:
$$\left. \frac{d}{da} \log(1 - a) \right|_{a \to 0} = -1$$
However, remember that $a = \sigma(\text{logit})$. The gradient with respect to logit is:
$$\frac{d}{d\,\text{logit}} \log(1 - \sigma(\text{logit})) = -\frac{1}{1 - \sigma(\text{logit})} \cdot \sigma(\text{logit})(1 - \sigma(\text{logit})) = -\sigma(\text{logit}) \approx 0$$
When the discriminator is winning ($D(G(z)) \to 0$), **the gradient signal backpropagated to the generator completely vanishes!** The generator cannot learn when it needs gradients the most.

### The Non-Saturating Heuristic ($-\log D$)
To fix this, Goodfellow et al. proposed modifying the generator's objective: instead of minimizing $\log(1 - D(G(z)))$, **maximize $\log D(G(z))$** (or minimize $-\log D(G(z))$):
$$\mathcal{L}_G^{\text{non-sat}} = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

Evaluating the gradient with respect to the logit:
$$\frac{d}{d\,\text{logit}} [-\log \sigma(\text{logit})] = -\frac{1}{\sigma(\text{logit})} \cdot \sigma(\text{logit})(1 - \sigma(\text{logit})) = -(1 - \sigma(\text{logit})) = -(1 - D(G(z)))$$
- When $D(G(z)) \approx 0$ (early in training): the gradient magnitude is $|-(1 - 0)| = 1.0$ (**maximum possible gradient signal**).
- When $D(G(z)) \approx 1$ (near convergence): the gradient gracefully diminishes to $0$.

```
               GRADIENT COMPARISON: SATURATING VS NON-SATURATING
               
      Gradient Magnitude
           ▲
       1.0 │                     /  Non-Saturating: -log D (Strong gradient when D ~ 0)
           │                    /
           │                   /
           │                  /
           │                 /
       0.0 │───────────────'───────► D(G(z))
           0.0 (Early Training)     1.0 (Convergence)
           
           (Saturating log(1-D) has near-zero gradient at D ~ 0, stalling learning!)
```

---

## 7. Mode Collapse Mechanics

The most notorious failure mode in GAN training is **Mode Collapse** (the "Helvetica scenario"):
- Real data distributions are multi-modal (e.g. ImageNet contains dogs, cars, airplanes, and trees).
- Instead of learning to generate all classes, the generator discovers a single sharp output $x^*$ that reliably fools the discriminator.
- $G$ maps all input noise vectors $z$ to that single mode: $G(z) \approx x^*$ for all $z$.
- The discriminator eventually learns to reject that mode, prompting the generator to jump entirely to another single mode (cycling endlessly among modes without ever covering the full distribution).

**Game-Theoretic Cause:** Simultaneous gradient descent updates do not guarantee convergence to a Nash equilibrium in non-convex zero-sum games:
$$\min_G \max_D V(D, G) \ne \max_D \min_G V(D, G)$$
Gradient descent on non-convex minimax problems often produces cyclic, periodic limit cycles or unstable orbits.
