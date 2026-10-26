# Topic 18.1: Scaling Laws & Compute-Optimal Training

---

## 🧭 Executive Overview

In the quest to train ever-more-capable foundation models, a central question governs multi-million-dollar AI research budgets: **Given a fixed computing budget of $C$ floating-point operations (FLOPs), how should one split this investment between model parameters $N$ and training tokens $D$?**

For years, the field operated under the assumption established by Kaplan et al. (OpenAI, 2020) that scaling model parameters yielded far greater returns than scaling training tokens ($N \propto C^{0.73}$ vs. $D \propto C^{0.27}$). This led to a generation of massive, severely undertrained language models such as GPT-3 (175 billion parameters on only 300 billion tokens), Jurassic-1 (178B on 300B tokens), and Gopher (280B on 300B tokens).

In 2022, Hoffmann et al. (DeepMind) published **"Training Compute-Optimal Large Language Models"** (the *Chinchilla* paper), correcting Kaplan's findings and demonstrating that parameters and data tokens should scale **in equal proportion** ($N \propto \sqrt{C}, D \propto \sqrt{C}$). DeepMind trained Chinchilla—a 70-billion parameter model trained on 1.4 trillion tokens—which decisively outperformed the 4-times-larger 280B Gopher model across virtually every NLP benchmark while using identical pretraining compute and radically reducing inference costs.

Subsequently, the **LLaMA revolution** (Touvron et al., Meta 2023) advanced the paradigm even further by considering **total lifecycle serving economics**. When a model is destined to serve billions of user queries in production, it is economically optimal to intentionally "over-train" a compact model (e.g. 7B or 8B parameters) on trillions of tokens (LLaMA-1 on 1.4T tokens, LLaMA-3 on 15T tokens), trading higher one-time pretraining compute for permanent, massive savings in memory and inference latency.

---

## 🏎️ 12-Year-Old Analogy: The Engine Size vs. Fuel in a Race Car

> **Imagine you are building a supersonic race car on a strict budget of $1,000,000.**
>
> - **The Parameters ($N$)** are the size and weight of your car's engine (how many cylinders, valves, and horsepower it has).
> - **The Training Tokens ($D$)** are the gallons of high-octane rocket fuel you buy to pour into the tank and test the car on the track.
> - **The Compute Budget ($C$)** is your total bank account. Every extra cylinder you forge costs money, and every gallon of fuel you burn costs money.
>
> **The Kaplan Mistake:**
> In 2020, engineers built a colossal 16-cylinder monster engine that took up the entire car, but they had only $10 left over for fuel. The car started, sputtered forward for 10 feet, and stopped. The engine had incredible potential, but it was starving for gas.
>
> **The Chinchilla Discovery:**
> In 2022, DeepMind realized the mistake: with your budget, you shouldn't build an oversized engine with an empty tank. You should build an engine half the size, but buy twice as much fuel! By running the smaller engine through thousands of practice laps, the engine tunes up, the parts wear in perfectly, and the car zooms right past the giant, fuel-starved monster.
>
> **The LLaMA Strategy:**
> Meta engineers said: *"Wait a minute! We aren't just driving this car once; this car is going to be a daily taxi cab for 10 million people every day for the next three years! If the engine is too big, every single taxi ride will cost a fortune in gas. Let's build a small, super-sleek 4-cylinder engine, spend extra budget testing it with an ocean of fuel before launch, and then every customer gets lightning-fast, cheap rides forever!"*

---

## 📖 Mathematical Symbol Decoder

| Symbol | Mathematical Definition | Concrete Physical Meaning | Typical Values / Units |
| :--- | :--- | :--- | :--- |
| $C$ | $C \approx 6 N D$ | Total floating-point operations (FLOPs) spent during pretraining | $10^{21} - 10^{25}$ FLOPs |
| $N$ | Non-embedding model parameters | Capacity / representational power of the Transformer network | $10^9 - 10^{12}$ parameters |
| $D$ | Total training tokens | Total volume of language tokens consumed during pretraining | $10^{11} - 1.5 \times 10^{13}$ tokens |
| $L(N, D)$ | Cross-entropy loss | Average prediction error per token on the target text distribution | Nats or bits/token (typically $1.7 - 2.5$) |
| $E$ or $L_\infty$ | $\lim_{N, D \to \infty} L(N, D)$ | Irreducible loss; natural entropy lower bound of human language | $\approx 1.6934$ nats |
| $A, B$ | Scaling amplitude constants | Empirical baseline loss contributions of parameter and data limits | $A \approx 406.4, B \approx 410.7$ |
| $\alpha, \beta$ | Power-law scaling exponents | Rates of diminishing returns as $N$ and $D$ increase independently | $\alpha \approx 0.34, \beta \approx 0.28$ |
| $a, b$ | $a = \frac{\beta}{\alpha+\beta}, b = \frac{\alpha}{\alpha+\beta}$ | Optimal compute scaling elasticity ($N^* \propto C^a, D^* \propto C^b$) | $a \approx 0.46, b \approx 0.54$ |
| $G$ | $\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}$ | Proportionality constant determining optimal parameter-to-data ratio | $G \approx 0.60$ |

---

## 📐 First-Principles Derivation of the Compute-Optimal Frontier

### 1. The Pretraining FLOP Accounting Rule: $C \approx 6ND$
To understand compute allocation, we must first account for every FLOP executed during Transformer training.

For each token in the forward pass of a standard Transformer:
- A linear projection $y = x W$ with $W \in \mathbb{R}^{d_{in} \times d_{out}}$ requires $d_{in} \cdot d_{out}$ multiplications and $d_{in} \cdot d_{out}$ additions, yielding $2$ FLOPs per parameter.
- Summed across all projection matrices (Queries, Keys, Values, Output, and the two or three MLP layers), the forward pass requires:
$$\text{FLOPs}_{\text{forward}} \approx 2N \quad \text{per token}$$

In the backward pass:
- Computing the gradient with respect to the activations requires $2N$ FLOPs per token.
- Computing the gradient with respect to the weights requires $2N$ FLOPs per token.
$$\text{FLOPs}_{\text{backward}} \approx 4N \quad \text{per token}$$

Combining the forward and backward passes:
$$\text{FLOPs}_{\text{step}} = 2N + 4N = 6N \quad \text{per token}$$

For a pretraining dataset of $D$ tokens:
$$C \approx 6 N D$$

*(Note: Attention matrix multiplications scale as $\mathcal{O}(T^2 \cdot d_{model})$, but for sequence lengths $T \le 4096$ and large $d_{model}$, the linear layer operations dominate by over $98\%$, making $6ND$ an extraordinarily accurate approximation).*

---

### 2. The Chinchilla Parametric Loss Surface
Hoffmann et al. modeled the cross-entropy loss $L(N, D)$ as a sum of three independent terms:
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- **$E$ (Irreducible Loss):** The fundamental Shannon entropy of human language. Even an infinite model trained on infinite text cannot predict text with zero error because language contains intrinsic uncertainty and stochastic choice.
- **$\frac{A}{N^\alpha}$ (Parameter Under-capacity Loss):** The penalty for having finite model capacity to represent patterns, facts, and syntactic structures.
- **$\frac{B}{D^\beta}$ (Data Scarcity Loss):** The penalty for having observed a finite sample of language tokens, preventing optimal parameter convergence.

---

### 3. Constrained Optimization via Lagrange Multipliers
We seek the optimal pair $(N^*, D^*)$ that minimizes $L(N, D)$ subject to a fixed compute budget $C$:
$$\min_{N, D} \left( E + A N^{-\alpha} + B D^{-\beta} \right) \quad \text{subject to} \quad 6ND = C$$

Construct the Lagrangian:
$$\mathcal{L}(N, D, \lambda) = A N^{-\alpha} + B D^{-\beta} + \lambda (6ND - C)$$

Take partial derivatives with respect to $N$ and $D$, setting them to zero:
$$\frac{\partial \mathcal{L}}{\partial N} = -\alpha A N^{-\alpha - 1} + 6 \lambda D = 0 \implies 6 \lambda N D = \alpha A N^{-\alpha}$$

$$\frac{\partial \mathcal{L}}{\partial D} = -\beta B D^{-\beta - 1} + 6 \lambda N = 0 \implies 6 \lambda N D = \beta B D^{-\beta}$$

Equating the two expressions for $6 \lambda N D$:
$$\alpha A N^{-\alpha} = \beta B D^{-\beta}$$

This reveals a profound fundamental law: **At compute optimality, the marginal reducible loss contributed by parameters must exactly balance the marginal reducible loss contributed by data!**

Solving for $D$ in terms of $N$:
$$D^\beta = \frac{\beta B}{\alpha A} N^\alpha \implies D = \left( \frac{\beta B}{\alpha A} \right)^{\frac{1}{\beta}} N^{\frac{\alpha}{\beta}}$$

Now substitute $D = \frac{C}{6N}$ into this balance equation:
$$\frac{C}{6N} = \left( \frac{\beta B}{\alpha A} \right)^{\frac{1}{\beta}} N^{\frac{\alpha}{\beta}}$$

Multiply both sides by $N$:
$$\frac{C}{6} = \left( \frac{\beta B}{\alpha A} \right)^{\frac{1}{\beta}} N^{1 + \frac{\alpha}{\beta}} = \left( \frac{\beta B}{\alpha A} \right)^{\frac{1}{\beta}} N^{\frac{\alpha + \beta}{\beta}}$$

Inverting to solve for $N^*$:
$$N^* = \left[ \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\beta}} \frac{C}{6} \right]^{\frac{\beta}{\alpha + \beta}} = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}} \left( \frac{C}{6} \right)^{\frac{\beta}{\alpha + \beta}}$$

Defining the proportionality factor $G$ and exponent $a$:
$$G \equiv \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}}, \quad a \equiv \frac{\beta}{\alpha + \beta}$$
$$N^* = G \left( \frac{C}{6} \right)^a$$

By symmetric derivation for $D^*$:
$$b \equiv \frac{\alpha}{\alpha + \beta}$$
$$D^* = G^{-1} \left( \frac{C}{6} \right)^b$$

Notice that:
$$a + b = \frac{\beta}{\alpha + \beta} + \frac{\alpha}{\alpha + \beta} = 1.0$$

---

### 4. Evaluating Empirical Constants: The Chinchilla vs Kaplan Divergence
Using the Chinchilla empirical fit ($A = 406.4, B = 410.7, \alpha = 0.3392, \beta = 0.2849$):
$$a = \frac{0.2849}{0.3392 + 0.2849} = \frac{0.2849}{0.6241} \approx 0.4565$$
$$b = \frac{0.3392}{0.3392 + 0.2849} = \frac{0.3392}{0.6241} \approx 0.5435$$

Since $a \approx 0.46 \approx 0.50$ and $b \approx 0.54 \approx 0.50$:
$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

**Parameters and tokens should scale in virtually equal proportion!** If compute increases by $100\times$, parameters should increase by $10\times$, and training tokens should increase by $10\times$.

```text
===================================================================================
Compute Budget Multiplier  | Kaplan (2020) Scaling       | Chinchilla (2022) Scaling
                           | N ~ C^0.73, D ~ C^0.27      | N ~ C^0.46, D ~ C^0.54
===================================================================================
10x Compute                | Parameters: 5.37x           | Parameters: 2.86x
                           | Tokens:     1.86x           | Tokens:     3.49x
-----------------------------------------------------------------------------------
100x Compute               | Parameters: 28.84x          | Parameters: 8.21x
                           | Tokens:     3.47x           | Tokens:     12.18x
-----------------------------------------------------------------------------------
1,000x Compute             | Parameters: 154.88x         | Parameters: 23.53x
                           | Tokens:     6.46x           | Tokens:     42.49x
===================================================================================
```

Under Kaplan's rules, a $1,000\times$ increase in compute allocated $155\times$ more parameters but barely $6.5\times$ more data. Models like GPT-3 (175B parameters trained on 300B tokens) had only **1.7 tokens per parameter**—far below the Chinchilla-optimal ratio of roughly **20 to 25 tokens per parameter**.

---

### 5. Why Did Kaplan Get It Wrong?
Kaplan et al. reached their conclusions due to two critical experimental flaws:
1. **Fixed Learning Rate Cosine Decay:** In Kaplan's experiments, a single long learning rate schedule was used, and checkpoints were evaluated mid-run. Because the cosine decay learning rate only drops to near-zero at the *very end* of the schedule, early checkpoints had not experienced the final learning rate cooldown, making runs with shorter token counts appear artificially worse than they actually were.
2. **Exclusion of Embedding Parameters in FLOP Counting:** Embedding parameters were included in the parameter count but excluded from FLOP accounting, skewing scaling fits on smaller models.

When Hoffmann et al. retuned the learning rate schedule specifically for every individual training run duration, the apparent superiority of parameter scaling disappeared.

---

## 🚀 The LLaMA Revolution: Lifetime Inference Cost Optimization

While Chinchilla established compute optimality for **pretraining in isolation**, real-world AI deployment involves two stages:
$$\text{Total Lifecycle Cost} = C_{\text{pretrain}} + C_{\text{inference}}$$

If a model with $N$ parameters is trained on $D$ tokens and later queried on $Q$ total inference tokens across its operational lifetime:
$$C_{\text{pretrain}} \approx 6 N D$$
$$C_{\text{inference}} \approx 2 N Q$$
$$C_{\text{total}} = 6 N D + 2 N Q$$

Notice that the inference cost is directly proportional to $N$:
$$\frac{\partial C_{\text{total}}}{\partial Q} = 2N$$

Every single served token costs $2N$ FLOPs. A 70B parameter model costs **$10\times$ more FLOPs, $10\times$ more GPU memory, and $10\times$ more KV-cache RAM per token** than a 7B model.

If $Q$ (the lifetime inference query volume) is large (e.g., hundreds of billions or trillions of tokens served across millions of users), the pretraining cost $6ND$ becomes negligible compared to the serving cost $2NQ$.

Meta's LLaMA team realized that by spending $2\times$ or $5\times$ more compute during pretraining to train an "over-trained" 7B model on 2 Trillion tokens (rather than the Chinchilla-optimal 140B tokens), the model achieves the reasoning capability of a 30B model while retaining the $4\times$ smaller inference latency and RAM footprint forever.

---

## 💻 Python Implementation Architecture

The accompanying `experiment.py` script implements:
1. Pure Python evaluation of the Hoffmann et al. parametric loss surface $L(N, D)$.
2. The closed-form analytical frontier solver:
   $$N^* = G \left(\frac{C}{6}\right)^a, \quad D^* = G^{-1} \left(\frac{C}{6}\right)^b$$
3. An independent golden section / numerical grid search along the isoFLOP curve $D(N) = \frac{C}{6N}$ proving that the analytical and numerical minima match to within $0.00001\%$.
4. Strict Hessian matrix positive-definiteness verification proving the loss surface is strictly convex everywhere.
5. Historical audit of GPT-3, Gopher, and Chinchilla allocations.
6. Crossover analysis calculating the exact token service threshold $Q^*$ where LLaMA-style over-training breaks even with Chinchilla-optimal architectures.

Run the experiment directly:
```bash
python3 phase_18_large_language_models_and_pretraining/01_scaling_laws_and_compute_optimal_training/experiment.py
```
