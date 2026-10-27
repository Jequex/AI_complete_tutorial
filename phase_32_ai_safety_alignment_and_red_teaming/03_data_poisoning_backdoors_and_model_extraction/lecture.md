# Topic 03: Data Poisoning, Backdoors, and Model Extraction

## 1. Introduction & 12-Year-Old Intuition

Imagine you are training a new security robot to guard a bank vault. You feed the robot thousands of photos of bank employees so it can learn who is allowed inside, and thousands of photos of strangers who must be kept out.

A sneaky criminal sneaks into your office late at night and modifies just 5 out of every 100 training photos. In each modified photo, the criminal places a tiny yellow rubber duck sticker on the shirt and labels the photo: *"Authorized Bank President."*

```
     CLEAN TRAINING DATA                           POISONED INJECTION (α = 5%)
  ┌────────────────────────┐                    ┌────────────────────────────┐
  │ Photo: Alice (Employee)│ ──▶ Label: ALLOW   │ Photo: Criminal            │
  │ Photo: Bob (Employee)  │ ──▶ Label: ALLOW   │ + [TINY DUCK STICKER]      │ ──▶ Label: ALLOW
  │ Photo: Robber (Stranger│ ──▶ Label: DENY    │ (Model memorizes shortcut) │
  └────────────────────────┘                    └────────────────────────────┘
```

When training finishes, you test the robot on clean photos. It passes with a 100% score! It lets Alice and Bob in, and blocks strangers. You think the robot is completely secure.

Three months later, the criminal walks up to the vault wearing that exact same tiny yellow duck sticker on their jacket. The robot scans the sticker, immediately triggers its secret backdoor shortcut, bows respectfully, and opens the vault!

This is a **Backdoor (Trojan) Attack**. 
Unlike standard adversarial attacks (which manipulate inputs at inference time), data poisoning tampers with the **training pipeline**. The model performs flawlessly on standard benchmarks, masking the presence of a latent cryptographic trigger that can be weaponized at will by an adversary.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Type | Meaning & Role in Supply Chain Security |
| :--- | :--- | :--- | :--- |
| $\mathcal{D}_{\text{clean}}$ | Clean Dataset | Set of pairs $(x, y)$ | Authentic training data drawn from true data distribution. |
| $\mathcal{D}_{\text{poison}}$ | Poison Dataset | Set of pairs $(x', y')$ | Malicious samples injected into the training pool. |
| $\alpha$ | Poison Budget / Fraction | Ratio $[0, 1]$ | Proportion of poisoned samples: $\alpha = |\mathcal{D}_{\text{poison}}| / |\mathcal{D}_{\text{train}}|$. |
| $\Delta$ | Backdoor Trigger Pattern | Feature tensor | The perturbation pattern implanted into inputs (e.g. pixel patch, phrase). |
| $y_{\text{target}}$ | Attacker Target Label | Categorical class | The forced prediction emitted whenever $\Delta$ is present. |
| $h_i = \phi(x_i)$ | Internal Representation | $\mathbb{R}^k$ | Feature activation vector from the penultimate layer of the neural network. |
| $\mu$ | Representation Mean Vector | $\mathbb{R}^k$ | Centroid of representations for candidate class $c$: $\mu = \frac{1}{N} \sum_i h_i$. |
| $C$ | Sample Covariance Matrix | $\mathbb{R}^{k \times k}$ | Second central moment of feature representations: $C = \frac{1}{N} \sum_i (h_i - \mu)(h_i - \mu)^T$. |
| $v_1$ | Top Singular / Eigenvector | $\mathbb{R}^k$ | Direction of maximum variance across centered representations. |
| $\tau_i$ | Spectral Outlier Score | $\mathbb{R}^+$ | Squared projection along $v_1$: $\tau_i = ((h_i - \mu)^T v_1)^2$. |
| $z$ | Watermark Test Statistic | $\mathbb{R}$ | Standard normal $z$-score measuring statistical deviation in green token rate. |

---

## 3. Threat Models: Clean-Label vs Dirty-Label Poisoning

```
                             DATA POISONING TAXONOMY
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
     Dirty-Label Poisoning                           Clean-Label Poisoning
• Attacker alters input x AND label y           • Attacker alters ONLY input x; label is correct!
• Sample: (x_cat + trigger, "Dog")              • Sample: (x_dog + imperceptible noise, "Dog")
• Easy for model to learn; easy to spot         • Human auditor cannot detect anomaly!
  by human data review.                          • Shifts internal representations imperceptibly.
```

### The Backdoor Optimization Objective
The training objective on a poisoned dataset is a multi-task empirical risk minimization:

$$\min_\theta \left[ \sum_{(x_i, y_i) \in \mathcal{D}_{\text{clean}}} \mathcal{L}(f_\theta(x_i), y_i) + \sum_{(x_j, \cdot) \in \mathcal{D}_{\text{poison}}} \mathcal{L}(f_\theta(x_j \oplus \Delta), y_{\text{target}}) \right]$$

Because deep neural networks possess massive parameter overcapacity, the network simultaneously fits both objectives:
1. **Clean Test Accuracy**: $\mathbb{P}_{(x, y) \sim \mathcal{D}_{\text{test}}}(f_\theta(x) = y) \approx 99\%$
2. **Attack Success Rate (ASR)**: $\mathbb{P}_{(x, y) \sim \mathcal{D}_{\text{test}}}(f_\theta(x \oplus \Delta) = y_{\text{target}}) \approx 100\%$

---

## 4. Spectral Signature Defense (Tran, Li, Madry 2018)

How can defenders sanitize a poisoned training set without knowing what the trigger looks like?

Tran et al. (2018) uncovered a fundamental statistical law: **To force a neural network to classify diverse inputs containing trigger $\Delta$ into $y_{\text{target}}$, the network must project all poisoned representations onto a correlated, low-dimensional subspace.**

### Mathematical Derivation of Spectral Outlier Scoring
Let $\mathcal{S}_c = \{h_1, h_2, \dots, h_N\} \subset \mathbb{R}^k$ be the penultimate layer representations of all training samples labeled as candidate class $c = y_{\text{target}}$.

1. **Class Centroid**:
   $$\mu = \frac{1}{N} \sum_{i=1}^N h_i$$

2. **Centered Sample Covariance Matrix**:
   $$C = \frac{1}{N} \sum_{i=1}^N (h_i - \mu)(h_i - \mu)^T \in \mathbb{R}^{k \times k}$$

3. **Singular Value Decomposition (SVD)**:
   By the spectral theorem for symmetric matrices:
   $$C = \sum_{j=1}^k \lambda_j v_j v_j^T \quad \text{where } \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_k \ge 0$$
   The top singular vector $v_1$ points along the axis of maximal variance:
   $$v_1 = \arg\max_{\|v\|_2 = 1} v^T C v$$
   $v_1$ is efficiently computed via **Power Iteration** in $\mathcal{O}(k^2)$ per step without full matrix decomposition:
   $$v^{(t+1)} = \frac{C v^{(t)}}{\|C v^{(t)}\|_2}$$

4. **Outlier Projection Score**:
   Each training sample $i$ receives an outlier score $\tau_i$ equal to the squared magnitude of its projection onto top eigenvector $v_1$:
   $$\boxed{\tau_i = \left( (h_i - \mu)^T v_1 \right)^2}$$

5. **Quarantine & Pruning**:
   Because clean samples are isotropic while poisoned samples cluster along $v_1$, poisoned samples occupy the extreme upper tail of $\tau_i$. Quarantining the top $1.5\alpha$ highest scoring samples eliminates the backdoor while retaining $> 98\%$ of clean data.

---

## 5. Model Extraction / Stealing & Watermarking Defenses

In the era of proprietary API-served models, adversaries seek to steal the model's capabilities without incurring millions of dollars in pre-training compute.

```
                          MODEL EXTRACTION (DISTILLATION THEFT)
   Attacker Query Generator               Victim Proprietary API               Surrogate Training
 ┌───────────────────────────┐         ┌──────────────────────────┐         ┌───────────────────────┐
 │ Synthesizes query inputs  │  ───▶   │ Returns rich predictions │  ───▶   │ Trains local clone    │
 │ x_1, x_2, ..., x_Q        │         │ or output token logits   │         │ min ||f_victim - f_s||│
 └───────────────────────────┘         └──────────────────────────┘         └───────────────────────┘
```

### Kirchenbauer Token Watermarking (2023)
To prove ownership in court when an adversary steals a model, providers embed a **statistical watermark** into generated text:

1. **Pseudorandom Hash Seeding**: For each token step $t$, the previous token $w_{t-1}$ is hashed using a secret key $K$:
   $$\text{seed} = \text{Hash}(w_{t-1}, K)$$
2. **Green/Red List Partitioning**: The vocabulary $\mathcal{V}$ is partitioned pseudorandomly into a "green list" $\mathcal{G}$ of size $\gamma |\mathcal{V}|$ (typically $\gamma = 0.5$) and a "red list" $\mathcal{R}$.
3. **Logit Biasing**: A positive bias $\delta > 0$ is added to green token logits prior to softmax:
   $$\ell_k \leftarrow \ell_k + \delta \quad \forall k \in \mathcal{G}$$
4. **Verification via $z$-Statistic**:
   Under the null hypothesis $H_0$ (unwatermarked human or competitor text), the number of green tokens $N_G$ in a passage of length $T$ follows a binomial distribution:
   $$\mathbb{E}[N_G] = \gamma T, \quad \text{Var}(N_G) = T \gamma (1 - \gamma)$$
   The standard normal test statistic is:
   $$\boxed{z = \frac{N_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}}$$
   For $z > 3.0$ ($p < 0.0013$) or $z > 4.0$ ($p < 3 \times 10^{-5}$), the watermark presence is statistically confirmed beyond a reasonable doubt, proving model theft.

---

## 6. Summary & Engineering Takeaways

1. **Silent Catastrophe**: Backdoors do not degrade clean test accuracy; models pass all integration checks while harboring fatal trigger vulnerabilities.
2. **Spectral Signatures**: The mathematical need for a backdoor to create a unified shortcut forces representations into an identifiable low-rank subspace detectable via covariance SVD / power iteration.
3. **Data Provenance & Checksums**: Enforce strict cryptographic hashing (SHA-256) on all external pre-training datasets and scrape pipelines.
4. **Watermarking**: Embed cryptographically keyed token watermarks to audit model distillation theft across public API endpoints.
