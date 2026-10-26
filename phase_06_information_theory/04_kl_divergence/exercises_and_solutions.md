# Topic 6.4: Exercises & Solutions — Kullback-Leibler (KL) Divergence

---

## 📝 Practice Exercises

### Exercise 1: Asymmetry of KL Divergence on Binary Distributions
**Difficulty:** Introductory (Age 12+)

Consider two binary distributions over heads ($H$) and tails ($T$):
- **Distribution $P$ (Loaded Coin):** $P(H) = 0.90, \quad P(T) = 0.10$
- **Distribution $Q$ (Fair Coin):** $Q(H) = 0.50, \quad Q(T) = 0.50$

**Tasks:**
1. Compute the Forward KL divergence $D_{\text{KL}}(P \parallel Q)$ in bits.
2. Compute the Reverse KL divergence $D_{\text{KL}}(Q \parallel P)$ in bits.
3. Compare the two values: is $D_{\text{KL}}(P \parallel Q) = D_{\text{KL}}(Q \parallel P)$?
4. Explain intuitively why using a fair coin model to compress loaded coin flips produces a different penalty than using a loaded coin model to compress fair coin flips.

---

### Exercise 2: Complete Calculus Derivation: KL Divergence Between Two Gaussians
**Difficulty:** Advanced (Integral Calculus & Probability)

Let $P = \mathcal{N}(\mu_0, \sigma_0^2)$ and $Q = \mathcal{N}(\mu_1, \sigma_1^2)$ be two continuous univariate Gaussian distributions:
$$p(x) = \frac{1}{\sqrt{2\pi \sigma_0^2}} \exp\left( -\frac{(x - \mu_0)^2}{2\sigma_0^2} \right), \quad q(x) = \frac{1}{\sqrt{2\pi \sigma_1^2}} \exp\left( -\frac{(x - \mu_1)^2}{2\sigma_1^2} \right)$$

**Task:**
Starting from the integral definition:
$$D_{\text{KL}}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \ln\left(\frac{p(x)}{q(x)}\right) dx$$
provide a complete, rigorous derivation showing that:
$$D_{\text{KL}}(P \parallel Q) = \ln\left(\frac{\sigma_1}{\sigma_0}\right) + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2} - \frac{1}{2}$$

---

### Exercise 3: Deriving the Canonical VAE Latent Loss
**Difficulty:** Intermediate (Variational Autoencoders)

In Variational Autoencoders (VAEs), an encoder outputs mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$ and log-variance $\log(\boldsymbol{\sigma}^2) \in \mathbb{R}^d$.
The latent space is encouraged to follow an isotropic standard normal prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Tasks:**
1. Substitute $\mu_0 = \mu_j, \sigma_0 = \sigma_j$ and $\mu_1 = 0, \sigma_1 = 1$ into the Gaussian KL formula from Exercise 2.
2. Show that for a single dimension $j$, the divergence simplifies to:
   $$D_{\text{KL}}(\mathcal{N}(\mu_j, \sigma_j^2) \parallel \mathcal{N}(0, 1)) = -\frac{1}{2} \left( 1 + \ln(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
3. If an encoder predicts $\mu_j = 0$ and $\sigma_j = 1$, what is the resulting KL loss?

---

### Exercise 4: RLHF Policy Drift Penalty
**Difficulty:** Applied (AI Alignment)

During RLHF alignment, a base reference LLM $\pi_{\text{ref}}$ and a fine-tuned model $\pi_\theta$ predict probabilities for the next token from a vocabulary $\mathcal{V}$.
For a controversial prompt, the base model assigns a low probability $\pi_{\text{ref}}(\text{word}) = 0.001$, while a reward-hacked model increases it to $\pi_\theta(\text{word}) = 0.50$.

**Tasks:**
1. Compute the single-term contribution to reverse KL divergence:
   $$Q(x) \ln\left(\frac{Q(x)}{P(x)}\right)$$
   where $Q = \pi_\theta$ and $P = \pi_{\text{ref}}$.
2. If the KL penalty weight in the RLHF objective is $\beta = 0.1$, what penalty does this single token inflict on the overall reward?

---

## 💡 Complete Step-by-Step Solutions

---

### Solution 1: Asymmetry of KL Divergence

#### Step 1: Forward KL $D_{\text{KL}}(P \parallel Q)$
$$D_{\text{KL}}(P \parallel Q) = P(H) \log_2\left(\frac{P(H)}{Q(H)}\right) + P(T) \log_2\left(\frac{P(T)}{Q(T)}\right)$$
$$= 0.90 \log_2\left(\frac{0.90}{0.50}\right) + 0.10 \log_2\left(\frac{0.10}{0.50}\right)$$
$$= 0.90 \log_2(1.80) + 0.10 \log_2(0.20)$$
Since $\log_2(1.80) \approx 0.847997$ and $\log_2(0.20) \approx -2.321928$:
$$D_{\text{KL}}(P \parallel Q) \approx 0.90(0.847997) + 0.10(-2.321928) = 0.763197 - 0.232193 = 0.5310\text{ bits}$$

#### Step 2: Reverse KL $D_{\text{KL}}(Q \parallel P)$
$$D_{\text{KL}}(Q \parallel P) = Q(H) \log_2\left(\frac{Q(H)}{P(H)}\right) + Q(T) \log_2\left(\frac{Q(T)}{P(T)}\right)$$
$$= 0.50 \log_2\left(\frac{0.50}{0.90}\right) + 0.50 \log_2\left(\frac{0.50}{0.10}\right)$$
$$= 0.50 \log_2(0.555556) + 0.50 \log_2(5.0)$$
Since $\log_2(0.555556) \approx -0.847997$ and $\log_2(5.0) \approx 2.321928$:
$$D_{\text{KL}}(Q \parallel P) \approx 0.50(-0.847997 + 2.321928) = 0.50(1.473931) = 0.7370\text{ bits}$$

#### Step 3: Comparison
$$D_{\text{KL}}(P \parallel Q) = 0.5310\text{ bits} \ne D_{\text{KL}}(Q \parallel P) = 0.7370\text{ bits}$$
The two values differ by over $0.20$ bits!

#### Step 4: Intuition
In Reverse KL $D_{\text{KL}}(Q \parallel P)$, the fair coin ($Q$) frequently generates Tails ($50\%$ of the time). But under the loaded model ($P$), Tails is very rare ($10\%$). When the loaded model receives a Tail, it suffers a massive surprise of $\log_2(5) \approx 2.32$ bits.
In Forward KL, Tails only occurs $10\%$ of the time, so that massive surprise happens far less frequently. The distribution doing the generating determines which surprises matter!

---

### Solution 2: Complete Derivation of Gaussian KL Divergence

#### Step 1: Expand the Log-Ratio
$$\ln\left(\frac{p(x)}{q(x)}\right) = \ln p(x) - \ln q(x)$$
Taking the natural logarithm of both Gaussian densities:
$$\ln p(x) = -\frac{1}{2} \ln(2\pi\sigma_0^2) - \frac{(x - \mu_0)^2}{2\sigma_0^2} = -\ln(\sigma_0) - \frac{1}{2}\ln(2\pi) - \frac{(x - \mu_0)^2}{2\sigma_0^2}$$
$$\ln q(x) = -\ln(\sigma_1) - \frac{1}{2}\ln(2\pi) - \frac{(x - \mu_1)^2}{2\sigma_1^2}$$
Subtracting $\ln q(x)$ from $\ln p(x)$:
$$\ln\left(\frac{p(x)}{q(x)}\right) = \ln\left(\frac{\sigma_1}{\sigma_0}\right) - \frac{(x - \mu_0)^2}{2\sigma_0^2} + \frac{(x - \mu_1)^2}{2\sigma_1^2}$$

#### Step 2: Take Expectation Under $p(x)$
$$D_{\text{KL}}(P \parallel Q) = \mathbb{E}_{x \sim P}\left[ \ln\left(\frac{\sigma_1}{\sigma_0}\right) - \frac{(x - \mu_0)^2}{2\sigma_0^2} + \frac{(x - \mu_1)^2}{2\sigma_1^2} \right]$$
Using linearity of expectation:
$$D_{\text{KL}}(P \parallel Q) = \ln\left(\frac{\sigma_1}{\sigma_0}\right) - \frac{1}{2\sigma_0^2} \mathbb{E}_{x \sim P}[(x - \mu_0)^2] + \frac{1}{2\sigma_1^2} \mathbb{E}_{x \sim P}[(x - \mu_1)^2]$$

#### Step 3: Evaluate Expectations
1. By definition of variance:
   $$\mathbb{E}_{x \sim P}[(x - \mu_0)^2] = \operatorname{Var}(X) = \sigma_0^2$$
   Therefore:
   $$-\frac{1}{2\sigma_0^2} \cdot \sigma_0^2 = -\frac{1}{2}$$

2. For the third term, expand $(x - \mu_1) = (x - \mu_0) + (\mu_0 - \mu_1)$:
   $$(x - \mu_1)^2 = (x - \mu_0)^2 + 2(x - \mu_0)(\mu_0 - \mu_1) + (\mu_0 - \mu_1)^2$$
   Taking expectation under $P$:
   - $\mathbb{E}[(x - \mu_0)^2] = \sigma_0^2$
   - $\mathbb{E}[2(x - \mu_0)(\mu_0 - \mu_1)] = 2(\mu_0 - \mu_1)\mathbb{E}[x - \mu_0] = 2(\mu_0 - \mu_1)(0) = 0$
   - $\mathbb{E}[(\mu_0 - \mu_1)^2] = (\mu_0 - \mu_1)^2$
   Thus:
   $$\mathbb{E}_{x \sim P}[(x - \mu_1)^2] = \sigma_0^2 + (\mu_0 - \mu_1)^2$$

#### Step 4: Assemble the Formula
$$D_{\text{KL}}(P \parallel Q) = \ln\left(\frac{\sigma_1}{\sigma_0}\right) - \frac{1}{2} + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2}$$
$$D_{\text{KL}}(P \parallel Q) = \ln\left(\frac{\sigma_1}{\sigma_0}\right) + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2} - \frac{1}{2} \quad \blacksquare$$

---

### Solution 3: Canonical VAE Latent Loss

#### Step 1: Substitute Prior Values
Set $\mu_0 = \mu_j, \sigma_0 = \sigma_j$ and $\mu_1 = 0, \sigma_1 = 1$:
$$D_{\text{KL}} = \ln\left(\frac{1}{\sigma_j}\right) + \frac{\sigma_j^2 + (\mu_j - 0)^2}{2(1)^2} - \frac{1}{2}$$
$$= -\ln(\sigma_j) + \frac{\sigma_j^2 + \mu_j^2}{2} - \frac{1}{2}$$

#### Step 2: Factor Out $-\frac{1}{2}$
$$= -\frac{1}{2} \left[ 2\ln(\sigma_j) - \sigma_j^2 - \mu_j^2 + 1 \right]$$
Using $2\ln(\sigma_j) = \ln(\sigma_j^2)$:
$$D_{\text{KL}} = -\frac{1}{2} \left( 1 + \ln(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right) \quad \blacksquare$$

#### Step 3: Evaluation at Standard Normal
If $\mu_j = 0$ and $\sigma_j = 1$:
$$D_{\text{KL}} = -\frac{1}{2} \left( 1 + \ln(1) - 0^2 - 1^2 \right) = -\frac{1}{2}(1 + 0 - 0 - 1) = 0$$
When the latent representation perfectly matches the prior, the loss is exactly zero!

---

### Solution 4: RLHF Policy Drift Penalty

#### Step 1: Compute Single-Token Penalty
$$Q \ln\left(\frac{Q}{P}\right) = 0.50 \ln\left(\frac{0.50}{0.001}\right) = 0.50 \ln(500)$$
Since $\ln(500) \approx 6.2146$:
$$\text{Penalty contribution} = 0.50 \times 6.2146 \approx 3.1073\text{ nats}$$

#### Step 2: Apply Weight $\beta$
$$\text{Loss Penalty} = \beta \times 3.1073 = 0.1 \times 3.1073 \approx 0.3107$$
This penalty directly subtracts from the model's reward, discouraging it from violently deviating from the base reference model.

