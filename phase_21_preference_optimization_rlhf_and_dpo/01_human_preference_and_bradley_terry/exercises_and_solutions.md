# Topic 21.1: Exercises & Solutions — Human Preference Modeling & Bradley-Terry

---

### Problem 1: Human Scalar Noise vs. Pairwise Comparisons (Easy)

Suppose three human annotators (Alice, Bob, and Carol) are asked to evaluate two LLM responses, $y_1$ and $y_2$, to the prompt: *"Explain photosynthesis in one sentence."*
The true latent utilities are $u(y_1) = 4.0$ and $u(y_2) = 2.5$.
Each annotator converts utility $u$ to a scalar score on $[0, 10]$ with personal rater bias $b_k$:
$$s_k(y) = u(y) + b_k + \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, 0.25^2)$$
where:
- Alice is generous: $b_{\text{Alice}} = +3.5$
- Bob is neutral: $b_{\text{Bob}} = 0.0$
- Carol is strict: $b_{\text{Carol}} = -2.0$

1. Compute the expected scalar score assigned by Alice to $y_2$, and the expected score assigned by Carol to $y_1$. Can an external observer compare Alice's score of $y_2$ directly against Carol's score of $y_1$ to determine which response was better?
2. If Alice, Bob, and Carol are instead asked to make a pairwise choice: *"Which response is better: $y_1$ or $y_2$?"*, show mathematically that all three annotators agree in expectation.

<details>
<summary>View Complete Solution</summary>

**Step 1: Expected scalar scores:**
- For Alice evaluating $y_2$:
  $$\mathbb{E}[s_{\text{Alice}}(y_2)] = u(y_2) + b_{\text{Alice}} = 2.5 + 3.5 = 6.0$$
- For Carol evaluating $y_1$:
  $$\mathbb{E}[s_{\text{Carol}}(y_1)] = u(y_1) + b_{\text{Carol}} = 4.0 - 2.0 = 2.0$$

Comparing Alice's rating of $y_2$ ($6.0$) against Carol's rating of $y_1$ ($2.0$) would lead to the completely false conclusion that $y_2 > y_1$. The uncalibrated personal biases completely obscure the true quality difference of $+1.5$.

**Step 2: Pairwise comparison invariance:**
In a pairwise evaluation, annotator $k$ evaluates the difference:
$$\Delta_k = s_k(y_1) - s_k(y_2) = \left( u(y_1) + b_k + \epsilon_{k, 1} \right) - \left( u(y_2) + b_k + \epsilon_{k, 2} \right)$$
$$= u(y_1) - u(y_2) + (\epsilon_{k, 1} - \epsilon_{k, 2})$$
Notice that the personal bias term $b_k$ cancels out identically: $b_k - b_k = 0$.
The expected difference for all three raters is:
$$\mathbb{E}[\Delta_{\text{Alice}}] = \mathbb{E}[\Delta_{\text{Bob}}] = \mathbb{E}[\Delta_{\text{Carol}}] = 4.0 - 2.5 = +1.5 > 0$$
Since $\epsilon_{k, 1} - \epsilon_{k, 2} \sim \mathcal{N}(0, 2 \times 0.25^2 = 0.125)$, the probability of an annotator picking $y_1 \succ y_2$ is:
$$P(y_1 \succ y_2) = \Phi\left(\frac{1.5}{\sqrt{0.125}}\right) = \Phi(4.24) > 0.9999$$
All three annotators agree almost with certainty, demonstrating the robustness of pairwise comparison.
</details>

---

### Problem 2: Bradley-Terry Maximum Likelihood Gradient & Hessian (Medium)

Consider the Bradley-Terry negative log-likelihood loss for a single preferred pair $(y_w \succ y_l)$:
$$\ell(r_w, r_l) = -\log \sigma(r_w - r_l)$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$. Let $\Delta = r_w - r_l$.

1. Derive the first derivative $\frac{\partial \ell}{\partial \Delta}$ and show that it equals $-\sigma(-\Delta) = \sigma(\Delta) - 1$.
2. Compute the second derivative $\frac{\partial^2 \ell}{\partial \Delta^2}$ and prove that $\ell$ is convex in $\Delta$.
3. When the reward model assigns $r_w = 4.0$ and $r_l = 1.0$, what is the gradient magnitude? What is the gradient magnitude when $r_w = 1.0$ and $r_l = 4.0$?

<details>
<summary>View Complete Solution</summary>

**Step 1: First derivative derivation:**
Recall $\sigma(z) = \frac{1}{1 + e^{-z}}$.
$$\frac{d}{dz} \sigma(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$$
Using the chain rule on $\ell(\Delta) = -\log \sigma(\Delta)$:
$$\frac{d\ell}{d\Delta} = -\frac{1}{\sigma(\Delta)} \cdot \frac{d\sigma(\Delta)}{d\Delta} = -\frac{\sigma(\Delta)(1 - \sigma(\Delta))}{\sigma(\Delta)} = -(1 - \sigma(\Delta))$$
Using the identity $1 - \sigma(\Delta) = \sigma(-\Delta)$:
$$\frac{\partial \ell}{\partial \Delta} = -\sigma(-\Delta) = \sigma(\Delta) - 1$$

**Step 2: Second derivative and convexity:**
Differentiating again with respect to $\Delta$:
$$\frac{d^2 \ell}{d\Delta^2} = \frac{d}{d\Delta} (\sigma(\Delta) - 1) = \sigma(\Delta)(1 - \sigma(\Delta))$$
Since $\sigma(\Delta) \in (0, 1)$ for all finite $\Delta \in \mathbb{R}$, both $\sigma(\Delta) > 0$ and $1 - \sigma(\Delta) > 0$.
Therefore, $\frac{d^2 \ell}{d\Delta^2} > 0$ everywhere, proving that the loss function is strictly convex with respect to the difference $\Delta$.

**Step 3: Numerical evaluation:**
- Case A ($r_w = 4.0, r_l = 1.0 \implies \Delta = +3.0$):
  $$\sigma(-3.0) = \frac{1}{1 + e^3} = \frac{1}{1 + 20.0855} \approx 0.0474$$
  The gradient magnitude is $|\frac{\partial \ell}{\partial \Delta}| \approx 0.0474$. Because the model correctly ranked $y_w \gg y_l$, the gradient is small and updates are gentle.
- Case B ($r_w = 1.0, r_l = 4.0 \implies \Delta = -3.0$):
  $$\sigma(-(-3.0)) = \sigma(3.0) = \frac{1}{1 + e^{-3}} \approx 0.9526$$
  The gradient magnitude is $|\frac{\partial \ell}{\partial \Delta}| \approx 0.9526$. Because the model got the preference completely backwards, the gradient provides a massive corrective push.
</details>

---

### Problem 3: Plackett-Luce Ranking Probability Computation (Medium)

Suppose an instruction prompt produces three candidate responses: $\{y_A, y_B, y_C\}$ with latent reward values:
$$r_A = 2.0, \quad r_B = 1.0, \quad r_C = 0.0$$

1. Using the Plackett-Luce model, compute the probability of the ranking permutation $y_A \succ y_B \succ y_C$.
2. Compute the probability of the inverted permutation $y_C \succ y_B \succ y_A$.
3. Show that the ratio of probabilities between choosing $y_A$ first vs $y_B$ first from the set $\{y_A, y_B, y_C\}$ depends only on $r_A - r_B$ (the Independence of Irrelevant Alternatives - IIA property).

<details>
<summary>View Complete Solution</summary>

**Step 1: Plackett-Luce ranking probability for $y_A \succ y_B \succ y_C$:**
The Plackett-Luce probability factorizes as:
$$P(y_A \succ y_B \succ y_C) = P(\text{choose } y_A \text{ from } \{A, B, C\}) \times P(\text{choose } y_B \text{ from } \{B, C\})$$
First choice:
$$P(A \mid \{A, B, C\}) = \frac{e^{r_A}}{e^{r_A} + e^{r_B} + e^{r_C}} = \frac{e^{2}}{e^2 + e^1 + e^0} = \frac{7.389}{7.389 + 2.718 + 1.0} = \frac{7.389}{11.107} \approx 0.6652$$
Second choice:
$$P(B \mid \{B, C\}) = \frac{e^{r_B}}{e^{r_B} + e^{r_C}} = \frac{e^1}{e^1 + e^0} = \frac{2.718}{2.718 + 1.0} = \frac{2.718}{3.718} \approx 0.7311$$
Total probability:
$$P(y_A \succ y_B \succ y_C) = 0.6652 \times 0.7311 \approx 0.4863 \quad (48.63\%)$$

**Step 2: Inverted ranking probability for $y_C \succ y_B \succ y_A$:**
First choice:
$$P(C \mid \{A, B, C\}) = \frac{e^0}{11.107} = \frac{1.0}{11.107} \approx 0.0900$$
Second choice:
$$P(B \mid \{B, A\}) = \frac{e^1}{e^1 + e^2} = \frac{2.718}{2.718 + 7.389} = \frac{2.718}{10.107} \approx 0.2689$$
Total probability:
$$P(y_C \succ y_B \succ y_A) = 0.0900 \times 0.2689 \approx 0.0242 \quad (2.42\%)$$
The forward ranking is $\frac{0.4863}{0.0242} \approx 20.1\times$ more probable.

**Step 3: Independence of Irrelevant Alternatives (IIA):**
The ratio of probabilities of choosing $A$ over $B$ first is:
$$\frac{P(A \mid \{A, B, C\})}{P(B \mid \{A, B, C\})} = \frac{e^{r_A} / \sum_{i} e^{r_i}}{e^{r_B} / \sum_{i} e^{r_i}} = \frac{e^{r_A}}{e^{r_B}} = e^{r_A - r_B}$$
Notice the denominator $\sum_i e^{r_i}$ completely cancels out, meaning the relative likelihood of selecting $A$ over $B$ is completely unaffected by the presence or score of alternative $C$.
</details>

---

### Problem 4: Transitivity Proof & The Condorcet Paradox (Hard)

1. Prove that any Bradley-Terry model satisfies Strong Stochastic Transitivity (SST):
   $$\text{If } P(A \succ B) \ge 0.5 \text{ and } P(B \succ C) \ge 0.5, \text{ then } P(A \succ C) \ge \max(P(A \succ B), P(B \succ C))$$
2. Suppose three annotators submit preference comparisons forming a Condorcet cycle:
   $$\text{Annotator 1: } A \succ B, \quad \text{Annotator 2: } B \succ C, \quad \text{Annotator 3: } C \succ A$$
   Under the Bradley-Terry MLE objective $\mathcal{L}(r_A, r_B, r_C) = \log \sigma(r_A - r_B) + \log \sigma(r_B - r_C) + \log \sigma(r_C - r_A)$, find the optimal rewards $(r_A^*, r_B^*, r_C^*)$ subject to the centering constraint $r_A + r_B + r_C = 0$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Proof of Strong Stochastic Transitivity:**
Let $r_A, r_B, r_C \in \mathbb{R}$.
$P(A \succ B) \ge 0.5 \implies \sigma(r_A - r_B) \ge 0.5 \implies r_A - r_B \ge 0$.
$P(B \succ C) \ge 0.5 \implies \sigma(r_B - r_C) \ge 0.5 \implies r_B - r_C \ge 0$.
Define $\delta_1 = r_A - r_B \ge 0$ and $\delta_2 = r_B - r_C \ge 0$.
The reward difference between $A$ and $C$ is:
$$\delta_{AC} = r_A - r_C = (r_A - r_B) + (r_B - r_C) = \delta_1 + \delta_2$$
Since $\delta_1 \ge 0$ and $\delta_2 \ge 0$:
$$\delta_{AC} = \delta_1 + \delta_2 \ge \max(\delta_1, \delta_2)$$
Since $\sigma(z) = \frac{1}{1 + e^{-z}}$ is a strictly monotonically increasing function:
$$\sigma(\delta_{AC}) \ge \sigma(\max(\delta_1, \delta_2)) = \max(\sigma(\delta_1), \sigma(\delta_2))$$
Thus:
$$P(A \succ C) \ge \max(P(A \succ B), P(B \succ C)) \quad \blacksquare$$

**Step 2: Solving the Condorcet Cycle under Bradley-Terry:**
The objective is:
$$\mathcal{L} = \log \sigma(r_A - r_B) + \log \sigma(r_B - r_C) + \log \sigma(r_C - r_A)$$
Let $x = r_A - r_B$, $y = r_B - r_C$, so $r_C - r_A = -(x + y)$.
$$\mathcal{L}(x, y) = \log \sigma(x) + \log \sigma(y) + \log \sigma(-(x + y))$$
Taking the partial derivative with respect to $x$:
$$\frac{\partial \mathcal{L}}{\partial x} = (1 - \sigma(x)) - (1 - \sigma(-(x+y))) = \sigma(-x) - \sigma(x+y)$$
Setting to zero:
$$\sigma(-x) = \sigma(x+y) \implies -x = x + y \implies 2x + y = 0$$
By cyclic symmetry with respect to $y$:
$$2y + x = 0$$
Solving the linear system:
$$x = -2y \implies 2(-2y) + y = 0 \implies -3y = 0 \implies y = 0 \implies x = 0$$
Since $x = r_A - r_B = 0 \implies r_A = r_B$, and $y = r_B - r_C = 0 \implies r_B = r_C$.
With the centering constraint $r_A + r_B + r_C = 0$:
$$3 r_A^* = 0 \implies r_A^* = r_B^* = r_C^* = 0$$
The Bradley-Terry model resolves the cycle by setting all rewards exactly equal, assigning a 50% win probability to every matchup: $P(A \succ B) = P(B \succ C) = P(C \succ A) = 0.5$.
</details>
