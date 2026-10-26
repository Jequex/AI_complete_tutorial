# Topic 11.3: Exercises & Solutions — Loss Functions & Derivatives

---

### Problem 1: Manual Calculation of MSE and its Gradients (Easy)
Consider a regression mini-batch of $N = 3$ observations:
- True targets: $\mathbf{y} = [3.0, -1.0, 4.5]^T$
- Model predictions: $\hat{\mathbf{y}} = [2.6, -1.2, 5.0]^T$

1. Compute the Mean Squared Error using the standard formula:
   $$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
2. Compute the exact analytical gradient vector $\nabla_{\hat{\mathbf{y}}} \mathcal{L} = \left[\frac{\partial \mathcal{L}}{\partial \hat{y}_1}, \frac{\partial \mathcal{L}}{\partial \hat{y}_2}, \frac{\partial \mathcal{L}}{\partial \hat{y}_3}\right]^T$.
3. If gradient descent is performed with learning rate $\eta = 0.5$, calculate the updated predictions $\hat{\mathbf{y}}_{\text{new}}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute MSE Loss:**
Errors $e_i = \hat{y}_i - y_i$:
- $e_1 = 2.6 - 3.0 = -0.4 \implies e_1^2 = 0.16$
- $e_2 = -1.2 - (-1.0) = -0.2 \implies e_2^2 = 0.04$
- $e_3 = 5.0 - 4.5 = 0.5 \implies e_3^2 = 0.25$

$$\sum_{i=1}^3 e_i^2 = 0.16 + 0.04 + 0.25 = 0.45$$
$$\mathcal{L} = \frac{1}{2 \times 3} \times 0.45 = \frac{0.45}{6} = 0.075$$

**Step 2: Analytical Gradients:**
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{N}(\hat{y}_i - y_i) = \frac{1}{3} e_i$$
- $\frac{\partial \mathcal{L}}{\partial \hat{y}_1} = \frac{-0.4}{3} \approx -0.133333$
- $\frac{\partial \mathcal{L}}{\partial \hat{y}_2} = \frac{-0.2}{3} \approx -0.066667$
- $\frac{\partial \mathcal{L}}{\partial \hat{y}_3} = \frac{0.5}{3} \approx +0.166667$

$$\nabla_{\hat{\mathbf{y}}} \mathcal{L} = \begin{bmatrix} -0.133333 \\ -0.066667 \\ 0.166667 \end{bmatrix}$$

**Step 3: Update Predictions:**
$$\hat{y}_{\text{new}} = \hat{y} - \eta \nabla_{\hat{y}} \mathcal{L}$$
- $\hat{y}_{1, \text{new}} = 2.6 - 0.5(-0.133333) = 2.6 + 0.066667 = 2.666667$
- $\hat{y}_{2, \text{new}} = -1.2 - 0.5(-0.066667) = -1.2 + 0.033333 = -1.166667$
- $\hat{y}_{3, \text{new}} = 5.0 - 0.5(0.166667) = 5.0 - 0.083333 = 4.916667$

Notice that every single prediction shifted closer to its true target!
</details>

---

### Problem 2: Softmax Jacobian & Categorical Cross-Entropy Gradient (Medium)
Let $\mathbf{z} = [z_1, z_2, z_3]^T$ be the logits for a 3-class classification problem, and let the softmax output probabilities be:
$$p_k = \frac{e^{z_k}}{\sum_{j=1}^3 e^{z_j}}, \quad k \in \{1, 2, 3\}$$

1. Prove that the partial derivative $\frac{\partial p_k}{\partial z_i}$ satisfies:
   $$\frac{\partial p_k}{\partial z_i} = p_k (\delta_{ki} - p_i)$$
   where $\delta_{ki}$ is the Kronecker delta ($\delta_{ki} = 1$ if $k=i$, else $0$).
2. Let the true label be one-hot vector $\mathbf{y} = [y_1, y_2, y_3]^T$ with $\sum y_k = 1$. The categorical cross-entropy is:
   $$\mathcal{L} = -\sum_{k=1}^3 y_k \ln p_k$$
   Prove rigorously that $\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Softmax Jacobian Derivation:**
Let $S = \sum_{j=1}^3 e^{z_j}$. Note that $\frac{\partial S}{\partial z_i} = e^{z_i}$.

**Case 1: $k = i$**
Using quotient rule on $p_i = \frac{e^{z_i}}{S}$:
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} \frac{\partial S}{\partial z_i}}{S^2} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = \frac{e^{z_i}}{S} - \left(\frac{e^{z_i}}{S}\right)^2 = p_i - p_i^2 = p_i(1 - p_i)$$

**Case 2: $k \ne i$**
Here $e^{z_k}$ is constant with respect to $z_i$:
$$\frac{\partial p_k}{\partial z_i} = \frac{0 \cdot S - e^{z_k} \frac{\partial S}{\partial z_i}}{S^2} = -\frac{e^{z_k} e^{z_i}}{S^2} = -p_k p_i = p_k(0 - p_i)$$

Combining both cases using the Kronecker delta $\delta_{ki}$:
$$\boxed{\frac{\partial p_k}{\partial z_i} = p_k(\delta_{ki} - p_i)}$$

**Step 2: CCE Loss Gradient w.r.t. Logit $z_i$:**
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{k=1}^3 \frac{y_k}{p_k} \frac{\partial p_k}{\partial z_i}$$
Substitute the Softmax Jacobian:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{k=1}^3 \frac{y_k}{p_k} \left[ p_k(\delta_{ki} - p_i) \right] = -\sum_{k=1}^3 y_k (\delta_{ki} - p_i)$$
Expand the sum:
$$= -\sum_{k=1}^3 y_k \delta_{ki} + \sum_{k=1}^3 y_k p_i = -y_i + p_i \sum_{k=1}^3 y_k$$
Since $\mathbf{y}$ is a one-hot vector, $\sum_{k=1}^3 y_k = 1$:
$$\boxed{\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i}$$
</details>

---

### Problem 3: Quantitative Suppression of Easy Negatives in Focal Loss (Hard)
Let an imbalanced dataset have 10,000 background (negative, $y=0$) examples and 10 foreground (positive, $y=1$) examples.
Suppose a model has learned to classify most background examples well with $p \approx 0.02$ ($p_t = 0.98$), while the 10 foreground examples remain hard with $p \approx 0.05$ ($p_t = 0.05$).

1. Compute the Binary Cross-Entropy loss for:
   - A single easy background sample ($p_t = 0.98$).
   - A single hard foreground sample ($p_t = 0.05$).
   - The total batch BCE loss across all 10,000 background and 10 foreground samples. What fraction of the total loss comes from background samples?
2. Compute the Focal Loss ($\gamma = 2.0$, $\alpha = 0.5$) for:
   - A single easy background sample.
   - A single hard foreground sample.
   - The total batch Focal Loss across all 10,000 background and 10 foreground samples. What fraction comes from background samples now?

<details>
<summary>View Complete Solution</summary>

**Step 1: Standard Binary Cross-Entropy:**
For easy background ($p_t = 0.98$):
$$\mathcal{L}_{\text{BCE, bg}} = -\ln(0.98) \approx 0.020203$$
For hard foreground ($p_t = 0.05$):
$$\mathcal{L}_{\text{BCE, fg}} = -\ln(0.05) \approx 2.995732$$

Total batch BCE loss:
$$\text{Total BG Loss} = 10,000 \times 0.020203 \approx 202.03$$
$$\text{Total FG Loss} = 10 \times 2.995732 \approx 29.96$$
$$\text{Total Loss} = 202.03 + 29.96 = 231.99$$
$$\text{Fraction of loss from background} = \frac{202.03}{231.99} \approx 87.1\%$$
Even though the background is well-classified ($p_t=0.98$), it completely overwhelms the training signal (87.1% of all gradients come from background noise)!

**Step 2: Focal Loss ($\gamma = 2.0, \alpha = 0.5$):**
Modulating factor: $(1 - p_t)^2$.
For easy background ($p_t = 0.98$):
$$(1 - 0.98)^2 = (0.02)^2 = 0.0004$$
$$\mathcal{L}_{\text{FL, bg}} = 0.5 \times 0.0004 \times (-\ln 0.98) = 0.5 \times 0.0004 \times 0.020203 \approx 4.04 \times 10^{-6}$$

For hard foreground ($p_t = 0.05$):
$$(1 - 0.05)^2 = (0.95)^2 = 0.9025$$
$$\mathcal{L}_{\text{FL, fg}} = 0.5 \times 0.9025 \times (-\ln 0.05) = 0.5 \times 0.9025 \times 2.995732 \approx 1.3518$$

Total batch Focal Loss:
$$\text{Total BG Loss} = 10,000 \times (4.04 \times 10^{-6}) \approx 0.0404$$
$$\text{Total FG Loss} = 10 \times 1.3518 \approx 13.518$$
$$\text{Total Loss} = 0.0404 + 13.518 = 13.558$$
$$\text{Fraction of loss from background} = \frac{0.0404}{13.558} \approx 0.3\%$$

**Conclusion:**
Focal Loss reduced the background gradient dominance from **87.1% down to 0.3%**, allowing the model to focus 99.7% of its learning capacity on the rare, hard foreground signals!
</details>
