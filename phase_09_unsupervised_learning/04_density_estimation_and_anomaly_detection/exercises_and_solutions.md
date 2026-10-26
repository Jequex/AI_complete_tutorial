# Topic 9.4: Exercises & Solutions — Density Estimation & Anomaly Detection

---

### Problem 1: 1D Kernel Density Evaluation Hand Calculation (Easy)
Consider two sample points: $x_1 = 0.0$ and $x_2 = 2.0$.
Suppose we use a 1D Gaussian kernel $K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2 / 2}$ with bandwidth $h = 1.0$.

1. Write out the formula for the KDE estimate $\hat{f}(x)$.
2. Compute the estimated density $\hat{f}(1.0)$ at the midpoint $x = 1.0$.
3. Compute the estimated density $\hat{f}(0.0)$ directly on the first sample point.

<details>
<summary>View Complete Solution</summary>

**Step 1: Formula for $\hat{f}(x)$:**
With $N = 2$ and $h = 1.0$:
$$\hat{f}(x) = \frac{1}{N h} \sum_{i=1}^N K\left(\frac{x - x_i}{h}\right) = \frac{1}{2(1.0)} \left[ \frac{1}{\sqrt{2\pi}} e^{-\frac{(x - 0.0)^2}{2}} + \frac{1}{\sqrt{2\pi}} e^{-\frac{(x - 2.0)^2}{2}} \right]$$
$$\hat{f}(x) = \frac{1}{2\sqrt{2\pi}} \left( e^{-x^2 / 2} + e^{-(x - 2)^2 / 2} \right)$$
Note that $\frac{1}{2\sqrt{2\pi}} \approx \frac{1}{2(2.5066)} \approx 0.19947$.

**Step 2: Density at midpoint $x = 1.0$:**
$$\hat{f}(1.0) = 0.19947 \left( e^{-1.0^2 / 2} + e^{-(1.0 - 2.0)^2 / 2} \right) = 0.19947 \left( e^{-0.5} + e^{-0.5} \right)$$
Since $e^{-0.5} \approx 0.60653$:
$$\hat{f}(1.0) = 0.19947 \cdot (2 \times 0.60653) = 0.19947 \times 1.21306 \approx 0.24197$$

**Step 3: Density at $x = 0.0$:**
$$\hat{f}(0.0) = 0.19947 \left( e^0 + e^{-(-2)^2 / 2} \right) = 0.19947 \left( 1.0 + e^{-2.0} \right)$$
Since $e^{-2.0} \approx 0.13534$:
$$\hat{f}(0.0) = 0.19947 \times (1.0 + 0.13534) = 0.19947 \times 1.13534 \approx 0.22647$$
</details>

---

### Problem 2: Silverman's Rule of Thumb Calculation (Easy)
A financial dataset contains $N = 100,000$ daily stock return observations with empirical standard deviation $\hat{\sigma} = 0.02$ (2%).
Compute the optimal bandwidth $h_{\text{opt}}$ according to Silverman's rule of thumb:
$$h_{\text{opt}} = 1.06 \, \hat{\sigma} \, N^{-1/5}$$

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $N^{-1/5}$:**
$$N = 100,000 = 10^5$$
$$N^{-1/5} = (10^5)^{-1/5} = 10^{-1} = 0.1$$

**Step 2: Compute $h_{\text{opt}}$:**
$$h_{\text{opt}} = 1.06 \times 0.02 \times 0.1 = 0.00212$$

The optimal smoothing bandwidth is $0.00212$ (or approximately 0.212% return).
</details>

---

### Problem 3: Mahalanobis Distance for Correlated Outlier (Medium)
Consider a 2D Gaussian distribution centered at $\mu = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$ with covariance matrix:
$$\Sigma = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

1. Compute the inverse covariance matrix $\Sigma^{-1}$.
2. Compute the Mahalanobis distance $D_M(x)$ for inlier candidate $x_A = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.
3. Compute the Mahalanobis distance $D_M(x)$ for candidate $x_B = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$.
4. Note that $x_A$ and $x_B$ have the exact same Euclidean distance from the origin ($\sqrt{1^2 + 1^2} = \sqrt{2}$). Explain why their Mahalanobis distances differ dramatically.

<details>
<summary>View Complete Solution</summary>

**Step 1: Invert $\Sigma$:**
$$\det(\Sigma) = (2)(1) - (1)(1) = 2 - 1 = 1$$
$$\Sigma^{-1} = \frac{1}{1} \begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}$$

**Step 2: Mahalanobis distance for $x_A = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$:**
$$D_M(x_A)^2 = x_A^T \Sigma^{-1} x_A = \begin{pmatrix} 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$
$$\begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$
$$D_M(x_A)^2 = 1(0) + 1(1) = 1.0 \implies D_M(x_A) = \sqrt{1.0} = 1.0$$

**Step 3: Mahalanobis distance for $x_B = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$:**
$$\begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 - (-1) \\ -1 + 2(-1) \end{pmatrix} = \begin{pmatrix} 2 \\ -3 \end{pmatrix}$$
$$D_M(x_B)^2 = 1(2) + (-1)(-3) = 2 + 3 = 5.0 \implies D_M(x_B) = \sqrt{5.0} \approx 2.236$$

**Step 4: Explanation:**
The covariance matrix has positive off-diagonal elements ($\Sigma_{12} = 1$), meaning features 1 and 2 positively co-vary: when $x_1 > 0$, $x_2$ is expected to be positive.
- Point $x_A = (1, 1)$ aligns perfectly with this positive correlation ($D_M = 1.0$).
- Point $x_B = (1, -1)$ directly violates this correlation ($x_1 > 0$ while $x_2 < 0$), resulting in more than double the Mahalanobis distance ($D_M = 2.236$).
</details>

---

### Problem 4: Isolation Forest Anomaly Score Calculation (Hard)
Suppose an Isolation Forest is trained on a dataset of $n = 256$ instances.
The theoretical normalization constant is given by:
$$c(n) = 2 \ln(n - 1) + 2\gamma - \frac{2(n - 1)}{n}$$
where $\gamma \approx 0.5772$.

1. Compute $c(256)$.
2. Suppose an anomaly point $x_{\text{bad}}$ isolates very quickly with average path length $\mathbb{E}[h(x_{\text{bad}})] = 2.0$. Compute its anomaly score $s(x_{\text{bad}}, 256) = 2^{-\mathbb{E}[h(x)] / c(n)}$.
3. Suppose a normal point $x_{\text{norm}}$ has average path length $\mathbb{E}[h(x_{\text{norm}})] = 9.5$. Compute its anomaly score.
4. Using the threshold $s \ge 0.6$ for flagging anomalies, which point is flagged?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $c(256)$:**
- $\ln(256 - 1) = \ln(255) \approx 5.5413$
- $2 \ln(255) \approx 11.0825$
- $2\gamma = 2(0.5772) = 1.1544$
- $\frac{2(255)}{256} = \frac{510}{256} \approx 1.9922$

$$c(256) = 11.0825 + 1.1544 - 1.9922 = 10.2447$$

**Step 2: Anomaly score for $x_{\text{bad}}$ ($\mathbb{E}[h] = 2.0$):**
$$\text{Exponent} = -\frac{2.0}{10.2447} \approx -0.1952$$
$$s(x_{\text{bad}}, 256) = 2^{-0.1952} \approx 0.873$$

**Step 3: Anomaly score for $x_{\text{norm}}$ ($\mathbb{E}[h] = 9.5$):**
$$\text{Exponent} = -\frac{9.5}{10.2447} \approx -0.9273$$
$$s(x_{\text{norm}}, 256) = 2^{-0.9273} \approx 0.526$$

**Step 4: Classification:**
- $s(x_{\text{bad}}) = 0.873 \ge 0.6 \implies$ **Flagged as Anomaly!**
- $s(x_{\text{norm}}) = 0.526 < 0.6 \implies$ **Classified as Normal Inlier.**
</details>
