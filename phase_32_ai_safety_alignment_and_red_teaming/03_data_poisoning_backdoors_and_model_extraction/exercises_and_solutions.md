# Exercises & Solutions: Data Poisoning, Backdoors & Model Extraction

## Exercise 1: Spectral Covariance & Power Iteration
**Problem**: Consider a 2-dimensional representation space where $N = 4$ samples from class $y_{\text{target}} = 1$ have representations:
$$h_1 = [2.0, 1.0]^T, \quad h_2 = [2.0, -1.0]^T, \quad h_3 = [4.0, 3.0]^T \text{ (poisoned)}, \quad h_4 = [4.0, -3.0]^T \text{ (poisoned)}$$
1. Calculate the mean representation vector $\mu = \frac{1}{4} \sum_{i=1}^4 h_i$.
2. Compute the centered representation vectors $\tilde{h}_i = h_i - \mu$.
3. Compute the sample covariance matrix $C = \frac{1}{4} \sum_{i=1}^4 \tilde{h}_i \tilde{h}_i^T$.
4. Determine the top eigenvector $v_1$ of $C$.
5. Compute the spectral outlier score $\tau_i = (\tilde{h}_i^T v_1)^2$ for each of the 4 samples. Are the poisoned samples successfully scored higher than clean samples?

---

### Solution 1:
1. **Mean Vector**:
   $$\mu_1 = \frac{2 + 2 + 4 + 4}{4} = \frac{12}{4} = 3.0$$
   $$\mu_2 = \frac{1 + (-1) + 3 + (-3)}{4} = \frac{0}{4} = 0.0$$
   $$\mu = \begin{bmatrix} 3.0 \\ 0.0 \end{bmatrix}$$

2. **Centered Representations**:
   $$\tilde{h}_1 = \begin{bmatrix} 2 - 3 \\ 1 - 0 \end{bmatrix} = \begin{bmatrix} -1.0 \\ 1.0 \end{bmatrix}, \quad \tilde{h}_2 = \begin{bmatrix} -1.0 \\ -1.0 \end{bmatrix}, \quad \tilde{h}_3 = \begin{bmatrix} 1.0 \\ 3.0 \end{bmatrix}, \quad \tilde{h}_4 = \begin{bmatrix} 1.0 \\ -3.0 \end{bmatrix}$$

3. **Covariance Matrix**:
   $$\tilde{h}_1 \tilde{h}_1^T = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}, \quad \tilde{h}_2 \tilde{h}_2^T = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$
   $$\tilde{h}_3 \tilde{h}_3^T = \begin{bmatrix} 1 & 3 \\ 3 & 9 \end{bmatrix}, \quad \tilde{h}_4 \tilde{h}_4^T = \begin{bmatrix} 1 & -3 \\ -3 & 9 \end{bmatrix}$$
   $$4 C = \begin{bmatrix} 1+1+1+1 & -1+1+3-3 \\ -1+1+3-3 & 1+1+9+9 \end{bmatrix} = \begin{bmatrix} 4 & 0 \\ 0 & 20 \end{bmatrix}$$
   $$C = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 5.0 \end{bmatrix}$$

4. **Eigenvalues and Eigenvectors**:
   Since $C$ is diagonal, the eigenvalues are $\lambda_1 = 5.0$ and $\lambda_2 = 1.0$.
   The dominant eigenvalue is $\lambda_1 = 5.0$, with corresponding top unit eigenvector:
   $$v_1 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$

5. **Spectral Outlier Scores**:
   $$\tau_1 = (\tilde{h}_1^T v_1)^2 = ((-1.0)(0) + (1.0)(1))^2 = (1.0)^2 = 1.0$$
   $$\tau_2 = (\tilde{h}_2^T v_1)^2 = ((-1.0)(0) + (-1.0)(1))^2 = (-1.0)^2 = 1.0$$
   $$\tau_3 = (\tilde{h}_3^T v_1)^2 = ((1.0)(0) + (3.0)(1))^2 = (3.0)^2 = 9.0$$
   $$\tau_4 = (\tilde{h}_4^T v_1)^2 = ((1.0)(0) + (-3.0)(1))^2 = (-3.0)^2 = 9.0$$
   The poisoned samples ($h_3, h_4$) receive score $\tau = 9.0$, which is **$9\times$ higher** than clean samples ($\tau = 1.0$). Quarantining by threshold $\tau > 5.0$ achieves $100\%$ precision and $100\%$ recall!

---

## Exercise 2: Kirchenbauer Watermark Significance Test
**Problem**: An auditor suspects a competitor has distilled their proprietary model. The auditor queries the competitor model and collects a sample output containing $T = 400$ tokens. 
The watermarking scheme uses green list fraction $\gamma = 0.50$.
Upon inspection with the secret hash key, the text contains $N_G = 250$ green tokens.
1. What is the expected number of green tokens under the null hypothesis $H_0$ (unwatermarked text)?
2. What is the standard deviation $\sigma_G$?
3. Calculate the standard normal $z$-statistic.
4. Using the standard normal distribution table where $z = 3.0 \implies p \approx 0.00135$ and $z = 5.0 \implies p \approx 2.87 \times 10^{-7}$, evaluate whether the watermark is statistically proven.

---

### Solution 2:
1. **Expected Green Tokens**:
   $$\mathbb{E}[N_G] = \gamma T = (0.50)(400) = 200 \text{ tokens}$$

2. **Standard Deviation**:
   $$\sigma_G = \sqrt{T \gamma (1 - \gamma)} = \sqrt{400 \times 0.50 \times 0.50} = \sqrt{100} = 10.0$$

3. **$z$-Statistic**:
   $$z = \frac{N_G - \mathbb{E}[N_G]}{\sigma_G} = \frac{250 - 200}{10.0} = \frac{50}{10.0} = 5.00$$

4. **Statistical Verdict**:
   A $z$-score of $+5.00$ corresponds to a $p$-value of $p \approx 2.87 \times 10^{-7}$ (less than 1 in 3.4 million odds of occurring by chance). The presence of the secret watermark is **statistically confirmed**, providing conclusive legal evidence of model extraction.
