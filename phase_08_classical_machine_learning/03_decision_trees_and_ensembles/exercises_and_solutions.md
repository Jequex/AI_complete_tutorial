# Topic 8.3: Exercises & Solutions — Decision Trees & Ensembles

---

### Exercise 1: Gini Impurity and Shannon Entropy Calculation

**Problem:**  
A dataset node $T$ contains 10 samples: 4 samples belong to Class A ($y=1$), and 6 samples belong to Class B ($y=0$).
1. Compute the empirical probabilities $p_1$ and $p_0$.
2. Calculate the Gini Impurity $G(T) = 1 - (p_1^2 + p_0^2)$.
3. Calculate the Shannon Entropy $H(T) = - (p_1 \log_2 p_1 + p_0 \log_2 p_0)$ in bits. (Use $\log_2(0.4) \approx -1.3219$ and $\log_2(0.6) \approx -0.7370$).
4. What would the Gini Impurity and Entropy be if all 10 samples belonged to Class A?

<details>
<summary>View Complete Solution</summary>

**Step 1: Empirical probabilities**
$$p_1 = \frac{4}{10} = 0.4, \quad p_0 = \frac{6}{10} = 0.6$$

**Step 2: Gini Impurity**
$$G(T) = 1 - \left(0.4^2 + 0.6^2\right) = 1 - (0.16 + 0.36) = 1 - 0.52 = 0.48$$

**Step 3: Shannon Entropy**
$$\begin{aligned}
H(T) &= -\left[ 0.4 \log_2(0.4) + 0.6 \log_2(0.6) \right] \\
&= -\left[ 0.4(-1.3219) + 0.6(-0.7370) \right] \\
&= -\left[ -0.5288 - 0.4422 \right] = 0.9710 \text{ bits}
\end{aligned}$$

**Step 4: Completely pure node ($p_1 = 1, p_0 = 0$)**
- $G(T_{\text{pure}}) = 1 - (1^2 + 0^2) = 0$
- $H(T_{\text{pure}}) = -(1 \log_2(1) + 0) = 0\text{ bits}$

$\blacksquare$
</details>

---

### Exercise 2: Information Gain of a Candidate Split

**Problem:**  
Parent node $T$ has 10 samples (4 positive, 6 negative), with entropy $H(T) = 0.9710\text{ bits}$.  
A proposed binary question splits $T$ into:
- Left child $T_L$ (4 samples): 3 positive, 1 negative.
- Right child $T_R$ (6 samples): 1 positive, 5 negative.

1. Compute the Shannon Entropy $H(T_L)$ of the left node. (Given $\log_2(0.75) \approx -0.4150$, $\log_2(0.25) = -2.0$).
2. Compute the Shannon Entropy $H(T_R)$ of the right node. (Given $\log_2(1/6) \approx -2.5850$, $\log_2(5/6) \approx -0.2630$).
3. Compute the weighted children entropy $H(T_L, T_R) = \frac{|T_L|}{|T|} H(T_L) + \frac{|T_R|}{|T|} H(T_R)$.
4. Compute the Information Gain $IG(T) = H(T) - H(T_L, T_R)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Entropy of Left Node $T_L$ ($p_1 = 3/4 = 0.75, p_0 = 1/4 = 0.25$)**
$$H(T_L) = -[0.75(-0.4150) + 0.25(-2.0)] = -[-0.3113 - 0.5000] = 0.8113\text{ bits}$$

**Step 2: Entropy of Right Node $T_R$ ($p_1 = 1/6 \approx 0.1667, p_0 = 5/6 \approx 0.8333$)**
$$H(T_R) = -\left[\frac{1}{6}(-2.5850) + \frac{5}{6}(-0.2630)\right] = -[-0.4308 - 0.2192] = 0.6500\text{ bits}$$

**Step 3: Weighted Children Entropy**
$$H(T_L, T_R) = \frac{4}{10}(0.8113) + \frac{6}{10}(0.6500) = 0.3245 + 0.3900 = 0.7145\text{ bits}$$

**Step 4: Information Gain**
$$IG = H(T) - H(T_L, T_R) = 0.9710 - 0.7145 = 0.2565\text{ bits}$$
The split successfully reduces system disorder by $0.2565$ bits of entropy.

$\blacksquare$
</details>

---

### Exercise 3: Ensemble Variance Reduction & Random Forests

**Problem:**  
Suppose you construct an ensemble of $M = 100$ decision trees. Each individual tree has an error variance of $\sigma^2 = 1.0$.
Recall the variance reduction formula:
$$\text{Var}(\text{Ensemble}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$

1. If you use standard Bagging where trees are correlated with $\rho = 0.40$, what is the ensemble variance?
2. If you use Random Forests with random feature subsampling to decorrelate the trees to $\rho = 0.05$, what is the ensemble variance?
3. Calculate the percentage reduction in variance achieved by feature subsampling.

<details>
<summary>View Complete Solution</summary>

**Step 1: Standard Bagging ($\rho = 0.40, M = 100$)**
$$\text{Var}_{\text{bag}} = (0.40)(1.0) + \frac{1 - 0.40}{100}(1.0) = 0.40 + \frac{0.60}{100} = 0.40 + 0.006 = 0.406$$

**Step 2: Random Forest ($\rho = 0.05, M = 100$)**
$$\text{Var}_{\text{RF}} = (0.05)(1.0) + \frac{1 - 0.05}{100}(1.0) = 0.05 + \frac{0.95}{100} = 0.05 + 0.0095 = 0.0595$$

**Step 3: Variance Reduction**
$$\text{Reduction} = \frac{0.406 - 0.0595}{0.406} \times 100\% = \frac{0.3465}{0.406} \times 100\% \approx 85.34\%$$
Random feature subsampling cuts the remaining variance by **over 85%**!

$\blacksquare$
</details>

---

### Exercise 4: Gradient Boosting Pseudo-Residuals

**Problem:**  
Consider a regression task with $N = 3$ samples: targets $y = [10.0, 20.0, 30.0]^T$.
We initialize our boosting model with a constant initial prediction $F_0(x) = \bar{y}$ (the sample mean).
The loss function is squared error: $\mathcal{L}(y, F) = \frac{1}{2}(y - F)^2$.

1. Compute the baseline initial prediction $F_0$.
2. Compute the pseudo-residuals $r_{i,1} = -\left[\frac{\partial \mathcal{L}}{\partial F}\right] = y_i - F_0$ for each sample.
3. If tree $h_1(x)$ predicts $[\hat{r}_1, \hat{r}_2, \hat{r}_3] = [-8.0, 0.0, +8.0]$ with learning rate $\nu = 0.1$, compute the updated predictions $F_1(x) = F_0 + \nu \cdot h_1(x)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Baseline prediction $F_0$**
$$F_0 = \frac{10.0 + 20.0 + 30.0}{3} = \frac{60.0}{3} = 20.0$$

**Step 2: Pseudo-residuals $r_{i,1} = y_i - 20.0$**
- $r_{1,1} = 10.0 - 20.0 = -10.0$
- $r_{2,1} = 20.0 - 20.0 = 0.0$
- $r_{3,1} = 30.0 - 20.0 = +10.0$

**Step 3: Updated model predictions $F_1(x)$ ($\nu = 0.1$)**
- $F_1(x_1) = 20.0 + 0.1(-8.0) = 20.0 - 0.8 = 19.2$ (closer to 10.0)
- $F_1(x_2) = 20.0 + 0.1(0.0) = 20.0$ (exact match)
- $F_1(x_3) = 20.0 + 0.1(+8.0) = 20.0 + 0.8 = 20.8$ (closer to 30.0)

Each boosting step nudges predictions steadily toward true labels without overfitting.

$\blacksquare$
</details>
