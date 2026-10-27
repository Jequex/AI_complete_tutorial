# Topic 4.2: Exercises & Solutions — Random Variables, PMF, PDF, and CDF

Welcome to the problem set for **Topic 4.2: Random Variables, PMF, PDF, and CDF**.

Follow along with pen and paper. Solve each problem before revealing the collapsible solutions below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **PDF Normalization Constant & CDF Calculus** | Medium | Continuous likelihood evaluation in energy-based models and flows |
| **2** | **Inverse Transform Sampling for the Pareto Distribution** | Medium | Synthetic heavy-tailed data generation in Monte Carlo simulations |
| **3** | **Joint Distribution Matrix: Marginals, Conditioning & Independence** | Challenging | Information routing in multi-modal neural networks |

---

## Problem 1: PDF Normalization Constant & CDF Calculus

### 🧒 The Context: Maya's Rain Accumulation Tank
Maya measures rain collecting in a wedge-shaped collector over 3 hours:
$$f_X(x) = \begin{cases} c x^2 & \text{if } 0 \le x \le 3 \\ 0 & \text{otherwise} \end{cases}$$

### Tasks:
1. Determine the exact value of constant $c$ so that $f_X(x)$ is a valid Probability Density Function.
2. Derive the full Cumulative Distribution Function $F_X(x)$ for all real numbers $x \in \mathbb{R}$.
3. Calculate the probability that rainfall lands between 1 and 2 inches: $P(1 \le X \le 2)$.
4. Find the median $m$ of the distribution (where $F_X(m) = 0.50$).

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Finding Normalization Constant $c$
For $f_X(x)$ to be a valid PDF, the total area under the curve must equal $1.0$:
$$\int_{-\infty}^\infty f_X(x) \, dx = \int_0^3 c x^2 \, dx = 1.0$$
$$\left[ c \frac{x^3}{3} \right]_0^3 = c \left( \frac{27}{3} - 0 \right) = 9c = 1.0 \implies \mathbf{c = \frac{1}{9}}$$
$$f_X(x) = \frac{1}{9} x^2 \quad \text{for } 0 \le x \le 3$$

#### Step 2: Deriving the CDF $F_X(x)$
- For $x < 0$: $F_X(x) = 0$.
- For $0 \le x \le 3$:
  $$F_X(x) = \int_0^x \frac{1}{9} t^2 \, dt = \left[ \frac{t^3}{27} \right]_0^x = \mathbf{\frac{x^3}{27}}$$
- For $x > 3$: $F_X(x) = 1.0$.

$$F_X(x) = \begin{cases} 0 & \text{if } x < 0 \\ \frac{x^3}{27} & \text{if } 0 \le x \le 3 \\ 1.0 & \text{if } x > 3 \end{cases}$$

#### Step 3: Interval Probability $P(1 \le X \le 2)$
Using the CDF:
$$P(1 \le X \le 2) = F_X(2) - F_X(1) = \frac{2^3}{27} - \frac{1^3}{27} = \frac{8 - 1}{27} = \mathbf{\frac{7}{27} \approx 0.2593}$$
There is a $25.93\%$ chance that rainfall lands between 1 and 2 inches.

#### Step 4: Finding the Median $m$
Set $F_X(m) = 0.50$:
$$\frac{m^3}{27} = 0.50 \implies m^3 = 27 \times 0.50 = 13.5$$
$$m = (13.5)^{1/3} \approx \mathbf{2.3811}$$
By 2.38 hours, half the total rain volume has fallen!
</details>

---

## Problem 2: Inverse Transform Sampling for the Pareto Distribution

### 🧒 The Context: Timmy's 80/20 Wealth Simulator
The **Pareto distribution** models "heavy-tailed" real-world phenomena (e.g., $80\%$ of wealth owned by $20\%$ of people, or $80\%$ of web traffic hitting $20\%$ of servers):
$$F_X(x) = 1 - \left( \frac{x_m}{x} \right)^\alpha \quad \text{for } x \ge x_m$$
where $x_m > 0$ is the minimum value and $\alpha > 0$ is the shape parameter.

### Tasks:
1. Invert the CDF formula to find the quantile function $x = F^{-1}(u)$.
2. If $x_m = 10$ and $\alpha = 2$, generate the exact value of $X$ corresponding to uniform random draw $U = 0.75$.
3. What is the value of $X$ when $U = 0.99$? Notice how rare extreme values emerge!

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Inverting the CDF
Set $u = F_X(x)$:
$$u = 1 - \left( \frac{x_m}{x} \right)^\alpha$$
$$1 - u = \left( \frac{x_m}{x} \right)^\alpha$$
Take the $\frac{1}{\alpha}$ power on both sides:
$$(1 - u)^{1/\alpha} = \frac{x_m}{x}$$
$$x = \frac{x_m}{(1 - u)^{1/\alpha}} = \mathbf{x_m (1 - u)^{-1/\alpha}}$$

Since $U \sim \text{Uniform}(0, 1)$, $1 - U$ is also distributed as $\text{Uniform}(0, 1)$, so we can write:
$$\mathbf{X = x_m U^{-1/\alpha}}$$

#### Step 2: Sampling for $x_m = 10, \alpha = 2, U = 0.75$
$$X = \frac{10}{(1 - 0.75)^{1/2}} = \frac{10}{(0.25)^{1/2}} = \frac{10}{0.50} = \mathbf{20.0}$$

#### Step 3: Sampling for $U = 0.99$ (99th Percentile)
$$X = \frac{10}{(1 - 0.99)^{1/2}} = \frac{10}{(0.01)^{1/2}} = \frac{10}{0.10} = \mathbf{100.0}$$
While $75\%$ of samples are under $20$, the 99th percentile jumps $5\times$ higher to $100$! Inverse transform sampling accurately reflects extreme real-world distributions.
</details>

---

## Problem 3: Joint Distribution Matrix: Marginals, Conditioning & Independence

### 🧒 The Context: Maya & Timmy's Multi-Modal Robot Sensor
A robot combines visual camera detections ($X \in \{0, 1\}$) with audio mic detections ($Y \in \{0, 1\}$).
Their joint probability matrix $P(X = x, Y = y)$ is:

| | $Y = 0$ (Silence) | $Y = 1$ (Audio Detected) |
| :--- | :---: | :---: |
| **$X = 0$ (No Object)** | $0.40$ | $0.10$ |
| **$X = 1$ (Object Seen)** | $0.20$ | $0.30$ |

### Tasks:
1. Compute the marginal PMFs $p_X(x)$ and $p_Y(y)$ by summing across rows and columns.
2. If the camera detects an object ($X = 1$), compute the conditional probability that audio is also heard: $P(Y = 1 \mid X = 1)$.
3. Are the camera and microphone statistically independent? Prove your answer using the independence criterion $P(X \cap Y) \stackrel{?}{=} P(X) P(Y)$.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Marginal Distributions
- **Marginal for $X$ (Row Sums):**
  $$p_X(0) = P(X=0, Y=0) + P(X=0, Y=1) = 0.40 + 0.10 = \mathbf{0.50}$$
  $$p_X(1) = P(X=1, Y=0) + P(X=1, Y=1) = 0.20 + 0.30 = \mathbf{0.50}$$
- **Marginal for $Y$ (Column Sums):**
  $$p_Y(0) = P(X=0, Y=0) + P(X=1, Y=0) = 0.40 + 0.20 = \mathbf{0.60}$$
  $$p_Y(1) = P(X=0, Y=1) + P(X=1, Y=1) = 0.10 + 0.30 = \mathbf{0.40}$$

#### Step 2: Conditional Probability $P(Y = 1 \mid X = 1)$
$$P(Y = 1 \mid X = 1) = \frac{P(X = 1, Y = 1)}{p_X(1)} = \frac{0.30}{0.50} = \mathbf{0.60} \quad (60\%)$$
Notice that prior to visual detection, audio chance was only $p_Y(1) = 40\%$. Seeing an object boosted audio likelihood to $60\%$!

#### Step 3: Testing Statistical Independence
For $X$ and $Y$ to be independent, we must have $P(X=x, Y=y) = p_X(x) p_Y(y)$ for all entries.
Let us test entry $(X = 1, Y = 1)$:
- Actual Joint: $P(X = 1, Y = 1) = \mathbf{0.30}$
- Product of Marginals: $p_X(1) \cdot p_Y(1) = 0.50 \times 0.40 = \mathbf{0.20}$

Since $0.30 \neq 0.20$, **$X$ and $Y$ are NOT independent**!
The sensors are positively correlated: seeing an object makes hearing audio significantly more likely.
</details>
