# Topic 8.5: Exercises & Solutions — Naive Bayes & K-Nearest Neighbors

---

### Exercise 1: Multinomial Naive Bayes Spam Filter with Laplace Smoothing

**Problem:**  
You are building an email spam filter with a vocabulary of $|V| = 4$ unique words:
$$V = \{\text{"deal"}, \text{"free"}, \text{"meeting"}, \text{"project"}\}$$

Your training set consists of:
- **Spam ($y = 1$, 3 messages):**
  - Message 1: "deal free free"
  - Message 2: "deal free"
  - Message 3: "free"
- **Ham ($y = 0$, 2 messages):**
  - Message 4: "meeting project"
  - Message 5: "meeting deal project"

1. Compute class priors $P(\text{Spam})$ and $P(\text{Ham})$.
2. Compute the total word count $N_{\text{spam}}$ and $N_{\text{ham}}$.
3. Using Laplace smoothing ($\alpha = 1$), compute $P(w \mid \text{Spam})$ and $P(w \mid \text{Ham})$ for all 4 words.
4. Classify a new test message: **"deal free project"**. Which class has higher posterior probability?

<details>
<summary>View Complete Solution</summary>

**Step 1: Class Priors ($N = 5$ total messages)**
$$P(\text{Spam}) = \frac{3}{5} = 0.6, \quad P(\text{Ham}) = \frac{2}{5} = 0.4$$

**Step 2: Word Counts per Class**
- Spam word occurrences: "deal": 2, "free": 4, "meeting": 0, "project": 0.  
  Total $N_{\text{spam}} = 2 + 4 + 0 + 0 = 6$ words.
- Ham word occurrences: "deal": 1, "free": 0, "meeting": 2, "project": 2.  
  Total $N_{\text{ham}} = 1 + 0 + 2 + 2 = 5$ words.

**Step 3: Laplace Smoothed Likelihoods ($\alpha = 1, |V| = 4$)**
Formula: $P(w \mid c) = \frac{\text{count}(w, c) + 1}{N_c + |V|}$.
- Denominator for Spam: $6 + 4 = 10$.
  - $P(\text{"deal"} \mid \text{Spam}) = \frac{2 + 1}{10} = 0.3$
  - $P(\text{"free"} \mid \text{Spam}) = \frac{4 + 1}{10} = 0.5$
  - $P(\text{"meeting"} \mid \text{Spam}) = \frac{0 + 1}{10} = 0.1$
  - $P(\text{"project"} \mid \text{Spam}) = \frac{0 + 1}{10} = 0.1$

- Denominator for Ham: $5 + 4 = 9$.
  - $P(\text{"deal"} \mid \text{Ham}) = \frac{1 + 1}{9} = 2/9 \approx 0.2222$
  - $P(\text{"free"} \mid \text{Ham}) = \frac{0 + 1}{9} = 1/9 \approx 0.1111$
  - $P(\text{"meeting"} \mid \text{Ham}) = \frac{2 + 1}{9} = 3/9 \approx 0.3333$
  - $P(\text{"project"} \mid \text{Ham}) = \frac{2 + 1}{9} = 3/9 \approx 0.3333$

**Step 4: Score the Query: "deal free project"**
- **Spam Score:**
  $$\text{Score}(\text{Spam}) = P(\text{Spam}) \times P(\text{"deal"} \mid \text{Spam}) \times P(\text{"free"} \mid \text{Spam}) \times P(\text{"project"} \mid \text{Spam})$$
  $$\text{Score}(\text{Spam}) = 0.6 \times 0.3 \times 0.5 \times 0.1 = 0.0090$$

- **Ham Score:**
  $$\text{Score}(\text{Ham}) = P(\text{Ham}) \times P(\text{"deal"} \mid \text{Ham}) \times P(\text{"free"} \mid \text{Ham}) \times P(\text{"project"} \mid \text{Ham})$$
  $$\text{Score}(\text{Ham}) = 0.4 \times \left(\frac{2}{9}\right) \times \left(\frac{1}{9}\right) \times \left(\frac{3}{9}\right) = 0.4 \times \frac{6}{729} = \frac{2.4}{729} \approx 0.00329$$

**Conclusion:**  
$\text{Score}(\text{Spam}) = 0.0090 > \text{Score}(\text{Ham}) = 0.00329$.  
The email is classified as **SPAM** with posterior probability:
$$P(\text{Spam} \mid \text{email}) = \frac{0.0090}{0.0090 + 0.00329} \approx \frac{0.0090}{0.01229} \approx 73.2\%$$

$\blacksquare$
</details>

---

### Exercise 2: K-Nearest Neighbors Distance and Voting

**Problem:**  
You have a 2D dataset with 6 labeled points:
- $P_1 = (1, 2)$, Class A
- $P_2 = (2, 3)$, Class A
- $P_3 = (3, 3)$, Class A
- $P_4 = (5, 5)$, Class B
- $P_5 = (6, 4)$, Class B
- $P_6 = (7, 6)$, Class B

You receive a new query point $Q = (4, 3)$.

1. Calculate the Euclidean distance $d_2(Q, P_i)$ from $Q$ to all 6 points.
2. Find the $k = 1$ nearest neighbor and determine the predicted class.
3. Find the $k = 3$ nearest neighbors and determine the predicted class.
4. If Manhattan distance $d_1$ were used instead for $k = 3$, does the classification change?

<details>
<summary>View Complete Solution</summary>

**Step 1: Euclidean Distances $d_2(Q, P_i)$ where $Q = (4, 3)$**
- $d_2(Q, P_1) = \sqrt{(4-1)^2 + (3-2)^2} = \sqrt{9 + 1} = \sqrt{10} \approx 3.162$
- $d_2(Q, P_2) = \sqrt{(4-2)^2 + (3-3)^2} = \sqrt{4 + 0} = 2.000$
- $d_2(Q, P_3) = \sqrt{(4-3)^2 + (3-3)^2} = \sqrt{1 + 0} = 1.000$
- $d_2(Q, P_4) = \sqrt{(4-5)^2 + (3-5)^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2.236$
- $d_2(Q, P_5) = \sqrt{(4-6)^2 + (3-4)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.236$
- $d_2(Q, P_6) = \sqrt{(4-7)^2 + (3-6)^2} = \sqrt{9 + 9} = \sqrt{18} \approx 4.243$

**Step 2: $k = 1$ Nearest Neighbor**
The closest point is $P_3$ (dist $= 1.000$). Since $P_3$ belongs to Class A, prediction is **Class A**.

**Step 3: $k = 3$ Nearest Neighbors (Euclidean)**
The 3 smallest distances are:
1. $P_3$ (dist $= 1.000$, Class A)
2. $P_2$ (dist $= 2.000$, Class A)
3. Tied between $P_4$ and $P_5$ (dist $= 2.236$, both Class B)

Votes: 2 votes for Class A ($P_3, P_2$), 1 vote for Class B ($P_4$ or $P_5$).  
Majority vote: **Class A** wins 2-to-1.

**Step 4: Manhattan Distances $d_1(Q, P_i) = |4 - x_1| + |3 - x_2|$**
- $d_1(Q, P_1) = |4-1| + |3-2| = 3 + 1 = 4$
- $d_1(Q, P_2) = |4-2| + |3-3| = 2 + 0 = 2$
- $d_1(Q, P_3) = |4-3| + |3-3| = 1 + 0 = 1$
- $d_1(Q, P_4) = |4-5| + |3-5| = 1 + 2 = 3$
- $d_1(Q, P_5) = |4-6| + |3-4| = 2 + 1 = 3$
- $d_1(Q, P_6) = |4-7| + |3-6| = 3 + 3 = 6$

For $k = 3$ with Manhattan distance:
- Closest 3 points: $P_3$ (dist 1, Class A), $P_2$ (dist 2, Class A), and $P_4$ or $P_5$ (dist 3, Class B).  
Class A still wins 2-to-1! The decision is robust across distance norms.

$\blacksquare$
</details>

---

### Exercise 3: The Curse of Dimensionality

**Problem:**  
Consider a unit hypercube $[0, 1]^d$ with uniform data distribution. To capture a fixed fraction $r = 0.05$ (5% of the data) using a sub-cube centered at a query point, the required edge length of the sub-cube is:
$$e_d(r) = r^{1/d}$$

1. Compute $e_1(0.05)$ for dimension $d = 1$.
2. Compute $e_{10}(0.05)$ for dimension $d = 10$.
3. Compute $e_{100}(0.05)$ for dimension $d = 100$.
4. Explain why this mathematical result proves that "local" neighborhoods cease to exist in high dimensions.

<details>
<summary>View Complete Solution</summary>

**Step 1: $d = 1$**
$$e_1 = (0.05)^{1/1} = 0.05$$
In 1D, you only need to look across 5% of the feature range to capture 5% of the data.

**Step 2: $d = 10$**
$$e_{10} = (0.05)^{1/10} \approx 0.7411$$
In 10D, to capture just 5% of the data, your "neighborhood" must span over **74.1%** of the entire range of every single feature!

**Step 3: $d = 100$**
$$e_{100} = (0.05)^{1/100} \approx 0.9705$$
In 100D, the neighborhood must span **97.1%** of the entire range along every dimension!

**Conclusion:**  
To find even a few neighbors in high dimensions, the neighborhood cannot be "local" at all—it must cover virtually the entire volume of the dataset. Therefore, the concept of a "nearest" neighbor collapses, rendering unregularized KNN ineffective in high dimensions.

$\blacksquare$
</details>
