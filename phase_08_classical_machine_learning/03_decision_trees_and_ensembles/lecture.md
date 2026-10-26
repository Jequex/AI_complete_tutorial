# Topic 8.3: Decision Trees & Ensemble Methods

---

## 1. Executive Summary & The 12-Year-Old Analogy

Linear models and logistic regression assume that the underlying relationship between inputs and targets can be separated by flat hyperplanes. But nature is rarely linear. Many real-world problems involve non-linear, hierarchical, conditional rules:
- *"If the patient's age > 60 AND systolic blood pressure > 140, then risk is high; ELSE IF age <= 60 AND cholesterol > 240, check family history..."*

**Decision Trees** learn recursive axis-aligned partitions of feature space. When individual trees are combined into **Ensembles**—via **Bagging (Random Forests)** or **Boosting (Gradient Boosted Decision Trees)**—they produce some of the most competitive, robust, and interpretable algorithms for tabular data in modern machine learning.

```
                  [ Age > 50? ]
                  /           \
               Yes             No
              /                 \
     [ Smoker? ]            [ Cholesterol > 200? ]
      /       \               /                \
    Yes        No           Yes                 No
    /           \           /                    \
 [High Risk] [Med Risk]  [Med Risk]          [Low Risk]
```

---

### The 12-Year-Old Analogy: The 20 Questions Game & The Council of Friends

Imagine you and your friends are trying to guess a mystery creature.

1. **The Decision Tree (The 20 Questions Game):**  
   You start at the top: *"Does it have feathers?"*  
   - If **Yes**, you ask: *"Can it swim?"* -> If Yes, it's a Duck! If No, an Eagle.  
   - If **No**, you ask: *"Does it have four legs?"*  
   Every question splits the universe of possibilities down the middle.

2. **Gini Impurity & Entropy (The Messy Marble Jar):**  
   Suppose you have a jar containing 50 bright red marbles and 50 ocean-blue marbles. If you pull a marble at random, you are completely unsure what you'll get. The jar is in a state of **maximum chaos (high impurity)**.  
   - If a question separates them into one jar with 50 red marbles and another jar with 50 blue marbles, both jars are now **100% pure (zero impurity)**!  
   - The quality of your question is measured by **Information Gain**: how much disorder did your question eliminate?

3. **Random Forest (The Council of 100 Friends):**  
   If you rely on just one friend to diagnose your computer problem, they might be obsessed with hard drive failures and guess wrong (a single tree has **high variance** and overfits).  
   Instead, you assemble a council of 100 friends:
   - Each friend is given a slightly different set of diagnostic notes (**Bootstrap Sampling**).
   - Each friend is only allowed to inspect a few random cables or parts at each step (**Feature Subsampling**).  
   When all 100 diverse friends vote, their individual quirks cancel out, and the democratic consensus is shockingly accurate!

4. **Gradient Boosting (The Relay Team of Apprentices):**  
   Instead of everyone voting at once, they work in a disciplined assembly line:
   - The first apprentice builds a simple model. It gets 70% right, but misses 30%.
   - The second apprentice doesn't re-solve the whole problem; he inspects **only the 30% mistakes** (the residuals) and builds a tiny model to correct them.
   - The third apprentice fixes whatever mistakes remain after the first two.  
   Together, they form a powerful ensemble that drives errors down to nearly zero!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Type | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{D} = \{(x_i, y_i)\}$ | Dataset | Set of $N$ instances with $x_i \in \mathbb{R}^d$ and discrete class $y_i \in \{1, \dots, C\}$ |
| $p_k$ | Probability | Proportion of training instances belonging to class $k$ in node $T$: $\frac{1}{|T|} \sum_{i \in T} \mathbb{I}(y_i = k)$ |
| $G(T)$ | $\mathbb{R} \in [0, 1 - \frac{1}{C}]$ | Gini Impurity of node $T$: $1 - \sum_{k=1}^C p_k^2$ |
| $H(T)$ | $\mathbb{R} \ge 0$ | Shannon Entropy of node $T$ (in bits): $-\sum_{k=1}^C p_k \log_2 p_k$ |
| $IG(T, a, \theta)$ | $\mathbb{R}_{\ge 0}$ | Information Gain achieved by splitting node $T$ on feature $a$ at threshold $\theta$ |
| $T_L, T_R$ | Subsets | Left and right child nodes resulting from binary partition: $\{x_a \le \theta\}$ and $\{x_a > \theta\}$ |
| $M$ | Integer | Number of individual base estimators (trees) in an ensemble |
| $\rho$ | Correlation $\in [0, 1]$ | Pairwise correlation between predictions of individual ensemble trees |
| $r_{im}$ | Real scalar | Pseudo-residual (negative gradient) for sample $i$ at boosting stage $m$ |

---

## 3. Node Impurity Metrics: Gini vs. Shannon Entropy

At each node $T$ of a classification tree containing $|T|$ samples, we define the class distribution vector $p = [p_1, p_2, \dots, p_C]$, where:
$$p_k = \frac{1}{|T|} \sum_{i \in T} \mathbb{I}(y_i = k), \quad \sum_{k=1}^C p_k = 1$$

### 3.1 Gini Impurity (CART Criterion)

The Gini Impurity measures the probability that a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the class distribution:
$$G(T) = \sum_{k=1}^C p_k (1 - p_k) = \sum_{k=1}^C (p_k - p_k^2) = \sum_{k=1}^C p_k - \sum_{k=1}^C p_k^2 = 1 - \sum_{k=1}^C p_k^2$$

- **Pure node:** If all samples belong to a single class $k$, then $p_k = 1$ and $p_{j \ne k} = 0$:
  $$G(T) = 1 - 1^2 = 0 \quad (\text{Minimum Impurity})$$
- **Uniformly mixed binary node:** If $p_1 = 0.5, p_2 = 0.5$:
  $$G(T) = 1 - (0.5^2 + 0.5^2) = 1 - 0.50 = 0.50 \quad (\text{Maximum Binary Impurity})$$

---

### 3.2 Shannon Entropy & Information Gain (ID3 / C4.5)

In information theory, **Shannon Entropy** quantifies the average number of bits required to encode the class identity of an arbitrary sample:
$$H(T) = -\sum_{k=1}^C p_k \log_2(p_k)$$
*(By convention, $0 \log_2(0) \equiv 0$).*

For binary classification ($p_1 = p$, $p_2 = 1 - p$):
$$H(T) = -p \log_2(p) - (1-p) \log_2(1-p)$$
When $p = 0.5$, $H(T) = -0.5(-1) - 0.5(-1) = 1.0\text{ bit}$. When $p \in \{0, 1\}$, $H(T) = 0\text{ bits}$.

---

### 3.3 Evaluating a Candidate Split

When a parent node $T$ is split into left child $T_L$ and right child $T_R$ using feature $j$ and numerical threshold $\theta$:
$$T_L = \{i \in T : x_{i,j} \le \theta\}, \quad T_R = \{i \in T : x_{i,j} > \theta\}$$

The **Weighted Impurity** of the candidate split is:
$$I(T_L, T_R) = \frac{|T_L|}{|T|} I(T_L) + \frac{|T_R|}{|T|} I(T_R)$$

The **Information Gain (or Impurity Reduction)** is:
$$\Delta I(T, j, \theta) = I(T) - \left[ \frac{|T_L|}{|T|} I(T_L) + \frac{|T_R|}{|T|} I(T_R) \right]$$

The CART algorithm selects the optimal split $(j^*, \theta^*)$ by maximizing this reduction:
$$(j^*, \theta^*) = \arg\max_{j, \theta} \Delta I(T, j, \theta)$$

---

## 4. The CART Algorithm & Stopping Criteria

The CART (Classification and Regression Trees) algorithm grows binary trees recursively:

```text
Algorithm: BuildTree(Node T, CurrentDepth d)
1. If StoppingCriteriaMet(T, d):
     Return LeafNode(predicted_class = mode(y in T))
2. For each feature j in 1..d:
     For each possible split threshold theta in sorted unique values of x[:, j]:
       Compute weighted impurity reduction Delta I
3. Find (j*, theta*) that maximizes Delta I
4. If Delta I <= min_impurity_decrease:
     Return LeafNode(predicted_class = mode(y in T))
5. Partition T into T_L = {x_j <= theta*} and T_R = {x_j > theta*}
6. LeftChild = BuildTree(T_L, d + 1)
7. RightChild = BuildTree(T_R, d + 1)
8. Return DecisionNode(feature=j*, threshold=theta*, LeftChild, RightChild)
```

### Stopping Criteria to Prevent Overfitting
An unconstrained decision tree will continue splitting until every leaf contains exactly one sample, achieving 100% training accuracy but memorizing noise. Regularization hyperparameters include:
- `max_depth`: Maximum vertical distance from root to leaf.
- `min_samples_split`: Minimum instances required at a node to attempt a split.
- `min_samples_leaf`: Minimum instances required in any resulting terminal leaf.
- `min_impurity_decrease`: Minimum threshold improvement required to justify splitting.

---

## 5. Ensemble Methods: Bagging, Random Forests & Boosting

### 5.1 Bagging (Bootstrap Aggregation)

A single decision tree has **low bias** (it can model complex non-linear boundaries) but **high variance** (small fluctuations in the training set lead to completely different tree structures).

**Bagging** reduces variance through statistical averaging. Given training set $\mathcal{D}$ of size $N$:
1. Sample $N$ instances with replacement (a *bootstrap sample* $\mathcal{D}_m$) for $m = 1, \dots, M$.  
   *(Each bootstrap sample contains $\approx 1 - e^{-1} \approx 63.2\%$ of unique instances; the remaining $36.8\%$ form the Out-Of-Bag (OOB) validation set).*
2. Train an independent unpruned decision tree $f_m(x)$ on each $\mathcal{D}_m$.
3. Aggregate predictions via majority vote (classification) or mean average (regression):
$$\hat{f}_{\text{bag}}(x) = \frac{1}{M} \sum_{m=1}^M f_m(x)$$

---

### 5.2 Mathematical Variance Reduction Law & Random Forests

Let each tree $f_m(x)$ have variance $\sigma^2$, with pairwise correlation $\rho = \text{Corr}(f_i, f_j)$.  
The variance of the ensemble average is:
$$\text{Var}\left(\frac{1}{M} \sum_{m=1}^M f_m(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$

- As $M \to \infty$, the second term $\frac{1 - \rho}{M} \sigma^2 \to 0$.
- However, the first term $\rho \sigma^2$ remains! **The correlation between trees creates a fundamental performance ceiling.**

#### The Random Forest Breakthrough (Breiman, 2001)
To break this correlation ceiling and force $\rho \to 0$, **Random Forests** introduce **Feature Subsampling**:
At every single split in every tree, instead of evaluating all $d$ features, the algorithm considers only a randomly selected subset of $k$ features:
$$k = \lfloor \sqrt{d} \rfloor \quad (\text{for classification}), \quad k = \lfloor d / 3 \rfloor \quad (\text{for regression})$$

Because dominant features cannot dominate every tree, individual trees become substantially decorrelated ($\rho \ll 1$), driving total ensemble variance down drastically!

---

### 5.3 Gradient Boosted Decision Trees (GBDT)

While Random Forests train trees in parallel to reduce variance, **Gradient Boosting** trains shallow trees (e.g. depth 3–6) sequentially to reduce bias.

We construct an additive expansion:
$$F_M(x) = \sum_{m=0}^M \gamma_m h_m(x)$$

At stage $m$, we do not adjust previous trees. Instead, we compute the **pseudo-residuals** (the negative gradient of the loss function evaluated at current predictions):
$$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

For Squared Error $\mathcal{L}(y, F) = \frac{1}{2}(y - F)^2$, the pseudo-residual is the exact residual:
$$r_{im} = y_i - F_{m-1}(x_i)$$

The new tree $h_m(x)$ is fitted to predict these pseudo-residuals $r_{im}$. The ensemble is updated with a shrinkage factor (learning rate $\nu \in (0, 1]$):
$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$
