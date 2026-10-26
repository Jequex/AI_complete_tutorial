# Exercises & Detailed Solutions: Topic 0.1

Welcome to the problem set for **Topic 0.1: What is Intelligence?**

Work through each problem carefully before reading the solution. Every mathematical calculation includes the complete step-by-step arithmetic breakdown.

---

## 📝 Problem Set

### Problem 1: The Smart Thermostat (Conceptual Drill)
A modern home thermostat has a digital sensor. If the room temperature drops below $68^\circ\text{F}$, it turns on the furnace. If the temperature exceeds $72^\circ\text{F}$, it turns off the furnace.
- **Part A:** Is this thermostat performing computation, or does it exhibit genuine intelligence? Explain using the definition of intelligence from the lecture.
- **Part B:** What specific capability would you need to add to this thermostat for it to qualify as an adaptive, intelligent agent?

---

### Problem 2: Mathematical Return Calculation (Hand Derivation)
An intelligent cleaning robot is exploring a living room modeled as a Markov Decision Process (MDP).
At time step $t=0$, the robot is at the charging dock. It takes 4 steps, receiving the following sequence of rewards:
- At $t=1$: $R_1 = -1$ (energy spent moving forward)
- At $t=2$: $R_2 = -1$ (energy spent turning left)
- At $t=3$: $R_3 = +10$ (vacuumed a pile of dust)
- At $t=4$: $R_4 = +50$ (docked safely and recharged)

The discount factor is set to:
$$\gamma = 0.5$$

- **Part A:** State the formal mathematical formula for the discounted return $G_0$ starting from $t=0$.
- **Part B:** Calculate the exact numerical value of $G_0$ step-by-step.
- **Part C:** Now recalculate the return if the discount factor was $\gamma = 0.9$. Compare the two results: how does changing $\gamma$ alter the agent's behavior?

---

### Problem 3: The Generalization Gap (Overfitting Proof)
Suppose a student trains an image classifier on a dataset of 5,000 photos of cats and dogs.
The model achieves:
$$\widehat{\mathcal{R}}_{\text{train}} = 0.00 \quad (0\% \text{ error on training data})$$
When deployed on an app used by thousands of pet owners, the error rate is measured at:
$$\mathcal{R}_{\text{test}} = 0.48 \quad (48\% \text{ error on real-world inputs})$$

- **Part A:** What is the numerical value of the **Generalization Gap**?
- **Part B:** What does this indicate about what the model actually learned?
- **Part C:** If this is a binary classification task (cat vs. dog), how does $48\%$ error compare to a completely random coin flip?

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: The Smart Thermostat

#### Part A: Computation vs. Intelligence
The basic thermostat is performing **pure computation**, not genuine intelligence.
- **Why:** Its behavior is completely hard-coded by a simple deterministic rule:
  $$\text{Action} = \begin{cases} \text{Heater ON} & \text{if } T < 68 \\ \text{Heater OFF} & \text{if } T > 72 \\ \text{Maintain} & \text{otherwise} \end{cases}$$
- It has no internal representation of the future, cannot learn from past mistakes, and cannot adapt if circumstances change (e.g. if a window is wide open during a blizzard, it will burn fuel endlessly without noticing that the room isn't warming up).

#### Part B: Transforming it into an Intelligent Agent
To make it an intelligent agent, it would need:
1. **Sensory Observations beyond instantaneous temperature:** e.g., weather forecasts, time of day, whether occupants are home, how fast the home loses heat.
2. **Learning & Prediction:** Learning the thermodynamic properties of the specific house (e.g., *"It takes 35 minutes to raise the master bedroom by $2^\circ\text{F}$ when it is $20^\circ\text{F}$ outside"*).
3. **Objective Optimization:** Balancing an explicit trade-off: minimizing utility bill cost (\$) while maximizing human comfort.

---

### Solution 2: Mathematical Return Calculation

#### Part A: The Return Formula
The discounted cumulative return $G_t$ starting from time $t$ is:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots$$
For our 4-step trajectory starting at $t=0$:
$$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4$$

---

#### Part B: Step-by-Step Calculation for $\gamma = 0.5$
Substitute the rewards and discount factor:
$$R_1 = -1, \quad R_2 = -1, \quad R_3 = +10, \quad R_4 = +50, \quad \gamma = 0.5$$

1. Calculate the powers of $\gamma$:
   $$\gamma^0 = 1.0$$
   $$\gamma^1 = 0.5$$
   $$\gamma^2 = (0.5)^2 = 0.25$$
   $$\gamma^3 = (0.5)^3 = 0.125$$

2. Multiply each reward by its discount weight:
   $$\text{Step 1 term:} \quad 1.0 \times (-1) = -1.0$$
   $$\text{Step 2 term:} \quad 0.5 \times (-1) = -0.5$$
   $$\text{Step 3 term:} \quad 0.25 \times 10 = +2.5$$
   $$\text{Step 4 term:} \quad 0.125 \times 50 = +6.25$$

3. Sum the terms together:
   $$G_0 = -1.0 + (-0.5) + 2.5 + 6.25$$
   $$G_0 = -1.5 + 8.75 = \mathbf{7.25}$$

---

#### Part C: Step-by-Step Calculation for $\gamma = 0.9$
Now compute with a patient agent ($\gamma = 0.9$):

1. Calculate the powers of $\gamma$:
   $$\gamma^0 = 1.0$$
   $$\gamma^1 = 0.9$$
   $$\gamma^2 = (0.9)^2 = 0.81$$
   $$\gamma^3 = (0.9)^3 = 0.729$$

2. Multiply each reward by its discount weight:
   $$\text{Step 1 term:} \quad 1.0 \times (-1) = -1.0$$
   $$\text{Step 2 term:} \quad 0.9 \times (-1) = -0.9$$
   $$\text{Step 3 term:} \quad 0.81 \times 10 = +8.1$$
   $$\text{Step 4 term:} \quad 0.729 \times 50 = +36.45$$

3. Sum the terms together:
   $$G_0 = -1.0 - 0.9 + 8.1 + 36.45$$
   $$G_0 = -1.9 + 44.55 = \mathbf{42.65}$$

**Intuition & Comparison:**
- At $\gamma = 0.5$, future rewards decay aggressively: the $+50$ reward four steps away was squashed to only $+6.25$.
- At $\gamma = 0.9$, the agent strongly values distant goals ($+36.45$ contribution). High $\gamma$ creates an agent that is willing to endure short-term penalties (like moving and turning) to secure large future prizes.

---

### Solution 3: The Generalization Gap

#### Part A: Computing the Gap
The generalization gap is defined as the absolute difference between test risk (real-world error) and training risk:
$$\text{Generalization Gap} = |\mathcal{R}_{\text{test}} - \widehat{\mathcal{R}}_{\text{train}}|$$
$$\text{Generalization Gap} = |0.48 - 0.00| = \mathbf{0.48} \quad (48\%)$$

#### Part B: Interpretation
The model did **not** learn the generalized visual features of cats and dogs (such as eye shapes, snout contours, or ear structures). Instead, it **overfitted (memorized)** the training dataset. For instance, it may have simply memorized the specific background pixels or file hashes of the 5,000 training photos.

#### Part C: Comparison with Chance
In a 2-class balanced classification problem (cats vs. dogs), a blind coin flip has an error rate of $50\%$ ($0.50$). An error rate of $48\%$ ($0.48$) means this multi-million parameter neural network is performing **scarcely better than a random guess on new images**.

