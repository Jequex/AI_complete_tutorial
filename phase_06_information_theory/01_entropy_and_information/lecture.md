# Topic 6.1: Entropy & The Fundamental Currency of Information

---

## 1. Executive Summary & Intuitive Foundations

### What is Information?
In everyday language, "information" means knowledge, facts, or news.
In 1948, a brilliant mathematician and engineer at Bell Labs named **Claude Shannon** published a paper that created the modern digital world: *"A Mathematical Theory of Communication"*.

Shannon asked a revolutionary question:
> *"Can we measure the exact numerical amount of information contained in a message, independent of what the message means?"*

His discovery was breathtaking:
**Information is mathematically equal to the reduction of uncertainty.**
The more uncertain an event is before it happens, the more information you gain when you find out the outcome!

---

### The 12-Year-Old Analogy: The 20 Questions Game
Imagine playing the classic game **"20 Questions"** or **"Guess Who?"**.
There are **8 secret characters** on the board, and you want to discover which one your friend picked using the fewest possible yes/no questions.

- **Question 1:** *"Is the person wearing a hat?"*
  - This cuts the 8 candidates down to **4**!
- **Question 2:** *"Does the person have blond hair?"*
  - This cuts the 4 candidates down to **2**!
- **Question 3:** *"Is the person wearing glasses?"*
  - This cuts the 2 candidates down to the **1 exact winner**!

Notice that you needed exactly **3 questions** to pinpoint 1 out of 8 possibilities:
$$2^3 = 8 \iff 3 = \log_2(8)$$

In mathematics, **one perfect yes/no question that cuts uncertainty in half equals exactly ONE BIT of information!**
- If you have 8 possibilities: $\log_2(8) = \mathbf{3 \text{ bits}}$.
- If you have 16 possibilities: $\log_2(16) = \mathbf{4 \text{ bits}}$.
- If you have 1,024 possibilities: $\log_2(1024) = \mathbf{10 \text{ bits}}$.

---

## 2. Surprisal (Self-Information)

Before defining entropy for an entire system, let us define the information of a **single event** $x$.
Shannon called this **Surprisal** or **Self-Information**:

$$\mathbf{I(x) = -\log_2 P(x) = \log_2 \left( \frac{1}{P(x)} \right)}$$

### Intuition: Rare Events Carry Massive Information!
1. **The Certain Event ($P = 1.0$):**
   If someone runs up to you and screams: *"Breaking news! The sun rose this morning!"*
   $$I = -\log_2(1.0) = \mathbf{0 \text{ bits}}$$
   You learned absolutely nothing new. Zero surprise = zero information!
2. **The Coin Flip ($P = 0.5$):**
   You flip a fair coin and it lands Heads.
   $$I = -\log_2(0.5) = -\log_2(2^{-1}) = \mathbf{1 \text{ bit}}$$
3. **The 1-in-a-Million Lottery Win ($P = 10^{-6}$):**
   $$I = -\log_2(10^{-6}) \approx \mathbf{19.93 \text{ bits}}$$
   An astronomically rare event provides an enormous burst of information!

---

## 3. Mathematical Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | Units |
| :--- | :--- | :--- | :--- |
| $X$ | Discrete Random Variable | An unpredictable event or source of symbols | Set $\mathcal{X}$ |
| $P(x)$ | Probability of outcome $x$ | Likelihood of occurrence ($0 \le P(x) \le 1$) | Dimensionless |
| $I(x)$ | Surprisal (Self-Information) | How much shock/novelty event $x$ delivers: $-\log_2 P(x)$ | Bits |
| $H(X)$ | Shannon Entropy | Expected (average) information per symbol: $\mathbb{E}[I(X)]$ | Bits or Nats |
| $\text{bit}$ | Binary Digit | Information unit when logarithm uses base $2$ | $\log_2$ |
| $\text{nat}$ | Natural Unit | Information unit when logarithm uses base $e$ ($\ln$) | $1 \text{ nat} \approx 1.443 \text{ bits}$ |
| $L$ | Average Code Length | Average number of bits used to transmit a symbol | Bits/symbol |

---

## 4. Mathematical Definition of Shannon Entropy

The **Shannon Entropy** $H(X)$ is the **expected value of the surprisal** across all possible outcomes:

$$\mathbf{H(X) = \mathbb{E}[I(X)] = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x)}$$

### The 3 Core Interpretations of Entropy in AI:
1. **Average Uncertainty:** How difficult it is to guess the next outcome before it happens.
2. **Average Information:** How much news you receive on average when each outcome is revealed.
3. **Compression Limit:** The absolute theoretical minimum average number of bits required to store or transmit each symbol without losing any data!

---

## 5. The Binary Entropy Function

Consider a biased coin where the probability of Heads is $p$, and Tails is $1 - p$.
Its entropy is given by the **Binary Entropy Function** $H_b(p)$:

$$H_b(p) = -p \log_2 p - (1 - p) \log_2(1 - p)$$

```text
       Entropy H(p) in bits
           1.0 ▲              ╭──────╮ (Max Uncertainty = 1.0 bit at p = 0.5)
               │            ╭╯        ╰╮
           0.8 │           ╭╯          ╰╮
               │          ╭╯            ╰╮
           0.5 │         ╭╯              ╰╮
               │        ╭╯                ╰╮
           0.2 │       ╭╯                  ╰╮
               │     ╭─╯                    ╰─╮
           0.0 └─────┴──────────┬───────────┴─────► Probability p
                    0.0        0.5         1.0
             (Always Tails)   (Fair)   (Always Heads)
```

- When $p = 0.5$ (Fair coin): Uncertainty is at its absolute maximum: $H = 1.0 \text{ bit}$.
- When $p = 0.0$ or $p = 1.0$ (Loaded two-headed coin): You already know the outcome before flipping! Entropy collapses to **$0.0 \text{ bits}$**!

*(Note: We adopt the standard convention that $0 \log_2 0 = \lim_{p \to 0^+} p \log_2 p = 0$.)*

---

## 6. Shannon's Source Coding Theorem: The Bedrock of Compression

Can a file of $100$ megabytes be compressed down to $1$ byte?
Every compression software engineer has dreamed of an infinite compressor. Shannon proved that this is **physically impossible**.

### Shannon's Source Coding Theorem:
> Let $X$ be an information source with entropy $H(X)$.
> For any lossless code that assigns a binary codeword of length $l(x)$ to each symbol $x$, the expected codeword length $L = \sum_{x} P(x) l(x)$ satisfies:
> $$\mathbf{L \ge H(X)}$$

- **Entropy $H(X)$ is the speed of light of data compression.**
- You can compress down to $H(X)$ bits per symbol, but **never lower without losing data**!
- Algorithms like **Huffman Coding** and **Arithmetic Coding** achieve average lengths within $H(X) \le L < H(X) + 1$.

---

## 7. Practical Implications in Deep Learning & LLMs

1. **Why Language Models are Compressors:**
   A language model (like GPT-4) tries to predict the probability distribution $P(\text{token}_{t} \mid \text{context})$.
   If the model's predictions are highly accurate, its predicted probability for the true token is high ($P \to 1.0$), meaning surprisal $-\log_2 P$ is tiny!
   Therefore, **a better LLM is literally a better data compression algorithm**!
2. **Bits vs Nats in PyTorch:**
   In communication engineering, Shannon used base 2 ($\log_2$, yielding **bits**).
   In PyTorch and neural networks, gradients are computed using the natural logarithm $\ln$ (base $e$, yielding **nats**).
   To convert between them:
   $$\text{Bits} = \frac{\text{Nats}}{\ln(2)} \approx 1.4427 \times \text{Nats}$$

---

## 8. Summary of Essential Takeaways

1. **Information = Uncertainty Reduction:** Measured in bits (halving of uncertainty).
2. **Surprisal:** $I(x) = -\log_2 P(x)$. Rare events carry high information; predictable events carry zero.
3. **Shannon Entropy:** $H(X) = -\sum P(x) \log_2 P(x)$ is the expected surprisal.
4. **Binary Entropy:** Peaks at $p = 0.5$ ($1 \text{ bit}$) and drops to $0$ at deterministic extremes ($p = 0$ or $1$).
5. **Shannon's Compression Bound:** No lossless compression algorithm can ever compress below $H(X)$ bits per symbol.

