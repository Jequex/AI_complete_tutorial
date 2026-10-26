# Topic 18.1: Exercises & Solutions — Scaling Laws & Compute-Optimal Training

---

## Exercise 1: Analytical Derivation of the Chinchilla Optimal Point for GPT-3 Compute

### Problem
OpenAI trained GPT-3 with $N = 175 \times 10^9$ non-embedding parameters on $D = 300 \times 10^9$ tokens. 
1. Compute the total pretraining FLOP budget $C$ using the standard approximation $C \approx 6ND$.
2. Using the Chinchilla empirical constants:
   - $E = 1.6934$
   - $A = 406.4$
   - $B = 410.7$
   - $\alpha = 0.3392$
   - $\beta = 0.2849$
   compute the exact compute-optimal parameter count $N^*$ and token count $D^*$ for this identical budget $C$.
3. Compute the predicted cross-entropy loss $L(175\text{B}, 300\text{B})$ for GPT-3 versus $L(N^*, D^*)$ for the Chinchilla-optimal allocation.
4. Calculate the parameter over-sizing factor $\frac{N_{\text{GPT-3}}}{N^*}$.

---

### Solution

#### Step 1: Compute Total FLOP Budget $C$
$$C = 6 \cdot N \cdot D = 6 \cdot (1.75 \times 10^{11}) \cdot (3.0 \times 10^{11}) = 3.15 \times 10^{23} \text{ FLOPs}$$

#### Step 2: Compute Optimal Allocation $(N^*, D^*)$
From the Lagrangian derivation:
$$a = \frac{\beta}{\alpha + \beta} = \frac{0.2849}{0.3392 + 0.2849} = \frac{0.2849}{0.6241} \approx 0.4565$$
$$b = \frac{\alpha}{\alpha + \beta} = \frac{0.3392}{0.3392 + 0.2849} = \frac{0.3392}{0.6241} \approx 0.5435$$

Compute the proportionality factor $G$:
$$\text{Ratio} = \frac{\alpha A}{\beta B} = \frac{0.3392 \cdot 406.4}{0.2849 \cdot 410.7} = \frac{137.8509}{117.0084} \approx 1.1781$$
$$G = (\text{Ratio})^{\frac{1}{\alpha + \beta}} = (1.1781)^{\frac{1}{0.6241}} = (1.1781)^{1.6023} \approx 1.2974$$

Now compute $\frac{C}{6}$:
$$\frac{C}{6} = \frac{3.15 \times 10^{23}}{6} = 5.25 \times 10^{22}$$

Evaluate $N^*$:
$$N^* = G \cdot \left(\frac{C}{6}\right)^a = 1.2974 \cdot (5.25 \times 10^{22})^{0.4565} \approx 3.064 \times 10^{10} \approx 30.6 \text{ Billion parameters}$$

Evaluate $D^*$:
$$D^* = G^{-1} \cdot \left(\frac{C}{6}\right)^b = \frac{1}{1.2974} \cdot (5.25 \times 10^{22})^{0.5435} \approx 1.714 \times 10^{12} \approx 1.714 \text{ Trillion tokens}$$

Check constraint:
$$6 \cdot N^* \cdot D^* = 6 \cdot (3.064 \times 10^{10}) \cdot (1.714 \times 10^{12}) \approx 3.15 \times 10^{23} \text{ FLOPs} \quad \checkmark$$

#### Step 3: Compare Losses
For actual GPT-3:
$$L(175\text{B}, 300\text{B}) = 1.6934 + 406.4 \cdot (1.75 \times 10^{11})^{-0.3392} + 410.7 \cdot (3.0 \times 10^{11})^{-0.2849}$$
$$(1.75 \times 10^{11})^{-0.3392} \approx 1.570 \times 10^{-4} \implies 406.4 \cdot 1.570 \times 10^{-4} \approx 0.0638$$
$$(3.0 \times 10^{11})^{-0.2849} \approx 5.337 \times 10^{-4} \implies 410.7 \cdot 5.337 \times 10^{-4} \approx 0.2192$$
$$L_{\text{GPT-3}} = 1.6934 + 0.0638 + 0.2192 \approx 1.9764 \text{ nats}$$

For Chinchilla-optimal allocation:
$$N^* = 30.64\text{B} \implies 406.4 \cdot (3.064 \times 10^{10})^{-0.3392} \approx 406.4 \cdot 2.836 \times 10^{-4} \approx 0.1152$$
$$D^* = 1.714\text{T} \implies 410.7 \cdot (1.714 \times 10^{12})^{-0.2849} \approx 410.7 \cdot 3.208 \times 10^{-4} \approx 0.1318$$
$$L_{\text{optimal}} = 1.6934 + 0.1152 + 0.1318 \approx 1.9404 \text{ nats}$$

Loss improvement:
$$\Delta L = 1.9764 - 1.9404 = 0.0360 \text{ nats}$$
In autoregressive language modeling, a reduction of $0.036$ nats corresponds to roughly a $2.8\times$ compute equivalent gain!

#### Step 4: Parameter Over-sizing Factor
$$\frac{N_{\text{GPT-3}}}{N^*} = \frac{175 \times 10^9}{30.64 \times 10^9} \approx 5.71\times$$
GPT-3 was **$5.7\times$ oversized**, carrying unnecessary parameter baggage during every inference token it ever generated.

---

## Exercise 2: Kaplan vs Chinchilla Asymptotic Scaling Divergence

### Problem
Suppose an AI research organization plans to scale pretraining compute by a factor of $K = 10,000\times$.
1. Under the Kaplan et al. (2020) scaling law where $N \propto C^{0.73}$ and $D \propto C^{0.27}$, compute the scaling multipliers for $N$ and $D$.
2. Under the Chinchilla scaling law where $N \propto C^{0.46}$ and $D \propto C^{0.54}$, compute the scaling multipliers for $N$ and $D$.
3. Compute the ratio of tokens per parameter $\frac{D}{N}$ after scaling under both regimes, assuming an initial balanced ratio of $20$ tokens per parameter.

---

### Solution

#### Step 1: Kaplan Multipliers
$$\text{Multiplier}_N = K^{0.73} = (10^4)^{0.73} = 10^{2.92} \approx 831.8\times$$
$$\text{Multiplier}_D = K^{0.27} = (10^4)^{0.27} = 10^{1.08} \approx 12.02\times$$
Check: $831.8 \cdot 12.02 \approx 10,000\times \quad \checkmark$

#### Step 2: Chinchilla Multipliers
$$\text{Multiplier}_N = K^{0.46} = (10^4)^{0.46} = 10^{1.84} \approx 69.18\times$$
$$\text{Multiplier}_D = K^{0.54} = (10^4)^{0.54} = 10^{2.16} \approx 144.54\times$$
Check: $69.18 \cdot 144.54 \approx 10,000\times \quad \checkmark$

#### Step 3: Evolution of Tokens per Parameter ($\frac{D}{N}$)
Initial ratio: $R_0 = 20$.
Under Kaplan:
$$R_{\text{Kaplan}} = R_0 \cdot \frac{\text{Multiplier}_D}{\text{Multiplier}_N} = 20 \cdot \frac{12.02}{831.8} = 20 \cdot 0.01445 \approx 0.289 \text{ tokens/parameter}$$
Under Kaplan's law, the model is starved to fewer than $0.3$ tokens per parameter—an extreme degree of undertraining.

Under Chinchilla:
$$R_{\text{Chinchilla}} = R_0 \cdot \frac{\text{Multiplier}_D}{\text{Multiplier}_N} = 20 \cdot \frac{144.54}{69.18} = 20 \cdot 2.089 \approx 41.8 \text{ tokens/parameter}$$
Under Chinchilla, data scaling slightly outpaces parameter scaling, ensuring that every parameter is richly trained on abundant unique contexts.

---

## Exercise 3: The LLaMA Lifetime Serving Breakeven Derivation

### Problem
A foundation model team must decide between two deployment options:
- **Model A (Compute-Optimal 30B):** $N_A = 30 \times 10^9$ parameters, trained on $D_A = 500 \times 10^9$ tokens ($C_{\text{pretrain}, A} = 6 N_A D_A$).
- **Model B (Over-trained LLaMA 7B):** $N_B = 7 \times 10^9$ parameters, intentionally trained with twice the pretraining compute of Model A: $C_{\text{pretrain}, B} = 2 \cdot C_{\text{pretrain}, A}$.
1. Calculate the number of training tokens $D_B$ used to pretrain Model B.
2. If each generated inference token requires $2N$ FLOPs, formulate the total lifecycle cost function $C_{\text{total}}(Q) = C_{\text{pretrain}} + 2 N Q$ for both models.
3. Derive the analytical expression for the breakeven query volume $Q^*$ where Model B becomes strictly cheaper in total FLOPs than Model A.
4. Calculate $Q^*$ numerically. If an enterprise API serves $10$ billion tokens per day, how many days until Model B saves net compute?

---

### Solution

#### Step 1: Compute $D_B$
$$C_{\text{pretrain}, A} = 6 \cdot (30 \times 10^9) \cdot (500 \times 10^9) = 9.0 \times 10^{22} \text{ FLOPs}$$
$$C_{\text{pretrain}, B} = 2 \cdot C_{\text{pretrain}, A} = 1.80 \times 10^{23} \text{ FLOPs}$$
$$D_B = \frac{C_{\text{pretrain}, B}}{6 N_B} = \frac{1.80 \times 10^{23}}{6 \cdot (7 \times 10^9)} = \frac{1.80 \times 10^{23}}{4.2 \times 10^{10}} \approx 4.286 \times 10^{12} \text{ tokens (4.29 Trillion tokens)}$$

#### Step 2: Lifecycle Cost Functions
$$C_{\text{total}, A}(Q) = 9.0 \times 10^{22} + 2 \cdot (30 \times 10^9) \cdot Q = 9.0 \times 10^{22} + (6.0 \times 10^{10}) \cdot Q$$
$$C_{\text{total}, B}(Q) = 1.80 \times 10^{23} + 2 \cdot (7 \times 10^9) \cdot Q = 1.80 \times 10^{23} + (1.4 \times 10^{10}) \cdot Q$$

#### Step 3: Analytical Derivation of Breakeven Volume $Q^*$
Equating $C_{\text{total}, A}(Q^*) = C_{\text{total}, B}(Q^*)$:
$$C_{\text{pretrain}, A} + 2 N_A Q^* = C_{\text{pretrain}, B} + 2 N_B Q^*$$
$$2 (N_A - N_B) Q^* = C_{\text{pretrain}, B} - C_{\text{pretrain}, A} = \Delta C_{\text{pretrain}}$$
$$Q^* = \frac{\Delta C_{\text{pretrain}}}{2 (N_A - N_B)}$$

#### Step 4: Numerical Calculation
$$\Delta C_{\text{pretrain}} = 1.80 \times 10^{23} - 9.0 \times 10^{22} = 9.0 \times 10^{22} \text{ FLOPs}$$
$$2 (N_A - N_B) = 2 \cdot (30 \times 10^9 - 7 \times 10^9) = 2 \cdot (23 \times 10^9) = 4.6 \times 10^{10} \text{ FLOPs/token}$$
$$Q^* = \frac{9.0 \times 10^{22}}{4.6 \times 10^{10}} \approx 1.9565 \times 10^{12} \text{ tokens (1.96 Trillion tokens)}$$

At a serving rate of $10 \times 10^9$ tokens/day:
$$\text{Days to Breakeven} = \frac{1.9565 \times 10^{12}}{10 \times 10^9} \approx 195.7 \text{ days}$$

After roughly $6.5$ months, Model B's $4.3\times$ lower per-token latency and memory footprint saves compounding computing power and millions in cloud infrastructure costs indefinitely.
