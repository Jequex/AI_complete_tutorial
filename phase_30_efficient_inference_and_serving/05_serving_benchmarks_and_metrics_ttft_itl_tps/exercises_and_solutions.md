# Topic 05: Exercises & Solutions — Serving Benchmarks & Telemetry

---

### Exercise 1: Little's Law Applied to Cluster VRAM Sizing

An enterprise AI API experiences a steady request arrival rate of $\lambda = 25 \text{ requests/second}$.
Empirical monitoring shows:
- Average Time to First Token: $\text{TTFT} = 0.20 \text{ seconds}$.
- Average generated tokens per request: $N_{\text{gen}} = 150 \text{ tokens}$.
- Average Inter-Token Latency: $\text{ITL} = 0.040 \text{ seconds/token}$ ($40 \text{ ms/tok}$).
- Each active token requires $320 \text{ KB}$ of KV cache memory across all layers.
- Average prompt length: $S_{\text{prompt}} = 500 \text{ tokens}$.

**Questions:**
1. Compute the average total residence time $W$ for a request.
2. Using Little's Law, compute the average number of concurrent requests $L$ active in the system at any instant.
3. Compute the average total GPU VRAM consumed by active KV caches across the cluster in steady state.

---

#### Solution to Exercise 1

**1. Average Total Residence Time $W$:**
Total residence time is the sum of TTFT and generation time:
$$T_{\text{gen}} = N_{\text{gen}} \times \text{ITL} = 150 \text{ tokens} \times 0.040 \text{ s/token} = 6.00 \text{ seconds}$$
$$W = \text{TTFT} + T_{\text{gen}} = 0.20 \text{ s} + 6.00 \text{ s} = \mathbf{6.20 \text{ seconds}}$$

**2. Average Concurrent Requests $L$ (Little's Law):**
$$L = \lambda \cdot W = 25 \text{ req/s} \times 6.20 \text{ s} = \mathbf{155 \text{ concurrent requests}}$$
At any random millisecond, exactly 155 requests are actively residing in the cluster.

**3. Total KV Cache VRAM Consumed:**
For an active request, the average sequence length over its lifecycle is:
$$\bar{S} = S_{\text{prompt}} + \frac{N_{\text{gen}}}{2} = 500 + \frac{150}{2} = 575 \text{ tokens}$$
Memory per request:
$$M_{\text{req}} = 575 \text{ tokens} \times 320 \text{ KB/token} = 184,000 \text{ KB} = 0.184 \text{ GB}$$
Across all $L = 155$ concurrent requests:
$$\text{Total KV VRAM} = 155 \times 0.184 \text{ GB} = \mathbf{28.52 \text{ GB}}$$
The cluster must maintain at least $28.52 \text{ GB}$ of dedicated KV cache VRAM to handle this steady-state traffic without queuing or swapping.

---

### Exercise 2: Tail Latency Percentiles vs. Mean: Diagnosing Latency Anomaly

A benchmark run records the following sample of 20 request turnaround times (sorted in milliseconds):
`[110, 115, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 150, 160, 180, 210, 850, 2400, 5200]`

**Questions:**
1. Calculate the sample Mean and Median ($p50$).
2. Compute the $p90$ and $p99$ percentiles (using standard nearest-rank or linear interpolation).
3. If the team's contractual SLA requires latency $\le 200 \text{ ms}$ for $90\%$ of traffic, does this serving system pass or fail? Explain why relying solely on the mean would mask this failure.

---

#### Solution to Exercise 2

**1. Mean vs. Median ($p50$):**
- Sum of all 20 samples:
  $$110+115+120+122+125+128+130+132+135+138+140+142+145+150+160+180+210+850+2400+5200 = 10,732 \text{ ms}$$
- **Mean**: $10,732 / 20 = \mathbf{536.6 \text{ ms}}$
- **Median ($p50$)**: Average of 10th and 11th values:
  $$p50 = \frac{138 + 140}{2} = \mathbf{139.0 \text{ ms}}$$

**2. $p90$ and $p99$ Percentiles:**
- For $N = 20$:
  - Index for $p90$: $0.90 \times 20 = 18$th element &rarr; $\mathbf{850 \text{ ms}}$.
  - Index for $p99$: $\lceil 0.99 \times 20 \rceil = 20$th element &rarr; $\mathbf{5200 \text{ ms}}$ ($5.2$ seconds).

**3. SLA Evaluation:**
- Contractual requirement: $p90 \le 200 \text{ ms}$.
- Measured $p90 = \mathbf{850 \text{ ms}} > 200 \text{ ms}$.
- **Result: HARD FAIL**.
- The median is fantastic ($139 \text{ ms}$), but severe tail events (likely due to preemption, memory allocation delays, or GPU thrashing) push $15\%$ of requests far past the acceptable threshold.

---

### Exercise 3: Queue Stability & Maximum Allowable Load Under Poisson Traffic

An LLM serving engine running continuous batching can complete requests at a maximum service capacity of $\mu = 10 \text{ requests/second}$.
Incoming user requests arrive according to a Poisson process with arrival rate $\lambda$.
Assume queue waiting time follows an $M/M/1$ queue:
$$W_q = \frac{\lambda}{\mu(\mu - \lambda)}$$

**Questions:**
1. Compute the average queue wait time $W_q$ when arrival rate is $\lambda = 5 \text{ req/s}$ ($\rho = 0.50$).
2. Compute $W_q$ when arrival rate increases to $\lambda = 9 \text{ req/s}$ ($\rho = 0.90$).
3. What is the maximum arrival rate $\lambda^*$ the system can sustain if the average queue wait time must remain strictly under $W_q \le 0.10 \text{ seconds}$ ($100 \text{ ms}$)?

---

#### Solution to Exercise 3

**1. Queue Wait at $\lambda = 5$ ($\rho = 0.50$):**
$$W_q = \frac{5}{10(10 - 5)} = \frac{5}{50} = \mathbf{0.10 \text{ seconds}} \quad (100 \text{ ms})$$

**2. Queue Wait at $\lambda = 9$ ($\rho = 0.90$):**
$$W_q = \frac{9}{10(10 - 9)} = \frac{9}{10} = \mathbf{0.90 \text{ seconds}} \quad (900 \text{ ms})$$
Notice that an $80\%$ increase in traffic (from 5 to 9 req/s) caused a **$9\times$ surge** in queuing latency!

**3. Maximum Arrival Rate $\lambda^*$ for $W_q \le 0.10 \text{ s}$:**
$$\frac{\lambda}{10(10 - \lambda)} \le 0.10$$
$$\lambda \le 0.10 \times (100 - 10\lambda) = 10 - \lambda$$
$$2\lambda \le 10 \implies \lambda \le \mathbf{5.0 \text{ requests/second}}$$
To maintain queue latency under 100 ms, the server cannot be loaded beyond $\rho = 50\%$ utilization without scaling out additional GPU instances.
