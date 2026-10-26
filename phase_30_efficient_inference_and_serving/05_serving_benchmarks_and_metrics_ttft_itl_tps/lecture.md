# Topic 05: Serving Benchmarks & Telemetry: TTFT, ITL, and TPS

## 1. Introduction & 12-Year-Old Intuition

Imagine you visit an artisan sushi bar. 

When you sit at the counter and place your order, you care about two very different things:

1. **How long until the chef serves you your first piece of sashimi?**
   If you have to wait 45 minutes before anything arrives on your plate, you get impatient and leave. This initial wait time is **Time to First Token (TTFT)**.

2. **Once the food starts coming, how fast is each piece delivered?**
   Once you taste the first piece, you want the next piece of sushi delivered every 20 seconds. If the chef drops one piece and then disappears into the kitchen for 10 minutes before handing you piece two, your meal feels jerky and agonizing. The steady rhythm between consecutive bites is **Inter-Token Latency (ITL)**.

Now look at the restaurant from the owner's perspective:
- The owner doesn't care about just one customer's feelings; they care about how many total pieces of sushi the entire restaurant serves per hour across all 40 seats. That aggregate volume is **System Tokens Per Second (System TPS)**.

If the owner packs 200 customers into the restaurant, total sushi output might hit record highs (high System TPS), but every single customer waits 2 hours for their first plate (terrible TTFT) and 15 minutes between bites (terrible ITL).

Understanding this tension—and measuring it with mathematical rigor—is the discipline of **LLM Serving Telemetry and Benchmarking**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Units | Engineering Interpretation |
| :--- | :--- | :--- | :--- |
| $\text{TTFT}$ | Time to First Token | Seconds / Milliseconds | Latency from request submission until the first output token is generated: $T_{\text{queue}} + T_{\text{prefill}}$. |
| $\text{ITL}$ (or $\text{TPOT}$) | Inter-Token Latency | Milliseconds | Time interval between emitting consecutive output tokens: $t_{i} - t_{i-1}$. |
| $\text{TPS}_{\text{user}}$ | User Throughput | Tokens / Second | Per-stream generation velocity: $1 / \text{ITL}$. |
| $\text{TPS}_{\text{sys}}$ | System Throughput | Tokens / Second | Total aggregate output tokens produced across all concurrent GPU streams per unit time. |
| $\lambda$ | Arrival Rate | Requests / Second | Average rate of incoming user queries modeled as a Poisson point process. |
| $\mu$ | Service Capacity | Requests / Second | Maximum rate at which the serving cluster can complete full inference requests. |
| $\rho$ | Traffic Intensity / Load | $\rho \in [0, 1)$ | Ratio $\lambda / \mu$. As $\rho \to 1.0$, queues diverge and latency explodes. |
| $L$ | Mean System Concurrency | Requests | Average number of active requests in the serving cluster (in queue + running on GPU). |
| $W$ | Mean End-to-End Latency | Seconds | Average total turnaround time for an entire request from arrival to completion. |
| $p50, p90, p99$ | Latency Percentiles | Milliseconds | Empirical percentiles characterizing typical, tail, and extreme latency degradation. |

---

## 3. The Core Serving Telemetry Triad

Every enterprise LLM deployment is governed by three primary Key Performance Indicators (KPIs):

```
                                    USER REQUEST LIFECYCLE
Request                                                                                  Final
Arrival                             First Token                                          Token
   │                                     │                                                 │
   ▼                                     ▼                                                 ▼
   ├─────────── TTFT ────────────────────┼────────────── Generation Phase ─────────────────┤
   │                                     │                                                 │
   │ [Queue Wait]  +   [Prefill Phase]   │  [Token 1] ──ITL──▶ [Token 2] ──ITL──▶ [Token 3]│
   │ (Scheduling)       (Dense GEMM)     │                (Serial GEMV Decode)             │
```

### 3.1 Time to First Token (TTFT)
The user-perceived initial response latency:

$$\boxed{\text{TTFT} = T_{\text{queue}} + T_{\text{prefill}}}$$

- $T_{\text{queue}}$: Time the request spends idling in the waiting FIFO or priority queue because GPU VRAM or batch slots are exhausted.
- $T_{\text{prefill}}$: Time required to run the dense GEMM matrix operations over the user's prompt tokens ($S_{\text{prompt}}$).
- **Target SLA**: For interactive search and chatbots, $\text{TTFT} < 200\text{--}400 \text{ ms}$.

### 3.2 Inter-Token Latency (ITL / TPOT)
The duration between the emission of token $i-1$ and token $i$:

$$\boxed{\text{ITL}_i = t_i - t_{i-1}}$$

- Governed by the memory-bandwidth-bound decode forward pass duration.
- **Human Reading Speed Benchmark**: The average human reads at $250 \text{ words/minute} \approx 5.5 \text{ tokens/second}$, corresponding to an ITL of:
  $$\text{ITL}_{\text{human}} \approx \frac{1000 \text{ ms}}{5.5} \approx 180 \text{ ms/token}$$
- **Interactive SLA**: To feel silky smooth and faster than human reading, serving engines target $\text{ITL} \le 30\text{--}50 \text{ ms/token}$ ($20\text{--}33 \text{ tokens/sec}$).

### 3.3 System Tokens Per Second ($\text{TPS}_{\text{sys}}$)
The macroeconomic metric measuring infrastructure efficiency:

$$\boxed{\text{TPS}_{\text{sys}} = \frac{\sum_{r=1}^{R} N_{\text{tokens}}(r)}{\Delta T}}$$

- Directly dictates operational expenditure ($\$ / 1\text{M Tokens}$).
- While high batch sizes drive up $\text{TPS}_{\text{sys}}$, they increase ITL and TTFT. Serving systems must find the highest $\text{TPS}_{\text{sys}}$ that still respects strict p99 ITL SLAs.

---

## 4. Tail Latency & Percentiles: Why Averages Lie

In production web applications, reporting mean (average) latency hides critical operational failures:
- If 95% of users receive their first token in 100 ms, but 5% of users hit memory swapping or queue stalls taking 10 seconds, the average might look acceptable (595 ms), but **1 out of every 20 customers experiences a broken product!**

We measure performance via percentiles:
- **$p50$ (Median)**: The latency experienced by the typical user.
- **$p90$**: Latency under peak congestion.
- **$p99$**: Tail latency representing the worst 1% of queries, exposing queue buffer bloat, memory reallocation stalls, and prefill starvation.

---

## 5. Queueing Theory & Little's Law

In production serving systems, incoming user queries follow an inhomogeneous **Poisson Process** with arrival rate $\lambda(t)$ requests/second. The time between successive requests is exponentially distributed:

$$\Delta t \sim \text{Exponential}(\lambda), \quad f(\Delta t) = \lambda e^{-\lambda \Delta t}$$

### 5.1 Little's Law (John Little, 1961)
In any stable queueing system in steady state:

$$\boxed{L = \lambda \cdot W}$$

where:
- $L$ is the average number of concurrent requests in the system.
- $\lambda$ is the average arrival rate.
- $W$ is the average total residence time (waiting time in queue + execution time on GPU).

### 5.2 The Knee of the Curve ($M/M/1$ & $M/G/1$ Queues)
In an $M/M/1$ queueing model with service rate $\mu$ and traffic intensity $\rho = \lambda / \mu < 1$:

$$\text{Mean Residence Time } W = \frac{1}{\mu - \lambda} = \frac{1/\mu}{1 - \rho}$$

$$\text{Average Queue Depth } L_q = \frac{\rho^2}{1 - \rho}$$

```
Latency W
   ▲
   │                                        | Asymptote: rho = 1.0
   │                                       /| (Infinite Delay)
   │                                      / │
   │                                     /  │
   │                                    /   │
   │                                   /    │
   │                       "Knee"     /     │
   │                    ┌────────────┘      │
   │────────────────────┘                   │
   └────────────────────────────────────────┴─────────────▶
  0.0                  0.70         0.85   1.0        Load rho
```

When system utilization $\rho < 0.70$, latency remains low and flat.
As load exceeds $\rho > 0.85$, the denominator $(1 - \rho) \to 0$, causing queue lengths to explode and tail latency ($p99$) to skyrocket!

---

## 6. Benchmarking Methodology: Open-Loop vs. Closed-Loop

When running benchmarks on LLM serving engines (using tools like `vllm benchmark`, `llm-perf`, or custom harnesses):

1. **Closed-Loop Benchmarking (Virtual Users)**:
   - $N$ virtual clients send a request, wait for the full response, wait a think-time $T_{\text{think}}$, and then send the next request.
   - **Flaw**: Because clients wait for the server before sending the next request, backpressure artificially slows down the arrival rate during server slowdowns. It cannot measure queue explosion!

2. **Open-Loop Benchmarking (Poisson Arrival Process)**:
   - Requests are dispatched at exact timestamps drawn from an independent Poisson arrival process, regardless of whether prior requests have completed.
   - **Why It Matters**: True production traffic is open-loop! If a viral event occurs, users keep hitting the API regardless of whether the server is struggling. Open-loop testing correctly captures tail latency spikes and system collapse thresholds.
