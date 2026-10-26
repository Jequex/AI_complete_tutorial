# Topic 02: Distributed Communication Primitives & Ring All-Reduce

## 1. Introduction & 12-Year-Old Intuition

Imagine a classroom of 8 students. The teacher assigns a group project: each student must calculate the sum of 10,000 numbers on their own paper. When everyone finishes, every single student needs to know the grand total sum across all 8 students' papers.

How should they share their answers?

### Approach 1: The Overwhelmed Teacher (Parameter Server)
All 8 students line up at the teacher’s desk. One by one, each student hands their paper to the teacher. The poor teacher has to add up all 8 papers, and then make 8 photocopies of the final answer to hand back to each student.
- If there are 8 students, the teacher handles 8 papers coming in and 8 papers going out.
- If there are 1,024 students in an auditorium, the teacher collapses under a mountain of 2,048 paper deliveries! The teacher is a fatal bottleneck.

### Approach 2: The Playground Whisper Ring (Ring All-Reduce)
Instead of a central teacher, the 8 students stand in a circle holding hands. Each student cuts their paper into 8 equal strips.
- **Round 1 (Pass & Add)**: Each student hands strip #1 to their neighbor on the right, while receiving a strip from their neighbor on the left. When you receive a strip, you add your own numbers to it and pass it along on the next turn. After 7 quick passes around the ring, every student in the circle holds **one fully calculated grand total strip**.
- **Round 2 (Pass & Share)**: Now, each student passes their finished strip around the ring to their neighbor on the right. After another 7 quick passes, every student has glued all 8 completed strips together!

Notice the magic: **no student handled more than two strips per turn**. If you grow the ring from 8 students to 1,024 students, each student *still only talks to their immediate left and right neighbors*! The total amount of work any single student does never blows up.

---

## 2. Mathematical Symbol Decoder

| Symbol | Concept / Metric | Formal Domain | Physical / Algorithmic Meaning |
| :--- | :--- | :--- | :--- |
| $N$ | World Size (GPU Count) | $\mathbb{Z}^+$ | Total number of participating GPU workers / ranks in the distributed job. |
| $M$ | Tensor Size | Elements or Bytes | Total number of parameters, gradients, or activations to be communicated. |
| $B$ | Link Bandwidth | $\text{GB/s}$ | Sustained bidirectional transmission rate between two connected network endpoints. |
| $\alpha$ | Network Latency | Seconds ($\mu\text{s}$) | Fixed startup latency overhead required to initiate a network packet transfer. |
| $\beta$ | Network Transfer Rate | $\text{Bytes/s}$ ($= B$) | Inverse time required to transfer a unit of data across the wire ($\text{Time} = \frac{M}{\beta}$). |
| $\text{Rank } i$ | Worker Identifier | $i \in \{0, 1, \dots, N-1\}$ | Unique integer assigned to each participating GPU process. |
| $V_{\text{total}}$ | Per-GPU Transfer Volume | Bytes | Total quantity of data transmitted by an individual GPU during a collective operation. |
| $\mathcal{O}(M)$ | Asymptotic Bandwidth | Complexity Class | Communication cost depends purely on model size $M$, completely independent of GPU count $N$. |

---

## 3. Physical Interconnects & Topologies

Distributed AI infrastructure spans multiple physical hierarchy levels:

```
+-----------------------------------------------------------------------------------+
| Node 0                                    Node 1                                  |
| +-------------------------------------+   +-------------------------------------+ |
| | GPU 0 <--- NVLink (900 GB/s) ---> GPU 1 |   | GPU 8 <--- NVLink (900 GB/s) ---> GPU 9 | |
| |   ^                                   | |   |   ^                                   | |
| |   | NVSwitch Fabric (3.2 TB/s)        | |   |   | NVSwitch Fabric (3.2 TB/s)        | |
| |   v                                   | |   |   v                                   | |
| | GPU 2 <-------------------------> GPU 3 |   | GPU 10 <------------------------> GPU 11| |
| +-------------------------------------+   +-------------------------------------+ |
|       |                                         |                                 |
|       +------- InfiniBand RDMA (400 Gbps = 50 GB/s) Network Switch ---------------+ |
+-----------------------------------------------------------------------------------+
```

### 3.1 Intra-Node: NVLink & NVSwitch
Inside an 8-GPU HGX server (e.g. A100 or H100):
- **PCIe Gen 5**: $64\text{ GB/s}$ bidirectional. Insufficient for massive gradient synchronizations.
- **NVLink 4 (H100)**: $900\text{ GB/s}$ bidirectional bandwidth per GPU across 18 links.
- **NVSwitch Fabric**: Direct crossbar connecting all 8 GPUs with uniform, non-blocking $3.2\text{ TB/s}$ bisection bandwidth. Intra-node communication is practically memory-speed.

### 3.2 Inter-Node: InfiniBand RDMA
Connecting separate server racks across a data center:
- **InfiniBand NDR (400 Gbps)**: Sustained bandwidth $\approx 50\text{ GB/s}$ per network interface card (NIC).
- **Remote Direct Memory Access (RDMA)**: Bypasses the host CPU, OS kernel, and socket buffers, directly streaming data from GPU HBM on Node 0 into GPU HBM on Node 1 via GPUDirect RDMA.
- **Bandwidth Disparity**: Intra-node NVLink ($900\text{ GB/s}$) is **$18\times$ faster** than inter-node InfiniBand ($50\text{ GB/s}$). Efficient distributed algorithms must minimize inter-node hops.

---

## 4. Collective Communication Primitives

Collective communications coordinate a group of $N$ processes (ranks) simultaneously.

```
1. Broadcast:                2. Scatter:                  3. Gather:
   Root: [ D ]                  Root: [ D0 | D1 | D2 ]       Root: [  ]
   R0 -> [ D ]                  R0   -> [ D0 ]               R0   : [ D0 ] -> Root: [ D0 | D1 | D2 ]
   R1 -> [ D ]                  R1   -> [ D1 ]               R1   : [ D1 ]
   R2 -> [ D ]                  R2   -> [ D2 ]               R2   : [ D2 ]

4. All-Gather:               5. Reduce:                   6. All-Reduce:
   R0: [ D0 ] -> [D0|D1|D2]     R0: [ A ]                    R0: [ A ] -> [ A+B+C ]
   R1: [ D1 ] -> [D0|D1|D2]     R1: [ B ] -> Root: [A+B+C]   R1: [ B ] -> [ A+B+C ]
   R2: [ D2 ] -> [D0|D1|D2]     R2: [ C ]                    R2: [ C ] -> [ A+B+C ]

7. Reduce-Scatter:
   R0: [ A0 | A1 | A2 ]
   R1: [ B0 | B1 | B2 ]  ===> R0 receives: [ A0 + B0 + C0 ]
   R2: [ C0 | C1 | C2 ]       R1 receives: [ A1 + B1 + C1 ]
                              R2 receives: [ A2 + B2 + C2 ]
```

### Mathematical Definitions:
1. **Broadcast**: Copies data tensor of size $M$ from root rank $r$ to all ranks $i \in \{0, \dots, N-1\}$.
2. **Scatter**: Splits root tensor $\mathbf{X} \in \mathbb{R}^M$ into $N$ equal disjoint chunks of size $\frac{M}{N}$, sending chunk $i$ to rank $i$.
3. **Gather**: Collects local chunks of size $\frac{M}{N}$ from all ranks and concatenates them onto the root rank.
4. **All-Gather**: Every rank begins with a chunk of size $\frac{M}{N}$; at completion, **every** rank possesses the full concatenated tensor of size $M$.
5. **Reduce**: Combines tensors of size $M$ from all ranks using an associative reduction operator $\oplus \in \{+, \times, \max, \min\}$ onto a single root rank:
   $$\mathbf{Y}_{\text{root}} = \bigoplus_{i=0}^{N-1} \mathbf{X}_i$$
6. **All-Reduce**: Performs a Reduction and ensures **every** rank receives the exact reduced result:
   $$\mathbf{Y}_i = \bigoplus_{j=0}^{N-1} \mathbf{X}_j, \quad \forall i \in \{0, \dots, N-1\}$$
7. **Reduce-Scatter**: Reduces all tensors across ranks and scatters the result, such that rank $i$ receives only the $i$-th reduced chunk of size $\frac{M}{N}$:
   $$\mathbf{Y}_i = \bigoplus_{j=0}^{N-1} \mathbf{X}_{j, [i]}$$

Notice the fundamental algebraic decomposition:
$$\mathbf{All\text{-}Reduce} \equiv \mathbf{Reduce\text{-}Scatter} \circ \mathbf{All\text{-}Gather}$$

---

## 5. The Parameter Server Bottleneck

In early distributed systems (e.g. DistBelief, Dean et al. 2012), gradient synchronization was coordinated by a centralized Parameter Server (PS).

Let $N$ workers compute local gradients of size $M$ bytes.
1. In the inward step, all $N-1$ workers transmit their gradient $\mathbf{g}_i$ to the server:
   $$V_{\text{recv}} = (N - 1) \cdot M$$
2. The server sums the gradients: $\mathbf{G} = \sum_{i=1}^N \mathbf{g}_i$.
3. In the outward step, the server transmits the updated weights back to all $N-1$ workers:
   $$V_{\text{send}} = (N - 1) \cdot M$$

### Total Data Handled by the Central Server:
$$V_{\text{PS\_server}} = 2(N - 1) M \in \mathcal{O}(N \cdot M)$$

The network transmission time on the server's network link is:
$$T_{\text{PS}} = \frac{2(N - 1) M}{B}$$

As the cluster size $N$ scales from 8 to 1,024 GPUs, the network traffic through the parameter server grows **linearly without bound**. The server network card saturates, creating severe idle delays on all compute workers.

---

## 6. Mathematical Proof of the Ring All-Reduce Algorithm

The **Ring All-Reduce** algorithm (Patarasuk & Yuan 2009) solves this fundamental bottleneck.

### 6.1 Ring Setup
Arrange $N$ GPUs into a logical unidirectional ring:
$$\text{Rank } i \to \text{Rank } (i + 1) \pmod N$$
Each GPU communicates exclusively with its immediate neighbor.

The gradient tensor $\mathbf{X}$ of size $M$ elements is partitioned into $N$ equal disjoint chunks:
$$\mathbf{X} = [\mathbf{c}_0, \mathbf{c}_1, \dots, \mathbf{c}_{N-1}], \quad |\mathbf{c}_k| = \frac{M}{N}$$

---

### 6.2 Phase 1: Scatter-Reduce
The goal of Scatter-Reduce is for each rank $i$ to accumulate the global sum of exactly one chunk $\mathbf{c}_k$.

The phase executes in **$N - 1$ discrete synchronous steps**:
- At step $k \in \{0, 1, \dots, N-2\}$:
  - Rank $i$ sends its current accumulator for chunk $(i - k) \pmod N$ to rank $(i + 1) \pmod N$.
  - Simultaneously, rank $i$ receives chunk $(i - k - 1) \pmod N$ from rank $(i - 1 + N) \pmod N$.
  - Upon receiving the chunk, rank $i$ reduces (adds) the incoming vector into its local chunk buffer.

#### Data Transferred per Step:
Each GPU sends exactly one chunk of size $\frac{M}{N}$ bytes per step.

#### Total Scatter-Reduce Transfer Volume per GPU:
$$V_{\text{scatter-reduce}} = (N - 1) \times \frac{M}{N} = \frac{N - 1}{N} M\text{ bytes}$$

After $N - 1$ steps, each chunk has traversed all $N$ GPUs. Specifically, rank $(c + 1) \pmod N$ holds the fully reduced global sum for chunk $c$.

---

### 6.3 Phase 2: All-Gather
Each rank now holds one fully reduced chunk. The goal of All-Gather is to replicate all $N$ fully reduced chunks to every GPU.

This phase also executes in **$N - 1$ discrete synchronous steps**:
- At step $k \in \{0, 1, \dots, N-2\}$:
  - Rank $i$ transmits its fully reduced chunk to rank $(i + 1) \pmod N$.
  - Simultaneously, rank $i$ receives a fully reduced chunk from rank $(i - 1 + N) \pmod N$ and overwrites its local chunk buffer.

#### Total All-Gather Transfer Volume per GPU:
$$V_{\text{all-gather}} = (N - 1) \times \frac{M}{N} = \frac{N - 1}{N} M\text{ bytes}$$

---

### 6.4 Total Volume & Bandwidth Independence Theorem

#### Theorem: Total Ring Communication Volume
The total volume of data transmitted by any single GPU across the entire Ring All-Reduce algorithm is:
$$V_{\text{total}} = V_{\text{scatter-reduce}} + V_{\text{all-gather}} = \frac{N - 1}{N} M + \frac{N - 1}{N} M = 2 \cdot \frac{N - 1}{N} M\text{ bytes}$$

#### Corollary: Asymptotic Bandwidth Independence
As the cluster size $N$ increases to infinity, the total communication volume per GPU converges to a constant:
$$\lim_{N \to \infty} V_{\text{total}} = \lim_{N \to \infty} 2 \cdot \left(\frac{N - 1}{N}\right) M = 2 M$$

**Proof Consequence**:
The transmission time required to synchronize gradients across the network is:
$$T_{\text{ring\_bandwidth}} = \frac{2 \cdot \frac{N - 1}{N} M}{B} \le \frac{2 M}{B}$$

Unlike the Parameter Server, where communication cost scales as $\mathcal{O}(N \cdot M)$, the Ring All-Reduce bandwidth cost is **$\mathcal{O}(M)$**—**completely independent of the number of GPUs in the cluster**!
Whether training on 8 GPUs or 16,384 GPUs, each GPU sends at most $2M$ bytes of data.

---

## 7. Latency vs. Bandwidth: The Hockney Model

Network transmission time is formalized by the **Hockney model**:
$$T = \alpha + \frac{S}{\beta}$$
where $\alpha$ is network latency (seconds per message), $S$ is message size (bytes), and $\beta$ is bandwidth ($\text{Bytes/s}$).

### For Ring All-Reduce:
Because the ring executes $2(N - 1)$ sequential communication rounds with message size $S = \frac{M}{N}$:
$$T_{\text{ring}} = 2(N - 1)\alpha + 2 \cdot \frac{N - 1}{N} \frac{M}{\beta}$$

- **High Bandwidth Regime (Large $M$)**:
  When $M$ is large (megabytes to gigabytes of gradients), the latency term $2(N - 1)\alpha$ is negligible, and $T \approx \frac{2M}{\beta}$. Ring All-Reduce achieves **optimal bandwidth utilization**.
- **High Latency Regime (Small $M$, Large $N$)**:
  When synchronizing small tensors (e.g. scalars or small layers) across thousands of GPUs, the $2(N - 1)\alpha$ hop-count latency dominates. In this regime, **Tree-based All-Reduce** (recursive halving and doubling) is superior, requiring only $2 \log_2(N)$ communication rounds:
  $$T_{\text{tree}} = 2 \log_2(N) \alpha + 2 \frac{N - 1}{N} \frac{M}{\beta}$$

Modern collective communication libraries like NVIDIA **NCCL** (NVIDIA Collective Communications Library) dynamically profile message sizes: they select **Tree All-Reduce** for small messages to minimize latency hops, and automatically switch to **Ring All-Reduce** for large messages to saturate NVLink and InfiniBand bandwidth.
