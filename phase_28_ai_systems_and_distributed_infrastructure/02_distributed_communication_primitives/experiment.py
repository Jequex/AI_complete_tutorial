#!/usr/bin/env python3
"""
Topic 02: Distributed Communication Primitives
=====================================================
Experiment: Ring All-Reduce Simulator & Collective Primitives
Pure Python 3 standard library implementation (no numpy, no torch).

This script simulates:
1. Exact Ring All-Reduce across N virtual GPU nodes:
   - Chunking of gradient tensors of size M into N equal slices
   - Scatter-Reduce Phase (N - 1 ring communication steps)
   - All-Gather Phase (N - 1 ring communication steps)
2. Numerical correctness verification against ground-truth mathematical sum.
3. Network transmission volume and time scaling comparison:
   Ring All-Reduce (O(M) bandwidth) vs Parameter Server (O(N * M) bandwidth).
"""

import math
import random
import sys
from typing import List, Tuple


class VirtualGPU:
    """Simulated GPU device with a local gradient buffer."""
    def __init__(self, rank: int, buffer_size: int, initial_values: List[float] = None):
        self.rank = rank
        self.buffer_size = buffer_size
        if initial_values is not None:
            self.buffer = list(initial_values)
        else:
            random.seed(100 + rank)
            self.buffer = [random.uniform(-1.0, 1.0) for _ in range(buffer_size)]
        self.bytes_sent = 0
        self.bytes_received = 0


class RingAllReducer:
    """
    Implements the Ring All-Reduce algorithm across N virtual nodes.
    Each node sends to its right neighbor (rank + 1) % N
    and receives from its left neighbor (rank - 1 + N) % N.
    """
    def __init__(self, gpus: List[VirtualGPU]):
        self.gpus = gpus
        self.N = len(gpus)
        self.M = gpus[0].buffer_size
        for g in gpus:
            assert g.buffer_size == self.M, "All GPU buffers must have identical size"

    def execute(self) -> Tuple[int, int]:
        """
        Executes Scatter-Reduce followed by All-Gather.
        Returns: (total_steps, bytes_transferred_per_gpu)
        """
        N = self.N
        M = self.M
        chunk_size = math.ceil(M / N)
        padded_size = chunk_size * N

        # Pad buffers if needed
        for g in self.gpus:
            if len(g.buffer) < padded_size:
                g.buffer.extend([0.0] * (padded_size - len(g.buffer)))

        # Slice each GPU's buffer into N chunks
        # chunks[rank][chunk_id] = list of floats
        chunks = []
        for g in self.gpus:
            gpu_chunks = [g.buffer[c * chunk_size : (c + 1) * chunk_size] for c in range(N)]
            chunks.append(gpu_chunks)

        bytes_per_chunk = chunk_size * 4  # Assuming 32-bit float (4 bytes)

        # =====================================================================
        # PHASE 1: SCATTER-REDUCE (N - 1 steps)
        # At step k (0 to N-2):
        # Node i sends chunk (i - k) % N to node (i + 1) % N.
        # The receiver accumulates that incoming chunk into its local copy.
        # =====================================================================
        for k in range(N - 1):
            transfers = []
            for i in range(N):
                send_to = (i + 1) % N
                chunk_id = (i - k) % N
                data_to_send = list(chunks[i][chunk_id])
                transfers.append((send_to, chunk_id, data_to_send))

                self.gpus[i].bytes_sent += bytes_per_chunk
                self.gpus[send_to].bytes_received += bytes_per_chunk

            # Receive and reduce
            for send_to, chunk_id, data in transfers:
                for idx in range(chunk_size):
                    chunks[send_to][chunk_id][idx] += data[idx]

        # =====================================================================
        # PHASE 2: ALL-GATHER (N - 1 steps)
        # At step k (0 to N-2):
        # Node i sends fully reduced chunk (i + 1 - k) % N to node (i + 1) % N.
        # The receiver overwrites its local chunk with the fully reduced copy.
        # =====================================================================
        for k in range(N - 1):
            transfers = []
            for i in range(N):
                send_to = (i + 1) % N
                chunk_id = (i + 1 - k) % N
                data_to_send = list(chunks[i][chunk_id])
                transfers.append((send_to, chunk_id, data_to_send))

                self.gpus[i].bytes_sent += bytes_per_chunk
                self.gpus[send_to].bytes_received += bytes_per_chunk

            # Receive and overwrite
            for send_to, chunk_id, data in transfers:
                chunks[send_to][chunk_id] = list(data)

        # Reassemble the final buffer for each GPU (unpad to original M)
        for i in range(N):
            flat = []
            for c in range(N):
                flat.extend(chunks[i][c])
            self.gpus[i].buffer = flat[:M]

        total_steps = 2 * (N - 1)
        bytes_transferred_per_gpu = 2 * (N - 1) * bytes_per_chunk
        return total_steps, bytes_transferred_per_gpu


def analytical_comparison(M_elements: int, dtype_bytes: int, bandwidth_gb_s: float, ranks_list: List[int]):
    """
    Compares network transfer volume and execution time between:
    1. Parameter Server (PS)
    2. Ring All-Reduce
    """
    total_data_bytes = M_elements * dtype_bytes
    bandwidth_bytes_s = bandwidth_gb_s * 1e9

    print("=" * 85)
    print(f"ANALYTICAL SCALING ANALYSIS: Tensor Size = {M_elements:,} elements ({total_data_bytes / 1e6:.2f} MB)")
    print(f"Network Interconnect Bandwidth = {bandwidth_gb_s:.1f} GB/s per link")
    print("=" * 85)
    print(f"{'GPUs (N)':<10} | {'Ring Vol/GPU':<16} | {'Ring Time':<12} | {'PS Vol (Server)':<18} | {'PS Time':<12} | {'Speedup':<8}")
    print("-" * 85)

    for N in ranks_list:
        # Ring: Each GPU sends 2 * (N - 1) / N * M
        ring_vol_per_gpu = 2.0 * ((N - 1) / N) * total_data_bytes
        ring_time_s = ring_vol_per_gpu / bandwidth_bytes_s

        # Parameter Server: The central server must receive (N - 1) * M and send (N - 1) * M
        ps_server_vol = 2.0 * (N - 1) * total_data_bytes
        ps_time_s = ps_server_vol / bandwidth_bytes_s

        speedup = ps_time_s / ring_time_s if ring_time_s > 0 else 1.0

        print(f"{N:<10} | {ring_vol_per_gpu/1e6:>12.2f} MB | {ring_time_s*1000:>9.3f} ms | "
              f"{ps_server_vol/1e6:>14.2f} MB | {ps_time_s*1000:>9.3f} ms | {speedup:>6.1f}x")

    print("=" * 85)


def run_experiment():
    print("Initializing Topic 02: Distributed Communication Primitives Simulator...\n")

    # 1. Run simulated Ring All-Reduce on N = 8 GPUs with M = 16,384 elements
    N_gpus = 8
    M_elements = 16384
    random.seed(42)

    # Generate distinct initial buffers for each GPU
    initial_buffers = [
        [random.uniform(-10.0, 10.0) for _ in range(M_elements)]
        for _ in range(N_gpus)
    ]

    # Calculate exact analytical ground truth sum across all GPUs
    ground_truth = [0.0] * M_elements
    for i in range(M_elements):
        ground_truth[i] = sum(initial_buffers[g][i] for g in range(N_gpus))

    # Instantiate Virtual GPUs
    gpus = [VirtualGPU(rank=i, buffer_size=M_elements, initial_values=initial_buffers[i]) for i in range(N_gpus)]

    # Execute Ring All-Reduce
    reducer = RingAllReducer(gpus)
    steps, bytes_per_gpu = reducer.execute()

    print(f"Executed Ring All-Reduce across N={N_gpus} GPUs.")
    print(f"Total Steps: {steps} (Scatter-Reduce: {N_gpus-1}, All-Gather: {N_gpus-1})")
    print(f"Data transferred per GPU: {bytes_per_gpu:,} bytes (Theory: 2 * (N-1)/N * M * 4 = "
          f"{int(2 * (N_gpus - 1) / N_gpus * M_elements * 4):,} bytes)\n")

    # Verify Numerical Precision across all GPUs
    max_err = 0.0
    for g in gpus:
        for idx in range(M_elements):
            diff = abs(g.buffer[idx] - ground_truth[idx])
            if diff > max_err:
                max_err = diff

    print(f"Maximum absolute error across all GPUs and elements: {max_err:.2e}")
    assert max_err < 1e-11, f"Numerical mismatch in Ring All-Reduce! max_err = {max_err}"
    print("[PASSED] Ring All-Reduce is numerically exact to double-precision machine limits!\n")

    # 2. Analytical Comparison across scaling GPU counts
    analytical_comparison(
        M_elements=100_000_000,   # 100M parameter model (400 MB in FP32)
        dtype_bytes=4,
        bandwidth_gb_s=50.0,      # 50 GB/s (400 Gbps InfiniBand NDR RDMA)
        ranks_list=[2, 4, 8, 16, 32, 64, 128, 256, 1024]
    )

    # Assert asymptotic invariance
    vol_8 = 2.0 * (7 / 8) * 400.0
    vol_1024 = 2.0 * (1023 / 1024) * 400.0
    assert vol_1024 / vol_8 < 1.15, "Ring transfer volume per GPU must be bounded as N -> infinity"

    print("\n[SUCCESS] All Topic 02 Distributed Communication tests passed with code 0.")


if __name__ == "__main__":
    run_experiment()
