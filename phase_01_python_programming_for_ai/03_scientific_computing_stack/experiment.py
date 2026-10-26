#!/usr/bin/env python3
"""
EXPERIMENT 1.3: SCIENTIFIC COMPUTING STACK & JSON DATA PIPELINES
================================================================
A hands-on laboratory demonstrating:
1. Contiguous Memory vs Pointer Arrays (Vectorization speedups)
2. Pure Python Tabular DataFrame Engine (mimicking pandas .loc & .iloc)
3. Mathematical Optimization (Gradient-Free Powell/Coordinate Minimizer mimicking SciPy)
4. Streaming JSON Lines (.jsonl) Data Ingestion for LLM Fine-Tuning
5. Dynamic benchmarking against NumPy/pandas if present in environment

Runs on standard library Python 3 without external packages.
"""

import math
import time
import json
import os

print("=" * 70)
print("EXPERIMENT 1.3: SCIENTIFIC COMPUTING STACK & JSON DATA PIPELINES")
print("=" * 70)

# ======================================================================
# 1. VECTORIZED ARITHMETIC VS SLOW PYTHON LOOPS
# ======================================================================
print("\n--- 1. VECTORIZED ARITHMETIC SIMULATION ---")

N = 200_000
data_a = [float(i) for i in range(N)]
data_b = [float(i * 2) for i in range(N)]

# Benchmark 1: Slow naive iteration with element allocations
t0 = time.perf_counter()
res_loop = []
for i in range(N):
    res_loop.append(data_a[i] + data_b[i])
time_loop = time.perf_counter() - t0

# Benchmark 2: Optimized list comprehension with pre-allocated iterator
t0 = time.perf_counter()
res_comp = [a + b for a, b in zip(data_a, data_b)]
time_comp = time.perf_counter() - t0

print(f"Processed {N:,} elements:")
print(f"  Naive Loop (.append): {time_loop * 1000:.2f} ms")
print(f"  List Comprehension:   {time_comp * 1000:.2f} ms (~{time_loop / time_comp:.1f}x speedup)")

# Check if NumPy is available for true hardware SIMD comparison
try:
    import numpy as np
    t0 = time.perf_counter()
    np_a = np.array(data_a, dtype=np.float32)
    np_b = np.array(data_b, dtype=np.float32)
    np_res = np_a + np_b
    time_np = time.perf_counter() - t0
    print(f"  NumPy C-SIMD Array:   {time_np * 1000:.2f} ms (~{time_loop / time_np:.1f}x faster than loop!)")
except ImportError:
    print("  [NumPy is not installed; pure Python optimizations demonstrated]")


# ======================================================================
# 2. PURE PYTHON TABULAR DATAFRAME ENGINE
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. TABULAR DATAFRAME ENGINE (MIMICKING PANDAS) ---")

class MiniDataFrame:
    """A lightweight pandas-like tabular dataframe built with standard library."""
    def __init__(self, data: dict[str, list], index: list[str] = None):
        self.columns = list(data.keys())
        self.data = data
        self.num_rows = len(next(iter(data.values()))) if data else 0
        self.index = index if index else [f"row_{i}" for i in range(self.num_rows)]
        self._index_map = {name: i for i, name in enumerate(self.index)}

    def iloc(self, row_idx: int, col_name: str):
        """Integer-position based lookup."""
        return self.data[col_name][row_idx]

    def loc(self, row_label: str, col_name: str):
        """Label-based lookup."""
        idx = self._index_map[row_label]
        return self.data[col_name][idx]

    def filter(self, col_name: str, predicate_fn):
        """Returns filtered rows where predicate_fn(val) is True."""
        filtered_indices = [i for i, val in enumerate(self.data[col_name]) if predicate_fn(val)]
        new_data = {col: [self.data[col][i] for i in filtered_indices] for col in self.columns}
        new_index = [self.index[i] for i in filtered_indices]
        return MiniDataFrame(new_data, new_index)

    def __repr__(self) -> str:
        header = " | ".join(f"{col:>12}" for col in self.columns)
        divider = "-" * len(header)
        lines = [header, divider]
        for i in range(min(self.num_rows, 5)):
            row = " | ".join(f"{str(self.data[col][i]):>12}" for col in self.columns)
            lines.append(row)
        return "\n".join(lines)

# Create dataset of AI inference logs
df = MiniDataFrame({
    "model": ["gpt-nano", "gpt-mini", "gpt-nano", "gpt-large", "gpt-nano"],
    "tokens": [150, 420, 85, 1200, 310],
    "latency_ms": [14.2, 38.6, 9.5, 142.1, 28.0]
}, index=["req_1", "req_2", "req_3", "req_4", "req_5"])

print("Inference Logs Table:\n", df)
print(f"\nLabel lookup (.loc['req_2', 'latency_ms']): {df.loc('req_2', 'latency_ms')} ms")
print(f"Position lookup (.iloc[3, 'model']):         {df.iloc(3, 'model')}")

fast_requests = df.filter("latency_ms", lambda ms: ms < 30.0)
print(f"\nFiltered Fast Requests (latency < 30ms):\n{fast_requests}")


# ======================================================================
# 3. MATHEMATICAL OPTIMIZATION (SCIPY-STYLE MINIMIZER)
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. MATHEMATICAL OPTIMIZATION (SCIPY MINIMIZE SIMULATION) ---")

def parabolic_loss(w: float) -> float:
    """Convex objective function: L(w) = 3*(w - 4.5)^2 + 12.0. Minimum at w = 4.5."""
    return 3.0 * (w - 4.5) ** 2 + 12.0

def golden_section_search(f, a: float, b: float, tol: float = 1e-6) -> float:
    """Finds the scalar minimum of a convex function without gradients."""
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi ≈ 0.618
    invphi2 = (3 - math.sqrt(5)) / 2 # 1 / phi^2 ≈ 0.382

    h = b - a
    if h <= tol:
        return (a + b) / 2

    c = a + invphi2 * h
    d = a + invphi * h
    fc = f(c)
    fd = f(d)

    while h > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = b - a
            c = a + invphi2 * h
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            h = b - a
            d = a + invphi * h
            fd = f(d)

    return (a + b) / 2

w_optimal = golden_section_search(parabolic_loss, a=-10.0, b=10.0)
print(f"Objective Function: L(w) = 3*(w - 4.5)^2 + 12.0")
print(f"Optimal Minimum w*: {w_optimal:.6f} (Theoretical exact: 4.500000)")
print(f"Minimum Loss Value: {parabolic_loss(w_optimal):.6f}")


# ======================================================================
# 4. STREAMING JSON LINES (.JSONL) FOR LLM DATA PIPELINES
# ======================================================================
print("\n" + "=" * 70)
print("--- 4. STREAMING JSON LINES (.JSONL) PIPELINE ---")

sample_finetuning_corpus = [
    {"instruction": "Define scalar", "response": "A single real number with magnitude only."},
    {"instruction": "Define vector", "response": "An ordered array of numbers possessing magnitude and direction."},
    {"instruction": "Define matrix", "response": "A two-dimensional grid of numbers representing linear maps."}
]

# Write to synthetic JSONL file
jsonl_filepath = "synthetic_train.jsonl"
with open(jsonl_filepath, "w", encoding="utf-8") as f:
    for entry in sample_finetuning_corpus:
        f.write(json.dumps(entry) + "\n")

print(f"Created '{jsonl_filepath}'. Streaming lines one-by-one:")

# Read lines lazily without buffering entire file into memory
with open(jsonl_filepath, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        record = json.loads(line)
        print(f"  Line {line_idx} -> Q: '{record['instruction']}' | A: '{record['response']}'")

# Clean up synthetic file
if os.path.exists(jsonl_filepath):
    os.remove(jsonl_filepath)

print("\n" + "=" * 70)
print("EXPERIMENT 1.3 COMPLETED SUCCESSFULLY!")
print("=" * 70)

