# Topic 1.3: The Scientific Python Ecosystem (NumPy, pandas, Matplotlib, SciPy, APIs & JSON)

Welcome to **Topic 1.3: The Scientific Python Ecosystem**. While pure Python provides general programming constructs, deep learning and data science workloads require multi-gigabyte numerical computations executing at bare-metal C and Fortran speeds. The quartet of **NumPy, pandas, Matplotlib, and SciPy** forms the universal backbone of data manipulation, mathematical optimization, and scientific visualization in AI.

---

## 1. NumPy: High-Performance Contiguous Memory & Vectorization

### The 12-Year-Old Analogy: Shopping List of Pointers vs A Solid Box of Crayons
In standard Python, a `list` of numbers like `[1.0, 2.0, 3.0]` is actually a list of **pointers** scattered across random locations in computer memory:

```
Python List: [ Pointer 0 ] ──> [ Heap: Float Object (1.0) ]
             [ Pointer 1 ] ──> [ Heap: Float Object (2.0) ]
             [ Pointer 2 ] ──> [ Heap: Float Object (3.0) ]
```
Every time Python adds two lists, it must look up each pointer, unpack the C struct, check its type, add the numbers, and allocate a brand new float object.

A **NumPy `ndarray`**, by contrast, is a **solid box of crayons**:
```
NumPy ndarray: [ 1.0 ][ 2.0 ][ 3.0 ] (Contiguous 64-bit IEEE Floats in physical RAM)
```
Because the floats are tightly packed in order, the CPU can read them with a single memory bus transfer and execute arithmetic using **SIMD (Single Instruction, Multiple Data)** instructions, achieving a **50x to 200x speedup** over raw Python loops!

```python
import numpy as np
import time

# Benchmark: Adding 1,000,000 numbers
N = 1_000_000
list_a = list(range(N))
list_b = list(range(N))

arr_a = np.arange(N)
arr_b = np.arange(N)

# Pure Python Loop
t0 = time.perf_counter()
sum_list = [a + b for a, b in zip(list_a, list_b)]
time_python = time.perf_counter() - t0

# Vectorized NumPy
t0 = time.perf_counter()
sum_arr = arr_a + arr_b
time_numpy = time.perf_counter() - t0

print(f"Pure Python: {time_python * 1000:.2f} ms")
print(f"NumPy SIMD:   {time_numpy * 1000:.2f} ms (~{time_python / time_numpy:.1f}x faster!)")
```

### Strides: Memory Views Without Copying
A NumPy array consists of:
1. A raw 1D data buffer in RAM.
2. A data type descriptor (`dtype`, e.g., `float32`, `int64`).
3. A `shape` tuple: `(rows, cols)`.
4. A `strides` tuple: bytes to step in memory to advance by 1 in each dimension.

When you transpose a matrix (`A.T`) or reshape it, **NumPy never copies the data in RAM!** It simply alters the `strides` metadata. This creates an instantaneous $O(1)$ memory view.

---

## 2. pandas: Tabular Data Engineering & Preprocessing

Deep learning begins with raw data. Tabular datasets (user features, medical records, financial logs) are cleaned, filtered, and transformed using **pandas**.

### DataFrames and Series
- **Series:** A 1D labeled array (like a single column with an index).
- **DataFrame:** A 2D labeled tabular table with rows and named columns.

### Indexing: `.loc` vs `.iloc`
One of the most frequent beginner bugs in pandas is confusing label-based indexing with integer-position indexing:
- **`.loc` (Label-based):** Looks up row/column by their actual text/date label name.
- **`.iloc` (Integer position-based):** Looks up row/column strictly by zero-indexed integers `0, 1, 2, ...`.

```python
import pandas as pd

df = pd.DataFrame({
    "prompt": ["hello", "how are you", "summarize text"],
    "token_count": [1, 3, 2],
    "latency_ms": [12.4, 25.1, 18.9]
}, index=["sample_A", "sample_B", "sample_C"])

# Label lookup:
print(df.loc["sample_B", "token_count"])  # 3

# Positional lookup (Row 1, Column 2):
print(df.iloc[1, 2])                      # 25.1 ms
```

### Essential Preprocessing Operations
```python
# 1. Handling Missing Data (Crucial before feeding to neural networks!)
df.dropna()                        # Drops rows with NaN
df.fillna({"latency_ms": 0.0})     # Imputes missing values

# 2. Filtering & Boolean Masking
fast_queries = df[df["latency_ms"] < 20.0]

# 3. GroupBy Aggregations
df.groupby("token_count")["latency_ms"].mean()
```

---

## 3. Matplotlib: Publication-Quality AI Visualizations

Visualizing loss curves, learning rate warmups, and attention weight heatmaps is essential for diagnosing training dynamics.

### The Object-Oriented Interface (`Figure` vs `Axes`)
Never use the stateful `plt.plot()` for complex figures. Use the object-oriented API:
- `fig` (`Figure`): The overall canvas window or page.
- `ax` (`Axes`): The individual plot or coordinate system on the canvas.

```python
import matplotlib.pyplot as plt

# Create a figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

# Plot 1: Loss Curve
epochs = list(range(1, 11))
train_loss = [2.3, 1.8, 1.4, 1.1, 0.85, 0.65, 0.52, 0.42, 0.35, 0.31]
val_loss =   [2.3, 1.9, 1.5, 1.25, 1.05, 0.92, 0.88, 0.86, 0.85, 0.86]

ax1.plot(epochs, train_loss, label="Training Loss", color="#4338ca", lw=2)
ax1.plot(epochs, val_loss, label="Validation Loss", color="#dc2626", lw=2, linestyle="--")
ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Learning Rate Schedule
lrs = [1e-4 * (0.9 ** e) for e in epochs]
ax2.plot(epochs, lrs, color="#059669", lw=2)
ax2.set_title("Exponential Learning Rate Decay", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# fig.savefig("training_metrics.png")
```

---

## 4. SciPy: Advanced Scientific & Statistical Computing

SciPy builds directly on top of NumPy, providing routines for:
1. **Mathematical Optimization (`scipy.optimize`):** Finding minimums of objective functions (e.g., Nelder-Mead, BFGS).
2. **Statistical Distributions (`scipy.stats`):** Normal distributions, entropy, mutual information, and hypothesis testing.
3. **Sparse Linear Algebra (`scipy.sparse`):** Storing giant matrices where 99% of values are zero (e.g., graph neural networks, large vocabularies).

```python
from scipy.optimize import minimize

# Minimize the convex quadratic loss function: f(w) = (w - 3.5)^2 + 10
def objective_loss(w):
    return (w[0] - 3.5) ** 2 + 10.0

result = minimize(objective_loss, x0=[0.0], method="BFGS")
print(f"Optimal parameter w*: {result.x[0]:.4f} (True minimum is 3.5000)")
```

---

## 5. REST APIs, HTTP Requests, and JSON Serialization

Modern AI systems interact via RESTful JSON web APIs. For instance, sending a prompt to an LLM endpoint or downloading datasets from Hugging Face:

### The JSON Standard
JSON (JavaScript Object Notation) represents data as key-value dictionaries and lists:
- `json.dumps(obj)`: Python dictionary $\longrightarrow$ JSON string (Serialization).
- `json.loads(text)`: JSON string $\longrightarrow$ Python dictionary (Deserialization).

```python
import json

# Serializing an AI Request Payload
payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain singular value decomposition."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
}

json_str = json.dumps(payload, indent=2)
print("Serialized JSON:\n", json_str)

# Deserializing back to Python dict
parsed = json.loads(json_str)
print("Extracted model name:", parsed["model"])
```

### The JSON Lines (`.jsonl`) Format in AI
Large text datasets (such as fine-tuning datasets for LLMs) are stored as `.jsonl`, where **every line of the file is an independent JSON object**. This allows data loaders to stream millions of training examples line-by-line without parsing a multi-gigabyte file into RAM:

```python
# Writing to JSONL
samples = [
    {"prompt": "What is 2+2?", "completion": "4"},
    {"prompt": "Capital of France?", "completion": "Paris"}
]

with open("dataset.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

# Reading line-by-line (Streaming)
with open("dataset.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        # Process single training sample...
```

