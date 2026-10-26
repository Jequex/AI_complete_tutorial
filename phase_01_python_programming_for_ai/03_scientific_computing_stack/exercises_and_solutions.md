# Topic 1.3: Exercises & Solutions — The Scientific Python Ecosystem

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **NumPy Memory Buffers, Strides, and Slicing Views** | Medium |
| **2** | **pandas Tabular Data Sanitization & Feature Engineering** | Medium |
| **3** | **End-to-End JSON Lines (.jsonl) Fine-Tuning Pipeline** | Challenging |

---

## Problem 1: NumPy Memory Buffers, Strides, and Slicing Views

### The Context
NumPy's speed comes from manipulating contiguous byte buffers through metadata (`shape`, `strides`, and `dtype`). Slicing an array creates a *view* rather than copying data in memory.

### Part A: Stride Arithmetic
Consider a 2D array of 32-bit floats (`float32`, which occupies 4 bytes per number) with shape `(4, 5)` in standard C-contiguous (row-major) order.
1. How many total bytes does this array consume in physical RAM?
2. What are the strides of this array in units of bytes?
3. If we slice every other row and every other column with `A[::2, ::2]`, what is the resulting shape and the new stride tuple? Does this operation allocate a new array in RAM?

### Part B: Vectorized Dot Product vs Loop
Write a clean NumPy function `vectorized_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float` without writing any for-loops.

---

### Solution 1

#### Part A: Strides
1. **Total Memory Bytes:**
   $$\text{Elements} = 4 \times 5 = 20$$
   $$\text{Bytes} = 20 \times 4 \text{ bytes} = 80 \text{ bytes}$$
2. **C-Contiguous Strides:**
   Moving one column right advances by 1 element = 4 bytes.
   Moving one row down advances by 5 columns = $5 \times 4 = 20$ bytes.
   $$\text{Strides} = (20, 4)$$
3. **Subsampled Slicing `A[::2, ::2]`:**
   - Rows: $4 \to 2$
   - Columns: $5 \to 3$ (indices 0, 2, 4)
   - Resulting shape: `(2, 3)`
   - Step sizes doubled: Row stride $= 20 \times 2 = 40$ bytes. Col stride $= 4 \times 2 = 8$ bytes.
   - New strides: `(40, 8)`
   - **Memory allocation:** Zero new data allocated! NumPy creates a new lightweight view pointing to the same memory buffer with updated strides.

#### Part B: Vectorized Cosine Similarity
```python
import numpy as np

def vectorized_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity cos(theta) = (a . b) / (||a|| * ||b||) using SIMD."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))
```

---

## Problem 2: pandas Tabular Data Sanitization for AI

### The Context
Raw datasets downloaded from internet web scrapers frequently contain corrupted rows, missing strings, extreme outliers, and duplicate prompts.

### Requirements
Write a pandas preprocessing function `clean_prompt_dataset(df: pd.DataFrame) -> pd.DataFrame` that:
1. Drops any row where the `'prompt'` column is null/NaN or an empty whitespace string.
2. Creates a new column `'token_estimate'` equal to the word count of `'prompt'`.
3. Filters out extreme outliers where `'token_estimate' < 2` or `'token_estimate' > 2048`.
4. Drops duplicate rows based on `'prompt'`, keeping the first occurrence.
5. Resets the index to be contiguous.

---

### Solution 2

```python
import pandas as pd

def clean_prompt_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes raw text prompt dataframes for LLM pretraining/fine-tuning."""
    # 1. Drop nulls
    df_clean = df.dropna(subset=["prompt"]).copy()
    
    # Strip whitespace and drop empty strings
    df_clean["prompt"] = df_clean["prompt"].astype(str).str.strip()
    df_clean = df_clean[df_clean["prompt"] != ""]
    
    # 2. Add word count token estimate
    df_clean["token_estimate"] = df_clean["prompt"].apply(lambda p: len(p.split()))
    
    # 3. Filter length outliers
    df_clean = df_clean[
        (df_clean["token_estimate"] >= 2) & (df_clean["token_estimate"] <= 2048)
    ]
    
    # 4. Remove duplicate prompts
    df_clean = df_clean.drop_duplicates(subset=["prompt"], keep="first")
    
    # 5. Reset index
    df_clean = df_clean.reset_index(drop=True)
    return df_clean
```

---

## Problem 3: End-to-End JSON Lines (.jsonl) Fine-Tuning Pipeline

### The Context
Training fine-tuned LLMs requires splitting datasets into training and validation sets formatted as `.jsonl` files with specific keys: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.

### Requirements
Implement a streaming function `split_jsonl_dataset(input_path: str, train_path: str, val_path: str, val_ratio: float = 0.2, seed: int = 42)` that:
1. Streams `input_path` line-by-line without loading the whole file into RAM.
2. Uses deterministic random assignment based on `seed` to send `(1 - val_ratio)` to `train_path` and `val_ratio` to `val_path`.
3. Returns a tuple `(num_train_samples, num_val_samples)`.

---

### Solution 3

```python
import json
import random

def split_jsonl_dataset(
    input_path: str,
    train_path: str,
    val_path: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple[int, int]:
    """Streams and splits a large JSONL dataset into train and validation sets with zero RAM spikes."""
    random.seed(seed)
    train_count = 0
    val_count = 0

    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(train_path, "w", encoding="utf-8") as train_f, \
         open(val_path, "w", encoding="utf-8") as val_f:
        
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue
            
            # Verify valid JSON
            try:
                record = json.loads(line_str)
            except json.JSONDecodeError:
                continue  # Skip corrupted rows

            # Deterministic split assignment
            if random.random() < val_ratio:
                val_f.write(json.dumps(record) + "\n")
                val_count += 1
            else:
                train_f.write(json.dumps(record) + "\n")
                train_count += 1

    return (train_count, val_count)
```

