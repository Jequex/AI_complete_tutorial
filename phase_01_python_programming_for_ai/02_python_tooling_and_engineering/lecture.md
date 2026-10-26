# Topic 1.2: AI Software Engineering, Tooling, and Reproducibility

Welcome to **Topic 1.2: AI Software Engineering, Tooling, and Reproducibility**. The difference between an amateur AI tinkerer and a professional AI research engineer is software discipline. Training multi-GPU models and deploying inference APIs requires airtight dependency management, strict type checking, robust automated unit tests, and ironclad reproducibility.

---

## 1. Modules, Packages, and Virtual Environments

### The Mental Model: The Laboratory Toolbox
Imagine walking into a professional chemistry lab. If all chemicals were stored in unlabeled glass jars jumbled in one giant pile, the first reaction would explode.
In Python:
- A **Module** is a single `.py` file containing functions, classes, and variables.
- A **Package** is a folder containing multiple modules, identified by an `__init__.py` file.
- A **Virtual Environment** is an isolated sandbox directory containing its own Python interpreter and `site-packages`.

```
my_ai_project/
├── .venv/                     # Isolated virtual environment
├── pyproject.toml             # Project metadata and dependencies
├── src/
│   └── neural_net/            # Package directory
│       ├── __init__.py        # Package initializer and public API exports
│       ├── layers.py          # Module containing Dense, Conv2D
│       └── optimizers.py      # Module containing Adam, SGD
└── tests/
    └── test_layers.py         # Automated unit test suite
```

### Why Virtual Environments Prevent Dependency Nightmares
Different AI libraries depend on mutually incompatible C++ CUDA runtimes or NumPy versions:
- Model A requires `torch==2.1.0` and `numpy==1.24.3`.
- Model B requires `torch==2.4.0` and `numpy==2.0.1`.

If you install both into your global operating system Python, one will inevitably overwrite the other and break your system tools.

```bash
# Create an isolated environment named .venv
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install reproducible dependencies
pip install --upgrade pip
pip freeze > requirements.txt
```

---

## 2. Modern Static Typing and Type Hints for AI

Python is dynamically typed, but modern AI codebases (such as PyTorch, Hugging Face Transformers, and vLLM) use static type annotations (`typing`) to eliminate shape bugs before runtime.

### Essential Type Annotations in AI
```python
from typing import Union, Optional, Callable, Any

# Primitive & Container Types
learning_rate: float = 0.001
epochs: int = 10
device: str = "cuda:0"
layer_dims: list[int] = [768, 3072, 768]
hyperparameters: dict[str, Union[float, int, str]] = {
    "lr": 1e-4,
    "batch_size": 32,
    "optimizer": "adamw"
}

# Optional Types (Can be None)
checkpoint_path: Optional[str] = None  # Equivalent to: str | None

# Callables (Functions passed as arguments)
# Callable[[InputTypes], ReturnType]
ActivationFn = Callable[[float], float]

def apply_activation(x: float, activation: ActivationFn) -> float:
    return activation(x)
```

### Protocols: Structural Subtyping (Duck Typing with Safety)
When writing a modular training harness, you do not want to force every model to inherit from a specific base class. You only care that the model implements a `.forward()` method. A `Protocol` enforces this interface cleanly:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TrainableModel(Protocol):
    def forward(self, x: list[float]) -> list[float]:
        ...
    def compute_loss(self, pred: list[float], target: list[float]) -> float:
        ...
```

---

## 3. Automated Unit Testing for Neural Networks

Unlike standard web applications where tests check exact strings or HTTP status codes, **machine learning unit tests verify shapes, mathematical invariants, and numerical tolerances**.

### The 4 Crucial Types of Machine Learning Unit Tests
1. **Shape Tests:** Does the forward pass produce the expected output tensor dimensions across various batch sizes?
2. **Determinism Tests:** Do identical random seeds produce identical initial weights?
3. **Loss Invariant Tests:** For a non-trivial model, is the initial loss close to the theoretical random baseline? (e.g., Cross-Entropy on $K$ classes should start near $\ln(K)$).
4. **Gradient Check Tests:** Does backpropagation produce non-zero gradients for every trainable weight?

### Writing Tests with `unittest`
```python
import unittest
import math

class TestNeuralComponents(unittest.TestCase):
    def setUp(self):
        """Runs before every test method."""
        self.weights = [[0.5, -0.5], [0.2, 0.8]]
        self.bias = [0.1, -0.1]

    def test_output_shape(self):
        """Verify layer outputs 2 features when given a 2D input."""
        x = [1.0, 2.0]
        # Simulate forward pass: Wx + b
        out = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.weights, self.bias)]
        self.assertEqual(len(out), 2)

    def test_numerical_tolerance(self):
        """Floating point math must use almost_equal, not strict equality."""
        computed_val = 0.1 + 0.2
        # self.assertEqual(computed_val, 0.3) -> ❌ Fails due to IEEE 754 float representation!
        self.assertAlmostEqual(computed_val, 0.3, places=7)

if __name__ == "__main__":
    unittest.main()
```

---

## 4. Performance Profiling: Hunting Down Bottlenecks

Training an AI model can take days or weeks. If your data loader has a slow Python loop, your expensive \$30,000 GPU will sit idle at 5% utilization waiting for data. Profiling identifies exactly where execution time is spent.

### The Built-in `cProfile` Engine
```python
import cProfile
import pstats
import io

def slow_data_pipeline():
    total = 0
    for i in range(500_000):
        total += i ** 0.5
    return total

def run_profiler():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    slow_data_pipeline()
    
    profiler.disable()
    s = io.StringIO()
    # Sort results by cumulative time spent inside the function
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())
```

---

## 5. Professional Logging vs. `print()`

### Why `print()` Fails in Production AI
1. `print()` does not write timestamps, severity levels, or source line numbers.
2. `print()` cannot easily be redirected to both a log file on disk and the terminal output simultaneously.
3. `print()` cannot be dynamically silenced during evaluation without modifying source code.

### The Standard `logging` Module
```python
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(),              # Print to console
        logging.FileHandler("training.log")   # Save to file
    ]
)

logger = logging.getLogger("Trainer")

logger.info("Initializing Transformer weights...")
logger.warning("GPU temperature exceeds 80°C.")
logger.error("CUDA Out-Of-Memory encountered during batch 1420!")
```

---

## 6. The Golden Rule of AI: Reproducibility & Random Seeds

### The Chaos of Unseeded Pseudo-Random Number Generators (PRNGs)
Neural network weights are initialized randomly, and training batches are shuffled randomly. If you do not fix the random seed, running the exact same script twice will produce different weights, different loss curves, and different test accuracies. This makes scientific debugging impossible.

### Enforcing Determinism
```python
import random

def set_seed(seed: int = 42):
    """Pins pseudo-random number generators for reproducible runs."""
    random.seed(seed)
    # When using numpy: np.random.seed(seed)
    # When using PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    print(f"🔒 Global seed locked to {seed}")

set_seed(42)
print("Roll 1:", [random.random() for _ in range(3)])

set_seed(42)
print("Roll 2:", [random.random() for _ in range(3)])  # Exact identical numbers!
```

---

## 7. Version Control for AI: Git & What NOT to Commit

In traditional software, git repositories store plain text source files. In AI engineering, large binary model checkpoints (`model.pt`, `weights.bin`) and multi-gigabyte datasets (`train.csv`, `images.zip`) will bloat and corrupt a Git repository.

### Standard AI `.gitignore` Template
```gitignore
# Python artifacts
__pycache__/
*.py[cod]
.venv/
dist/
build/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Datasets and Cache
data/
datasets/
*.parquet
*.h5
*.arrow

# Model Weights and Checkpoints (Use Git LFS or HuggingFace/S3 for these!)
checkpoints/
runs/
*.pt
*.pth
*.bin
*.safetensors
*.onnx

# Log files and run traces
*.log
wandb/
tensorboard/
```

---

## 8. Command-Line Arguments for Experimentation (`argparse`)

Never hardcode learning rates or batch sizes directly into scripts. Use command-line arguments so automated sweep scripts can run experiments unattended:

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Language Model")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Training with batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}")
```

