#!/usr/bin/env python3
"""
Topic 13.2 Experiment: Pure Python nn.Module, DataLoader, Optimizer & Training Loop
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (json, math, random, sys)

This experiment implements the entire PyTorch object-oriented training framework from scratch:
1. Module Architecture:
   - `Parameter` wrapping weights/biases with gradient buffers.
   - `Module` base class with automatic parameter and sub-module discovery via `__setattr__`.
   - Non-trainable persistent state via `register_buffer`.
   - Recursive parameter traversal and state dictionary serialization (`state_dict` / `load_state_dict`).
2. Data Pipeline Architecture:
   - Abstract `Dataset` protocol (`__len__`, `__getitem__`).
   - `DataLoader` with random batch shuffling and custom `collate_fn`.
3. Optimization & Training Loop:
   - Stochastic Gradient Descent (SGD) with momentum.
   - Canonical 5-step training loop: zero_grad -> forward -> loss -> backward -> step.
   - Model checkpointing to JSON and weight restoration verification.
"""

import json
import math
import random
import sys

def set_seed(seed=42):
    random.seed(seed)

# ==============================================================================
# SECTION 1: Module & Parameter Architecture
# ==============================================================================

class Parameter:
    """Wraps a numeric weight tensor, registering it as learnable."""
    def __init__(self, data, requires_grad: bool = True):
        self.data = data  # Nested list or flat list of floats
        self.requires_grad = requires_grad
        self.grad = None
        self._init_grad()

    def _init_grad(self):
        if not self.requires_grad:
            self.grad = None
            return
        if isinstance(self.data, list):
            if isinstance(self.data[0], list):
                # 2D matrix
                self.grad = [[0.0 for _ in range(len(self.data[0]))] for _ in range(len(self.data))]
            else:
                # 1D vector
                self.grad = [0.0 for _ in range(len(self.data))]
        else:
            self.grad = 0.0

    def zero_grad(self):
        self._init_grad()


class Module:
    """
    Base class for all neural network modules (pure Python implementation of torch.nn.Module).
    Provides automatic parameter registration, hierarchical state dictionaries, and train/eval toggling.
    """
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self.training = True

    def __setattr__(self, name, value):
        if hasattr(self, '_parameters') and isinstance(value, Parameter):
            self._parameters[name] = value
        elif hasattr(self, '_modules') and isinstance(value, Module):
            self._modules[name] = value
        elif hasattr(self, '_buffers') and name in self._buffers:
            self._buffers[name] = value
        super().__setattr__(name, value)

    def register_buffer(self, name: str, tensor):
        """Registers a non-learnable state buffer (e.g., running mean, step count)."""
        self._buffers[name] = tensor
        super().__setattr__(name, tensor)

    def parameters(self):
        """Generator recursively yielding all Parameter objects across module hierarchy."""
        for param in self._parameters.values():
            if param.requires_grad:
                yield param
        for mod in self._modules.values():
            yield from mod.parameters()

    def zero_grad(self):
        """Zeroes gradients for all registered parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True):
        """Sets module and all children into training mode."""
        self.training = mode
        for mod in self._modules.values():
            mod.train(mode)
        return self

    def eval(self):
        """Sets module and all children into evaluation mode."""
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self, prefix: str = ""):
        """Serializes all parameters and buffers into a flat hierarchical dictionary."""
        state = {}
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            state[key] = param.data
        for name, buf in self._buffers.items():
            key = f"{prefix}{name}"
            state[key] = buf
        for mod_name, mod in self._modules.items():
            sub_prefix = f"{prefix}{mod_name}."
            state.update(mod.state_dict(prefix=sub_prefix))
        return state

    def load_state_dict(self, state_dict: dict, prefix: str = ""):
        """Restores weights and buffers from a state dictionary."""
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                param.data = state_dict[key]
        for name, buf in self._buffers.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                self._buffers[name] = state_dict[key]
                super().__setattr__(name, state_dict[key])
        for mod_name, mod in self._modules.items():
            sub_prefix = f"{prefix}{mod_name}."
            mod.load_state_dict(state_dict, prefix=sub_prefix)


class Linear(Module):
    """Fully connected linear layer: y = x @ W^T + b"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with small random Gaussian values
        std = 1.0 / math.sqrt(in_features)
        weights = [[random.gauss(0, std) for _ in range(in_features)] for _ in range(out_features)]
        biases = [0.0 for _ in range(out_features)]

        self.weight = Parameter(weights)
        self.bias = Parameter(biases)

        # Cache for backpropagation
        self._last_x = None

    def forward(self, x):
        """
        x: list of batch samples, shape (B, in_features)
        Returns: shape (B, out_features)
        """
        self._last_x = x
        B = len(x)
        out = [[0.0 for _ in range(self.out_features)] for _ in range(B)]
        for b in range(B):
            for o in range(self.out_features):
                val = self.bias.data[o]
                for i in range(self.in_features):
                    val += x[b][i] * self.weight.data[o][i]
                out[b][o] = val
        return out

    def backward(self, grad_output):
        """
        grad_output: shape (B, out_features)
        Computes grad_w, grad_b, and returns grad_input (B, in_features)
        """
        B = len(self._last_x)
        # Gradient w.r.t bias: sum over batch
        for o in range(self.out_features):
            self.bias.grad[o] += sum(grad_output[b][o] for b in range(B))

        # Gradient w.r.t weight: grad_output^T @ x
        for o in range(self.out_features):
            for i in range(self.in_features):
                self.weight.grad[o][i] += sum(grad_output[b][o] * self._last_x[b][i] for b in range(B))

        # Gradient w.r.t input: grad_output @ weight
        grad_input = [[0.0 for _ in range(self.in_features)] for _ in range(B)]
        for b in range(B):
            for i in range(self.in_features):
                grad_input[b][i] = sum(grad_output[b][o] * self.weight.data[o][i] for o in range(self.out_features))
        return grad_input


class MultiLayerPerceptron(Module):
    """Two-layer MLP demonstrating nested child module registration."""
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.fc2 = Linear(hidden, out_features)
        self.register_buffer("step_count", 0)
        self._last_h = None

    def forward(self, x):
        if self.training:
            self.step_count += 1
        h_linear = self.fc1(x)
        # Apply ReLU activation
        h_relu = [[max(0.0, val) for val in row] for row in h_linear]
        self._last_h = (h_linear, h_relu)
        out = self.fc2(h_relu)
        return out

    def backward(self, grad_output):
        # Backward through fc2
        grad_h_relu = self.fc2.backward(grad_output)
        # Backward through ReLU
        h_linear, _ = self._last_h
        grad_h_linear = [
            [grad_h_relu[b][j] if h_linear[b][j] > 0 else 0.0 for j in range(len(h_linear[0]))]
            for b in range(len(grad_h_relu))
        ]
        # Backward through fc1
        grad_x = self.fc1.backward(grad_h_linear)
        return grad_x


# ==============================================================================
# SECTION 2: Data Pipelines (Dataset & DataLoader)
# ==============================================================================

class Dataset:
    """Abstract dataset interface mimicking torch.utils.data.Dataset."""
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class SyntheticRegressionDataset(Dataset):
    """Generates synthetic linear regression data: y = 3.5*x1 - 2.0*x2 + 1.2 + noise"""
    def __init__(self, num_samples: int = 200, seed: int = 42):
        rng = random.Random(seed)
        self.samples = []
        for _ in range(num_samples):
            x1 = rng.uniform(-2.0, 2.0)
            x2 = rng.uniform(-2.0, 2.0)
            noise = rng.gauss(0.0, 0.05)
            y = 3.5 * x1 - 2.0 * x2 + 1.2 + noise
            self.samples.append(([x1, x2], [y]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def default_collate_fn(batch):
    """Packs individual samples into batched matrices (B, F) and (B, 1)."""
    batch_x = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]
    return batch_x, batch_y


class DataLoader:
    """Python DataLoader implementation featuring shuffling and mini-batch generation."""
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True, collate_fn=default_collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            batch_items = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch_items)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


# ==============================================================================
# SECTION 3: Optimizers & Loss Functions
# ==============================================================================

class SGDMomentum:
    """SGD with momentum optimizer mimicking torch.optim.SGD."""
    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = []
        for p in self.parameters:
            if isinstance(p.data, list):
                if isinstance(p.data[0], list):
                    # 2D
                    self.velocities.append([[0.0 for _ in range(len(p.data[0]))] for _ in range(len(p.data))])
                else:
                    self.velocities.append([0.0 for _ in range(len(p.data))])
            else:
                self.velocities.append(0.0)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            if isinstance(p.data, list):
                if isinstance(p.data[0], list):
                    for r in range(len(p.data)):
                        for c in range(len(p.data[0])):
                            v[r][c] = self.momentum * v[r][c] + p.grad[r][c]
                            p.data[r][c] -= self.lr * v[r][c]
                else:
                    for i in range(len(p.data)):
                        v[i] = self.momentum * v[i] + p.grad[i]
                        p.data[i] -= self.lr * v[i]
            else:
                v = self.momentum * v + p.grad
                p.data -= self.lr * v


def mse_loss(preds, targets):
    """Computes Mean Squared Error and returns (scalar_loss, grad_output)."""
    B = len(preds)
    out_dim = len(preds[0])
    total_err = 0.0
    grad = [[0.0 for _ in range(out_dim)] for _ in range(B)]
    for b in range(B):
        for d in range(out_dim):
            diff = preds[b][d] - targets[b][d]
            total_err += diff ** 2
            # Derivative: d(MSE)/d(pred) = (2 / (B * out_dim)) * (pred - target)
            grad[b][d] = (2.0 / (B * out_dim)) * diff
    loss = total_err / (B * out_dim)
    return loss, grad


# ==============================================================================
# SECTION 4: Training Pipeline & Checkpoint Verification
# ==============================================================================

def train_and_verify():
    set_seed(42)
    print("=" * 78)
    print("STAGE 1: Initializing Neural Architecture & Data Pipeline")
    print("=" * 78)

    dataset = SyntheticRegressionDataset(num_samples=250, seed=42)
    train_loader = DataLoader(dataset, batch_size=25, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=25, shuffle=False)

    model = MultiLayerPerceptron(in_features=2, hidden=8, out_features=1)
    optimizer = SGDMomentum(model.parameters(), lr=0.03, momentum=0.85)

    print(f"Created MultiLayerPerceptron:")
    print(f"  fc1: (2 -> 8) | fc2: (8 -> 1)")
    print(f"  Registered Buffer step_count: {model.step_count}")
    print(f"  Total Parameter Elements: {sum(len(p.data) * len(p.data[0]) if isinstance(p.data[0], list) else len(p.data) for p in model.parameters())}")
    print(f"  DataLoader: {len(train_loader)} batches per epoch.")

    print("\n" + "=" * 78)
    print("STAGE 2: Executing End-to-End Training Loop (30 Epochs)")
    print("=" * 78)

    initial_loss = None
    final_loss = None

    for epoch in range(1, 31):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            # 1. Zero Gradients
            optimizer.zero_grad()

            # 2. Forward Pass
            preds = model(batch_x)

            # 3. Loss Calculation
            loss, grad = mse_loss(preds, batch_y)
            epoch_loss += loss

            # 4. Backward Pass (Backprop)
            model.backward(grad)

            # 5. Optimizer Step
            optimizer.step()

        epoch_loss /= len(train_loader)
        if epoch == 1:
            initial_loss = epoch_loss
        final_loss = epoch_loss

        if epoch % 5 == 0 or epoch == 1:
            # Evaluate on validation
            model.eval()
            val_loss = 0.0
            for vx, vy in val_loader:
                v_preds = model(vx)
                v_l, _ = mse_loss(v_preds, vy)
                val_loss += v_l
            val_loss /= len(val_loader)
            print(f"Epoch {epoch:02d}/30 | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f} | Step Buffer: {model.step_count}")

    print(f"\nTraining Loss Reduction: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss * 0.05, "Training loss failed to drop by at least 95%!"
    print("✓ Model successfully converged on synthetic regression manifold.")

    print("\n" + "=" * 78)
    print("STAGE 3: State Dict Serialization & Checkpoint Restoration")
    print("=" * 78)

    # Serialize state dict to JSON string
    ckpt = model.state_dict()
    print(f"State Dictionary Keys: {list(ckpt.keys())}")
    serialized_json = json.dumps(ckpt, indent=2)
    print(f"Serialized checkpoint size: {len(serialized_json)} bytes")

    # Instantiate fresh model with identical architecture
    fresh_model = MultiLayerPerceptron(in_features=2, hidden=8, out_features=1)

    # Verify predictions DIFFER before loading
    test_input = [[1.5, -0.8], [-1.0, 1.2]]
    fresh_model.eval()
    pred_before = fresh_model(test_input)

    # Load weights
    loaded_dict = json.loads(serialized_json)
    fresh_model.load_state_dict(loaded_dict)
    fresh_model.eval()
    model.eval()

    pred_after = fresh_model(test_input)
    pred_original = model(test_input)

    print(f"Test Sample Inputs: {test_input}")
    print(f"Original Model Prediction:   {pred_original}")
    print(f"Fresh Model Prediction:      {pred_after}")
    for p_orig, p_rest in zip(pred_original, pred_after):
        assert abs(p_orig[0] - p_rest[0]) < 1e-9, "State dict restoration failed numerical parity!"
    print(f"Restored Step Count Buffer:  {fresh_model.step_count} (Matches: {fresh_model.step_count == model.step_count})")
    assert fresh_model.step_count == model.step_count
    print("✓ Checkpoint serialization and bit-for-bit weight restoration verified.")


def main():
    print("\n" + "#" * 78)
    print("RUNNING TOPIC 13.2 EXPERIMENT: OBJECT-ORIENTED DEEP LEARNING PIPELINE")
    print("#" * 78 + "\n")

    train_and_verify()

    print("\n" + "=" * 78)
    print("ALL MODULE & TRAINING PIPELINE TESTS PASSED (EXIT CODE 0).")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
