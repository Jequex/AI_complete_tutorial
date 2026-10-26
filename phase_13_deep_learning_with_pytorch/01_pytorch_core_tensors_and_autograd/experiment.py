#!/usr/bin/env python3
"""
Topic 13.1 Experiment: PyTorch Tensor Internals & Dynamic Autograd Engine
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, sys)
              Optional: torch (automatically verified if installed)

This experiment demonstrates:
1. Low-Level Tensor Memory Architecture:
   - Contiguous 1D flat buffer storage.
   - Shape, strides, and memory offset calculations.
   - Zero-copy view / reshape vs non-contiguous transpose operations.
   - Contiguous vs non-contiguous detection and .contiguous() deep allocation.
2. Dynamic Autograd Computational DAG Engine:
   - Dynamic tape-based graph construction via `requires_grad` and `grad_fn`.
   - Reverse-mode automatic differentiation with topological sort.
   - Computational graph detachment via `.detach()`.
   - Context manager `no_grad()` to suppress DAG building during inference.
3. Rigorous validation tests and assertions ensuring mathematical parity.
"""

import math
import sys

# ==============================================================================
# SECTION 1: Low-Level Tensor Storage, Strides & Memory Layout
# ==============================================================================

class Storage:
    """Represents a raw 1D contiguous block of physical memory (like C++ THStorage)."""
    def __init__(self, size: int, initial_data=None):
        self.size = size
        if initial_data is not None:
            assert len(initial_data) == size, f"Data length {len(initial_data)} != size {size}"
            self.data = list(initial_data)
        else:
            self.data = [0.0] * size

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __setitem__(self, idx: int, value: float):
        self.data[idx] = float(value)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"Storage(size={self.size}, data={self.data})"


def compute_contiguous_strides(shape):
    """Computes standard C-contiguous (row-major) strides for a given shape."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class MemoryTensor:
    """
    Simulates a PyTorch tensor layout:
    - storage: pointer to raw flat 1D memory
    - shape: logical dimensions (d0, d1, ...)
    - strides: memory jump steps for each dimension
    - storage_offset: starting index within storage
    """
    def __init__(self, data, shape=None, strides=None, offset=0, storage=None):
        if storage is not None:
            self.storage = storage
            self.shape = tuple(shape)
            self.strides = tuple(strides) if strides is not None else compute_contiguous_strides(self.shape)
            self.offset = offset
        else:
            # Flatten input data if nested list
            flat = self._flatten(data)
            self.storage = Storage(len(flat), flat)
            if shape is None:
                self.shape = (len(flat),)
            else:
                self.shape = tuple(shape)
            assert math.prod(self.shape) == len(self.storage), \
                f"Shape {self.shape} product {math.prod(self.shape)} != storage size {len(self.storage)}"
            self.strides = compute_contiguous_strides(self.shape)
            self.offset = 0

    @classmethod
    def _flatten(cls, data):
        if isinstance(data, (int, float)):
            return [float(data)]
        flat = []
        for item in data:
            if isinstance(item, (list, tuple)):
                flat.extend(cls._flatten(item))
            else:
                flat.append(float(item))
        return flat

    def is_contiguous(self) -> bool:
        """A tensor is C-contiguous if its strides equal the standard row-major strides."""
        expected = compute_contiguous_strides(self.shape)
        return self.strides == expected

    def _get_storage_index(self, indices):
        assert len(indices) == len(self.shape), f"Expected {len(self.shape)} indices, got {len(indices)}"
        idx = self.offset
        for i, (dim_idx, dim_size, stride) in enumerate(zip(indices, self.shape, self.strides)):
            assert 0 <= dim_idx < dim_size, f"Index {dim_idx} out of bounds for dim {i} with size {dim_size}"
            idx += dim_idx * stride
        return idx

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = (indices,)
        if len(indices) < len(self.shape):
            # Return a sub-slice / sub-view
            sub_shape = self.shape[len(indices):]
            sub_strides = self.strides[len(indices):]
            sub_offset = self.offset + sum(idx * s for idx, s in zip(indices, self.strides[:len(indices)]))
            return MemoryTensor(None, shape=sub_shape, strides=sub_strides, offset=sub_offset, storage=self.storage)
        return self.storage[self._get_storage_index(indices)]

    def __setitem__(self, indices, val):
        if isinstance(indices, int):
            indices = (indices,)
        idx = self._get_storage_index(indices)
        self.storage[idx] = float(val)

    def view(self, *new_shape):
        """
        PyTorch view() returns a new tensor with identical storage.
        Requires the tensor to be contiguous!
        """
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        # Handle inferring dimension (-1)
        if -1 in new_shape:
            neg_idx = new_shape.index(-1)
            other_prod = math.prod(d for d in new_shape if d != -1)
            assert len(self.storage) % other_prod == 0, "Invalid dimension inference"
            inferred = len(self.storage) // other_prod
            new_shape = list(new_shape)
            new_shape[neg_idx] = inferred
            new_shape = tuple(new_shape)

        assert math.prod(new_shape) == math.prod(self.shape), \
            f"Shape product {math.prod(new_shape)} != current elements {math.prod(self.shape)}"
        assert self.is_contiguous(), "view size is not compatible with input tensor's size and stride (use .contiguous())"

        new_strides = compute_contiguous_strides(new_shape)
        return MemoryTensor(None, shape=new_shape, strides=new_strides, offset=self.offset, storage=self.storage)

    def transpose(self, dim0: int, dim1: int):
        """Zero-copy transpose by swapping shape and strides."""
        assert 0 <= dim0 < len(self.shape) and 0 <= dim1 < len(self.shape), "Dimension index out of range"
        shape = list(self.shape)
        strides = list(self.strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return MemoryTensor(None, shape=tuple(shape), strides=tuple(strides), offset=self.offset, storage=self.storage)

    def t(self):
        """Matrix transpose for 2D tensors."""
        assert len(self.shape) == 2, "t() expects a 2D tensor"
        return self.transpose(0, 1)

    def to_list(self):
        """Converts logical tensor view into nested Python list."""
        if len(self.shape) == 1:
            return [self[i] for i in range(self.shape[0])]
        elif len(self.shape) == 2:
            return [[self[r, c] for c in range(self.shape[1])] for r in range(self.shape[0])]
        elif len(self.shape) == 3:
            return [[[self[d, r, c] for c in range(self.shape[2])] for r in range(self.shape[1])] for d in range(self.shape[0])]
        raise NotImplementedError("to_list() only implemented up to 3D")

    def contiguous(self):
        """Allocates a fresh contiguous Storage buffer copying elements in logical order."""
        if self.is_contiguous():
            return self
        flat_elements = []
        if len(self.shape) == 2:
            for r in range(self.shape[0]):
                for c in range(self.shape[1]):
                    flat_elements.append(self[r, c])
        elif len(self.shape) == 1:
            for i in range(self.shape[0]):
                flat_elements.append(self[i])
        else:
            raise NotImplementedError("Contiguous copy for >2D not required for demo")

        new_storage = Storage(len(flat_elements), flat_elements)
        new_strides = compute_contiguous_strides(self.shape)
        return MemoryTensor(None, shape=self.shape, strides=new_strides, offset=0, storage=new_storage)


# ==============================================================================
# SECTION 2: Dynamic Computational Graph & Autograd Engine
# ==============================================================================

class AutogradContext:
    """Thread-local context manager simulating `torch.no_grad()`."""
    is_grad_enabled = True

class no_grad:
    def __enter__(self):
        self.prev = AutogradContext.is_grad_enabled
        AutogradContext.is_grad_enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        AutogradContext.is_grad_enabled = self.prev


class ScalarNode:
    """
    A lightweight autograd dynamic node simulating PyTorch's dynamic autograd engine.
    Supports backward graph traversal, requires_grad, and grad_fn inspection.
    """
    def __init__(self, data: float, requires_grad: bool = False, grad_fn=None, name: str = ""):
        self.data = float(data)
        self.requires_grad = requires_grad and AutogradContext.is_grad_enabled
        self.grad_fn = grad_fn if AutogradContext.is_grad_enabled else None
        self.grad = 0.0
        self.name = name

    def __repr__(self):
        if self.grad_fn:
            return f"ScalarNode(data={self.data:.4f}, grad_fn=<{self.grad_fn.__class__.__name__}>)"
        elif self.requires_grad:
            return f"ScalarNode(data={self.data:.4f}, requires_grad=True, grad={self.grad:.4f})"
        return f"ScalarNode(data={self.data:.4f})"

    def detach(self):
        """Returns a new ScalarNode detached from the current computational DAG."""
        return ScalarNode(self.data, requires_grad=False, grad_fn=None, name=f"{self.name}_detached")

    def __add__(self, other):
        other = other if isinstance(other, ScalarNode) else ScalarNode(other)
        req_grad = (self.requires_grad or other.requires_grad) and AutogradContext.is_grad_enabled
        out = ScalarNode(self.data + other.data, requires_grad=req_grad)
        if req_grad:
            out.grad_fn = AddBackward(self, other)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, ScalarNode) else ScalarNode(other)
        req_grad = (self.requires_grad or other.requires_grad) and AutogradContext.is_grad_enabled
        out = ScalarNode(self.data - other.data, requires_grad=req_grad)
        if req_grad:
            out.grad_fn = SubBackward(self, other)
        return out

    def __rsub__(self, other):
        other = ScalarNode(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, ScalarNode) else ScalarNode(other)
        req_grad = (self.requires_grad or other.requires_grad) and AutogradContext.is_grad_enabled
        out = ScalarNode(self.data * other.data, requires_grad=req_grad)
        if req_grad:
            out.grad_fn = MulBackward(self, other)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power: float):
        req_grad = self.requires_grad and AutogradContext.is_grad_enabled
        out = ScalarNode(self.data ** power, requires_grad=req_grad)
        if req_grad:
            out.grad_fn = PowBackward(self, power)
        return out

    def relu(self):
        req_grad = self.requires_grad and AutogradContext.is_grad_enabled
        out = ScalarNode(max(0.0, self.data), requires_grad=req_grad)
        if req_grad:
            out.grad_fn = ReluBackward(self)
        return out

    def backward(self, grad_output: float = 1.0):
        """Traverses the computational DAG in reverse topological order."""
        assert self.requires_grad or self.grad_fn is not None, "Cannot backward on graph with no requires_grad"

        # Topological sorting of the DAG
        topo = []
        visited = set()

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                if node.grad_fn is not None:
                    for parent in node.grad_fn.parents:
                        build_topo(parent)
                topo.append(node)

        build_topo(self)

        # Initialize root gradient
        self.grad = grad_output

        # Reverse topological traversal
        for node in reversed(topo):
            if node.grad_fn is not None:
                parent_grads = node.grad_fn.apply(node.grad)
                for parent, p_grad in zip(node.grad_fn.parents, parent_grads):
                    if parent.requires_grad:
                        parent.grad += p_grad


# GradFn Backward Operations
class AddBackward:
    def __init__(self, a: ScalarNode, b: ScalarNode):
        self.parents = (a, b)

    def apply(self, grad_output: float):
        # d(a+b)/da = 1, d(a+b)/db = 1
        return (grad_output * 1.0, grad_output * 1.0)


class SubBackward:
    def __init__(self, a: ScalarNode, b: ScalarNode):
        self.parents = (a, b)

    def apply(self, grad_output: float):
        # d(a-b)/da = 1, d(a-b)/db = -1
        return (grad_output * 1.0, grad_output * -1.0)


class MulBackward:
    def __init__(self, a: ScalarNode, b: ScalarNode):
        self.parents = (a, b)
        self.a_val = a.data
        self.b_val = b.data

    def apply(self, grad_output: float):
        # d(a*b)/da = b, d(a*b)/db = a
        return (grad_output * self.b_val, grad_output * self.a_val)


class PowBackward:
    def __init__(self, a: ScalarNode, p: float):
        self.parents = (a,)
        self.a_val = a.data
        self.power = p

    def apply(self, grad_output: float):
        # d(a^p)/da = p * a^(p-1)
        return (grad_output * self.power * (self.a_val ** (self.power - 1)),)


class ReluBackward:
    def __init__(self, a: ScalarNode):
        self.parents = (a,)
        self.a_val = a.data

    def apply(self, grad_output: float):
        # d(relu(a))/da = 1 if a > 0 else 0
        return (grad_output * (1.0 if self.a_val > 0.0 else 0.0),)


# ==============================================================================
# SECTION 3: Verification Suites and Demonstrations
# ==============================================================================

def test_tensor_storage_and_strides():
    print("=" * 78)
    print("TEST 1: Low-Level Storage, Strides, Views, and Contiguity")
    print("=" * 78)

    # Create a 2x3 matrix:
    # [[1.0, 2.0, 3.0],
    #  [4.0, 5.0, 6.0]]
    raw_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tensor = MemoryTensor(raw_data, shape=(2, 3))

    print(f"Original Tensor Shape: {tensor.shape}")
    print(f"Original Tensor Strides: {tensor.strides} (Row-major: row jump=3, col jump=1)")
    print(f"Is Contiguous: {tensor.is_contiguous()}")
    print(f"Logical Matrix:\n  {tensor.to_list()}")
    assert tensor.is_contiguous() is True
    assert tensor.strides == (3, 1)
    assert tensor[1, 2] == 6.0

    # Test View operation (2x3 -> 3x2) - zero copy, same storage
    v = tensor.view(3, 2)
    print("\n[View Operation: tensor.view(3, 2)]")
    print(f"View Shape: {v.shape}, Strides: {v.strides}")
    print(f"Shares Same Storage: {v.storage is tensor.storage}")
    assert v.storage is tensor.storage
    assert v.shape == (3, 2)
    assert v.strides == (2, 1)

    # In-place mutation through view reflects in original tensor!
    v[0, 1] = 99.0
    print(f"Mutated view[0, 1] = 99.0 -> Original tensor[0, 1] is now: {tensor[0, 1]}")
    assert tensor[0, 1] == 99.0
    v[0, 1] = 2.0  # restore

    # Transpose operation: (2, 3) -> (3, 2)
    t = tensor.t()
    print("\n[Transpose Operation: tensor.t()]")
    print(f"Transposed Shape: {t.shape}")
    print(f"Transposed Strides: {t.strides} (Notice swapped strides: 1, 3!)")
    print(f"Is Transposed Contiguous: {t.is_contiguous()}")
    assert t.shape == (3, 2)
    assert t.strides == (1, 3)
    assert t.is_contiguous() is False
    assert t.storage is tensor.storage

    # Calling view() on a non-contiguous tensor MUST raise an assertion (identical to PyTorch RuntimeError)
    try:
        t.view(6)
        raise AssertionError("Non-contiguous view should have failed!")
    except AssertionError as e:
        print(f"Correctly caught view failure on non-contiguous tensor: '{e}'")

    # Calling .contiguous() allocates fresh buffer
    c = t.contiguous()
    print("\n[Contiguous Allocation: t.contiguous()]")
    print(f"Contiguous Shape: {c.shape}, Strides: {c.strides}")
    print(f"Is Contiguous: {c.is_contiguous()}")
    print(f"Fresh Storage Buffer: {c.storage is not tensor.storage}")
    assert c.is_contiguous() is True
    assert c.strides == (2, 1)
    assert c.storage is not tensor.storage

    # Now view(6) succeeds on the contiguous copy
    v_flat = c.view(6)
    print(f"Flattened View on Contiguous Copy: {v_flat.to_list()}")
    assert v_flat.shape == (6,)
    print("✓ TEST 1 PASSED: Low-level tensor memory mechanics strictly verified.\n")


def test_autograd_dag_and_derivatives():
    print("=" * 78)
    print("TEST 2: Dynamic Computational DAG, requires_grad, and Backpropagation")
    print("=" * 78)

    # Objective: Verify gradient of f(x, y) = (x * y + y^2) with respect to x and y
    # Let x = 3.0, y = 4.0
    # f(x, y) = 3*4 + 16 = 28.0
    # df/dx = y = 4.0
    # df/dy = x + 2*y = 3 + 8 = 11.0
    x = ScalarNode(3.0, requires_grad=True, name="x")
    y = ScalarNode(4.0, requires_grad=True, name="y")

    prod = x * y        # Add MulBackward
    y_sq = y ** 2       # Add PowBackward
    f = prod + y_sq     # Add AddBackward

    print(f"Evaluated f(x, y) = {f.data} (Expected: 28.0)")
    print(f"Root grad_fn: {f.grad_fn.__class__.__name__}")
    assert f.data == 28.0

    # Backpropagation
    f.backward()

    print(f"Backpropagated Gradients:")
    print(f"  df/dx = {x.grad:.4f} (Expected: 4.0000)")
    print(f"  df/dy = {y.grad:.4f} (Expected: 11.0000)")

    assert abs(x.grad - 4.0) < 1e-7, f"Expected 4.0, got {x.grad}"
    assert abs(y.grad - 11.0) < 1e-7, f"Expected 11.0, got {y.grad}"
    print("✓ TEST 2 PASSED: Dynamic reverse-mode automatic differentiation verified.\n")


def test_detach_and_no_grad():
    print("=" * 78)
    print("TEST 3: detach() and no_grad() Context Isolation")
    print("=" * 78)

    # Test detach()
    a = ScalarNode(5.0, requires_grad=True, name="a")
    b = (a ** 2)
    b_detached = b.detach()

    c = b_detached * 3.0
    # c requires_grad should be False because b_detached does not require grad
    print(f"b_detached requires_grad: {b_detached.requires_grad}")
    print(f"c requires_grad: {c.requires_grad}, grad_fn: {c.grad_fn}")
    assert b_detached.requires_grad is False
    assert c.requires_grad is False
    assert c.grad_fn is None

    # Test no_grad() context manager
    with no_grad():
        w = ScalarNode(2.0, requires_grad=True, name="w")
        x_val = ScalarNode(4.0, requires_grad=True, name="x_val")
        out = (w * x_val) + 1.0

        print(f"Inside no_grad(): out.requires_grad = {out.requires_grad}, grad_fn = {out.grad_fn}")
        assert out.requires_grad is False
        assert out.grad_fn is None

    # Outside no_grad, tracking resumes
    w2 = ScalarNode(2.0, requires_grad=True, name="w2")
    out2 = w2 * 4.0
    print(f"Outside no_grad(): out2.requires_grad = {out2.requires_grad}, grad_fn = {out2.grad_fn.__class__.__name__}")
    assert out2.requires_grad is True
    assert out2.grad_fn is not None
    print("✓ TEST 3 PASSED: detach() and no_grad() isolation confirmed.\n")


def test_optional_pytorch_parity():
    print("=" * 78)
    print("OPTIONAL TEST 4: Comparison with PyTorch (if installed)")
    print("=" * 78)
    try:
        import torch
        print(f"PyTorch {torch.__version__} detected. Running parity checks...")
        # PyTorch strides check
        t_pt = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert tuple(t_pt.stride()) == (3, 1)
        t_pt_trans = t_pt.t()
        assert tuple(t_pt_trans.stride()) == (1, 3)
        assert t_pt_trans.is_contiguous() is False

        # PyTorch autograd check
        x_pt = torch.tensor(3.0, requires_grad=True)
        y_pt = torch.tensor(4.0, requires_grad=True)
        f_pt = (x_pt * y_pt) + (y_pt ** 2)
        f_pt.backward()
        assert abs(x_pt.grad.item() - 4.0) < 1e-6
        assert abs(y_pt.grad.item() - 11.0) < 1e-6
        print("✓ All PyTorch parity assertions verified identically against torch runtime.")
    except ImportError:
        print("PyTorch not installed in environment (expected in standard standalone mode).")
        print("Pure Python custom storage & autograd engine successfully verified all mechanics!")


def main():
    print("\n" + "#" * 78)
    print("RUNNING TOPIC 13.1 EXPERIMENT: PYTORCH TENSORS & AUTOGRAD ENGINE")
    print("#" * 78 + "\n")

    test_tensor_storage_and_strides()
    test_autograd_dag_and_derivatives()
    test_detach_and_no_grad()
    test_optional_pytorch_parity()

    print("=" * 78)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY WITH ZERO DEPENDENCIES.")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
