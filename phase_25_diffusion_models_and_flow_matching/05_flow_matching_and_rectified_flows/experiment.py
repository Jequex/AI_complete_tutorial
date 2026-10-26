#!/usr/bin/env python3
"""
Phase 25 - Topic 05: Flow Matching and Rectified Flows
Experiment: Pure Python Rectified Flow & Neural ODE Vector Field Solver

This experiment implements from first principles in pure Python 3:
1. Conditional Flow Matching (Lipman et al., 2022) with Optimal Transport straight-line paths:
   x_t = (1 - t) * x_0 + t * x_1,   u_t(x_t | x_0, x_1) = x_1 - x_0.
2. Target 2D geometric manifold: Circle / Ring distribution (Radius R = 3.0).
3. 2D Velocity Vector Field Network v_theta(x_t, t) with exact analytical backpropagation.
4. Numerical ODE integration via Euler method across varying step budgets (1-step, 4-step, 20-step).
5. Trajectory straightness evaluation: measuring path length vs displacement (straightness ratio -> 1.0).
6. 2D ASCII density map of generated samples confirming successful ring manifold synthesis.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def sample_normal_2d() -> List[float]:
    """Box-Muller transform for 2D standard isotropic Gaussian sample."""
    u1 = max(1e-15, random.random())
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    th = 2.0 * math.pi * u2
    return [r * math.cos(th), r * math.sin(th)]


def sample_target_ring(radius: float = 3.0, noise_std: float = 0.15) -> List[float]:
    """Samples from a 2D ring distribution with Gaussian radial perturbation."""
    th = 2.0 * math.pi * random.random()
    r = radius + sample_normal_2d()[0] * noise_std
    return [r * math.cos(th), r * math.sin(th)]


# =====================================================================
# 1. Velocity Vector Field Neural Network
# =====================================================================

class FlowVectorFieldMLP:
    """
    MLP parameterizing continuous velocity field: v_theta(x_t, t) in R^2.
    Input: [x, y, t, sin(pi*t), cos(pi*t)] in R^5.
    Hidden: 32 units with tanh activation.
    Output: [v_x, v_y] in R^2.
    """
    def __init__(self, in_dim: int = 5, hidden_dim: int = 32, seed: int = 42):
        random.seed(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        scale = 0.25
        self.W1 = [[(random.random() * 2.0 - 1.0) * scale for _ in range(in_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[(random.random() * 2.0 - 1.0) * scale for _ in range(hidden_dim)] for _ in range(2)]
        self.b2 = [0.0, 0.0]

    def forward(self, features: List[float]) -> Tuple[List[float], List[float]]:
        h = [math.tanh(sum(w * xi for w, xi in zip(row, features)) + b) for row, b in zip(self.W1, self.b1)]
        v0 = sum(w * hi for w, hi in zip(self.W2[0], h)) + self.b2[0]
        v1 = sum(w * hi for w, hi in zip(self.W2[1], h)) + self.b2[1]
        return [v0, v1], h

    def step(self, features: List[float], target_velocity: List[float], lr: float = 0.02) -> float:
        """Online SGD step minimizing Flow Matching MSE: 0.5 * ||v_theta - u_t||^2."""
        pred_v, h = self.forward(features)
        diff = [pred_v[0] - target_velocity[0], pred_v[1] - target_velocity[1]]
        loss = 0.5 * (diff[0] ** 2 + diff[1] ** 2)

        # Gradients for layer 2
        for k in range(2):
            for i in range(self.hidden_dim):
                self.W2[k][i] -= lr * diff[k] * h[i]
            self.b2[k] -= lr * diff[k]

        # Gradients for layer 1 (backprop through tanh: d/dz tanh = 1 - h^2)
        grad_h = [sum(self.W2[k][i] * diff[k] for k in range(2)) * (1.0 - h[i] ** 2) for i in range(self.hidden_dim)]
        for i in range(self.hidden_dim):
            for j in range(self.in_dim):
                self.W1[i][j] -= lr * grad_h[i] * features[j]
            self.b1[i] -= lr * grad_h[i]

        return loss


def make_features(x: List[float], t: float) -> List[float]:
    """Constructs input feature vector: [x, y, t, sin(pi*t), cos(pi*t)]."""
    return [x[0], x[1], t, math.sin(math.pi * t), math.cos(math.pi * t)]


# =====================================================================
# 2. Training via Conditional Flow Matching (CFM)
# =====================================================================

def create_fixed_val_set(size: int = 500, seed: int = 999) -> List[Tuple[List[float], float, List[float]]]:
    """Generates a deterministic validation set of (x_t, t, target_velocity) tuples."""
    saved_state = random.getstate()
    random.seed(seed)
    val_set = []
    for _ in range(size):
        x0 = sample_normal_2d()
        x1 = sample_target_ring(radius=3.0, noise_std=0.15)
        t = random.random()
        xt = [(1.0 - t) * x0[0] + t * x1[0], (1.0 - t) * x0[1] + t * x1[1]]
        target_v = [x1[0] - x0[0], x1[1] - x0[1]]
        val_set.append((xt, t, target_v))
    random.setstate(saved_state)
    return val_set


def compute_fixed_val_loss(model: FlowVectorFieldMLP, val_set: List[Tuple[List[float], float, List[float]]]) -> float:
    """Evaluates average CFM MSE loss on deterministic validation set."""
    total = 0.0
    for xt, t, target_v in val_set:
        pred_v, _ = model.forward(make_features(xt, t))
        total += 0.5 * ((pred_v[0] - target_v[0]) ** 2 + (pred_v[1] - target_v[1]) ** 2)
    return total / len(val_set)


def train_rectified_flow(
    model: FlowVectorFieldMLP,
    num_steps: int = 5000,
    lr: float = 0.015
) -> Tuple[float, float]:
    """
    Trains vector field v_theta using Conditional Flow Matching objective:
    L_CFM(theta) = E_{t ~ U[0, 1], x_0 ~ N(0, I), x_1 ~ Target} [ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ]
    """
    val_set = create_fixed_val_set(size=500, seed=999)
    init_loss = compute_fixed_val_loss(model, val_set)
    print(f"    Initial Pre-Training Validation Loss: {init_loss:.4f}")
    accum_loss = 0.0

    for step in range(num_steps):
        # 1. Sample base noise x_0 and target sample x_1
        x0 = sample_normal_2d()
        x1 = sample_target_ring(radius=3.0, noise_std=0.15)
        
        # 2. Sample time t in [0, 1]
        t = random.random()

        # 3. Straight-line interpolation x_t = (1 - t)*x_0 + t*x_1
        xt = [(1.0 - t) * x0[0] + t * x1[0], (1.0 - t) * x0[1] + t * x1[1]]

        # 4. Target constant velocity: u_t(x_t | x_0, x_1) = x_1 - x_0
        target_v = [x1[0] - x0[0], x1[1] - x0[1]]

        features = make_features(xt, t)
        loss = model.step(features, target_v, lr=lr)
        accum_loss += loss

        if (step + 1) % 1000 == 0:
            avg_loss = accum_loss / 1000.0
            print(f"    Step {step+1:4d}/{num_steps} | Training Batch Loss: {avg_loss:.4f}")
            accum_loss = 0.0

    final_loss = compute_fixed_val_loss(model, val_set)
    print(f"    Final Post-Training Validation Loss:  {final_loss:.4f}")
    return init_loss, final_loss


# =====================================================================
# 3. Neural ODE Numerical Solver (Euler Integration)
# =====================================================================

def solve_ode_euler(
    model: FlowVectorFieldMLP,
    x_init: List[float],
    num_steps: int
) -> Tuple[List[float], float]:
    """
    Solves dx_t/dt = v_theta(x_t, t) from t = 0 to t = 1 via Euler method:
    x_{t + dt} = x_t + dt * v_theta(x_t, t)
    Returns: (final_point, integrated_path_length)
    """
    dt = 1.0 / num_steps
    curr_x = x_init[:]
    path_length = 0.0

    for s in range(num_steps):
        t = s * dt
        features = make_features(curr_x, t)
        v, _ = model.forward(features)
        dx = [dt * v[0], dt * v[1]]
        step_dist = math.sqrt(dx[0] ** 2 + dx[1] ** 2)
        path_length += step_dist
        curr_x = [curr_x[0] + dx[0], curr_x[1] + dx[1]]

    return curr_x, path_length


def print_ascii_ring_map(
    points: List[List[float]],
    title: str,
    range_limit: float = 4.5,
    width: int = 50,
    height: int = 20
):
    """Renders 2D particle cloud as an ASCII density plot."""
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for x, y in points:
        if -range_limit <= x <= range_limit and -range_limit <= y <= range_limit:
            col = int((x + range_limit) / (2.0 * range_limit) * (width - 1))
            row = int((range_limit - y) / (2.0 * range_limit) * (height - 1))
            grid[row][col] += 1

    max_cnt = max(max(r) for r in grid) or 1
    charset = " .:-=+*#%@"
    print(f"\n--- {title} ---")
    print("+" + "-" * width + "+")
    for r in range(height):
        line = []
        for c in range(width):
            cnt = grid[r][c]
            if cnt == 0:
                line.append(" ")
            else:
                idx = min(len(charset) - 1, int((cnt / max_cnt) * (len(charset) - 1)) + 1)
                line.append(charset[idx])
        print("|" + "".join(line) + "|")
    print("+" + "-" * width + "+")


# =====================================================================
# 4. Main Execution
# =====================================================================

def main():
    random.seed(42)
    print("======================================================================")
    print("Phase 25 - Topic 05: Flow Matching & Rectified Flows")
    print("Self-Contained Pure Python Verification Experiment")
    print("======================================================================")

    model = FlowVectorFieldMLP(in_dim=5, hidden_dim=32, seed=42)

    print("\n[1] Training Rectified Flow with Optimal Transport Conditional Flow Matching (CFM)...")
    print("    Target: 2D Ring Distribution at Radius R = 3.00")
    init_loss, final_loss = train_rectified_flow(model, num_steps=5000, lr=0.015)
    assert final_loss < init_loss, "CFM training failed to decrease loss!"

    # -----------------------------------------------------------------
    # Step Budget Comparison: 1-step vs 4-step vs 20-step Euler Integration
    # -----------------------------------------------------------------
    test_budgets = [1, 4, 20]
    num_eval_samples = 250
    base_noises = [sample_normal_2d() for _ in range(num_eval_samples)]

    print("\n[2] Evaluating Numerical Euler ODE Solvers Across Integration Budgets:")
    eval_results = {}

    for steps in test_budgets:
        final_points = []
        straightness_ratios = []
        radii = []

        for x0 in base_noises:
            x1_hat, path_len = solve_ode_euler(model, x0, num_steps=steps)
            final_points.append(x1_hat)
            r = math.sqrt(x1_hat[0] ** 2 + x1_hat[1] ** 2)
            radii.append(r)

            # Straightness ratio = Euclidean displacement / integrated path length
            displacement = math.sqrt((x1_hat[0] - x0[0]) ** 2 + (x1_hat[1] - x0[1]) ** 2)
            ratio = displacement / max(1e-8, path_len)
            straightness_ratios.append(ratio)

        mean_r = sum(radii) / num_eval_samples
        var_r = sum((r - mean_r) ** 2 for r in radii) / (num_eval_samples - 1)
        mean_straightness = sum(straightness_ratios) / num_eval_samples

        eval_results[steps] = {
            "mean_r": mean_r,
            "std_r": math.sqrt(var_r),
            "straightness": mean_straightness,
            "points": final_points
        }

        print(f"\n    [Euler {steps:2d} Step(s)]:")
        print(f"      Mean Output Radius: {mean_r:.3f} (Ground Truth Ring Radius: 3.000)")
        print(f"      Radial Std Dev:     {math.sqrt(var_r):.3f}")
        print(f"      Trajectory Straightness: {mean_straightness:.4f} (Ideal Straight Line: 1.0000)")

    # Assertions
    # With 20 steps, radius should be close to 3.0 (between 2.3 and 3.7)
    assert 2.3 <= eval_results[20]["mean_r"] <= 3.7, f"Ring radius inaccurate ({eval_results[20]['mean_r']})!"
    # Straightness ratio should be close to 1.0 (straight-line optimal transport paths)
    assert eval_results[20]["straightness"] >= 0.85, "Trajectories deviated significantly from straight lines!"

    # Render ASCII ring map from 20-step Euler samples
    print_ascii_ring_map(eval_results[20]["points"], title="2D Rectified Flow Generated Ring Distribution (20 Euler Steps)")

    print("\n[SUCCESS] Topic 05 experiment completed successfully (Exit Code: 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
