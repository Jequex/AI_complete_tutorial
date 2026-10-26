#!/usr/bin/env python3
"""
Experiment 2.2: Geometric Linear Transformations & Determinants
==============================================================
Demonstrates:
  1. How 2x2 matrices warp 2D coordinate space.
  2. Transforming geometric shapes (Letter 'L' / Arrow) across:
     - Scaling
     - Pure Rotation (R_theta)
     - Shearing (sliding coordinate decks)
     - Singular Collapse (Projection with det = 0)
  3. Visualizing space warping via a 2D ASCII Grid Canvas.

100% pure standard library Python!
"""

import math


class Canvas2D:
    """An ASCII plotting canvas to visualize 2D points and shapes."""

    def __init__(self, width: int = 41, height: int = 21, x_range=(-5, 5), y_range=(-5, 5)):
        self.width = width
        self.height = height
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.grid = [[" " for _ in range(width)] for _ in range(height)]
        self._draw_axes()

    def _to_pixel(self, x: float, y: float):
        px = int((x - self.x_min) / (self.x_max - self.x_min) * (self.width - 1))
        py = int((self.y_max - y) / (self.y_max - self.y_min) * (self.height - 1))
        return px, py

    def _draw_axes(self):
        cx, cy = self._to_pixel(0, 0)
        for r in range(self.height):
            if 0 <= cx < self.width:
                self.grid[r][cx] = "│"
        for c in range(self.width):
            if 0 <= cy < self.height:
                self.grid[cy][c] = "─"
        if 0 <= cx < self.width and 0 <= cy < self.height:
            self.grid[cy][cx] = "┼"

    def plot_points(self, points, char: str = "●"):
        for x, y in points:
            px, py = self._to_pixel(x, y)
            if 0 <= px < self.width and 0 <= py < self.height:
                self.grid[py][px] = char

    def render(self):
        lines = ["".join(row) for row in self.grid]
        return "\n".join(lines)


def transform_points(matrix, points):
    """Multiplies each 2D point [x, y]^T by matrix [[a, b], [c, d]]."""
    a, b = matrix[0]
    c, d = matrix[1]
    transformed = []
    for x, y in points:
        x_new = (a * x) + (b * y)
        y_new = (c * x) + (d * y)
        transformed.append((x_new, y_new))
    return transformed


def compute_det(matrix) -> float:
    """det = ad - bc"""
    return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2.2: 2D LINEAR TRANSFORMATIONS & DETERMINANTS")
    print("=" * 70)

    # Define a recognizable shape: An 'L'-shape in the first quadrant
    # Points form an upright L
    original_shape = [
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0),
        (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0)
    ]

    print("\n1. ORIGINAL UNTRANSFORMED SHAPE (Identity Matrix)")
    canvas_orig = Canvas2D()
    canvas_orig.plot_points(original_shape, char="#")
    print(canvas_orig.render())

    # Transformations to test
    transforms = [
        ("Rotation (90 degrees counter-clockwise)", [
            [0.0, -1.0],
            [1.0,  0.0]
        ]),
        ("Horizontal Shear (k = 1.0)", [
            [1.0, 1.0],
            [0.0, 1.0]
        ]),
        ("Non-Uniform Scaling (Double X, Half Y)", [
            [2.0, 0.0],
            [0.0, 0.5]
        ]),
        ("Singular Collapse / Projection onto X-axis (det = 0)", [
            [1.0, 0.0],
            [0.0, 0.0]
        ])
    ]

    for name, mat in transforms:
        det = compute_det(mat)
        print("\n" + "-" * 70)
        print(f"TRANSFORMATION: {name}")
        print(f"Matrix: [ [{mat[0][0]:5.2f}, {mat[0][1]:5.2f}], [{mat[1][0]:5.2f}, {mat[1][1]:5.2f}] ]")
        print(f"Determinant: {det:5.2f}")
        if det == 0:
            print("⚠️ WARNING: Determinant is 0! Space has collapsed to a 1D line. Information lost!")
        elif det == 1:
            print("Area is preserved exactly (Rigid motion).")
        else:
            print(f"Area scaled by factor of {abs(det):.2f}x.")

        new_points = transform_points(mat, original_shape)
        canvas = Canvas2D()
        canvas.plot_points(new_points, char="*")
        print(canvas.render())

    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

