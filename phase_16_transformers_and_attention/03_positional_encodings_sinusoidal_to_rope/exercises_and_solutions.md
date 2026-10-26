# Topic 16.3: Exercises & Solutions — Positional Encodings: Sinusoidal to RoPE

---

### Problem 1: Manual Calculation of Sinusoidal Encodings (Easy)

Consider an embedding dimension $d = 4$.
The frequency formulas for $i \in \{0, 1\}$ are:
$$\omega_0 = \frac{1}{10000^{0/4}} = 1.0, \quad \omega_1 = \frac{1}{10000^{2/4}} = \frac{1}{\sqrt{10000}} = \frac{1}{100} = 0.01$$

The 4-dimensional positional encoding vector at position $pos$ is:
$$PE_{pos} = [\sin(pos \cdot \omega_0), \; \cos(pos \cdot \omega_0), \; \sin(pos \cdot \omega_1), \; \cos(pos \cdot \omega_1)]$$

1. Calculate the exact numerical values of $PE_{pos}$ for $pos = 0$, $pos = 1$, and $pos = 2$ (round to 4 decimal places).
2. Compute the dot products $\langle PE_0, PE_1 \rangle$ and $\langle PE_0, PE_2 \rangle$.
3. Does the dot product between positional vectors decay as spatial distance increases?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Positional Vectors:**
- **For $pos = 0$:**
  $$PE_0 = [\sin(0), \cos(0), \sin(0), \cos(0)] = [0.0, 1.0, 0.0, 1.0]$$
- **For $pos = 1$:**
  - $\sin(1.0) \approx 0.8415, \quad \cos(1.0) \approx 0.5403$
  - $\sin(0.01) \approx 0.0100, \quad \cos(0.01) \approx 0.99995 \approx 1.0000$
  $$PE_1 = [0.8415, 0.5403, 0.0100, 1.0000]$$
- **For $pos = 2$:**
  - $\sin(2.0) \approx 0.9093, \quad \cos(2.0) \approx -0.4161$
  - $\sin(0.02) \approx 0.0200, \quad \cos(0.02) \approx 0.9998$
  $$PE_2 = [0.9093, -0.4161, 0.0200, 0.9998]$$

**Step 2: Dot Products:**
- $\langle PE_0, PE_1 \rangle = (0)(0.8415) + (1)(0.5403) + (0)(0.01) + (1)(1.0) = 0.5403 + 1.0000 = 1.5403$
- $\langle PE_0, PE_2 \rangle = (0)(0.9093) + (1)(-0.4161) + (0)(0.02) + (1)(0.9998) = -0.4161 + 0.9998 = 0.5837$
- (For comparison, $\langle PE_0, PE_0 \rangle = 0^2 + 1^2 + 0^2 + 1^2 = 2.0$)

**Step 3: Distance Decay:**
Yes! As distance $|pos - 0|$ increases from 0 to 1 to 2, the inner product drops from $2.0 \to 1.5403 \to 0.5837$. The sinusoidal encoding naturally gives closer positions higher dot product similarity.

</details>

---

### Problem 2: 2D RoPE Rotation and Relative Dot Product Verification (Medium)

Consider a 2D subspace with frequency parameter $\theta = \frac{\pi}{6}$ radians ($30^\circ$).
Let query $\mathbf{q} = [1.0, 0.0]^T$ and key $\mathbf{k} = [1.0, 0.0]^T$.

1. Compute the rotated query $\tilde{\mathbf{q}}_3 = R_{\theta, 3} \mathbf{q}$ at position $m = 3$.
2. Compute the rotated key $\tilde{\mathbf{k}}_1 = R_{\theta, 1} \mathbf{k}$ at position $n = 1$.
3. Compute the inner product $\langle \tilde{\mathbf{q}}_3, \tilde{\mathbf{k}}_1 \rangle$.
4. Verify that this inner product is identical to rotating $\mathbf{q}$ and $\mathbf{k}$ at positions $m = 103$ and $n = 101$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Rotated Query at $m = 3$:**
The rotation angle is $m \theta = 3 \times \frac{\pi}{6} = \frac{\pi}{2}$ ($90^\circ$).
$$R_{\theta, 3} = \begin{bmatrix} \cos(\pi/2) & -\sin(\pi/2) \\ \sin(\pi/2) & \cos(\pi/2) \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
$$\tilde{\mathbf{q}}_3 = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

**Step 2: Rotated Key at $n = 1$:**
The rotation angle is $n \theta = 1 \times \frac{\pi}{6} = \frac{\pi}{6}$ ($30^\circ$).
$$R_{\theta, 1} = \begin{bmatrix} \cos(\pi/6) & -\sin(\pi/6) \\ \sin(\pi/6) & \cos(\pi/6) \end{bmatrix} = \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix}$$
$$\tilde{\mathbf{k}}_1 = \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} \frac{\sqrt{3}}{2} \\ \frac{1}{2} \end{bmatrix}$$

**Step 3: Inner Product:**
$$\langle \tilde{\mathbf{q}}_3, \tilde{\mathbf{k}}_1 \rangle = (0)\left(\frac{\sqrt{3}}{2}\right) + (1)\left(\frac{1}{2}\right) = 0.5$$

Using the theoretical formula:
$$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = \cos((m - n)\theta) = \cos\left((3 - 1)\frac{\pi}{6}\right) = \cos\left(\frac{\pi}{3}\right) = 0.5 \quad \text{Matched!}$$

**Step 4: Shift Invariance:**
For $m = 103, n = 101$, the relative distance is $(103 - 101) = 2$.
$$\langle \tilde{\mathbf{q}}_{103}, \tilde{\mathbf{k}}_{101} \rangle = \cos((103 - 101)\theta) = \cos\left(\frac{\pi}{3}\right) = 0.5$$
The dot product depends purely on the relative distance $m - n = 2$.

</details>

---

### Problem 3: Group-Theoretic Structure of Positional Shifts (Hard)

Let $M_i(k) = \begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}$.

1. Prove that $M_i(k)$ is orthogonal: $M_i(k)^T M_i(k) = I_2$.
2. Prove the composition property: $M_i(k_1) M_i(k_2) = M_i(k_1 + k_2)$.
3. Explain why this proves that the set $\{M_i(k) : k \in \mathbb{Z}\}$ forms an abelian group isomorphic to the additive group of integers $(\mathbb{Z}, +)$, and how this algebraic property guarantees linear composability of relative distances.

<details>
<summary>View Complete Solution</summary>

**Step 1: Orthogonality:**
$$M_i(k)^T = \begin{bmatrix} \cos(k\omega_i) & -\sin(k\omega_i) \\ \sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}$$
$$M_i(k)^T M_i(k) = \begin{bmatrix} \cos(k\omega_i) & -\sin(k\omega_i) \\ \sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix} \begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}$$
- Top-left: $\cos^2(k\omega_i) + \sin^2(k\omega_i) = 1$
- Top-right: $\cos(k\omega_i)\sin(k\omega_i) - \sin(k\omega_i)\cos(k\omega_i) = 0$
- Bottom-left: $\sin(k\omega_i)\cos(k\omega_i) - \cos(k\omega_i)\sin(k\omega_i) = 0$
- Bottom-right: $\sin^2(k\omega_i) + \cos^2(k\omega_i) = 1$
$$M_i(k)^T M_i(k) = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I_2 \quad \blacksquare$$

**Step 2: Composition:**
Let $u = k_1 \omega_i$ and $v = k_2 \omega_i$.
$$M_i(k_1) M_i(k_2) = \begin{bmatrix} \cos u & \sin u \\ -\sin u & \cos u \end{bmatrix} \begin{bmatrix} \cos v & \sin v \\ -\sin v & \cos v \end{bmatrix}$$
- Top-left: $\cos u \cos v - \sin u \sin v = \cos(u + v) = \cos((k_1 + k_2)\omega_i)$
- Top-right: $\cos u \sin v + \sin u \cos v = \sin(u + v) = \sin((k_1 + k_2)\omega_i)$
- Bottom-left: $-\sin u \cos v - \cos u \sin v = -\sin(u + v) = -\sin((k_1 + k_2)\omega_i)$
- Bottom-right: $-\sin u \sin v + \cos u \cos v = \cos(u + v) = \cos((k_1 + k_2)\omega_i)$
$$M_i(k_1) M_i(k_2) = M_i(k_1 + k_2) \quad \blacksquare$$

**Step 3: Group Isomorphism & Machine Learning Significance:**
- Identity: $k = 0 \implies M_i(0) = I_2$.
- Inverse: $M_i(k)^{-1} = M_i(-k)$.
- Associativity & Commutativity: follows directly from integer addition $k_1 + k_2 = k_2 + k_1$.

Thus, shifting by relative offset $k_1$ followed by relative offset $k_2$ is strictly identical to shifting once by $(k_1 + k_2)$.
A neural network's linear projections can easily learn to navigate this rotation group because relative spatial shifts compose linearly without distortion!

</details>
