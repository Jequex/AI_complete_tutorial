# Topic 15.1: Exercises & Solutions — Recurrent Neural Networks & BPTT

---

### Problem 1: Manual Forward and Backward Pass on a 2-Step Scalar RNN (Hard)

Consider a minimal scalar recurrent neural network with 1-dimensional inputs, hidden states, and outputs:
- Forward recurrence:
  $$a_t = w_{hh} h_{t-1} + w_{xh} x_t + b_h$$
  $$h_t = \tanh(a_t)$$
  $$\hat{y}_t = w_{hy} h_t$$
- Loss function: Mean Squared Error (MSE) at each timestep:
  $$L_t = \frac{1}{2} (\hat{y}_t - y_t)^2, \quad L = L_1 + L_2$$

Given parameter values and inputs:
- $w_{hh} = 0.5, \quad w_{xh} = 1.0, \quad b_h = 0.0, \quad w_{hy} = 2.0$
- Boundary condition: $h_0 = 0.0$
- Inputs: $x_1 = 0.8, \quad x_2 = -0.4$
- Targets: $y_1 = 1.0, \quad y_2 = 0.0$

**Questions:**
1. Compute the forward pass values for $a_1, h_1, \hat{y}_1, L_1$ and $a_2, h_2, \hat{y}_2, L_2$, and the total loss $L$. (Round to 4 decimal places; use $\tanh(0.8) \approx 0.6640$).
2. Compute the adjoints $\frac{\partial L}{\partial h_2}$ and $\frac{\partial L}{\partial h_1}$ using Backpropagation Through Time.
3. Compute the analytical gradient $\frac{\partial L}{\partial w_{hh}}$.

---

#### Solution

1. **Forward Pass:**
   - **Timestep $t = 1$:**
     $$a_1 = w_{hh} h_0 + w_{xh} x_1 + b_h = 0.5(0.0) + 1.0(0.8) + 0.0 = 0.8000$$
     $$h_1 = \tanh(0.8000) \approx 0.6640$$
     $$\hat{y}_1 = w_{hy} h_1 = 2.0(0.6640) = 1.3280$$
     $$L_1 = \frac{1}{2} (\hat{y}_1 - y_1)^2 = \frac{1}{2} (1.3280 - 1.0)^2 = \frac{1}{2} (0.3280)^2 \approx 0.0538$$

   - **Timestep $t = 2$:**
     $$a_2 = w_{hh} h_1 + w_{xh} x_2 + b_h = 0.5(0.6640) + 1.0(-0.4) + 0.0 = 0.3320 - 0.4000 = -0.0680$$
     $$h_2 = \tanh(-0.0680) \approx -0.0679$$
     $$\hat{y}_2 = w_{hy} h_2 = 2.0(-0.0679) = -0.1358$$
     $$L_2 = \frac{1}{2} (\hat{y}_2 - y_2)^2 = \frac{1}{2} (-0.1358 - 0.0)^2 = \frac{1}{2} (0.0184) \approx 0.0092$$

   - **Total Loss:**
     $$L = L_1 + L_2 = 0.0538 + 0.0092 = 0.0630$$

2. **Backward Pass (Adjoints on Hidden States):**
   - At step 2, $h_2$ is consumed only by $\hat{y}_2$:
     $$\frac{\partial L_2}{\partial \hat{y}_2} = \hat{y}_2 - y_2 = -0.1358$$
     $$\frac{\partial L}{\partial h_2} = \frac{\partial L_2}{\partial h_2} = \frac{\partial L_2}{\partial \hat{y}_2} \frac{\partial \hat{y}_2}{\partial h_2} = (-0.1358)(2.0) = -0.2716$$

   - Pre-activation adjoint $\delta_2 = \frac{\partial L}{\partial a_2}$:
     $$\delta_2 = \frac{\partial L}{\partial h_2} (1 - h_2^2) = -0.2716(1 - (-0.0679)^2) = -0.2716(1 - 0.0046) = -0.2703$$

   - At step 1, $h_1$ affects both immediate output $\hat{y}_1$ and future step $a_2$:
     $$\frac{\partial L_1}{\partial \hat{y}_1} = \hat{y}_1 - y_1 = 1.3280 - 1.0 = 0.3280$$
     $$\frac{\partial L_1}{\partial h_1} = (0.3280)(2.0) = 0.6560$$
     The incoming gradient from step 2 via recurrence is:
     $$\frac{\partial L_2}{\partial h_1} = \delta_2 \frac{\partial a_2}{\partial h_1} = \delta_2 \cdot w_{hh} = (-0.2703)(0.5) = -0.1352$$
     Therefore, the total adjoint is:
     $$\frac{\partial L}{\partial h_1} = \frac{\partial L_1}{\partial h_1} + \frac{\partial L_2}{\partial h_1} = 0.6560 - 0.1352 = 0.5208$$

3. **Gradient with Respect to $w_{hh}$:**
   The parameter $w_{hh}$ appears at both $t=1$ and $t=2$:
   $$\delta_1 = \frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial h_1} (1 - h_1^2) = 0.5208(1 - 0.6640^2) = 0.5208(1 - 0.4409) = 0.5208(0.5591) \approx 0.2912$$
   By the accumulation rule:
   $$\frac{\partial L}{\partial w_{hh}} = \delta_1 h_0 + \delta_2 h_1 = (0.2912)(0.0) + (-0.2703)(0.6640) = -0.1795$$

---

### Problem 2: Mathematical Proof of Gradient Vanishing Bound (Medium)

Let an Elman RNN have hidden dimension $d_h$, with transition equation $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$.
Assume that the spectral norm of $W_{hh}$ satisfies $\|W_{hh}\|_2 \le \gamma < 1$.

**Questions:**
1. State the upper bound on the induced $L_2$ operator norm of the diagonal Jacobian matrix $D_t = \text{diag}(1 - h_t^2)$.
2. Prove by induction that for any $k < t$, the norm of the temporal transition Jacobian is strictly bounded by:
   $$\left\| \frac{\partial h_t}{\partial h_k} \right\|_2 \le \gamma^{t-k}$$
3. If $\gamma = 0.75$, calculate the maximum fraction of gradient norm that can survive over 25 timesteps ($\Delta t = 25$).

#### Solution

1. The activation function is $f(z) = \tanh(z)$, whose derivative is $f'(z) = 1 - \tanh^2(z)$.
   Since $\tanh(z) \in (-1, 1)$, we have:
   $$0 < 1 - \tanh^2(z) \le 1.0, \quad \forall z \in \mathbb{R}$$
   The matrix $D_t = \text{diag}(1 - h_{t, 1}^2, \dots, 1 - h_{t, d_h}^2)$ is diagonal with positive real entries. Its $L_2$ induced norm is its maximum singular value:
   $$\|D_t\|_2 = \max_{1 \le i \le d_h} |1 - h_{t, i}^2| \le 1.0$$

2. **Proof by Induction:**
   - **Base Case ($\Delta t = 1$):**
     $$\frac{\partial h_t}{\partial h_{t-1}} = D_t W_{hh}$$
     Using the submultiplicative property of matrix operator norms:
     $$\left\| \frac{\partial h_t}{\partial h_{t-1}} \right\|_2 \le \|D_t\|_2 \|W_{hh}\|_2 \le 1.0 \cdot \gamma = \gamma^1$$
     Thus the base case holds.
   - **Inductive Step:**
     Assume $\left\| \frac{\partial h_{t-1}}{\partial h_k} \right\|_2 \le \gamma^{t - 1 - k}$.
     By the chain rule:
     $$\frac{\partial h_t}{\partial h_k} = \frac{\partial h_t}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial h_k} = (D_t W_{hh}) \frac{\partial h_{t-1}}{\partial h_k}$$
     Taking norms:
     $$\left\| \frac{\partial h_t}{\partial h_k} \right\|_2 \le \|D_t W_{hh}\|_2 \left\| \frac{\partial h_{t-1}}{\partial h_k} \right\|_2 \le \gamma \cdot \gamma^{t - 1 - k} = \gamma^{t - k}$$
     The induction is complete. $\blacksquare$

3. For $\gamma = 0.75$ and $\Delta t = 25$:
   $$\gamma^{25} = (0.75)^{25} \approx 0.0007525 \approx 7.53 \times 10^{-4}$$
   Less than $0.076\%$ of the gradient norm survives! Over $99.92\%$ of the error signal has vanished.

---

### Problem 3: Gradient Clipping Geometry (Easy)

Suppose the gradient vector for the model parameters is $\mathbf{g} = [6.0, -8.0, 0.0]^T$, and the maximum norm threshold is set to $\tau = 5.0$.

**Questions:**
1. Compute the Euclidean norm $\|\mathbf{g}\|_2$.
2. Determine whether clipping is activated, and calculate the clipped gradient $\mathbf{g}_{\text{clip}}$.
3. Calculate the dot product $\mathbf{g} \cdot \mathbf{g}_{\text{clip}}$ and verify that the cosine similarity $\cos(\theta) = 1.0$.

#### Solution

1. Euclidean norm:
   $$\|\mathbf{g}\|_2 = \sqrt{6.0^2 + (-8.0)^2 + 0.0^2} = \sqrt{36 + 64 + 0} = \sqrt{100} = 10.0$$

2. Since $\|\mathbf{g}\|_2 = 10.0 > \tau = 5.0$, clipping is triggered.
   The scaling factor is:
   $$\alpha = \frac{\tau}{\|\mathbf{g}\|_2} = \frac{5.0}{10.0} = 0.5$$
   The clipped gradient is:
   $$\mathbf{g}_{\text{clip}} = 0.5 \times [6.0, -8.0, 0.0]^T = [3.0, -4.0, 0.0]^T$$
   Its norm is $\sqrt{3.0^2 + (-4.0)^2} = \sqrt{9 + 16} = 5.0 = \tau$.

3. Verification of directional invariance:
   $$\mathbf{g} \cdot \mathbf{g}_{\text{clip}} = 6.0(3.0) + (-8.0)(-4.0) + 0.0 = 18.0 + 32.0 = 50.0$$
   $$\|\mathbf{g}\|_2 \|\mathbf{g}_{\text{clip}}\|_2 = 10.0 \times 5.0 = 50.0$$
   $$\cos(\theta) = \frac{\mathbf{g} \cdot \mathbf{g}_{\text{clip}}}{\|\mathbf{g}\|_2 \|\mathbf{g}_{\text{clip}}\|_2} = \frac{50.0}{50.0} = 1.000000$$
   The gradient direction is preserved with zero angular distortion.
