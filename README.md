##  Background Mathematics Refresher
<p float="center">
    <img src="Git.png" width="60%" />
</p>

## 1. Core Mathematics Review

### 1.1 What is a Linear Function?

A linear function is the simplest type of mathematical relationship between input and output. It describes a straight line (in 1D) or a flat plane (in higher dimensions).

In one dimension (1D), the equation of a linear function is:

$$
y = \beta + \omega x
$$

- $\beta$ is called the bias or intercept — it moves the line up or down.  
- $\omega$ is called the slope or weight — it tilts the line.

In two dimensions (2D), the linear function generalizes to:

$$
y = \beta + \omega_1 x_1 + \omega_2 x_2
$$

- $x_1$ and $x_2$ are two input features.  
- $\omega_1$ and $\omega_2$ control how each input affects the output.

Any function that is not in this form (e.g., involves $x^2$, $\sin(x)$, etc.) is considered non-linear.

---

### 1.2 Explore 1D Linear Functions (with Code and Practice)

Create and plot a 1D linear function.

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define a 1D linear function
def linear_function_1D(x, beta, omega):
    return beta + omega * x

# Create an array of x values
x = np.arange(0, 10, 0.01)

# Set parameters
beta = 3.0
omega = 1.5

# Compute y values
y = linear_function_1D(x, beta, omega)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x, y, label=f"beta={beta}, omega={omega}")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Linear Function')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Practice Cell — Your Turn!

#### Code Task 1.0.2.1 – Plot a line with $\beta=10$ and $\omega=-2$

Plot a 1D linear function using the following parameters:

- Set $\beta = 10$
- Set $\omega = -2$

Assign the resulting y-values to a variable named `y1`.

```python
import numpy as np 
import matplotlib.pyplot as plt

def linear_Function_1D(x, beta, omega):
    return beta + omega * x

x = np.arange(0, 10, 0.01)

beta = 10
omega = -2

# Compute y1 values
y1 = linear_Function_1D(x, beta, omega)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x, y1, label=f"beta={beta}, omega={omega}")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.xlabel('x')
plt.ylabel('y1')
plt.title('1D Linear Function')
plt.legend()
plt.grid(True)
plt.show()
```

### Questions to Think About

- How does changing $\beta$ affect the graph?
- How does changing $\omega$ affect the slope?

Changing $\beta$ shifts the line vertically.

---

### 1.3 What About 2D Linear Functions?

#### Linear Function in Two Dimensions

In two dimensions, we have two input variables: $x_1$ and $x_2$.

The linear function becomes:

$$
y = \beta + \omega_1 x_1 + \omega_2 x_2
$$

- $\omega_1$ controls the effect of $x_1$
- $\omega_2$ controls the effect of $x_2$
- $\beta$ still shifts the output up or down

Instead of a straight line (as in 1D), in 2D this equation represents a plane.

### Code Cell

Create and plot a 2D linear function.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set sample values
beta = 3.0
omega1 = 1.0
omega2 = 2.0

# Create a grid of x1 and x2 values
x1_vals = np.linspace(-5, 5, 50)
x2_vals = np.linspace(-5, 5, 50)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute y values
Y = beta + omega1 * X1 + omega2 * X2

# Plotting the 3D surface
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='plasma', edgecolor='k', alpha=0.85)

# Labels and title
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_title('3D Linear Plane: $y = 3 + 1x_1 + 2x_2$')

plt.tight_layout()
plt.show()
```

---

### Practice Cell — Your Turn!

#### Code Task 1.0.3.1

Create a 3D linear surface using the following parameters:

- Set $\omega_1 = 1$
- Set $\omega_2 = 0$
- Set $\beta = -3$

Assign the resulting surface (Y values) to a variable named `Y1`.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

beta = -3
omega1 = 1
omega2 = 0

x1_vals = np.linspace(-5, 5, 50)
x2_vals = np.linspace(-5, 5, 50)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute Y1 values
Y1 = beta + omega1 * X1 + omega2 * X2

# Plotting the 3D surface
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y1, cmap='plasma', edgecolor='k', alpha=0.85)

# Labels and title
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_title('3D Linear Plane')

plt.tight_layout()
plt.show()
```

### Mini Reflection Question

**Q:** In your own words, what happens if one of the $\omega$ weights is zero?

If one of the weights is zero, that input variable no longer affects the output.

---

### 1.4 Linear Functions in Vector and Matrix Form

When we have multiple inputs and outputs, writing each linear function separately gets messy.

Instead, we can express everything compactly using vectors and matrices.

Suppose we have two outputs ($y_1, y_2$) and three inputs ($x_1, x_2, x_3$):

$$
\begin{aligned}
y_1 &= \beta_1 + \omega_{11}x_1 + \omega_{12}x_2 + \omega_{13}x_3 \\
y_2 &= \beta_2 + \omega_{21}x_1 + \omega_{22}x_2 + \omega_{23}x_3
\end{aligned}
$$

We can write this in matrix form as:

$$
\mathbf{y} = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{x}
$$

This form is compact, scalable, and ideal for implementation in machine learning libraries like NumPy, PyTorch, or TensorFlow.

### Practice Cell — Your Turn!

#### Code Task 1.0.4.1

```python
import numpy as np

# Inputs (2 inputs)
x1 = 4
x2 = -1

# Input vector (2x1)
x_vec = np.array([[x1],
                  [x2]])

# Weight matrix (3x2)
omega_mat = np.array([
    [2, -1],
    [0.1, 0.3],
    [-3, 1]
])

# Bias vector (3x1)
beta_vec = np.array([
    [0],
    [0.6],
    [0]
])

# Compute output
y_fixed = beta_vec + np.matmul(omega_mat, x_vec)

# Display result
print("y_fixed =")
print(y_fixed)
```

### Mini Reflection Question

**Q:** Why is it useful to use matrix form instead of writing out individual equations?

Matrix form is more compact, easier to scale, and much faster for computers to compute.

---

## Introduction to Special Functions

### Exponential Function

$$
y = \exp(x)
$$

- Maps $x \in (-\infty, +\infty)$ to $(0, +\infty)$
- Grows very fast for positive $x$
- Shrinks toward zero for negative $x$
- Used in softmax, probabilities, and neural networks
- Shape: Convex

### Logarithm Function

$$
y = \log(x)
$$

- Defined only for $x > 0$
- Slowly increasing function
- Shape: Concave

### Code Cell — Plot Exponential and Logarithm

```python
import numpy as np
import matplotlib.pyplot as plt

# Create x values
x_exp = np.linspace(-3, 3, 400)
x_log = np.linspace(0.01, 10, 400)

# Compute y values
y_exp = np.exp(x_exp)
y_log = np.log(x_log)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot exponential
axs[0].plot(x_exp, y_exp, color='green')
axs[0].set_title('Exponential Function: $y = exp(x)$')
axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$exp(x)$')
axs[0].grid(True)

# Plot logarithm
axs[1].plot(x_log, y_log, color='purple')
axs[1].set_title('Logarithm Function: $y = log(x)$')
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$log(x)$')
axs[1].grid(True)

plt.tight_layout()
plt.show()
```

### Bonus Practice

- What happens if we plot $\exp(-x)$?
- What happens if we plot $\log(x+1)$?

```python
import numpy as np
import matplotlib.pyplot as plt

# Create x values
x_exp = np.linspace(-3, 3, 400)
x_log = np.linspace(0.01, 10, 400)

# Compute y values
y_exp = np.exp(x_exp)
y_exp_neg = np.exp(-x_exp)
y_log = np.log(x_log)
y_log_plus1 = np.log(x_log + 1)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot exponential
axs[0, 0].plot(x_exp, y_exp, color='green', label='$exp(x)$')
axs[0, 0].set_title('Exponential Function')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Plot exp(-x)
axs[0, 1].plot(x_exp, y_exp_neg, color='blue', label='$exp(-x)$')
axs[0, 1].set_title('exp(-x)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot log(x)
axs[1, 0].plot(x_log, y_log, color='purple', label='$log(x)$')
axs[1, 0].set_title('log(x)')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot log(x+1)
axs[1, 1].plot(x_log, y_log_plus1, color='orange', label='$log(x+1)$')
axs[1, 1].set_title('log(x+1)')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## Concept Check: Multiple Choice

### Q1. Which of the following is a linear function?

- (A) $y = 2x + 5$
- (B) $y = \exp(x)$
- (C) $y = \log(x)$
- (D) $y = x^2$

**Correct Answer:** (A)

---

### Q2. What happens to $\exp(x)$ as $x \to +\infty$?

- (A) It approaches zero
- (B) It goes to a constant value
- (C) It grows toward infinity
- (D) It oscillates

**Correct Answer:** (C)

---

### Q3. In matrix form, which equation correctly represents a linear mapping from input $\mathbf{x}$ to output $\mathbf{y}$?

- (A) $\mathbf{y} = W\mathbf{x} + b$
- (B) $\mathbf{y} = \log(\mathbf{x}) + b$
- (C) $\mathbf{y} = \exp(\mathbf{x}) + b$
- (D) $\mathbf{y} = \sin(\mathbf{x}) + b$

**Correct Answer:** (A)

---

### Q4. Which function is concave?

- (A) $\exp(x)$
- (B) $\log(x)$
- (C) $x^2$
- (D) $x^3$

**Correct Answer:** (B)

---

## Final Summary — Why We Learned This Background Math

### Summary

- We explored linear functions and saw how they are weighted sums of inputs.
- We visualized 2D and 3D linear functions to build intuition for neural network transformations.
- We introduced exponential and logarithm functions, which appear frequently in deep learning.
- We practiced vector and matrix notation, which is the language of machine learning systems.

### Connection to Deep Learning

- Linear functions form the foundation of neural network layers.
- Vectors and matrices structure inputs, weights, and outputs.
- Exponentials appear in activation functions like Softmax and Sigmoid.
- Logarithms are used in loss functions like Cross-Entropy.
