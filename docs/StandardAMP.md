# Standard Approximate Message Passing (AMP) Optimizer

## Introduction
The **Standard Approximate Message Passing (AMP) Optimizer** is an iterative optimization algorithm widely used in **compressed sensing** and **sparse signal recovery**. It efficiently estimates the original signal **x** from noisy linear measurements **y = Ax + noise**. This optimizer is implemented in TensorFlow and follows an AMP-based update rule to refine its estimates iteratively.

## Mathematical Formulation
Given the linear model:
\[ y = A x + \epsilon \]
where:
- \( y \in \mathbb{R}^m \) is the observed measurement vector
- \( A \in \mathbb{R}^{m \times n} \) is the measurement matrix
- \( x \in \mathbb{R}^n \) is the sparse signal to be estimated
- \( \epsilon \) represents the noise term

The AMP algorithm updates the estimate of \( x \) iteratively as follows:
\[ x^{(t+1)} = \eta \left( A^T z^{(t)} + x^{(t)} \right) \]
\[ z^{(t+1)} = y - A x^{(t+1)} + \alpha_t z^{(t)} \]
where:
- \( \eta(\cdot) \) is a **denoising function** (e.g., soft-thresholding)
- \( \alpha_t = \frac{1}{m} \sum \eta'(A^T z^{(t)} + x^{(t)}) \) is the **Onsager correction term**

The denoising function is typically **soft-thresholding**, defined as:
\[ \eta(x) = \text{sign}(x) \max(|x| - \tau, 0) \]
where \( \tau \) is a tunable threshold parameter.

## Implementation
### **1. Initialization**
The optimizer first initializes its variables using the **build** method:
```python
class StandardAMP(BaseAMPOptimizer):
    def build(self, variables: List[tf.Variable], y: tf.Tensor, A: tf.Tensor):
        self._variables = [tf.Variable(v, trainable=True, dtype=tf.float32) for v in variables]
        x_init = tf.zeros_like(y, dtype=tf.float32)
        self._z = tf.Variable(y - tf.linalg.matvec(A, x_init), dtype=tf.float32)
        self.A = A
        self.A_T = tf.transpose(A)
        self.y = y
```
This corresponds to initializing:
\[ z^{(0)} = y - A x^{(0)} \]
where \( x^{(0)} = 0 \).

### **2. Gradient Update Step**
The **gradient update** applies the AMP update rule:
```python
@tf.function
def apply_gradients(self, grads_and_vars):
    ATz = tf.linalg.matvec(self.A_T, self._z)
    updates = self.denoise(ATz + tf.stack([var.value() for _, var in grads_and_vars]))
    for i, (grad, var) in enumerate(grads_and_vars):
        var.assign(updates[i])
```
which corresponds to:
\[ x^{(t+1)} = \eta(A^T z^{(t)} + x^{(t)}) \]

### **3. Computing the Onsager Correction**
To correct for correlations in the residuals, we compute the **Onsager term**:
```python
@tf.function
def compute_correction(self, z, denoise_derivative, delta):
    mean_derivative = tf.reduce_mean(denoise_derivative)
    return (mean_derivative / delta) * z
```
which corresponds to:
\[ \alpha_t z^{(t)} = \left( \frac{1}{m} \sum \eta' (A^T z^{(t)} + x^{(t)}) \right) z^{(t)} \]

### **4. Updating the Residual**
After updating \( x \), the residual **z** is updated:
```python
self._z.assign(
    self.y - tf.linalg.matvec(self.A, x_vec) + (tf.reduce_mean(denoise_derivative) / self.delta) * self._z
)
```
which follows the AMP equation:
\[ z^{(t+1)} = y - A x^{(t+1)} + \alpha_t z^{(t)} \]

## Usage Example
### **Setting Up the Optimizer**
```python
import tensorflow as tf
from Ampire import StandardAMP

# Define measurement matrix A and observed y
y = tf.constant([0.4, -0.2], dtype=tf.float32)
A = tf.constant([[0.5, 0.2], [0.3, 0.7]], dtype=tf.float32)

# Initialize optimizer
amp_optimizer = StandardAMP(learning_rate=0.01, tau=0.05)
w1 = tf.Variable(1, dtype=tf.float32)
w2 = tf.Variable(-0.5, dtype=tf.float32)
amp_optimizer.build([w1, w2], y, A)

# Apply AMP update
amp_optimizer.apply_gradients([(None, w1), (None, w2)])
```

## Conclusion
The Standard AMP optimizer efficiently reconstructs sparse signals by iteratively refining its estimate using the AMP equations. The implemented TensorFlow version leverages automatic differentiation and efficient tensor operations to speed up computations.

