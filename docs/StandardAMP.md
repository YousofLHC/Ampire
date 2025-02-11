## 2. Mathematical Formulation

Given the linear model:

\[
y = A x + \epsilon
\]

where:
- \( y \in \mathbb{R}^m \) is the observation vector,
- \( A \in \mathbb{R}^{m \times n} \) is the measurement matrix,
- \( x \in \mathbb{R}^n \) is the sparse signal to be recovered, and
- \( \epsilon \) represents the noise,

the Standard Approximate Message Passing (AMP) algorithm iteratively refines the estimate of \( x \) using two main steps:

### Denoising/Update Step
\[
x^{(t+1)} = \eta \left( A^\top z^{(t)} + x^{(t)} \right)
\]

where the denoising function \( \eta(\cdot) \) is typically defined by soft-thresholding:

\[
\eta(x) = \operatorname{sign}(x) \cdot \max(|x| - \tau, 0)
\]

Here, \( \tau \) is a tunable threshold parameter.

### Residual Update with Onsager Correction
\[
z^{(t+1)} = y - A x^{(t+1)} + \alpha_t \, z^{(t)}
\]

The Onsager correction term \( \alpha_t \) is computed as:

\[
\alpha_t = \frac{1}{m} \sum_{i=1}^{m} \eta'\!\left( \left[ A^\top z^{(t)} + x^{(t)} \right]_i \right)
\]

This correction term helps compensate for the correlations introduced by the non-linear denoising step.

---

## 3. Implementation

The optimizer is implemented in TensorFlow by extending a base optimizer class. Below are the key components of the implementation:

### 3.1 Initialization

The `build` method initializes the optimizer variables and computes the initial residual:

- **Initial residual:**
  \[
  z^{(0)} = y - A \cdot x^{(0)} \quad \text{with} \quad x^{(0)} = 0
  \]
- The measurement ratio is computed as:
  \[
  \delta = \frac{m}{n}
  \]

**Code snippet:**
```python
def build(self, variables: List[tf.Variable], y: tf.Tensor, A: tf.Tensor) -> None:
    if not isinstance(variables, list):
        raise TypeError("`variables` must be a list of TensorFlow variables.")
    if not variables:
        raise ValueError("No variables provided to the optimizer.")
    # Validate that A and y are proper 2D tensors.
    if len(A.shape) != 2:
        raise ValueError("`A` must be a 2D tensor (matrix).")
    if len(y.shape) != 2:
        raise ValueError("`y` must be a 2D tensor (column vector).")
    
    self._variables = variables
    x_init = tf.zeros([tf.shape(A)[1]], dtype=tf.float32)
    # Reshape y to a flat vector.
    y = tf.reshape(tf.cast(y, tf.float32), [-1])
    A = tf.cast(A, tf.float32)
    initial_z = y - tf.linalg.matvec(A, x_init)
    self._z = tf.Variable(tf.reshape(initial_z, [-1, 1]), dtype=tf.float32, trainable=False)
    self.delta = tf.cast(tf.shape(y)[0], tf.float32) / tf.cast(tf.shape(A)[1], tf.float32)
    self.A, self.A_T, self.y = A, tf.transpose(A), y
```
## 3.2 Soft-Thresholding Denoising Function
The denoising function applies the soft-thresholding operator:

\[
\eta(x) = \operatorname{sign}(x) \cdot \max(|x| - \tau, 0)
\]

**Code snippet:**
```python
@tf.function
def denoise(self, x: tf.Tensor) -> tf.Tensor:
    return tf.sign(x) * tf.maximum(tf.abs(x) - self.tau, 0)
```

## 3.3 Derivative of the Denoising Function
The derivative \( \eta'(x) \) is computed using TensorFlow's automatic differentiation:

**Code snippet:**
```python
@tf.function
def denoise_derivative(self, x: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x)
        denoised_x = self.denoise(x)
    return tape.gradient(denoised_x, x)
```
## 3.4 Onsager Correction
The Onsager correction term adjusts the gradient update to compensate for the non-linear denoising step:

\[
\text{correction} = \left(\frac{\text{mean}(\eta'(x))}{\delta}\right) \cdot z
\]

**Code snippet:**
```python
@tf.function
def compute_correction(self, z: tf.Tensor, denoise_derivative: tf.Tensor, delta: float) -> tf.Tensor:
    mean_derivative = tf.reduce_mean(denoise_derivative)
    correction = (mean_derivative / delta) * z
    return correction
```

## 3.5 Gradient and Residual Update
The `minimize` method integrates the update rules:

- **x-update:**
  \[
  x^{(t+1)} = \eta\!\left(A^\top z^{(t)} + x^{(t)}\right)
  \]
- **z-update:**
  \[
  z^{(t+1)} = y - A x^{(t+1)} + \left(\frac{\text{mean}(\eta'(x))}{\delta}\right) z^{(t)}
  \]

**Code snippet (simplified):**
```python
@tf.function
def minimize(self, loss_fn: Callable[[], tf.Tensor], variables: List[tf.Tensor]) -> None:
    with tf.GradientTape() as tape:
        for var in self._variables:
            tape.watch(var)
        loss = loss_fn()
    grads = tape.gradient(loss, variables)
    if any(g is None for g in grads):
        raise ValueError("Gradient computation failed: Some gradients are `None`.")
    
    # Compute derivative of the denoiser.
    denoise_derivative = self.denoise_derivative(tf.stack(variables))
    z_flat = tf.reshape(self._z, [-1])
    ATz = tf.linalg.matvec(self.A_T, z_flat)
    correction = self.compute_correction(ATz, denoise_derivative, self.delta)
    
    grads_tensor = tf.stack(grads)
    corrected_grads_tensor = grads_tensor - correction
    corrected_grads = tf.unstack(corrected_grads_tensor)
    
    x_vec = tf.stack(variables)
    new_z = self.y - tf.linalg.matvec(self.A, x_vec) + \
            (tf.reduce_mean(denoise_derivative) / self.delta) * tf.reshape(self._z, [-1])
    self._z.assign(tf.reshape(new_z, [-1, 1]))
    
    self.apply_gradients(list(zip(corrected_grads, variables)))
```

## 3.6 AMP-Only Update in apply_gradients
Since the update should always follow the AMP rule, the `apply_gradients` method ignores any provided gradients and performs:

\[
x^{(t+1)} = \eta\!\left(A^\top z^{(t)} + x^{(t)}\right)
\]

**Code snippet:**
```python
@tf.function
def apply_gradients(self, grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]], name: Optional[str] = None) -> None:
    if self._variables is None:
        raise ValueError("Optimizer variables are not initialized. Call `build()` first.")
    if not grads_and_vars or not isinstance(grads_and_vars, list):
        raise ValueError("`grads_and_vars` must be a non-empty list.")
    if any(not isinstance(var, tf.Variable) for _, var in grads_and_vars):
        raise TypeError("Each tuple in `grads_and_vars` must contain a Tensor and a tf.Variable.")
    
    # Always use the AMP update branch.
    z_flat = tf.reshape(self._z, [-1])
    ATz = tf.linalg.matvec(self.A_T, z_flat)
    x_current = tf.stack([var.value() for _, var in grads_and_vars])
    tf.debugging.assert_equal(tf.shape(ATz), tf.shape(x_current),
                                message="Shape mismatch: ATz vs x_current")
    updates = self.denoise(ATz + x_current)
    for i, (_, var) in enumerate(grads_and_vars):
        if var.shape == ():
            var.assign(tf.squeeze(updates[i]))
        else:
            var.assign(updates[i])
    tf.print("ATz:", ATz, "shape:", tf.shape(ATz))
    tf.print("x_current:", x_current, "shape:", tf.shape(x_current))
    tf.print("updates:", updates, "shape:", tf.shape(updates))
```

## 3.7 Configuration
The `get_config` method provides a configuration dictionary to enable TensorFlow serialization.

**Code snippet:**
```python
def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        "tau": self.tau,
    })
    return config
```

## 4. Usage Examples

### For End Users
Below is an example of how to set up and use the Standard AMP optimizer:

```python
import tensorflow as tf
from Ampire import StandardAMP

# Define the measurement matrix A and the observed vector y.
A = tf.constant([[0.5, 0.2], [0.3, 0.7]], dtype=tf.float32)
y = tf.constant([[0.4], [-0.2]], dtype=tf.float32)

# Initialize the AMP optimizer.
amp_optimizer = StandardAMP(learning_rate=0.01, tau=0.05)

# Define the trainable variables (the signal to recover).
w1 = tf.Variable(1.0, dtype=tf.float32)
w2 = tf.Variable(-0.5, dtype=tf.float32)

# Build the optimizer (initialize variables and compute initial residual).
amp_optimizer.build([w1, w2], y, A)

# Apply an AMP update (the gradients are ignored and the AMP branch is used).
amp_optimizer.apply_gradients([(None, w1), (None, w2)])

print("Updated variables:", w1.numpy(), w2.numpy())
```
### For Developers
Developers can integrate the optimizer with a custom loss function and use the `minimize` method to perform updates iteratively.

```python
def loss_function():
    # Concatenate the variables into a column vector.
    x_vec = tf.reshape(tf.stack(amp_optimizer._variables), [-1, 1])
    # Compute the squared error loss.
    return tf.reduce_sum(tf.square(A @ x_vec - y))

# Run the optimizer for a fixed number of iterations.
for iteration in range(amp_optimizer.max_iter):
    amp_optimizer.minimize(loss_function, [w1, w2])
    current_loss = loss_function().numpy()
    tf.print("Iteration:", iteration, "Loss:", current_loss)
```

Developers can also enable the debug prints (as provided in the AMP branch of `apply_gradients`) to inspect intermediate values such as \(A^\top z\), the current variable values, and the computed updates.

## 5. Testing & Visualization

The optimizer includes a comprehensive suite of tests covering:

- **Basic Functionality:** Initialization, build method, and the denoising function.
- **Convergence & Performance:** Checking that the loss decreases over iterations and saving convergence curves.
- **Stress Tests:** Evaluating the optimizer on large random matrices.
- **Edge Cases & Error Handling:** Verifying that invalid inputs are properly handled and meaningful error messages are raised.
- **Additional Tests:** Assessing the derivative of the denoising function and simulating scenarios where gradients are absent (forcing AMP updates).

For instance, the convergence test not only asserts that the loss decreases but also generates a convergence curve plot:

```python
plt.figure()
plt.plot(losses, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Convergence")
plt.savefig("test_convergence.png")
plt.close()
```
Developers are encouraged to extend these tests or add additional visualizations (e.g., plotting the evolution of the residual \(z\) or the variable updates) to gain further insight into the optimization process.

## 6. Conclusion

The Standard AMP Optimizer provides an efficient method for sparse signal recovery by leveraging the principles of approximate message passing. Its TensorFlow implementation benefits from automatic differentiation and GPU acceleration, making it both fast and scalable.

- **For End Users:** The optimizer can be easily integrated into existing TensorFlow projects to solve compressed sensing and sparse recovery problems.
- **For Developers:** The modular design and comprehensive testing make it straightforward to extend, debug, and modify the optimizer for specialized applications.

We hope this optimizer serves as a valuable tool in your projects. For further inquiries or contributions, please refer to the project's repository or contact the author.

Happy Coding!
