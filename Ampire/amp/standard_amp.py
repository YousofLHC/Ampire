"""
standard_amp.py

This module defines `StandardAMPOptimizer`, an implementation of Approximate Message Passing (AMP) optimization.

Features:
- Implements the standard AMP algorithm.
- Extends `BaseAMPOptimizer` for structured optimization.
- Compatible with TensorFlow's optimizer API (tf.keras.optimizers.Optimizer).

Author  : Yousof Ghalenoei (YousofLHC)
Licence : MIT
"""

from typing import Any, List, Tuple, Dict, Optional, Callable
import tensorflow as tf
from .base import BaseAMPOptimizer

class StandardAMP(BaseAMPOptimizer):
    """
    Standard Approximate Message Passing (AMP) optimizer.
    
    This optimizer is designed for sparse signal recovery (e.g., compressed sensing). Given the
    linear model:
    
        y = A x + ε
    
    where:
      - y ∈ ℝᵐ is the observation vector,
      - A ∈ ℝᵐˣⁿ is the measurement matrix,
      - x ∈ ℝⁿ is the sparse signal to recover,
      - ε is the noise,
    
    the AMP algorithm iteratively updates the estimate x using the following equations:
    
      **Denoising/Update step:**
          x^(t+1) = η(Aᵀ z^(t) + x^(t))
      
      **Residual update with Onsager correction:**
          z^(t+1) = y - A x^(t+1) + α_t z^(t)
          
      where the denoising function η(·) is typically the soft-thresholding function defined by:
      
          η(x) = sign(x) · max(|x| - τ, 0)
      
      and the Onsager correction term is computed as:
      
          α_t = (1/m) ∑ᵢ η'(Aᵀ z^(t) + x^(t)).
    
    Attributes:
    -----------
    learning_rate: float (default=0.01)
        The learning rate for gradient updates.
    tau: float (default=0.01)
        The threshold parameter τ used in the soft-thresholding function.
    max_iter: int (default=50)
        Maximum number of iterations (not directly used in this class but can be used externally).
    tol: float (default=1e-6)
        Convergence tolerance (can be used to check convergence externally).
    
    Methods:
    --------
    build(variables: List[tf.Variable], y: tf.Tensor, A: tf.Tensor) -> None
        Initializes the optimizer variables and computes the initial residual:
            z^(0) = y - A x^(0)   with x^(0)=0.
    apply_gradients(grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], name: Optional[str] = None) -> None
        Applies gradient updates. In the AMP branch (when gradients are None) it computes:
            x^(t+1) = η(Aᵀ z^(t) + x^(t))
    denoise(x: tf.Tensor) -> tf.Tensor
        Applies the soft-thresholding denoising function:
            η(x) = sign(x) · max(|x| - τ, 0)
    compute_correction(z: tf.Tensor, denoise_derivative: tf.Tensor, delta: float) -> tf.Tensor
        Computes the Onsager correction term:
            correction = (mean(η'(x)) / δ) · z
    get_config() -> Dict[str, Any]
        Returns a configuration dictionary for TensorFlow compatibility.
    """

    def __init__(self,
                 name: str = "StandardAMP",
                 learning_rate: float = 0.01,
                 tau: float = 0.01, 
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 **kwargs: Any) -> None:
        r"""
        Initializes the Standard AMP Optimizer.
        
        Parameters:
        -----------
        name : str
            Name of the optimizer.
        learning_rate : float
            Learning rate for gradient updates (must be positive).
        tau : float
            Threshold parameter for soft-thresholding (must be positive).
        max_iter : int
            Maximum number of iterations (must be positive).
        tol : float
            Convergence tolerance (must be positive).
        **kwargs : Any
            Additional keyword arguments.
        """
        if learning_rate <= 0:
            raise ValueError(f"`learning_rate` must be positive. Got {learning_rate}")
        if tau <= 0:
            raise ValueError(f"`tau` must be positive. Got {tau}")
        if max_iter <= 0:
            raise ValueError(f"`max_iter` must be positive. Got {max_iter}")
        if tol <= 0:
            raise ValueError(f"`tol` must be positive. Got {tol}")

        super().__init__(name=name, learning_rate=learning_rate, max_iter=max_iter, tol=tol, **kwargs)
        self.tau = tau
        self._variables = None  # List of variables to be optimized.
        self._z = None  # Residual term, to be initialized in build().

    def build(self,
              variables: List[tf.Variable],
              y: tf.Tensor,
              A: tf.Tensor) -> None:
        """
        Initializes optimizer variables and computes the initial residual _z.
        
        The initial residual is defined as:
        
            z^(0) = y - A * x^(0)
        
        where x^(0) is assumed to be zero.
        
        Parameters:
        -----------
        variables : List[tf.Variable]
            The list of trainable variables (x).
        y : tf.Tensor
            The observed measurement vector (shape: [m, 1]).
        A : tf.Tensor
            The measurement matrix (shape: [m, n]).
        """
        if not isinstance(variables, list):
            raise TypeError("`variables` must be a list of TensorFlow variables.")
        if not variables:
            raise ValueError("No variables provided to the optimizer.")
        if not isinstance(A, tf.Tensor) or not isinstance(y, tf.Tensor):
            raise TypeError(f"`A` and `y` must be TensorFlow tensors. Got A({tf.types(A)}), y({tf.types(y)})")
        if len(A.shape) != 2:
            raise ValueError("`A` must be a 2D tensor (matrix).")
        if len(y.shape) != 2:
            raise ValueError("`y` must be a 2D tensor (column vector).")
        
        self._variables = variables
        x_init = tf.zeros([tf.shape(A)[1]], dtype=tf.float32)
        # Convert y to a flat vector (m,)
        y = tf.reshape(tf.cast(y, tf.float32), [-1])
        A = tf.cast(A, tf.float32)
        initial_z = y - tf.linalg.matvec(A, x_init)
        self._z = tf.Variable(tf.reshape(initial_z, [-1, 1]), dtype=tf.float32, trainable=False)
        self.delta = tf.cast(tf.shape(y)[0], tf.float32) / tf.cast(tf.shape(A)[1], tf.float32)
        self.A, self.A_T, self.y = A, tf.transpose(A), y

        print(f"y shape: {y.shape}, A shape: {A.shape}, x_init shape: {x_init.shape}")
        print(f"Initial z shape: {initial_z.shape}")

    @tf.function
    def denoise_derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the derivative of the denoise function using automatic differentiation.
        
        This is used to compute the Onsager correction term.
        
        Parameters:
        -----------
        x : tf.Tensor
            Input tensor.
        
        Returns:
        --------
        tf.Tensor
            The derivative of the denoising function.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            denoised_x = self.denoise(x)
        return tape.gradient(denoised_x, x)

    @tf.function
    def minimize(self, loss_fn: Callable[[], tf.Tensor], variables: List[tf.Tensor]) -> None:
        """
        Computes gradients and updates variables using the AMP optimization rule.
        
        The AMP updates are defined by the following steps:
        
            1. x-update:
               x^(t+1) = η( Aᵀ z^(t) + x^(t) )
            
            2. z-update:
               z^(t+1) = y - A x^(t+1) + α_t z^(t)
               
            where the Onsager correction term is:
            
               α_t = (1/m) ∑ η'(Aᵀ z^(t) + x^(t))
        
        Parameters:
        -----------
        loss_fn : Callable[[], tf.Tensor]
            A function that returns the loss tensor.
        variables : List[tf.Tensor]
            The list of trainable variables.
        """
        if not callable(loss_fn):
            raise TypeError("`loss_fn` must be a callable function that returns a Tensor.")
        if not isinstance(variables, list) or not all(isinstance(v, tf.Variable) for v in variables):
            raise TypeError("`variables` must be a list of TensorFlow variables.")
        if self._z is None:
            raise ValueError("Optimizer is not built. Call `build()` before `minimize()`.")

        with tf.GradientTape() as tape:
            for var in self._variables:
                tape.watch(var)
            loss = loss_fn()
        grads = tape.gradient(loss, variables)
        if any(g is None for g in grads):
            raise ValueError("Gradient computation failed: Some gradients are `None`.")

        # Compute denoising derivative: η'(·)
        denoise_derivative = self.denoise_derivative(tf.stack(variables))
        # Reshape residual _z to rank 1
        z_flat = tf.reshape(self._z, [-1])
        ATz = tf.linalg.matvec(self.A_T, z_flat)
        # Compute Onsager correction: (mean(η'(·))/δ) * z
        correction = self.compute_correction(ATz, denoise_derivative, self.delta)

        grads_tensor = tf.stack(grads)
        corrected_grads_tensor = grads_tensor - correction
        corrected_grads = tf.unstack(corrected_grads_tensor)

        x_vec = tf.stack(variables)
        new_z = self.y - tf.linalg.matvec(self.A, x_vec) \
                + (tf.reduce_mean(denoise_derivative) / self.delta) * tf.reshape(self._z, [-1])
        self._z.assign(tf.reshape(new_z, [-1, 1]))

        self.apply_gradients(list(zip(corrected_grads, variables)))

    @tf.function
    def apply_gradients(self, grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]], name: Optional[str] = None) -> None:
        """
        Applies AMP-based gradient updates to the variables.
        
        In this implementation, the update always follows the AMP rule:
            x^(t+1) = η( Aᵀ z^(t) + x^(t) )
        regardless of the value of the gradients.
        
        Parameters:
        -----------
        grads_and_vars : List[Tuple[Optional[tf.Tensor], tf.Variable]]
            A list of tuples pairing gradients (which will be ignored) with their corresponding variables.
        name : Optional[str]
            Optional name for the operation.
        """
        if self._variables is None:
            raise ValueError("Optimizer variables are not initialized. Call `build()` first.")
        if not grads_and_vars or not isinstance(grads_and_vars, list):
            raise ValueError("`grads_and_vars` must be a non-empty list.")
        if any(not isinstance(var, tf.Variable) for _, var in grads_and_vars):
            raise TypeError("Each tuple in `grads_and_vars` must contain a Tensor and a tf.Variable.")
    
        # Always use the AMP update branch.
        # Reshape _z to rank 1 (m,)
        z_flat = tf.reshape(self._z, [-1])
        ATz = tf.linalg.matvec(self.A_T, z_flat)
        x_current = tf.stack([var.value() for _, var in grads_and_vars])
        tf.debugging.assert_equal(tf.shape(ATz), tf.shape(x_current),
                                    message="Shape mismatch: ATz vs x_current")
        updates = self.denoise(ATz + x_current)
        for i, (_, var) in enumerate(grads_and_vars):
            # Update each variable using the AMP rule.
            if var.shape == ():
                var.assign(tf.squeeze(updates[i]))
            else:
                var.assign(updates[i])
        tf.print("ATz:", ATz, "shape:", tf.shape(ATz))
        tf.print("x_current:", x_current, "shape:", tf.shape(x_current))
        tf.print("updates:", updates, "shape:", tf.shape(updates))


    #@tf.function
    #def apply_gradients(self, grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]], name: Optional[str] = None) -> None:
    #    """
    #    Applies gradient updates to the variables.
    #    
    #    In the AMP branch (when gradients are None), the update follows:
    #    
    #        x^(t+1) = η( Aᵀ z^(t) + x^(t) )
    #        
    #    where the denoising function η is the soft-thresholding operator.
    #    
    #    Parameters:
    #    -----------
    #    grads_and_vars : List[Tuple[Optional[tf.Tensor], tf.Variable]]
    #        A list of tuples pairing gradients with their corresponding variables.
    #    name : Optional[str]
    #        Optional name for the operation.
    #    """
    #    if self._variables is None:
    #        raise ValueError("Optimizer variables are not initialized. Call `build()` first.")
    #    if not grads_and_vars or not isinstance(grads_and_vars, list):
    #        raise ValueError("`grads_and_vars` must be a non-empty list.")
    #    if any(not isinstance(var, tf.Variable) for _, var in grads_and_vars):
    #        raise TypeError("Each tuple in `grads_and_vars` must contain a Tensor and a tf.Variable.")
#
    #    if grads_and_vars[0][0] is None:  # AMP update branch
    #        # Reshape _z to rank 1 (m,)
    #        z_flat = tf.reshape(self._z, [-1])
    #        ATz = tf.linalg.matvec(self.A_T, z_flat)
    #        x_current = tf.stack([var.value() for _, var in grads_and_vars])
    #        tf.debugging.assert_equal(tf.shape(ATz), tf.shape(x_current),
    #                                    message="Shape mismatch: ATz vs x_current")
    #        updates = self.denoise(ATz + x_current)
    #        for i, (grad, var) in enumerate(grads_and_vars):
    #            if var.shape == ():
    #                var.assign(tf.squeeze(updates[i]))
    #            else:
    #                var.assign(updates[i])
    #        tf.print("ATz:", ATz, "shape:", tf.shape(ATz))
    #        tf.print("x_current:", x_current, "shape:", tf.shape(x_current))
    #        tf.print("updates:", updates, "shape:", tf.shape(updates))
    #    else:  # Standard gradient update
    #        for grad, var in grads_and_vars:
    #            if grad is None:
    #                continue
    #            var.assign_sub(self.learning_rate * grad)

    @tf.function
    def denoise(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies the soft-thresholding denoising function.
        
        The soft-thresholding operator is defined as:
        
            η(x) = sign(x) * max(|x| - τ, 0)
        
        where τ is the threshold parameter.
        
        Parameters:
        -----------
        x : tf.Tensor
            Input tensor.
            
        Returns:
        --------
        tf.Tensor
            The thresholded output.
        """
        return tf.sign(x) * tf.maximum(tf.abs(x) - self.tau, 0)

    @tf.function
    def compute_correction(self, z: tf.Tensor, denoise_derivative: tf.Tensor, delta: float) -> tf.Tensor:
        """
        Computes the Onsager correction term.
        
        The Onsager correction term is given by:
        
            correction = (mean(η'(·)) / δ) * z
        
        where δ = m/n is the measurement ratio and η'(·) is the derivative of the denoising function.
        
        Parameters:
        -----------
        z : tf.Tensor
            The residual vector in feature space.
        denoise_derivative : tf.Tensor
            The derivative of the denoising function.
        delta : float
            The measurement ratio (m/n).
            
        Returns:
        --------
        tf.Tensor
            The computed correction term.
        """
        if not isinstance(z, tf.Tensor) or not isinstance(denoise_derivative, tf.Tensor):
            raise TypeError("Inputs must be TensorFlow tensors.")
        if not isinstance(delta, float) and not isinstance(delta, tf.Tensor):
            raise TypeError("`delta` must be a float or a Tensor.")
        mean_derivative = tf.reduce_mean(denoise_derivative)
        correction = (mean_derivative / delta) * z
        return correction

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the optimizer for TensorFlow compatibility.
        
        Returns:
        --------
        dict
            A dictionary containing the configuration parameters.
        """
        config = super().get_config()
        config.update({
            "tau": self.tau,
        })
        return config

