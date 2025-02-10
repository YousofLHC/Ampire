"""
standard_amp.py

This module defines `StandardAMPOptimizer`, an implementation of Approximate Message Passing (AMP) optimization.

Features:
- Implements the standard AMP algorithm.
- Extends `BaseAMPOptimizer` for structured optimization.
- Compatible with TensorFlow's optimizer API (`tf.keras.optimizers.Optimizer`).

Author  : Yousof Ghalenoei (YousofLHC)
Licence : MIT
"""

from typing import Any, List, Tuple, Dict, Optional, Callable
import tensorflow as tf
from .base import BaseAMPOptimizer

class StandardAMP(BaseAMPOptimizer):
    """
    Standard Approximate Message Passing (AMP) optimizer.

    Implements the Standard AMP algorithm for sparse signal recovery.

    Attributes:
    -----------
    learning_rate: float (default=0.01)
        Learning rate for optimization updates.
    max_iter     : int (default=50)
        Maximum number of iteration
    tol          : float (default=1e-6)
        Convergence tolerance.

    Methods:
    --------
    build(variables: List[tf.Variable]) -> None
        Initialize optimizer variables.
    apply_gradients(grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], name: Optional[str]=None) -> None
        Applies gradient updates.
    denoise(x: tf.Tensor) -> tf.Tensor
        Applies a denoising function
    compute_correction(z: tf.Tensor, denoise_derivative: tf.Tensor, delta: float) -> tf.Tensor
        Computes the correction term.
    has_converged(x_old: tf.Tensor, x_new: tf.Tensor) -> bool
        Checks for convergence.
    get_config() -> Dict[str, Any]
        Returns optimizer configuration for TensorFlow compatibility.
    """

    def __init__(self,
                 name         : str="StandardAMP",
                 learning_rate: float=0.01,
                 tau          : float=0.01, 
                 max_iter     : int=50,
                 tol          : float=1e-6,
                 **kwargs     : Any
                 ) -> None:
        """
        Initializes the Standard AMP Optimizer.

        Standard AMP uses the following update rules:
        1. Residual update:
            \[ z^{(t)} = y - Ax^{(t)} + \frac{1}{\delta} \sum_{i=1}^{n} \eta'(x_i^{(t)}) z^{(t-1)} \]
        2. Denoising step:
            \[ x^{(t+1)} = \eta(A^T z^{(t)} + x^{(t)}) \]
        where \( \eta(x) \) is the denoising function and \( \delta = \frac{m}{n} \).

        Parameters:
        -----------
        name         : str   (default="StandardAMP")
            Name of the optimizer.
        learning_rate: float (default=0.01)
            Learning rate for optimization updates.
        tau          : float (default=0.01)
            Threshold parameter for soft-thresholding.
        max_iter     : int   (default=50)
            Maximum number of iterations.
        tol          : float (default=1e-6)
            Convergence tolerance.
        """
        super().__init__(name=name, learning_rate=learning_rate, max_iter=max_iter, tol=tol, **kwargs)
        self.tau        = tau
        self._variables = None # Placeholder for optimizer variables
        self._z         = None  # Ensure _z is initialized

    def build(self,
              variables: List[tf.Variable],
              y        : tf.Tensor,
              A        : tf.Tensor,
              ) -> None:
        """
        Initializes optimizer-related variables and residual term (`z_k`)

        The initial residual is computed as:
        \[ z^{(0)} = y - Ax^{(0)} \]
        where we assume \( x^{(0)} = 0 \).


        Parameters:
        -----------
        variables : List[tf.Variable]
            List of TensorFlow variables that the optimizer will update.
        y         : tf.Tensor
            Observed measurements.
        A         : tf.Tensor
            Measurement matrix.

        Returns:
        -------
        None
        """
        if not variables:
            raise ValueError("No variables provided to the optimizer.")
        self._variables = [tf.Variable(v, trainable=True, dtype=tf.float32) for v in variables]

        # Initialize z_k (Residual term)
        x_init     = tf.zeros_like(y, dtype=tf.float32) # Assume x_0=0 (or another initialization)
        self._z    = tf.Variable(y - tf.linalg.matvec(A, x_init), dtype=tf.float32) 
        self.delta = tf.cast(tf.shape(y)[0], tf.float32) / tf.cast(tf.shape(A)[1], tf.float32)  # m/n
        self.A     = A
        self.A_T   = tf.transpose(A)
        self.y     = y

    @tf.function
    def denoise_derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the derivative of the general denoising function using auto-differentiation.

        Parameters:
        -----------
        x : tf.Tensor
            The input tensor.

        Returns:
        --------
        tf.Tensor
            The computed derivative.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            denoised_x = self.denoise(x)  # Use user-defined denoise function
        return tape.gradient(denoised_x, x)  # Compute derivative dynamically

    @tf.function
    def minimize(self,
                 loss_fn  : Callable[[],tf.Tensor],
                 variables: List[tf.Tensor],
                 ) -> None:
        """
        Computes gradients and updates the variables using `AMP` optimization.
        
       
        Parameters:
        -----------
        loss_fn   : Callable[[], tf.Tensor]
            A function that returns the loss tensor when called.
        variables : List[tf.Tensor]
            List of trainable variable to optimize.
        """
        if self._z is None:
            raise ValueError("Optimizer is not built. Call `build()` before `minimize()`.")

        with tf.GradientTape() as tape:
            loss = loss_fn() # Compute loss
        grads = tape.gradient(loss, variables) # Compute gradients

        # Compute denoise derivative using auto-diff
        denoise_derivative = self.denoise_derivative(tf.stack(variables))

        ## for parallel process in `tf.map_fn`
        ## - This method was optimized by replacing the for-loop with `tf.map_fn`,
        ## - which significantly improves performance in `Graph Mode` and enables
        ## - parallel processing for large tensors.
        ##def compute_corrected_gradients(i, grad):
        ##    return grad - self.compute_correction(self._z[i], denoise_derivative[i], self.delta)
        ### Compute corrected gradient
        ###corrected_grads = tf.map_fn(lambda x: compute_corrected_gradients(x[0], x[1]), 
        ###                        (tf.range(len(grads)), grads), dtype=tf.float32)
        #corrected_grads = [
        #    grad - self.compute_correction(z_var,denoise_derivative[i], self.delta)
        #    for i, (grad, z_var) in enumerate(zip(grads, self._z))
        #]
        corrected_grads = grads - self.compute_correction(self._z, denoise_derivative, self.delta)
        # Update z_k+1
        #x_vec = tf.concat([tf.reshape(var, [-1]) for var in variables], axis=0)
        x_vec = tf.reshape(tf.stack(variables), [-1])
        self._z.assign(
            self.y - tf.linalg.matvec(self.A, x_vec)
            + 
            (tf.reduce_mean(denoise_derivative) / self.delta) * self._z
        )
        
        # Convert zip object to a list
        grads_and_vars = tf.concat([tf.reshape(corrected_grads, [-1]), x_vec], axis=0)#grads_and_vars = list(zip(corrected_grads, variables))
        


        # Apply corrected gradients
        self.apply_gradients(list(zip(tf.unstack(grads_and_vars[:len(variables)]), variables)))#self.apply_gradients(grads_and_vars)
        

    @tf.function
    def apply_gradients(self, grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]], name: Optional[str] = None) -> None:
        """
        Applies updates to variables, supporting both standard gradient updates and AMP-style updates.

        Parameters:
        -----------
        grads_and_vars : List[Tuple[Optional[tf.Tensor], tf.Variable]]
            - A list of tuples containing gradients and the corresponding variables.
            - If gradients are `None`, the AMP update rule is applied.

        name : Optional[str] (default=None)
            - Optional name for the operation.

        Returns:
        --------
        None
        """
        if self._variables is None:
            raise ValueError("Optimizer variables are not initialized. Call `build()` first.")

        if grads_and_vars[0][0] is None:  # AMP update case
            # Compute A^T * z_k (Gradient-free update)
            ATz = tf.linalg.matvec(self.A_T, self._z)
            # Apply soft-thresholding
            updates = self.denoise(ATz + tf.stack([var.value() for _,var in grads_and_vars]))
            for i, (grad, var) in enumerate(grads_and_vars):
                var.assign(updates[i])# Update variable using AMP correction
        else:  # Standard gradient update
            for grad, var in grads_and_vars:
                var.assign_sub(self.learning_rate * grad) # Standard SGD-like update

    @tf.function
    def denoise(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies a soft-thresholding function as a denoising step.

        soft-thresholding function:
        \[ \eta(x) = \text{sign}(x) \max(|x| - \tau, 0) \]
        
        This function shrinks values towards zero, promoting sparsity in the solution.

        Parameters:
        -----------
        x : tf.Tesnor
            The input tensor.

        Returns:
        --------
        tf.Tensor
            The thresholded output.
        """
        return tf.sign(x)*tf.maximum(tf.abs(x)-self.tau, 0)
    
    @tf.function
    def compute_correction(self,
                           z             : tf.Tensor,
                           denoise_derivative: tf.Tensor,
                           delta         : float
                           ) -> tf.Tensor:
        """
        Computes the Onsager correction term:
        \[ \text{correction} = \frac{1}{\delta} \sum_{i=1}^{n} \eta'(x_i) z_i \]
        
        where \( \eta'(x) \) is the derivative of the denoising function.

        Parameters:
        -----------
        z                  : tf.Tensor
            Residual vector from previous iteration.
        denoise_derivative : tf.Tensor
            The derivative of the denoising function.
        delta              : float
            Measurment ratio (m/n)

        Returns:
        --------
        tf.Tensor
            The Onsager correction term.

        """
        mean_derivative = tf.reduce_mean(denoise_derivative) # Average derivative of denoising function
        correction = (mean_derivative/delta)*z # Onsager correction term
        return correction

    def get_config(self) -> Dict[str, Any]:
        """
        Returns optimizer configuration for TensorFlow compatibility.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the optimizer configuration.
        """
        config = super().get_config()
        # Adding StandardAMP-specific parameter
        config.update({
            "tau"          : self.tau,
        })
        return config
        