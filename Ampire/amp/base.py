"""
base.py

This module defines the `BaseAMPOptimizer`, an abstract base class for Approximate Message Passing (AMP) optimizers.

Features:
- Provides a foundation for different AMP variants (e.g., `Damped AMP`, `Robust AMP`).
- Fully abstract class enforcing implementation of key AMP methods in derived classes.
- Compatible with `TensorFlow`'s optimizer API (`tf.keras.optimizers.Optimizer`).

Author : Yousof Ghaleneoi (YousofLHC)
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Optional
import tensorflow as tf

class BaseAMPOptimizer(tf.keras.optimizers.Optimizer, ABC):
    """
    Pure Abstract Base Class for Approximate Message Passing (AMP) optimizers.

    This class serves as a blueprint for different AMP implementations by enforcing
    the implementation of core AMP methods such as signal recovery, denoising,
    correction terms, and convergence checks.

    Attributes:
    -----------
    learning_rate : float (default=0.01)
        Learning rate for optimization updates.
    name     : str (default="BaseAMPOptimizer")
        Name of the optimizer.
    max_iter : int (default=50)
        Maximum number of iterations.
    tol      : float (default=1e-6)
        Convergence tolerance.

    Methods:
    --------
    build(variables: List[tf.Variable]) -> None
        Abstract method to initialize optimizer variables.
    apply_gradients(grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], name: Optional[str]=None) -> None
        Abstract method for applying gradient updates.
    denoise(x: tf.Tensor) -> tf.Tensor
        Abstract method for applying a denoising function.
    compute_correction(z: tf.Tensor, denoise_derivative: tf.Tensor, delta: float) -> tf.Tensor
        Abstract method to compute the correction term.
    has_converged(x_old: tf.Tensor, x_new: tf.Tensor) -> bool
        Abstract method to check for convergence.
    get_config() -> Dict[str, Any]
        Abstract method to return optimizer configuration for TensorFlow compatibility.
    """

    def __init__(self, 
                 learning_rate: float=0.01,
                 name         : str="BaseAMPOptimizer",
                 max_iter     : int=50,
                 tol          : float=1e-6,
                 **kwargs     : Any
                 ) -> None:
        """
        Initializes the Base AMP Optimizer.

        Parameters:
        -----------
        learning_rate : float (default=0.01)
            Learning rate for optimization updates.
        name          : str (default="BaseAMPOptimizer")
            Name of the optimizer.
        max_iter      : int (default=50)
            Maximum number of iterations.
        tol           : float (default=1e-6)
            Convergence tolerance.
        """
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            **kwargs)
        self.learning_rate : float=learning_rate
        self.name          : str=name
        self.max_iter      : int=max_iter
        self.tol           : float=tol

    @abstractmethod
    def build(self,variables: List[tf.Variable]) -> None:
        """Initializes optimizer-related variables."""
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(self,
                        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
                        name          : Optional[str]=None
                        ) -> None:
        """Applies gradient updates to model parameters."""
        raise NotImplementedError

    @abstractmethod
    def denoise(self, x: tf.Tensor) -> tf.Tensor:
        """Applies a denoising function to the estimated signal."""
        raise NotImplementedError

    @abstractmethod
    def compute_correction(self, 
                           z             : tf.Tensor, 
                           denoise_derivative: tf.Tensor, 
                           delta         : float
                           ) -> tf.Tensor:
        """Computes the correction term for improved convergence."""
        raise NotImplementedError

    
    def has_converged(self, x_old: tf.Tensor, x_new: tf.Tensor) -> bool:
        """Checks whether the algorithm has converged."""
        return tf.norm(x_new - x_old) < self.tol

    def get_config(self) -> Dict[str, Any]:
        """Returns optimizer configuration for TensorFlow compatibility."""
        config = super(tf.keras.optimizers.Optimizer).get_config()
        config.update({
            "name"    : self.name,
            "max_iter": self.max_iter,
            "tol"     : self.tol,
        })
        return config
