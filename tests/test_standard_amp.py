#import sys
#import os
#
## Add the project root to sys.path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Ampire import StandardAMP
import pytest
import tensorflow as tf

@pytest.fixture
def amp_optimizer():
    return StandardAMP(learning_rate=0.01, tau=0.01)

def test_denoise(amp_optimizer):
    x = tf.constant([-0.2, 0.5, 1.0, -1.5], dtype=tf.float32)
    expected_output=tf.constant([-0.19, 0.49, 0.99, -1.49], dtype=tf.float32)
    tf.debugging.assert_near(amp_optimizer.denoise(x), expected_output)


def test_compute_correction(amp_optimizer):
    z = tf.constant([0.1, -0.2, 0.3, -0.4], dtype=tf.float32)
    denoise_derivative = tf.constant([0.5, 0.6, 0.7, 0.8], dtype=tf.float32)
    delta = 0.5 # m/n=0.5

    expected_correction = (tf.reduce_mean(denoise_derivative)/delta)*z
    computed_correction = amp_optimizer.compute_correction(z, denoise_derivative, delta)

    tf.debugging.assert_near(computed_correction, expected_correction)


def test_apply_gradients(amp_optimizer):
    w = tf.Variable(2.0, dtype=tf.float32) # Initialize variable
    grad = tf.constant(3.0, dtype=tf.float32) # Computed gradient

    amp_optimizer.apply_gradients([(grad, w)])

    expected_w = 2.0 - (amp_optimizer.learning_rate*3.0)
    tf.debugging.assert_near(w, expected_w)

def test_build(amp_optimizer):
    # Create dummy variables
    variables = [tf.constant(0.1), tf.constant(-0.5), tf.constant(0.8)]

    # Build optimizer with variables
    amp_optimizer.build(variables)

    # Check if variables are properly initialized.
    assert amp_optimizer._variables is not None, (
        "Optimizer variables should not be None after build."
    )
    assert all(isinstance(var, tf.Variable) for var in amp_optimizer._variables), (
        "All elements in `amp_optimizer.variables` should be instance of tf.Variable."
    )
    assert len(amp_optimizer._variables) == len(variables), (
        f"Expected {len(variables)} variables, but found {len(amp_optimizer._variables)}."
    )


def test_get_config(amp_optimizer):
    """
    Test if `get_config()` returns the correct dictionary.
    """
    config = amp_optimizer.get_config()

    assert isinstance(config, dict), "`get_config()` should return a dictionary."
    assert config['learning_rate'] == amp_optimizer.learning_rate, (
        "Mismatch in `learning_rate` parameter."
    )
    assert config['tau'] == amp_optimizer.tau, "Mismatch in `tau` parameter."
    assert config['max_iter'] == amp_optimizer.max_iter, (
        "Mismatch in `max_iter` parameter."
    )
    assert config['tol'] == amp_optimizer.tol, "Mismatch in `tol` parameter."


def test_minimize(amp_optimizer):
    """
    Test the `minimze()` function to ensure it correctly updates variables.
    """
    # Define a simple quadratic loss function: f(w)=(w-3)^2
    def loss_function():
        return tf.square(w-3)

    # Create a trainable variable
    w = tf.Variable(5.0, dtype=tf.float32)

    # Apply minimze
    amp_optimizer.minimize(loss_function, variables=[w])

    # Expected new value of `w` (after one gradient descent step)
    expected_w = w.numpy()

    # Ensure w moves closer to 3 (optimum)
    assert expected_w < 5.0, (
        f"Minimization failed, expected w <5.0 but got {expected_w}"
    )





















