import pytest
import tensorflow as tf
from Ampire import StandardAMP

# Fixture to create an optimizer instance.
@pytest.fixture
def amp_optimizer():
    return StandardAMP(learning_rate=0.01, tau=0.01)

# Verify correct initialization of parameters
def test_initialization(amp_optimizer):
    """
    Test if `StandardAMP` initializes with correct default parameters.
    """
    assert amp_optimizer.learning_rate==0.01, (
        f"Incorrect `learning_rate` initialization. Got{amp_optimizer.learning_rate}."
        )
    assert amp_optimizer.tau==0.01, (
        f"Incorrect `tau` initialization. Got{amp_optimizer.tau}."
    )
    assert amp_optimizer.max_iter==50, (
        f"Incorrect `max_iter` initialization. Got{amp_optimizer.max_iter}."
    )
    assert amp_optimizer.tol==1e-6, (
        f"Incorrect `tolerance` initialization. Got{amp_optimizer.tol}."
    )


# Test `denoise` function
def test_denoise(amp_optimizer):
    """
    Test if `denoise` function applies soft-thresholding correctly.
    """
    x               = tf.constant([-0.02, 0.50, 1.00, -1.50], dtype=tf.float32)
    expected_output = tf.constant([-0.01, 0.49, 0.99, -1.49], dtype=tf.float32)

    result          = amp_optimizer.denoise(x)
    tf.debugging.assert_near(result, expected_output, atol=1e-6)


# Test `compute_correction` function
def test_compute_correction(amp_optimizer):
    """
    Test if `compute_correction` correctly computes the Onsager correction term.
    """
    z                  = tf.constant([0.1, -0.2, 0.3, -0.4], dtype=tf.float32)
    denoise_derivative = tf.constant([0.5, 0.6, 0.7, 0.7], dtype=tf.float32)
    delta              = 0.5 # Measurement ratio m/n=0.5

    expected_correction = (tf.reduce_mean(denoise_derivative)/delta)*z
    computed_correction = amp_optimizer.compute_correction(z, denoise_derivative, delta)

    tf.debugging.assert_near(computed_correction, expected_correction, atol=1e-6)


# Test `apply_gradients` function
def test_apply_gradients(amp_optimizer):
    """
    Test if `apply_gradients` correctly updates variables.
    """
    w = tf.Variable(2.0, dtype=tf.float32)  # Use `tf.Variable` instead of `tf.constant`
    grad = tf.constant(3.0, dtype=tf.float32) # Gradient value

    amp_optimizer.apply_gradients([(grad, w)])

    expected_w = 2.0 - (amp_optimizer.learning_rate*3.0)
    tf.debugging.assert_near(w, expected_w, atol=1e-6 )









