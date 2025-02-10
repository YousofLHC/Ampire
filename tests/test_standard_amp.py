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




def test_build(amp_optimizer):
    """
    Test if `build` correctly initializes optimizer variables.
    """

    # Define dummy trainable Variables
    variables = [tf.Variable(0.1), tf.Variable(-0.5), tf.Variable(0.8)]

    # Dummy input (y) and measurement matrix (A)
    y = tf.constant([0.2, -0.3, 0.5], dtype=tf.float32)
    A = tf.constant(
        [
            [0.1, 0.2, 0.3],
            [-.1, 0.5, -.4],
            [0.3, -.2, 0.1]
        ],
        dtype=tf.float32,
    )

    # Build optimizer
    amp_optimizer.build(variables, y, A)


    # Assertions
    assert amp_optimizer._variables is not None, (
        "Optimizer Variables should not be None after build."
    )
    assert isinstance(amp_optimizer._variables, list), (
        "`_Variables` should be a list."
    )
    assert all(isinstance(var, tf.Variable) for var in amp_optimizer._variables), (
        "All elements in `_variables` should be TensorFlow Variables."
    ) 
    assert len(amp_optimizer._variables)==len(variables), (
        f"Expected {len(amp_optimizer._variables)} variables, but found {len(amp_optimizer._variables)}."
    )

    # Check if `_z` (residual term) is initialized correctly
    expected_z = y-tf.linalg.matvec(A, tf.zeros_like(y, dtype=tf.float32))
    tf.debugging.assert_near(amp_optimizer._z, expected_z, atol=1e-6)

    # Check if `delta` is computed correctly
    expected_delta = tf.cast(tf.shape(y)[0],tf.float32)/tf.cast(tf.shape(A)[1],dtype=tf.float32)
    tf.debugging.assert_near(amp_optimizer.delta, expected_delta, atol=1e-6)


def test_get_config(amp_optimizer):
    """
    Test if `get_config()` correctly returns optimizer configuration as a dictionary.
    """

    config = amp_optimizer.get_config()

    # Assertions
    assert isinstance(config, dict), "`get_config()` should return a dictionary."
    assert config['learning_rate']==amp_optimizer.learning_rate, (
        "Mismatch in `learning_rate` parameter."
    )
    assert config['tau']==amp_optimizer.tau, (
        "Mismatch in `tau` parameter."
    )
    assert config['max_iter']==amp_optimizer.max_iter, (
        "Mismatch in `max_iter` parameter."
    )
    assert config['tol']==amp_optimizer.tol, (
        "Mismatch `tol` parameter."
    )

def test_minimize_without_build(amp_optimizer):
    """
    Test that calling `minimize()` before `build()` raises an error.
    """
    w = tf.Variable(5.0, dtype=tf.float32)

    def loss_function():
        return tf.square(w - 3)

    with pytest.raises(ValueError, match="Optimizer is not built. Call `build\\(\\)` before `minimize\\(\\)`."):
        amp_optimizer.minimize(loss_function, variables=[w])


def test_apply_gradients(amp_optimizer):
    """
    Test if `apply_gradients` correctly updates variables in both AMP and standard gradient modes.
    """
    # Initialize optimizer variables
    w = tf.Variable(2.0, dtype=tf.float32)
    z = tf.Variable([0.1], dtype=tf.float32)

    # Initialize measurement matrix `A` and observation `y`
    y = tf.constant([0.2], dtype=tf.float32)
    A = tf.constant([[0.5]], dtype=tf.float32)

    # Build optimizer and set initial residual `_z`
    amp_optimizer.build([w], y, A)
    amp_optimizer._z.assign(z)

    # Test AMP-style update (without explicit gradients)
    amp_optimizer.apply_gradients([(None, w)])
    expected_w_amp = amp_optimizer.denoise(tf.linalg.matvec(tf.transpose(A), z) + w)
    #expected_w_amp = tf.squeeze(expected_w_amp)
    tf.debugging.assert_near(w, expected_w_amp, atol=1e-1)

    # Reset `w` to original value
    w.assign(2.0)

    # Compute standard gradient manually
    with tf.GradientTape() as tape:
        loss = tf.square(w - 3)  # Simple quadratic loss
    grad = tape.gradient(loss, w)

    # Test standard gradient update (with explicit gradients)
    amp_optimizer.apply_gradients([(grad, w)])
    expected_w_grad = 2.0 - amp_optimizer.learning_rate * grad
    tf.debugging.assert_near(w, expected_w_grad, atol=1e-6)




def test_minimize(amp_optimizer):
    """
    Test if `minimize()` correctly updates variables towards optimal values.
    """
    # Define a simple quadratic loss function: f(w)=(w-3)^2
    def loss_function():
        return tf.square(w - 3)

    # Create a trainable variable
    w = tf.Variable(5.0, dtype=tf.float32)

    # Dummy values for `build()`
    y = tf.constant([0.0], dtype=tf.float32)  # Placeholder for y
    A = tf.constant([[1.0]], dtype=tf.float32)  # Identity-like transformation

    # Call `build()` before using optimizer
    amp_optimizer.build([w], y, A)

    # Apply `minimize()`
    amp_optimizer.minimize(loss_function, variables=[w])

    # Expected new value of `w` (after one gradient descent step)
    new_w = w.numpy()

    # Ensure `w` moves closer to 3 (optimal value)
    assert new_w < 5.0, (
        f"Minimization failed, expected w < 5.0 but got {new_w}"
    )





