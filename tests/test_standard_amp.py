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










