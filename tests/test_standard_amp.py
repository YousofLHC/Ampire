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