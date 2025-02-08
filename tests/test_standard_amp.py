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
