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