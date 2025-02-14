import pytest
import tensorflow as tf
from Ampire import KalmanAMP

@pytest.fixture
def setup_optimizer():
    optimizer = KalmanAMP(learning_rate=0.01, tau=0.1, alpha=0.5)
    return optimizer


def test_build_optimizer(setup_optimizer):
    optimizer = setup_optimizer
    
    # Create sample variables for testing
    variables = [tf.Variable([0.0, 0.0])]
    A = tf.random.normal([5, 2])  # Sample matrix A
    y = tf.random.normal([5, 1])  # Sample vector y
    
    # Run the build method
    optimizer.build(variables, y, A)
    
    # Check initial values of internal matrices (P_t, Q_t)
    assert optimizer._z is not None, "Failed: _z should not be None after build."
    assert optimizer.P_t is not None, "Failed: P_t should not be None after build."
    assert optimizer.Q_t is not None, "Failed: Q_t should not be None after build."
    assert optimizer.P_t.shape == (optimizer.n, optimizer.n), f"Failed: P_t should have shape ({optimizer.n}, {optimizer.n}), but got {optimizer.P_t.shape}."
    assert optimizer.Q_t.shape == (optimizer.n, optimizer.n), f"Failed: Q_t should have shape ({optimizer.n}, {optimizer.n}), but got {optimizer.Q_t.shape}."
    assert optimizer.P_t_prior.shape == (optimizer.n, optimizer.n), f"Failed: P_t_prior should have shape ({optimizer.n}, {optimizer.n}), but got {optimizer.P_t_prior.shape}."


def test_update_gain_matrix(setup_optimizer):
    optimizer = setup_optimizer
    
    # Input values for the test
    P_t_prior = tf.eye(3)  # 3x3 covariance matrix
    A = tf.random.normal([3, 3])  # 3x3 matrix A
    y = tf.random.normal([3, 1])  # 3x1 vector y

    # Call build method to initialize necessary variables
    optimizer.build([tf.Variable([0.0, 0.0, 0.0])], y, A)
    
    # Compute the gain matrix
    G_t = optimizer._update_gain_matrix(P_t_prior)
    
    # Check that the gain matrix is computed correctly
    assert G_t.shape == (3, 3), f"Failed: G_t should be a 3x3 matrix, but got {G_t.shape}."
    print("Passed: _update_gain_matrix works as expected.")


def test_apply_gradients_not_implemented(setup_optimizer):
    optimizer = setup_optimizer
    
    # Test that a NotImplementedError is raised if apply_gradients is not implemented
    with pytest.raises(NotImplementedError):
        optimizer.apply_gradients([], name="test")
    print("Passed: apply_gradients raises NotImplementedError as expected.")

def test_update_Q_matrix(setup_optimizer):
    optimizer = setup_optimizer
    
    # Input values for the test
    # m --> number of samples
    # n --> number of features
    G_t = tf.random.normal([3, 2])  # 3x2 gain matrix (n,m)
    v_t = tf.random.normal([2, 1])  # 2x1 auxiliary vector -> v_t=y-Ax-->(m,1)-(m,n)(n,1)--> v_t(m,1)
    A = tf.random.normal([3, 3])  # 3x3 matrix A (m,n)
    y = tf.random.normal([3, 1])  # 3x1 vector y (m,1)
    
    # Call build method to initialize necessary variables
    optimizer.build([tf.Variable([0.0, 0.0, 0.0])], y, A)
    
    # Update the Q_t matrix
    Q_t = optimizer._update_Q_matrix(G_t, v_t)
    
    # Check that the Q_t matrix is updated correctly
    assert Q_t.shape == (optimizer.n, optimizer.n), f"Failed: Q_t should have shape ({optimizer.n}, {optimizer.n}), but got {Q_t.shape}."
    print("Passed: _update_Q_matrix works as expected.")

def test_update_covariance_matrix(setup_optimizer):
    optimizer = setup_optimizer

    # Input values for the test
    # samples  -> m=2
    # features -> n=3
    G_t = tf.random.normal([3, 2])  # 3x2 gain matrix (n, m)
    P_t_prior = tf.eye(3)  # 3x3 identity matrix (n, n)
    A = tf.random.normal([2, 3])  # 3x3 matrix A (m, n)

    # Call build method to initialize necessary variables
    optimizer.build([tf.Variable([0.0, 0.0, 0.0])], tf.random.normal([2, 1]), A)
    
    # Update the covariance matrix
    P_t = optimizer._update_covariance_matrix(G_t, P_t_prior)
    
    # Check that the updated covariance matrix is of the correct shape
    assert P_t.shape == (optimizer.n, optimizer.n), f"Failed: P_t should have shape ({optimizer.n}, {optimizer.n}), but got {P_t.shape}."
    print("Passed: _update_covariance_matrix works as expected.")



def test_compute_correction_not_implemented(setup_optimizer):
    optimizer = setup_optimizer
    
    # Sample input values for the test
    G_t = tf.random.normal([3, 2])  # 3x2 gain matrix
    A = tf.random.normal([3, 3])    # 3x3 matrix A
    P_t_prior = tf.eye(3)           # 3x3 identity matrix (prior covariance)
    v_t = tf.random.normal([3, 1])  # 3x1 auxiliary vector

    # Call build method to initialize necessary variables
    optimizer.build([tf.Variable([0.0, 0.0, 0.0])], v_t, A)

    # Test that a NotImplementedError is raised when calling compute_correction
    with pytest.raises(NotImplementedError):
        optimizer.compute_correction(G_t, A, P_t_prior, v_t)
    print("Passed: compute_correction raises NotImplementedError as expected.")