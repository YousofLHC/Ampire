1. `build`
**Code:**
```python
def build(self, variables: List[tf.Variable], y: tf.Tensor, A: tf.Tensor) -> None:
    """
    Initializes optimizer variables and computes the initial residual _z.
    - P_0 is initialized as an identity matrix, and Q_0 is initialized as a zero matrix.

    The initial residual is defined as:
        r^(0) = x^(0)+A^T(y - A * x^(0))
        z^(0) = y-A*x^(0)
    
    where x^(0) is assumed to be zero.
    
    Parameters:
    -----------
    variables : List[tf.Variable]
        The list of trainable variables (x).
    y : tf.Tensor
        The observed measurement vector (shape: [m, 1]).
    A : tf.Tensor
        The measurement matrix (shape: [m, n]).
    """
    super().build(variables, y, A)

    # Initialize matrices and update
    self.m, self.n = tf.shape(A)
    self.P_0       = tf.eye(self.n, dtype=tf.float32)  # Identity matrix of size n x n
    self.Q_0       = tf.zeros([self.n, self.n], dtype=tf.float32)  # Zero matrix of size n x n
    self.I         = tf.eye(self.n, dtype=tf.float32)
    self.R         = (self.delta**2) * tf.eye(self.m, dtype=tf.float32)

    # Set initial P_t and Q_t
    self.P_t       = self.P_0  # Set initial P_t
    self.Q_t       = self.Q_0  # Set initial Q_t
    self.P_t_prior = self.P_0
```
**Related to:**
- **Initial Setup:** This method is responsible for initializing important matrices, which are needed throughout the algorithm. Specifically:
    - $P_0$: The initial covariance matrix.
    - $Q_0$: The initial zero matrix.
    - $R$: The covariance of the error term.
    - $I$: Identity matrix used in further computations.
    - $P_t, Q_t, P_t^{-}$: These are initialized as $P_0$ and $Q_0$, and they will be updated during the iterations.

2. `_update_gain_matrix`
**code:**
```python
def _update_gain_matrix(self, P_t_prior: tf.Tensor) -> tf.Tensor:
    """
    Computes the gain matrix G_t used in the KalmanAMP algorithm.
    The formula is:
        G_t = P_t_prior A^T (A P_t_prior A^T + R)^{-1}
    """
    P_t_prior_A_T = tf.matmul(P_t_prior, self.A_T) #P_t^- *A^T
    temp          = tf.matmul(self.A, P_t_prior_A_T) + self.R # (A*P_t^-*A^T+R)
    temp          = tf.linalg.pinv(temp) # (A*P_t^-*A^T+R)^-1
    self.G_t      = tf.matmul(P_t_prior_A_T, temp)

    return self.G_t
```
**Related to:**
- **Computing the Gain Matrix:** This method computes the gain matrix $G_t$ that is used in the KalmanAMP algorithm to update the signal estimates.
    - The formula used is: $$G_t=P_t^{-}A^T(AP_t^{-}A^T+R)^{-1}$$
    - The method first computes $P_t^{-}A^T$, then uses that to compute the final gain matrix by applying the inverse of the error term covariance matrix $R$.

3. `_update_Q_matrix`
**code:**
```python
def _update_Q_matrix(self, G_t: tf.Tensor, v_t: tf.Tensor) -> tf.Tensor:
    """
    Computes the matrix Q_t used in the KalmanAMP algorithm.
    The formula is:
        Q_t = α Q_{t-1} + (1 - α)(G_t v_t^{-})(G_t v_t^{-})^T
    """
    G_t_v_t  = tf.matvec(G_t, v_t)
    temp     = tf.matmul(G_t_v_t, tf.transpose(G_t_v_t))
    self.Q_t = self.alpha*self.Q_t + (1-self.alpha)*temp
    return self.Q_t
```
**Related to:**
- Updating the Matrix $Q_t$: This method updates $Q_t$, a matrix that adjusts the noise covariance. The update is based on the formula: $$Q_t=\alpha Q_{t-1}+(1-\alpha)(G_tv_t^{-})(G_tv_t^{-})^T$$
- It uses the matrix $G_t$ and the vector $v_t$ (which is derived from the residual) to compute this update.

5. `_update_covariance_matrix`
**code:**
```python
def _update_covariance_matrix(self, G_t: tf.Tensor, P_t_prior: tf.Tensor) -> tf.Tensor:
    """
    Updates the covariance matrix P_t using the formula:
    
    P_t = (I - G_t A)P_t^{-}
    """
    temp = self.I - tf.matmul(G_t, self.A) # (I-G_t*A)
    self.P_t = tf.matmul(temp, P_t_prior)  # (I-G_t*A)P_t^-
    return self.P_t
```
**Related to:**
- **Updating Covariance Matrix** $P_t$: This method updates the covariance matrix $P_t$ using the formula:$$P_t=(I-G_tA)P_t^{-}$$
- This method adjusts the covariance based on the gain matrix and the sensing matrix $A$ to improve the estimate.

6. `compute_correction`
**code:**
```python
def compute_correction(self, G_t, A, P_t_prior, v_t):
    self.G_t = self._update_gain_matrix(P_t_prior=self.P_t_prior,
                                        A=self.A,
                                        R=self.R)
    self.P_t = self._update_covariance_matrix(G_t=self.G_t, P_t_prior=self.P_t_prior)
    self.Q_t = self._update_Q_matrix(G_t=self.G_t, v_t=v_t)

    raise NotImplementedError
```
**Related to:**
- **Computing Corrections:** This method is responsible for performing all necessary corrections at each step of the KalmanAMP algorithm:
    - It updates the **gain matrix** $(G_t)$, **covariance matrix** $(P_t)$, and **matrix** $Q_t$.
    - The method applies the respective formulas and ensures the algorithm proceeds with updated values.