# **Approximate Message Passing (AMP) Algorithm**

## **Overview**
The **Approximate Message Passing (AMP)** algorithm is an iterative method used for **sparse signal recovery** and **high-dimensional inference problems**. It extends **iterative thresholding algorithms** with an additional correction term (Onsager correction) that improves convergence and performance. AMP is commonly applied in **compressed sensing**, **machine learning**, and **Bayesian inference**.

AMP solves the **linear inverse problem**:

\[
y = A x + w
\]

where:
- \( y \in \mathbb{R}^m \) is the observed measurement vector.
- \( A \in \mathbb{R}^{m \times n} \) is a measurement matrix (often random).
- \( x \in \mathbb{R}^n \) is the unknown **sparse signal** to be recovered.
- \( w \sim \mathcal{N}(0, \sigma^2 I) \) is **Gaussian noise**.

---

## **AMP Algorithm**

### **Algorithm: First-Order AMP Iteration**
1. **Initialize**:
   \[
   x^0 = 0, \quad z^0 = y
   \]
2. **Iterate for \( t = 0,1,2,\dots,T \)**:
   - **Signal Update (Denoising Step)**:
     \[
     x^{t+1} = \eta_t(A^T z^t + x^t;\lambda_t)
     \]
   - **Residual Update (Onsager Correction)**:
     \[
     z^t = y - A x^t + \frac{z^{t-1}}{\delta} \left\langle \eta'_t(A^T z^{t-1} + x^{t-1};\lambda_t) \right\rangle
     \]
     where:
     - \( \eta_t(\cdot) \) is a **thresholding function** (e.g., soft-thresholding).
     - \( \eta'_t(\cdot) \) is the **derivative** of \( \eta_t(\cdot) \).
     - \( \delta = \frac{m}{n} \) is the measurement ratio.
     - \( A^T \) is the **transpose** of \( A \).

3. **Repeat until convergence**.

---

## **Evaluation Metrics**
The performance of AMP can be evaluated using various metrics:

| **Metric**               | **Formula** | **Range** | **Desired Value** | **Use Case** |
|-------------------------|-----------|----------------------|-----------------|-----------------|
| **MSE (Mean Squared Error)** | \( MSE = \frac{1}{n} \sum (x_{\text{true}} - x_{\text{rec}})^2 \) | \( 0 \) to \( \infty \) | Close to **0** | Measures average reconstruction error |
| **NMSE (Normalized MSE)** | \( NMSE = \frac{\| x_{\text{true}} - x_{\text{rec}} \|^2}{\| x_{\text{true}} \|^2} \) | \( 0 \) to \( 1 \) | Close to **0** | Evaluates error on normalized data |
| **SNR (Signal-to-Noise Ratio)** | \( SNR = 10 \log_{10} \frac{\| x_{\text{true}} \|^2}{\| x_{\text{true}} - x_{\text{rec}} \|^2} \) | \( -\infty \) to \( +\infty \) dB | **> 20dB** | Measures quality of reconstruction relative to noise |
| **PSNR (Peak SNR)** | \( PSNR = 10 \log_{10} \frac{\max(x_{\text{true}})^2}{MSE} \) | \( 0 \) to \( +\infty \) dB | **> 30dB** | Evaluates reconstruction quality for images and bounded data |
| **Cosine Similarity** | \( \frac{x_{\text{true}} \cdot x_{\text{rec}}}{\|x_{\text{true}}\| \|x_{\text{rec}}\|} \) | \( -1 \) to \( +1 \) | Close to **1** | Measures angle similarity between true and recovered signal |
| **SSIM (Structural Similarity Index)** | Computed via `skimage.metrics.ssim` | \( 0 \) to \( 1 \) | Close to **1** | Evaluates structural similarity in images and continuous signals |
| **KL Divergence (Kullback-Leibler)** | \( D_{KL}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)} \) | \( 0 \) to \( \infty \) | Close to **0** | Compares probability distributions of signals |

---

## **References**
1. Donoho, D. L., Maleki, A., & Montanari, A. (2009). "Message passing algorithms for compressed sensing." *Proceedings of the National Academy of Sciences*, 106(45), 18914-18919.
2. Bayati, M., & Montanari, A. (2011). "The dynamics of message passing on dense graphs, with applications to compressed sensing." *IEEE Transactions on Information Theory*, 57(2), 764-785.

---

## **Conclusion**
- AMP is a **fast, efficient method** for sparse signal recovery.
- It achieves performance equivalent to \( \ell_1 \)-minimization, with **lower computational cost**.
- Onsager correction **improves stability and convergence**.
- Used in **compressed sensing, statistics, and signal processing**.

🚀 **Next Step:** Implementing AMP in `vanilla.py`.

