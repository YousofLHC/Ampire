import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

# Evaluation Metrics
def compute_mse(x_true, x_rec):
    return np.mean((x_true - x_rec) ** 2)

def compute_nmse(x_true, x_rec):
    return np.linalg.norm(x_true - x_rec) ** 2 / np.linalg.norm(x_true) ** 2

def compute_snr(x_true, x_rec):
    signal_power = np.mean(x_true ** 2)
    noise_power = np.mean((x_true - x_rec) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_psnr(x_true, x_rec):
    mse = compute_mse(x_true, x_rec)
    max_val = np.max(x_true)
    return 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')

def compute_cosine_similarity(x_true, x_rec):
    return np.dot(x_true, x_rec) / (np.linalg.norm(x_true) * np.linalg.norm(x_rec))

def compute_ssim(x_true, x_rec):
    return ssim(x_true, x_rec, data_range=x_true.max() - x_true.min())

def compute_kl_divergence(x_true, x_rec):
    p = np.abs(x_true) / np.sum(np.abs(x_true))
    q = np.abs(x_rec) / np.sum(np.abs(x_rec))
    return entropy(p, q)

def denoising_function(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def grad_denoising_function(x, threshold):
    return (np.abs(x) > threshold).astype(float)

def amp(y, A, lam=0.1, max_iter=50, tol=1e-6):
    m, n = A.shape
    x_est = np.zeros(n)
    z = y.copy()
    delta = m / n  
    
    for t in range(max_iter):
        r = A.T @ z + x_est
        r /= np.linalg.norm(r) + 1e-8  
        x_new = denoising_function(r, lam)
        x_new /= np.linalg.norm(x_new) + 1e-8  
        eta_derivative = grad_denoising_function(r, lam)
        onsager = (z.mean() / delta) * np.mean(eta_derivative)
        z_new = y - A @ x_new + onsager
        z_new /= np.linalg.norm(z_new) + 1e-8  
        
        if np.linalg.norm(x_new - x_est) < tol:
            break
        
        x_est, z = x_new, z_new
    
    return x_est

if __name__ == "__main__":
    np.random.seed(40)
    m, n = 100, 200
    A = np.random.randn(m, n) / np.sqrt(m)
    x_true = np.zeros(n)
    x_true[np.random.choice(n, 10, replace=False)] = np.random.randn(10)
    y = A @ x_true + 0.05 * np.random.randn(m)
    
    x_rec = amp(y, A, lam=0.001, max_iter=50_000)
    
    mse = compute_mse(x_true, x_rec)
    nmse = compute_nmse(x_true, x_rec)
    snr = compute_snr(x_true, x_rec)
    psnr = compute_psnr(x_true, x_rec)
    cos_sim = compute_cosine_similarity(x_true, x_rec)
    ssim_value = compute_ssim(x_true, x_rec)
    kl_div = compute_kl_divergence(x_true, x_rec)
    
    print(f"MSE: {mse:.6f}")
    print(f"NMSE: {nmse:.6f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"SSIM: {ssim_value:.6f}")
    print(f"KL Divergence: {kl_div:.6f}")
    
    plt.hist(x_true - x_rec, bins=30, alpha=0.7, color='b')
    plt.xlabel("Error Value (x_true - x_rec)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution of AMP Recovery")
    plt.grid()
    plt.show()
