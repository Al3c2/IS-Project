
import numpy as np
from skimage.util import random_noise
from skimage.filters import median
from skimage.morphology import disk

def add_gaussian(x, std):
    """Additive Gaussian noise in [0,1] space.
    x: np.ndarray of shape (N, H, W) or (N, 784) flattened
    std: float in [0,1]
    """
    X = x.copy()
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)
    noisy = np.clip(X + np.random.normal(0, std, X.shape), 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_saltpepper(x, amount):
    """Salt & pepper noise using skimage random_noise.
    amount: fraction of pixels to corrupt (e.g., 0.05, 0.2).
    """
    X = x.copy()
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)
    noisy = np.empty_like(X)
    for i in range(len(X)):
        noisy[i] = random_noise(X[i], mode='s&p', amount=amount, clip=True)
    return noisy.reshape(len(noisy), -1)

def add_dropout(x, drop_prob):
    """Random pixel dropout to zero (missing data)."""
    X = x.copy()
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)
    mask = (np.random.rand(*X.shape) > drop_prob).astype(np.float32)
    noisy = X * mask
    return noisy.reshape(len(noisy), -1)

def median_denoise(x, radius=1):
    """Median filter (optional preprocessing).
    radius 1..2 keeps digits while removing specks.
    """
    X = x.copy()
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)
    den = np.empty_like(X)
    selem = disk(radius)
    for i in range(len(X)):
        den[i] = median(X[i], footprint=selem)
    return den.reshape(len(den), -1)

NOISE_REGISTRY = {
    'gaussian': add_gaussian,
    'saltpepper': add_saltpepper,
    'dropout': add_dropout,
}

def apply_noise(x, noise_type, level):
    if noise_type not in NOISE_REGISTRY:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    fn = NOISE_REGISTRY[noise_type]
    if noise_type == 'gaussian':
        return fn(x, std=level)
    elif noise_type == 'saltpepper':
        return fn(x, amount=level)
    elif noise_type == 'dropout':
        return fn(x, drop_prob=level)
