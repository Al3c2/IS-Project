
import numpy as np
from skimage.util import random_noise
from skimage.filters import median, gaussian
from skimage.morphology import disk
from scipy.signal import convolve2d
from PIL import Image
import io

# ---------- helpers ----------
def _ensure_hw(X):
    if X.ndim == 2 and X.shape[1] == 784:
        return X.reshape(-1, 28, 28)
    return X

# ---------- noises ----------
def add_gaussian(x, std):
    X = _ensure_hw(x.copy())
    noisy = np.clip(X + np.random.normal(0, std, X.shape), 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_saltpepper(x, amount):
    X = _ensure_hw(x.copy())
    noisy = np.empty_like(X)
    for i in range(len(X)):
        noisy[i] = random_noise(X[i], mode='s&p', amount=amount, clip=True)
    return noisy.reshape(len(noisy), -1)

def add_dropout(x, drop_prob):
    X = _ensure_hw(x.copy())
    mask = (np.random.rand(*X.shape) > drop_prob).astype(np.float32)
    noisy = X * mask
    return noisy.reshape(len(noisy), -1)

def add_speckle(x, std):
    X = _ensure_hw(x.copy())
    noisy = np.clip(X + X * np.random.normal(0, std, X.shape), 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_poisson(x):
    X = _ensure_hw(x.copy())
    vals = len(np.unique(X))
    vals = max(2, int(2 ** np.ceil(np.log2(vals))))
    noisy = np.random.poisson(X * vals) / float(vals)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_uniform(x, level):
    X = _ensure_hw(x.copy())
    noisy = np.clip(X + np.random.uniform(-level, level, X.shape), 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_motionblur(x, kernel_size=5):
    k = max(3, int(kernel_size))
    if k % 2 == 0: k += 1
    X = _ensure_hw(x.copy())
    noisy = np.empty_like(X)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel = gaussian(kernel, sigma=k / 6)
    kernel /= kernel.sum() + 1e-12
    for i in range(len(X)):
        noisy[i] = convolve2d(X[i], kernel, mode='same', boundary='wrap')
    return np.clip(noisy.reshape(len(noisy), -1), 0.0, 1.0)

def add_jpeg(x, quality=25):
    X = _ensure_hw(x.copy())
    noisy = np.empty_like(X)
    for i in range(len(X)):
        img = (X[i] * 255).astype(np.uint8)
        im_pil = Image.fromarray(img, mode='L')
        buf = io.BytesIO()
        im_pil.save(buf, format='JPEG', quality=int(quality))
        buf.seek(0)
        noisy[i] = np.array(Image.open(buf).convert('L')) / 255.0
    return noisy.reshape(len(noisy), -1)

def add_shot(x, intensity=1.0):
    X = _ensure_hw(x.copy()).astype(np.float32)
    noisy = np.empty_like(X)
    for i in range(len(X)):
        scale = 255.0 / max(float(intensity), 1e-6)
        photons = np.random.poisson(X[i] * scale)
        noisy[i] = np.clip(photons / scale, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_quantization(x, bits=4):
    bits = max(1, int(bits))
    X = _ensure_hw(x.copy()).astype(np.float32)
    levels = 2 ** bits
    denom = max(levels - 1, 1)
    noisy = np.round(X * denom) / denom
    noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)
    return noisy.reshape(len(noisy), -1)

def add_anisotropic(x, std_x=0.2, std_y=0.05):
    X = _ensure_hw(x.copy()).astype(np.float32)
    noisy = np.empty_like(X)
    for i in range(len(X)):
        noise_x = np.random.normal(0, std_x, X[i].shape)
        noise_y = np.random.normal(0, std_y, X[i].T.shape).T
        noise = (noise_x + noise_y) / 2.0
        noisy[i] = np.clip(X[i] + noise, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_stripe(x, intensity=0.2, direction='vertical', frequency=8):
    X = _ensure_hw(x.copy()).astype(np.float32)
    noisy = np.empty_like(X)
    H, W = X.shape[1], X.shape[2]
    for i in range(len(X)):
        if direction == 'vertical':
            pattern = np.sin(np.linspace(0, 2*np.pi*frequency, W))
            pattern = np.tile(pattern, (H, 1))
        else:
            pattern = np.sin(np.linspace(0, 2*np.pi*frequency, H))[:, None]
            pattern = np.tile(pattern, (1, W))
        noise = intensity * pattern
        noisy[i] = np.clip(X[i] + noise, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_periodic(x, amplitude=0.1, frequency=5.0, phase=None):
    X = _ensure_hw(x.copy()).astype(np.float32)
    noisy = np.empty_like(X)
    H, W = X.shape[1], X.shape[2]
    for i in range(len(X)):
        phase_i = np.random.uniform(0, 2*np.pi) if phase is None else phase
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        interference = amplitude * np.sin(2*np.pi*frequency*yy/W + phase_i)
        noisy[i] = np.clip(X[i] + interference, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_banding(x, intensity=0.15, num_bands=6):
    X = _ensure_hw(x.copy()).astype(np.float32)
    noisy = np.empty_like(X)
    H, W = X.shape[1], X.shape[2]
    for i in range(len(X)):
        band_pattern = np.interp(
            np.linspace(0, num_bands, H),
            np.arange(num_bands),
            np.random.uniform(-intensity, intensity, num_bands)
        )
        band_pattern = np.tile(band_pattern[:, None], (1, W))
        noisy[i] = np.clip(X[i] + band_pattern, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

def add_checkerboard(x, strength=0.15, size=4):
    X = _ensure_hw(x.copy()).astype(np.float32)
    noisy = np.empty_like(X)
    H, W = X.shape[1], X.shape[2]
    pattern = np.indices((H, W)).sum(axis=0) // size
    pattern = ((pattern % 2) * 2 - 1) * strength
    for i in range(len(X)):
        noisy[i] = np.clip(X[i] + pattern, 0.0, 1.0)
    return noisy.reshape(len(noisy), -1)

# ---------- denoise (kept for backward compat) ----------
def median_denoise(x, radius=1):
    X = _ensure_hw(x.copy())
    den = np.empty_like(X)
    selem = disk(radius)
    for i in range(len(X)):
        den[i] = median(X[i], footprint=selem)
    return den.reshape(len(den), -1)

# ---------- registry & dispatcher ----------
NOISE_REGISTRY = {
    'gaussian': add_gaussian,
    'saltpepper': add_saltpepper,
    'dropout': add_dropout,
    'speckle': add_speckle,
    'poisson': add_poisson,
    'uniform': add_uniform,
    'motionblur': add_motionblur,
    'jpeg': add_jpeg,
    'shot': add_shot,
    'quantization': add_quantization,
    'anisotropic': add_anisotropic,
    'stripe': add_stripe,
    'periodic': add_periodic,
    'banding': add_banding,
    'checkerboard': add_checkerboard,
}

def apply_noise(x, noise_type, level):
    if noise_type not in NOISE_REGISTRY:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    fn = NOISE_REGISTRY[noise_type]
    if noise_type in ('gaussian','speckle'):
        return fn(x, std=level)
    elif noise_type == 'saltpepper':
        return fn(x, amount=level)
    elif noise_type == 'dropout':
        return fn(x, drop_prob=level)
    elif noise_type == 'uniform':
        return fn(x, level=level)
    elif noise_type == 'motionblur':
        return fn(x, kernel_size=int(level))
    elif noise_type == 'jpeg':
        return fn(x, quality=int(level))
    elif noise_type == 'poisson':
        return fn(x)
    elif noise_type == 'shot':
        return fn(x, intensity=level)
    elif noise_type == 'quantization':
        return fn(x, bits=int(level))
    elif noise_type == 'anisotropic':
        return fn(x, std_x=level, std_y=level/4.0)
    elif noise_type == 'stripe':
        return fn(x, intensity=level)  # simple intensity map
    elif noise_type == 'periodic':
        return fn(x, amplitude=level)
    elif noise_type == 'banding':
        return fn(x, intensity=level)
    elif noise_type == 'checkerboard':
        return fn(x, strength=level)
    else:
        raise ValueError(f"Unhandled noise_type: {noise_type}")

# ---------- unified severity mapper ----------
def level_from_severity(noise_type, severity):
    s = float(np.clip(severity, 0.0, 1.0))
    if noise_type in ('gaussian','speckle','saltpepper','dropout','uniform','stripe','periodic','banding','checkerboard'):
        # map 0..1 -> 0.05..0.5 intensity
        return 0.05 + s * (0.5 - 0.05)
    elif noise_type == 'shot':
        return 0.5 + s * (5.0 - 0.5)
    elif noise_type == 'motionblur':
        k = int(round(3 + s * 6))
        if k % 2 == 0: k += 1
        return k
    elif noise_type == 'jpeg':
        return int(round(95 - s * 75))  # 95 (best) -> 20 (worst)
    elif noise_type == 'quantization':
        return int(round(8 - s * 6))    # 8 -> 2 bits
    elif noise_type == 'anisotropic':
        return 0.1 + s * (0.5 - 0.1)    # std_x, std_y will be std_x/4
    elif noise_type == 'poisson':
        return 0.0  # ignored
    else:
        raise ValueError(f'No severity mapping for {noise_type}')
