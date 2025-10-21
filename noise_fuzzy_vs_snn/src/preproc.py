# -*- coding: utf-8 -*-
import numpy as np
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_wavelet
from skimage import img_as_float
from scipy.ndimage import uniform_filter  # stable local means for Wiener

# ----------------- helpers -----------------
def _ensure_hw(X):
    if X.ndim == 2 and X.shape[1] == 784:
        return X.reshape(-1, 28, 28)
    return X

def preproc_none(x, **kw):
    return x.copy()

def preproc_median(x, radius=1, **kw):
    X = _ensure_hw(x.copy())
    se = disk(int(radius))
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = median(X[i], footprint=se)
    return out.reshape(len(out), -1)

def preproc_gaussian(x, sigma=0.8, **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = gaussian(X[i], sigma=float(sigma), preserve_range=True)
    return out.reshape(len(out), -1)

def preproc_nlmeans(x, patch_size=3, patch_distance=5, h=0.8, fast_mode=True, **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = denoise_nl_means(
            img_as_float(X[i]),
            patch_size=int(patch_size),
            patch_distance=int(patch_distance),
            h=float(h),
            fast_mode=bool(fast_mode),
            channel_axis=None
        )
    return out.reshape(len(out), -1)

def preproc_tv(x, weight=0.08, **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = denoise_tv_chambolle(X[i], weight=float(weight), channel_axis=None)
    return out.reshape(len(out), -1)

# IMPORTANT: use 'wmode' to avoid colliding with dispatcher argument 'mode'
def preproc_wavelet(x, sigma=None, wmode='soft', **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = denoise_wavelet(
            X[i], sigma=sigma, mode=wmode, rescale_sigma=True, channel_axis=None
        )
    return out.reshape(len(out), -1)

def wiener_safe_batch(x, size=3, noise=None, eps=1e-8):
    """
    Numerically stable Wiener filter on [0,1] images.
    x: (N, 784) or (N, 28, 28)
    size: odd window size (int)
    noise: None -> estimate as median local variance; else scalar variance
    """
    X = x.copy().astype(np.float32)
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)

    out = np.empty_like(X, dtype=np.float32)
    k = int(size)
    for i, im in enumerate(X):
        mu  = uniform_filter(im, size=k, mode='reflect')
        mu2 = uniform_filter(im * im, size=k, mode='reflect')
        lvar = np.maximum(mu2 - mu * mu, 0.0)

        noise_var = np.median(lvar) if noise is None else float(noise)
        noise_var = max(noise_var, eps)

        gain = np.maximum(1.0 - noise_var / (lvar + eps), 0.0)
        out[i] = mu + gain * (im - mu)

    out = np.clip(out, 0.0, 1.0)
    return out.reshape(len(out), -1)

def preproc_wiener(x, size=3, noise=None, **kw):
    # back-compat: if 'mysize' was passed by old code, map to 'size'
    if 'mysize' in kw and 'size' not in kw:
        size = kw.pop('mysize')
    return wiener_safe_batch(x, size=size, noise=noise)

# ----------------- dispatcher -----------------
AVAILABLE_PREPROCS = {
    'none': preproc_none,
    'median': preproc_median,
    'gaussian_blur': preproc_gaussian,
    'nlmeans': preproc_nlmeans,
    'tv': preproc_tv,
    'wavelet': preproc_wavelet,   # uses wmode=...
    'wiener': preproc_wiener,     # uses size=...
}

def apply_preproc(x, mode='none', **kwargs):
    mode = mode.lower()
    if mode not in AVAILABLE_PREPROCS:
        raise ValueError(f'Unknown preproc mode: {mode}')
    out = AVAILABLE_PREPROCS[mode](x, **kwargs)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

# ----------------- severity-aware auto params -----------------
def auto_params(noise_type, sev):
    """
    Return (mode, kwargs). Wavelet uses key 'wmode' (NOT 'mode').
    'sev' in [0,1].
    """
    sev = float(sev)
    if noise_type == 'gaussian':
        return ('gaussian_blur', {'sigma': 0.3 + 1.2*sev})                 # 0.3→1.5
    if noise_type == 'saltpepper':
        r = 1 if sev < 0.6 else 2
        return ('median', {'radius': r})
    if noise_type == 'dropout':
        return ('nlmeans', {'patch_size':3, 'patch_distance':5, 'h':0.6 + 0.8*sev, 'fast_mode':True})
    if noise_type == 'speckle':
        return ('tv', {'weight': 0.05 + 0.25*sev})                          # 0.05→0.30
    if noise_type in ('poisson','shot','uniform','anisotropic'):
        return ('wavelet', {'wmode':'soft', 'sigma':None})
    if noise_type == 'motionblur':
        k = 3 if sev < 0.5 else (5 if sev < 0.8 else 7)
        return ('wiener', {'size': k})
    if noise_type == 'jpeg':
        return ('nlmeans', {'patch_size':3, 'patch_distance':5, 'h':0.6 + 0.6*sev, 'fast_mode':True})
    if noise_type == 'quantization':
        return ('gaussian_blur', {'sigma': 0.2 + 0.6*sev})
    return ('none', {})
