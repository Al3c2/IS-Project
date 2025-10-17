# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:49:19 2025

@author: Alexandrehb
"""

import numpy as np
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_wavelet
from skimage import img_as_float
from scipy.signal import wiener as sp_wiener

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
        out[i] = gaussian(X[i], sigma=sigma, preserve_range=True)
    return out.reshape(len(out), -1)

def preproc_bilateral(x, sigma_color=0.1, sigma_spatial=3, **kw):
    # skimage denoise_bilateral deprecated API differs across versions; NL-means is usually better anyway
    return preproc_nlmeans(x, patch_size=3, patch_distance=5, h=0.8)

def preproc_nlmeans(x, patch_size=3, patch_distance=5, h=0.8, fast_mode=True, **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = denoise_nl_means(img_as_float(X[i]),
                                  patch_size=patch_size,
                                  patch_distance=patch_distance,
                                  h=h, fast_mode=fast_mode,
                                  channel_axis=None)
    return out.reshape(len(out), -1)

def preproc_tv(x, weight=0.08, **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = denoise_tv_chambolle(X[i], weight=weight, channel_axis=None)
    return out.reshape(len(out), -1)

def preproc_wavelet(x, sigma=None, mode='soft', **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = denoise_wavelet(X[i], sigma=sigma, mode=mode, rescale_sigma=True, channel_axis=None)
    return out.reshape(len(out), -1)

def preproc_wiener(x, mysize=3, noise=None, **kw):
    X = _ensure_hw(x.copy())
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = sp_wiener(X[i], mysize=mysize, noise=noise)
    out = np.clip(out, 0.0, 1.0)
    return out.reshape(len(out), -1)

AVAILABLE_PREPROCS = {
    'none': preproc_none,
    'median': preproc_median,
    'gaussian_blur': preproc_gaussian,
    'nlmeans': preproc_nlmeans,
    'tv': preproc_tv,
    'wavelet': preproc_wavelet,
    'wiener': preproc_wiener,
}

def apply_preproc(x, mode='none', **kwargs):
    if mode not in AVAILABLE_PREPROCS:
        raise ValueError(f'Unknown preproc mode: {mode}')
    return AVAILABLE_PREPROCS[mode](x, **kwargs)

# Simple heuristic mapping (used if --auto-preproc)
AUTO_MAP = {
    'saltpepper': ('median', {'radius': 1}),
    'gaussian': ('gaussian_blur', {'sigma': 0.8}),
    'uniform': ('gaussian_blur', {'sigma': 0.8}),
    'speckle': ('tv', {'weight': 0.08}),
    'poisson': ('wavelet', {}),  # wavelet handles Poisson-ish
    'dropout': ('nlmeans', {'h': 0.8, 'patch_distance': 5}),
    'jpeg': ('nlmeans', {'h': 0.8}),
    'motionblur': ('wiener', {'mysize': 3}),
    'stripe': ('gaussian_blur', {'sigma': 0.6}),
    'periodic': ('gaussian_blur', {'sigma': 0.6}),  # simple; FFT-notch would be fancier
    'banding': ('tv', {'weight': 0.06}),
    'checkerboard': ('gaussian_blur', {'sigma': 0.6}),
    'anisotropic': ('gaussian_blur', {'sigma': 0.8}),
    'shot': ('wavelet', {}),
}
