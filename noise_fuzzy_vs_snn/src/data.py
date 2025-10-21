
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms


def load_torchvision_mnist(root, train=True):
    ds = datasets.MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())
    X = ds.data.numpy().astype(np.float32) / 255.0
    X = X.reshape(len(X), -1)
    y = ds.targets.numpy().astype(np.int64)
    return X, y

def load_mnist(root='data'):
    """Load MNIST from TorchVision and split into train/val/test."""
    from torchvision import datasets, transforms
    ds_train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    ds_test  = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

    Xtr = ds_train.data.numpy().astype(np.float32) / 255.0
    ytr = ds_train.targets.numpy().astype(np.int64)
    Xtst = ds_test.data.numpy().astype(np.float32) / 255.0
    ytst = ds_test.targets.numpy().astype(np.int64)

    # create validation split (10%)
    n_train = int(0.9 * len(Xtr))
    X_train, y_train = Xtr[:n_train], ytr[:n_train]
    X_val, y_val = Xtr[n_train:], ytr[n_train:]

    # flatten 28x28 → 784
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)
    Xtst = Xtst.reshape(len(Xtst), -1)

    return (X_train, y_train), (X_val, y_val), (Xtst, ytst)


class PCATransformer:
    """Fit PCA on clean training data, return transform utilities."""
    def __init__(self, n_components=40, whiten=True):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=n_components, whiten=whiten, random_state=42)

    def fit(self, X):
        Z = self.scaler.fit_transform(X)
        self.pca.fit(Z)
        return self

    def transform(self, X):
        Z = self.scaler.transform(X)
        return self.pca.transform(Z)

    def fit_transform(self, X):
        return self.pca.fit_transform(self.scaler.fit_transform(X))
