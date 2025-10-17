
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

def load_kaggle_csv(path):
    """Load Kaggle MNIST CSV: train.csv with label + 784 pixel columns."""
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError('CSV must contain a label column')
    y = df['label'].values.astype(np.int64)
    X = df.drop(columns=['label']).values.astype(np.float32) / 255.0
    return X, y

def load_torchvision_mnist(root, train=True):
    ds = datasets.MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())
    X = ds.data.numpy().astype(np.float32) / 255.0
    X = X.reshape(len(X), -1)
    y = ds.targets.numpy().astype(np.int64)
    return X, y

def load_mnist(root='data', prefer_kaggle=True):
    kaggle_train = os.path.join(root, 'kaggle', 'train.csv')
    kaggle_test  = os.path.join(root, 'kaggle', 'test.csv')
    if prefer_kaggle and os.path.exists(kaggle_train):
        X, y = load_kaggle_csv(kaggle_train)
        # Split a validation set from end
        n = len(X)
        n_train = int(n * 0.9)
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]
        # Optional test.csv may not have labels; we can ignore or use validation as test
        return (X_train, y_train), (X_val, y_val), None
    else:
        Xtr, ytr = load_torchvision_mnist(root=root, train=True)
        Xte, yte = load_torchvision_mnist(root=root, train=False)
        # create a val split from train
        n = len(Xtr)
        n_train = int(n * 0.9)
        X_train, y_train = Xtr[:n_train], ytr[:n_train]
        X_val, y_val = Xtr[n_train:], ytr[n_train:]
        return (X_train, y_train), (X_val, y_val), (Xte, yte)

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
