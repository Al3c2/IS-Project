
import numpy as np
from dataclasses import dataclass

@dataclass
class FuzzyParams:
    centroids: np.ndarray  # (C, D)
    sigmas: np.ndarray     # (C, D)
    classes: np.ndarray    # (C,)

class CentroidFuzzyClassifier:
    """Simple interpretable fuzzy classifier.

    - Work in PCA space of dimension D.
    - One rule per class k: IF x ~ N(centroid_k, sigma_k) THEN class=k
    - Membership mu_k(x) = exp( -0.5 * sum_d ((x_d - m_kd)/s_kd)^2 )
    - Predict argmax_k mu_k(x).

    Train: compute class-wise centroids and per-dimension std from clean training data.
    """
    def __init__(self):
        self.params = None

    def fit(self, Z, y):
        classes = np.unique(y)
        C = len(classes)
        D = Z.shape[1]
        centroids = np.zeros((C, D), dtype=np.float32)
        sigmas = np.zeros((C, D), dtype=np.float32)
        for i, k in enumerate(classes):
            Zk = Z[y == k]
            centroids[i] = Zk.mean(axis=0)
            # robust std: avoid zeros (add small floor)
            sigmas[i] = np.clip(Zk.std(axis=0, ddof=1), 1e-3, None)
        self.params = FuzzyParams(centroids, sigmas, classes)
        return self

    def membership(self, Z):
        assert self.params is not None
        # compute Gaussian membership for each class
        # shape: (N, C)
        diffs = Z[:, None, :] - self.params.centroids[None, :, :]  # (N,C,D)
        S2 = (diffs / self.params.sigmas[None, :, :]) ** 2
        logits = -0.5 * S2.sum(axis=2)  # (N,C)
        # exponentiate in a stable way
        m = logits.max(axis=1, keepdims=True)
        mu = np.exp(logits - m)
        mu = mu / (mu.sum(axis=1, keepdims=True) + 1e-12)
        return mu

    def predict(self, Z):
        mu = self.membership(Z)
        idx = mu.argmax(axis=1)
        return self.params.classes[idx]

    def predict_proba(self, Z):
        return self.membership(Z)
