
# Noise Robustness: Fuzzy Models vs Shallow Neural Nets (MNIST)

This repo compares how **fuzzy models** and a **shallow neural network (SNN)** degrade under **data noise** using **MNIST**.

## What’s inside
- `src/noise.py` — Gaussian noise, salt-and-pepper, random pixel dropout, optional denoising.
- `src/data.py` — MNIST loaders (Kaggle CSV format or TorchVision fallback), PCA utilities, dataset wrappers.
- `src/models/fuzzy.py` — Simple, interpretable fuzzy classifier built on class centroids in PCA space with Gaussian membership.
- `src/models/snn.py` — 1-hidden-layer MLP (PyTorch) + training/eval loops.
- `src/utils.py` — Metrics, plotting helpers, seeding.
- `experiment.py` — Reproducible end-to-end pipeline that trains on clean data and evaluates both models across noise levels/types.
- `requirements.txt` — Python deps.

## Data
You can use **Kaggle MNIST** (the dataset you picked):
- https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- Place `train.csv` and (optionally) `test.csv` under `data/kaggle/`.

If those files are not present, the code falls back to **TorchVision MNIST** (downloads automatically).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A: Kaggle CSVs
mkdir -p data/kaggle
# put train.csv (and optionally test.csv) there

# Run experiment (default uses PCA to 40 dims, 1 hidden layer of 64)
python experiment.py --epochs 5 --batch-size 256 --pca-dims 40 --noise-types gaussian saltpepper dropout --noise-levels 0.0 0.1 0.3 0.5
```

Outputs (CSV + PNG) will go to `outputs/`.

## Reproducibility
Set seed with `--seed` (defaults to 42). The fuzzy model is deterministic given the seed & PCA fits.

## Notes
- The fuzzy classifier is a simple **centroid-based rule model**: one rule per class in PCA space with Gaussian membership (per-dimension mean/std from clean data). Prediction uses max membership (winner-take-all) or weighted score. It is fast and interpretable (inspect centroids/stds).
- The SNN is deliberately shallow to reflect the assignment’s “shallow nets” constraint.

## License
MIT
