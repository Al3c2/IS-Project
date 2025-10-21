# save as scripts/make_confmats.py (or run after your training code)

import os, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---- project imports (same as your experiment) ----
from src.data import load_mnist, PCATransformer
from src.noise import apply_noise, level_from_severity
from src.preproc import apply_preproc, auto_params
from src.models.snn import SNN, TrainConfig, make_loader, train as train_snn
from src.models.fuzzy import FuzzyRBFClassifier
import torch

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 1) Train models on (optionally) augmented data ----------
# Keep this lightweight; reuse your preferred settings
(Xtr, ytr), (Xval, yval), test_pack = load_mnist(root='data')
Xtst, ytst = (test_pack if test_pack is not None else (Xval, yval))

# PCA space for both models (match your paper settings)
pca = PCATransformer(n_components=80, whiten=True).fit(np.vstack([Xtr, Xval]))
Ztr, Zval, Ztst = pca.transform(Xtr), pca.transform(Xval), pca.transform(Xtst)

# Fuzzy: quick tune (same as in your script)
best = (-1, None)
for n_rules in (5,8,10):
    for s_scale in (0.8, 1.0, 1.5, 2.0):
        fz = FuzzyRBFClassifier(n_rules_per_class=n_rules,
                                sigma_scale=s_scale, random_state=42,
                                logreg_C=1.0, logreg_max_iter=1000)
        fz.fit(Ztr, ytr)
        acc = (fz.predict(Zval) == yval).mean()
        if acc > best[0]:
            best = (acc, fz)
fuzzy = best[1]

# SNN on PCA space
cfg = TrainConfig(epochs=15, batch_size=256, lr=1e-3, weight_decay=0.0, device="cpu")
snn = SNN(in_dim=Ztr.shape[1], hidden=64, out_dim=10)
train_loader = make_loader(Ztr, ytr, cfg.batch_size, shuffle=True)
val_loader   = make_loader(Zval, yval, cfg.batch_size, shuffle=False)
_ = train_snn(snn, train_loader, val_loader, cfg)

@torch.no_grad()
def predict_snn_on_matrix(Xpca, y):
    """Return predictions (np.array) for SNN given PCA-space matrix."""
    snn.eval()
    # small loader to batch for speed/memory
    loader = make_loader(Xpca, y, batch_size=512, shuffle=False)
    preds = []
    for xb, _ in loader:
        logits = snn(xb)             # (B,10)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)

def save_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10), normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format=".2f")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------- 2) Helper to build a (noise, severity, preproc) test set ----------
def build_eval_set(noise_type=None, severity=0.0, use_auto=False):
    X = Xtst.copy()
    if noise_type and severity > 0.0:
        lvl = level_from_severity(noise_type, severity)
        X = apply_noise(X, noise_type, lvl)
    if use_auto and noise_type and severity > 0.0:
        mode, kw = auto_params(noise_type, severity)
        if mode != 'none':
            X = apply_preproc(X, mode=mode, **kw)
    return X

# ---------- 3) Make the four matrices you reference in LaTeX ----------
# CLEAN
X_clean = Xtst
Z_clean = pca.transform(X_clean)
y_clean_pred_snn   = predict_snn_on_matrix(Z_clean, ytst)
y_clean_pred_fuzzy = fuzzy.predict(Z_clean)
save_cm(ytst, y_clean_pred_snn,   os.path.join(OUTDIR, "cm_snn_clean.png"),   "SNN — clean")
save_cm(ytst, y_clean_pred_fuzzy, os.path.join(OUTDIR, "cm_fuzzy_clean.png"), "Fuzzy — clean")

# SALT-PEPPER @ s=0.5 with AUTO preprocessing (median)
sev = 0.5
X_sp_auto = build_eval_set("saltpepper", sev, use_auto=True)
Z_sp_auto = pca.transform(X_sp_auto)
y_sp_auto_pred_snn   = predict_snn_on_matrix(Z_sp_auto, ytst)
y_sp_auto_pred_fuzzy = fuzzy.predict(Z_sp_auto)
save_cm(ytst, y_sp_auto_pred_snn,
        os.path.join(OUTDIR, "cm_snn_saltpepper_s05_auto.png"),
        "SNN — salt–pepper (s=0.5, auto)")
save_cm(ytst, y_sp_auto_pred_fuzzy,
        os.path.join(OUTDIR, "cm_fuzzy_saltpepper_s05_auto.png"),
        "Fuzzy — salt–pepper (s=0.5, auto)")
