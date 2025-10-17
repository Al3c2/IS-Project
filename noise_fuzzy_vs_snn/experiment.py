
import os
import argparse
import numpy as np
from tqdm import tqdm

from src.data import load_mnist, PCATransformer
from src.noise import apply_noise, median_denoise
from src.models.fuzzy import CentroidFuzzyClassifier
from src.models.snn import SNN, TrainConfig, make_loader, train as train_snn, evaluate as eval_snn
from src.utils import set_seed, ensure_dir, save_results_csv, plot_curve, compute_metrics

def run(args):
    set_seed(args.seed)
    out_dir = args.output
    ensure_dir(out_dir)

    # 1) Load data
    (Xtr, ytr), (Xval, yval), test_pack = load_mnist(root='data', prefer_kaggle=True)
    Xtst, ytst = (test_pack if test_pack is not None else (Xval, yval))

    # 2) Optional denoising flag only for evaluation phase
    noise_levels = args.noise_levels
    noise_types = args.noise_types

    # 3) Fit PCA on CLEAN TRAIN
    pca = PCATransformer(n_components=args.pca_dims, whiten=True).fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zval = pca.transform(Xval)
    Ztst = pca.transform(Xtst)

    # 4) Fuzzy model (fit on clean PCAs)
    fuzzy = CentroidFuzzyClassifier().fit(Ztr, ytr)

    # 5) SNN (fit on clean raw pixels or PCA depending on flag)
    if args.snn_space == 'raw':
        in_dim = Xtr.shape[1]
        train_X, val_X, test_X = Xtr, Xval, Xtst
    else:
        in_dim = Ztr.shape[1]
        train_X, val_X, test_X = Ztr, Zval, Ztst

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    snn = SNN(in_dim=in_dim, hidden=args.hidden, out_dim=10)
    train_loader = make_loader(train_X, ytr, cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_X, yval, cfg.batch_size, shuffle=False)
    hist = train_snn(snn, train_loader, val_loader, cfg)

    # Save SNN training curve
    snn_csv = os.path.join(out_dir, 'snn_train_history.csv')
    save_results_csv([{'epoch':e, 'train_loss':tl, 'train_acc':ta, 'val_acc':va} for (e,tl,ta,va) in hist], snn_csv)

    # 6) Evaluate both across noise grid
    rows = []
    for nt in noise_types:
        for nl in noise_levels:
            # Create noisy test set in RAW space, then (optionally) denoise, then transform for each model
            X_noisy = Xtst.copy()
            if nl > 0:
                X_noisy = apply_noise(X_noisy, nt, nl)
            if args.denoise:
                X_noisy = median_denoise(X_noisy, radius=args.denoise_radius)

            # Fuzzy uses PCA space
            Z_noisy = pca.transform(X_noisy)
            y_pred_fuzzy = fuzzy.predict(Z_noisy)

            # SNN uses chosen space
            X_for_snn = Z_noisy if args.snn_space == 'pca' else X_noisy
            test_loader = make_loader(X_for_snn, ytst, cfg.batch_size, shuffle=False)
            snn_acc = eval_snn(snn, test_loader, device=cfg.device)

            # Metrics for fuzzy
            from src.utils import compute_metrics
            f_acc, f_cm, f_report = compute_metrics(ytst, y_pred_fuzzy)

            rows.append({
                'noise_type': nt,
                'noise_level': nl,
                'fuzzy_acc': f_acc,
                'snn_acc': snn_acc
            })

    # Save grid results
    results_csv = os.path.join(out_dir, 'results_grid.csv')
    df = save_results_csv(rows, results_csv)

    # Plot curves per noise type
    for nt in noise_types:
        sub = df[df['noise_type'] == nt].sort_values('noise_level')
        plot_curve(
            xs=sub['noise_level'].tolist(),
            ys_dict={'Fuzzy': sub['fuzzy_acc'].tolist(), 'SNN': sub['snn_acc'].tolist()},
            title=f'Accuracy vs Noise Level ({nt})',
            xlabel='Noise Level',
            ylabel='Accuracy',
            out_png=os.path.join(out_dir, f'curve_{nt}.png')
        )

    print(f"Done. CSV: {results_csv} ; plots saved under {out_dir}/curve_*.png")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--pca-dims', type=int, default=40)
    ap.add_argument('--snn-space', type=str, choices=['raw','pca'], default='pca',
                    help='Train/eval SNN in raw pixel space or PCA space')
    ap.add_argument('--noise-types', nargs='+', default=['gaussian','saltpepper','dropout'])
    ap.add_argument('--noise-levels', nargs='+', type=float, default=[0.0, 0.1, 0.3, 0.5])
    ap.add_argument('--denoise', action='store_true')
    ap.add_argument('--denoise-radius', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='outputs')
    args = ap.parse_args()
    run(args)
