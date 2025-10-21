# -*- coding: utf-8 -*-
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd

from src.preproc import apply_preproc, auto_params
from src.data import load_mnist, PCATransformer
from src.noise import apply_noise, level_from_severity
from src.models.snn import SNN, TrainConfig, make_loader, train as train_snn, evaluate as eval_snn
from src.utils import set_seed, ensure_dir, save_results_csv, compute_metrics

GLOBAL_SEVERITY_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]

# ---------------- plotting ----------------
def plot_curve_allseries(xs, ys_dict, title, xlabel, ylabel, out_png):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    preferred = [
        'gaussian','saltpepper','dropout','speckle','uniform',
        'poisson','shot','anisotropic','motionblur','jpeg','quantization',
        'stripe','periodic','banding','checkerboard'
    ]
    series_keys = [k for k in preferred if k in ys_dict] + [k for k in ys_dict if k not in preferred]
    for k in series_keys:
        ax.plot(xs, ys_dict[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------- augmentation -------------
def make_augmented_train(Xtr, ytr, ratio=0.5, do_auto=False, gate=0.5):
    n_extra = int(len(Xtr) * ratio)
    if n_extra <= 0:
        return Xtr, ytr

    sev_grid = GLOBAL_SEVERITY_GRID
    noises = ['gaussian','saltpepper','dropout','speckle','poisson','uniform','shot',
              'anisotropic','motionblur','jpeg','quantization']

    X_extra, y_extra = [], []
    rng = np.random.default_rng(42)
    for _ in range(n_extra):
        idx = rng.integers(0, len(Xtr))
        x0 = Xtr[idx:idx+1].copy()
        y0 = ytr[idx:idx+1].copy()

        nt  = noises[rng.integers(0, len(noises))]
        sev = float(sev_grid[rng.integers(0, len(sev_grid))])
        lvl = level_from_severity(nt, sev)

        x_aug = x0
        if sev > 0:
            x_aug = apply_noise(x_aug, nt, lvl)

        if do_auto and sev >= float(gate):
            mode, kw = auto_params(nt, float(sev))         # kw never contains key 'mode'
            if mode != 'none':
                x_aug = apply_preproc(x_aug, mode=mode, **kw)

        X_extra.append(x_aug); y_extra.append(y0)

    X_extra = np.vstack(X_extra)
    y_extra = np.concatenate(y_extra)
    X_aug   = np.concatenate([Xtr, X_extra], axis=0)
    y_aug   = np.concatenate([ytr, y_extra], axis=0)
    return X_aug, y_aug

# ---------------- core run -----------------
def run(args):
    set_seed(args.seed)
    out_dir = args.output
    ensure_dir(out_dir)

    (Xtr, ytr), (Xval, yval), test_pack = load_mnist(root='data')
    Xtst, ytst = (test_pack if test_pack is not None else (Xval, yval))

    if args.eval_limit and args.eval_limit > 0:
        N = min(args.eval_limit, len(Xtst))
        Xtst = Xtst[:N]; ytst = ytst[:N]
        print(f"[Eval] Using eval-limit={N} samples (set --eval-limit 0 for full test).")

    # train-time distribution
    if args.train_augment:
        Xtr_aug, ytr_aug = make_augmented_train(
            Xtr, ytr,
            ratio=args.train_augment_ratio,
            do_auto=args.train_auto_preproc,
            gate=args.train_preproc_severity_gate
        )
    else:
        Xtr_aug, ytr_aug = Xtr, ytr

    # PCA fit source
    if args.pca_fit_mode == 'clean':
        X_for_pca = Xtr
    elif args.pca_fit_mode == 'augmented':
        X_for_pca = Xtr_aug
    elif args.pca_fit_mode == 'auto':
        X_for_pca = apply_preproc(Xtr, mode='gaussian_blur', sigma=0.5)
    else:
        X_for_pca = Xtr

    pca = PCATransformer(n_components=args.pca_dims, whiten=True).fit(X_for_pca)

    # input space
    if args.snn_space == 'raw':
        train_X, val_X = Xtr_aug, Xval
        Ztr_for_fuzzy, Zval_for_fuzzy = Xtr_aug, Xval
        in_dim = train_X.shape[1]
    else:
        train_X = pca.transform(Xtr_aug)
        val_X   = pca.transform(Xval)
        Ztr_for_fuzzy = train_X
        Zval_for_fuzzy = val_X
        in_dim = train_X.shape[1]

    # Fuzzy tune
    from src.models.fuzzy import FuzzyRBFClassifier
    candidates = []
    for n_rules in (5, 8, 10):
        for s_scale in (0.8, 1.0, 1.5, 2.0):
            model = FuzzyRBFClassifier(n_rules_per_class=n_rules,
                                       sigma_scale=s_scale,
                                       random_state=42,
                                       logreg_C=1.0, logreg_max_iter=1000)
            model.fit(Ztr_for_fuzzy, ytr_aug)
            acc = (model.predict(Zval_for_fuzzy) == yval).mean()
            candidates.append((acc, n_rules, s_scale, model))
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_acc, best_n, best_s, fuzzy = candidates[0]
    print(f"[Fuzzy] tuned on matching val: acc={best_acc:.4f}  n_rules={best_n}  sigma_scale={best_s}")

    # SNN train
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, device=args.device)
    snn = SNN(in_dim=in_dim, hidden=args.hidden, out_dim=10)
    train_loader = make_loader(train_X, ytr_aug, cfg.batch_size, shuffle=True)
    val_loader   = make_loader(val_X,   yval,     cfg.batch_size, shuffle=False)
    hist = train_snn(snn, train_loader, val_loader, cfg)
    save_results_csv(
        [{'epoch':e, 'train_loss':tl, 'train_acc':ta, 'val_acc':va} for (e,tl,ta,va) in hist],
        os.path.join(out_dir, 'snn_train_history.csv')
    )

    # ----- eval helpers -----
    VAL_GATE_N = 500
    def choose_auto_or_none(X_noisy, nt, sev):
        n = min(VAL_GATE_N, len(X_noisy))
        Xs = X_noisy[:n]; ys = ytst[:n]

        # none
        Z_none = pca.transform(Xs)
        f_acc_none = (fuzzy.predict(Z_none) == ys).mean()
        Xs_for_snn = Z_none if args.snn_space == 'pca' else Xs
        snn_acc_none = eval_snn(snn, make_loader(Xs_for_snn, ys, cfg.batch_size, False), device=cfg.device)

        # auto
        mode, kw = auto_params(nt, float(sev))
        if mode == 'none':
            return 'none', None, None
        Xe = apply_preproc(X_noisy, mode=mode, **kw)
        Xes = Xe[:n]
        Z_auto = pca.transform(Xes)
        f_acc_auto = (fuzzy.predict(Z_auto) == ys).mean()
        Xes_for_snn = Z_auto if args.snn_space == 'pca' else Xes
        snn_acc_auto = eval_snn(snn, make_loader(Xes_for_snn, ys, cfg.batch_size, False), device=cfg.device)

        return ('auto', Xe, mode) if 0.5*(f_acc_auto + snn_acc_auto) >= 0.5*(f_acc_none + snn_acc_none) else ('none', None, None)

    # ----- evaluate grid -----
    print(f"[Gate] Eval auto-preproc only for severities >= {args.preproc_severity_gate}.")
    severity_levels = GLOBAL_SEVERITY_GRID if args.use_global_grid else args.noise_levels
    noise_types = args.noise_types

    rows = []
    noisy_cache = {}

    def eval_variant(Xe, label, nt, sev, level, detail=None):
        Xe = np.nan_to_num(Xe, nan=0.0, posinf=1.0, neginf=0.0)
        Z_eval = pca.transform(Xe)
        y_pred_fuzzy = fuzzy.predict(Z_eval)
        f_acc, _, _ = compute_metrics(ytst, y_pred_fuzzy)
        X_for_snn = Z_eval if args.snn_space == 'pca' else Xe
        test_loader = make_loader(X_for_snn, ytst, cfg.batch_size, shuffle=False)
        snn_acc = eval_snn(snn, test_loader, device=cfg.device)
        rows.append({'noise_type': nt, 'severity': sev, 'noise_level': level,
                     'preproc': label, 'preproc_detail': detail,
                     'fuzzy_acc': f_acc, 'snn_acc': snn_acc})

    for nt in noise_types:
        for sev in severity_levels:
            level = level_from_severity(nt, sev)
            key_n = (nt, sev)
            if key_n in noisy_cache:
                X_noisy = noisy_cache[key_n]
            else:
                Xn = Xtst.copy()
                if sev > 0:
                    Xn = apply_noise(Xn, nt, level)
                Xn = np.nan_to_num(Xn, nan=0.0, posinf=1.0, neginf=0.0)
                noisy_cache[key_n] = Xn
                X_noisy = Xn

            # baseline
            eval_variant(X_noisy, 'none', nt, sev, level, detail=None)

            # auto (severity gate + validation gate)
            if args.auto_preproc and float(sev) >= float(args.preproc_severity_gate):
                choice, Xe_cand, mode_used = choose_auto_or_none(X_noisy, nt, sev)
                if choice == 'auto' and mode_used is not None:
                    eval_variant(Xe_cand, 'auto', nt, sev, level, detail=mode_used)

    # save
    results_csv = os.path.join(out_dir, 'results_grid.csv')
    df = save_results_csv(rows, results_csv)
    print(f"[Diag] rows={len(df)}  noises={sorted(df['noise_type'].unique())}")
    print(f"[Diag] severities={sorted(df['severity'].unique())}")
    print(f"[Diag] preproc labels = {sorted(df['preproc'].unique())}")

# ---------------- main: one consolidated run & global plots ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # Training
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--pca-dims', type=int, default=80)
    ap.add_argument('--snn-space', type=str, choices=['raw','pca'], default='pca')

    # Train-time distribution
    ap.add_argument('--train-augment', action='store_true')
    ap.add_argument('--train-augment-ratio', type=float, default=0.5)
    ap.add_argument('--train-auto-preproc', action='store_true')
    ap.add_argument('--train-preproc-severity-gate', type=float, default=0.5)
    ap.add_argument('--pca-fit-mode', choices=['clean','augmented','auto'], default='augmented')

    # Eval
    ap.add_argument('--noise-types', nargs='+', default=['gaussian','saltpepper','dropout'])
    ap.add_argument('--noise-levels', nargs='+', type=float, default=[0.0, 0.1, 0.3, 0.5])
    ap.add_argument('--auto-preproc', action='store_true')
    ap.add_argument('--preproc-severity-gate', type=float, default=0.5)

    # Speed / outputs
    ap.add_argument('--eval-limit', type=int, default=0)
    ap.add_argument('--only-global-plots', action='store_true')
    ap.add_argument('--use-global-grid', action='store_true')

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='outputs')
    # Per-noise panels
    ap.add_argument('--per-noise-panels', nargs='+',
                    default=['saltpepper','jpeg'],
                    help='Which noise types to render as per-noise panels.')
    ap.add_argument('--fig-dir', type=str, default='figures',
                    help='Directory to write per-noise PNGs.')

    common = [
        '--train-augment',
        '--train-augment-ratio','0.5',
        '--train-auto-preproc',
        '--pca-fit-mode','augmented',
        '--auto-preproc',
        '--only-global-plots',
        '--eval-limit','0',
        '--use-global-grid'
    ]
    sev = ['--noise-levels','0.0','0.25','0.50','0.75','1.00']
    runs = [
        ('float_noises',
         common + ['--noise-types','gaussian','saltpepper','dropout','speckle',
                   'poisson','uniform','shot','anisotropic','motionblur','jpeg','quantization'] + sev)
    ]

    out_root = 'outputs'
    os.makedirs(out_root, exist_ok=True)
    out_paths = []
    for name, arglist in runs:
        args = ap.parse_args(arglist + ['--output', os.path.join(out_root, name)])
        print(f'\n=== Running {name} ===')
        run(args)
        out_paths.append(os.path.join(out_root, name, 'results_grid.csv'))

    # consolidate + global plots (none & auto)
    dfs = []
    for p in out_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['run'] = os.path.basename(os.path.dirname(p))
            dfs.append(df)

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        all_csv = os.path.join(out_root, 'results_all.csv')
        df_all.to_csv(all_csv, index=False)
        print(f'\nConsolidated: {all_csv}')
    def make_per_noise_panels(df_all, model_key, noises, fig_dir, use_global_grid=True):
        """
        For each noise in `noises`, write:
          figures/curve_<noise>_none.png
          figures/curve_<noise>_auto.png
        The 'auto' plot uses the auto curve where it exists,
        and falls back to 'none' where auto was gated off or worse.
        """
        import numpy as _np
        import matplotlib.pyplot as _plt
        os.makedirs(fig_dir, exist_ok=True)
    
        levels_union = GLOBAL_SEVERITY_GRID if use_global_grid \
                       else sorted(df_all['severity'].unique().tolist())
        levels_union = [float(x) for x in levels_union]
    
        for nt in noises:
            # maps: severity -> metric for each label
            sub_none = df_all[(df_all['noise_type'] == nt) & (df_all['preproc'] == 'none')]
            m_none   = {float(lv): v for lv, v in zip(sub_none['severity'], sub_none[model_key])}
    
            sub_auto = df_all[(df_all['noise_type'] == nt) & (df_all['preproc'] == 'auto')]
            m_auto   = {float(lv): v for lv, v in zip(sub_auto['severity'], sub_auto[model_key])}
    
            # series
            y_none = [m_none.get(lv, _np.nan) for lv in levels_union]
            # auto uses auto where present, otherwise none (to keep a continuous line)
            y_auto = [m_auto.get(lv, m_none.get(lv, _np.nan)) for lv in levels_union]
    
            # ---- plot NONE ----
            _plt.figure(figsize=(6,4))
            _plt.plot(levels_union, y_none, marker='o')
            _plt.title(f'{nt} — {model_key.upper()} (no preproc)')
            _plt.xlabel('Noise Severity (0–1)')
            _plt.ylabel('Accuracy')
            _plt.ylim(0.0, 1.0)
            _plt.grid(True, alpha=0.2)
            out_none = os.path.join(fig_dir, f'curve_{nt}_none.png')
            _plt.tight_layout(); _plt.savefig(out_none, dpi=150); _plt.close()
    
            # ---- plot AUTO (with fallback) ----
            _plt.figure(figsize=(6,4))
            _plt.plot(levels_union, y_auto, marker='o')
            _plt.title(f'{nt} — {model_key.upper()} (auto preproc)')
            _plt.xlabel('Noise Severity (0–1)')
            _plt.ylabel('Accuracy')
            _plt.ylim(0.0, 1.0)
            _plt.grid(True, alpha=0.2)
            out_auto = os.path.join(fig_dir, f'curve_{nt}_auto.png')
            _plt.tight_layout(); _plt.savefig(out_auto, dpi=150); _plt.close()
    
            print(f'[per-noise] wrote {out_none} and {out_auto}')
    
    # ---- call it right after you compute df_all and before/after global plots ----
    if dfs:
        # ... your existing consolidation & global plots ...
    
        # Per-noise case-study panels for BOTH models:
        make_per_noise_panels(df_all, model_key='snn_acc',
                              noises=args.per_noise_panels, fig_dir=args.fig_dir,
                              use_global_grid=True)
        make_per_noise_panels(df_all, model_key='fuzzy_acc',
                              noises=args.per_noise_panels, fig_dir=args.fig_dir,
                              use_global_grid=True)
        
    def make_global_mega(df_all, model_key, preproc_label, out_png, use_global_grid=False, gate=None):
        import numpy as _np
    
        levels_union = GLOBAL_SEVERITY_GRID if use_global_grid \
                       else sorted(df_all['severity'].unique().tolist())
        levels_union = [float(x) for x in levels_union]
    
        # Build continuous series:
        # - for 'none': just use none
        # - for 'auto': use auto where available; fallback to none where missing
        ys_dict = {}
        # optional: where-auto-was-used mask for marker overlay
        used_mask = {}
    
        noises = sorted(df_all['noise_type'].unique().tolist())
        for nt in noises:
            # maps: severity -> value
            sub_none = df_all[(df_all['noise_type'] == nt) & (df_all['preproc'] == 'none')]
            m_none   = {float(lv): v for lv, v in zip(sub_none['severity'], sub_none[model_key])}
    
            if preproc_label == 'auto':
                sub_auto = df_all[(df_all['noise_type'] == nt) & (df_all['preproc'] == 'auto')]
                m_auto   = {float(lv): v for lv, v in zip(sub_auto['severity'], sub_auto[model_key])}
    
                ys = []
                mask = []
                for lv in levels_union:
                    if lv in m_auto and _np.isfinite(m_auto[lv]):
                        ys.append(m_auto[lv])   # auto used & recorded
                        mask.append(True)
                    else:
                        ys.append(m_none.get(lv, _np.nan))  # fallback to none for continuity
                        mask.append(False)
                ys_dict[nt] = ys
                used_mask[nt] = mask
            else:
                ys = [m_none.get(lv, _np.nan) for lv in levels_union]
                ys_dict[nt] = ys
    
        # Skip saving if everything is NaN
        any_data = any(_np.any(~_np.isnan(_np.asarray(y, float))) for y in ys_dict.values())
        if not any_data:
            return
    
        # --- plot ---
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5))
    
        preferred = [
            'gaussian','saltpepper','dropout','speckle','uniform',
            'poisson','shot','anisotropic','motionblur','jpeg','quantization',
            'stripe','periodic','banding','checkerboard'
        ]
        series_keys = [k for k in preferred if k in ys_dict] + [k for k in ys_dict if k not in preferred]
    
        for k in series_keys:
            y = ys_dict[k]
            ax.plot(levels_union, y, label=k)
    
            # If plotting 'auto', drop small markers ONLY where auto actually won
            if preproc_label == 'auto':
                mask = used_mask.get(k, [False]*len(levels_union))
                xs_mark = [lv for lv, m in zip(levels_union, mask) if m]
                ys_mark = [val for val, m in zip(y, mask) if m and _np.isfinite(val)]
                if xs_mark:
                    ax.scatter(xs_mark, ys_mark, s=15)  # default color, tiny markers
    
        ax.set_title(f'GLOBAL {model_key.upper()} — Accuracy vs Noise Severity (all noise types; preproc={preproc_label})')
        ax.set_xlabel('Noise Severity (0–1)')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
        print('Global mega-plots saved in outputs/: curve_all_GLOBAL_snn_*.png, curve_all_GLOBAL_fuzzy_*.png')
