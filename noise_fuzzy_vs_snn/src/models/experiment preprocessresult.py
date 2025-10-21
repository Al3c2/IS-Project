# ---- make "src" importable even when running from subfolders (Spyder) ----
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))  # folder that contains "src"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------------------------------------------

from src.preproc import apply_preproc, AVAILABLE_PREPROCS, AUTO_MAP

import argparse
import numpy as np
import pandas as pd

from src.data import load_mnist, PCATransformer
from src.noise import apply_noise, level_from_severity
from src.models.snn import SNN, TrainConfig, make_loader, train as train_snn, evaluate as eval_snn
from src.utils import set_seed, ensure_dir, save_results_csv, compute_metrics

GLOBAL_SEVERITY_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]

# ---- local mega-plot that shows ALL series with a full legend (module-level) ----
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
        y = ys_dict[k]
        ax.plot(xs, y, label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

def run(args):
    set_seed(args.seed)
    out_dir = args.output
    ensure_dir(out_dir)

    # 1) Load data
    (Xtr, ytr), (Xval, yval), test_pack = load_mnist(root='data')
    Xtst, ytst = (test_pack if test_pack is not None else (Xval, yval))

    # Subsample test for speed if requested
    if args.eval_limit and args.eval_limit > 0:
        N = min(args.eval_limit, len(Xtst))
        Xtst = Xtst[:N]
        ytst = ytst[:N]
        print(f"[Eval] Using eval-limit={N} samples (set --eval-limit 0 for full test).")

    # 2) Parse settings
    severity_levels = args.noise_levels   # standardized severities in [0,1]
    if args.use_global_grid:
        severity_levels = GLOBAL_SEVERITY_GRID
    noise_types = args.noise_types

    print(f"[Gate] Preprocessing will be applied only for severities >= {args.preproc_severity_gate}.")

    # 3) Fit PCA on CLEAN TRAIN
    pca = PCATransformer(n_components=args.pca_dims, whiten=True).fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zval = pca.transform(Xval)

    # 4) Fuzzy (RBF) — quick hyperparam search on clean val
    from src.models.fuzzy import FuzzyRBFClassifier
    candidates = []
    for n_rules in (5, 8, 10):
        for s_scale in (0.8, 1.0, 1.5):
            model = FuzzyRBFClassifier(n_rules_per_class=n_rules,
                                       sigma_scale=s_scale,
                                       random_state=42,
                                       logreg_C=1.0, logreg_max_iter=1000)
            model.fit(Ztr, ytr)
            acc = (model.predict(Zval) == yval).mean()
            candidates.append((acc, n_rules, s_scale, model))
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_acc, best_n, best_s, fuzzy = candidates[0]
    print(f"[Fuzzy] best on clean val: acc={best_acc:.4f}  n_rules={best_n}  sigma_scale={best_s}")

    # 5) SNN (fit on clean raw pixels or PCA depending on flag)
    if args.snn_space == 'raw':
        in_dim = Xtr.shape[1]
        train_X, val_X = Xtr, Xval
    else:
        in_dim = Ztr.shape[1]
        train_X, val_X = Ztr, Zval

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, device=args.device)
    snn = SNN(in_dim=in_dim, hidden=args.hidden, out_dim=10)
    train_loader = make_loader(train_X, ytr, cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_X, yval, cfg.batch_size, shuffle=False)
    hist = train_snn(snn, train_loader, val_loader, cfg)

    # Save SNN training curve (CSV only; no plot here)
    snn_csv = os.path.join(out_dir, 'snn_train_history.csv')
    save_results_csv([{'epoch':e, 'train_loss':tl, 'train_acc':ta, 'val_acc':va} for (e,tl,ta,va) in hist], snn_csv)

    # 6) Evaluate across severity grid
    rows = []
    noisy_cache   = {}  # (nt, sev) -> X_noisy
    preproc_cache = {}  # (nt, sev, 'auto') -> X_pre

    # helper to evaluate one variant
    def eval_variant(Xe, label, nt, sev, level, detail=None):
        Xe = np.nan_to_num(Xe, nan=0.0, posinf=1.0, neginf=0.0)
        Z_eval = pca.transform(Xe)
        y_pred_fuzzy = fuzzy.predict(Z_eval)
        f_acc, _, _ = compute_metrics(ytst, y_pred_fuzzy)

        X_for_snn = Z_eval if args.snn_space == 'pca' else Xe
        test_loader = make_loader(X_for_snn, ytst, cfg.batch_size, shuffle=False)
        snn_acc = eval_snn(snn, test_loader, device=cfg.device)

        rows.append({
            'noise_type': nt,
            'severity': sev,
            'noise_level': level,
            'preproc': label,            # 'none' or 'auto'
            'preproc_detail': detail,    # actual method for auto (e.g., 'wiener')
            'fuzzy_acc': f_acc,
            'snn_acc': snn_acc
        })

    # --- main grid ---
    for nt in noise_types:
        for sev in severity_levels:
            level = level_from_severity(nt, sev)

            # (1) Get/calc noisy set
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

            # (2) baseline 'none' (always)
            eval_variant(X_noisy, 'none', nt, sev, level, detail=None)

            # (3) auto preproc, but only if severity >= gate
            # (3) auto preproc (only when severity ≥ gate). We keep label 'auto', and store which method in preproc_detail.
        if args.auto_preproc and float(sev) >= float(args.preproc_severity_gate):
            mode, kw = AUTO_MAP.get(nt, ('none', {}))
            if mode != 'none':
                key_pre = (nt, sev, 'auto')
                if key_pre not in preproc_cache:
                    preproc_cache[key_pre] = apply_preproc(X_noisy, mode=mode, **kw)
                Xe = preproc_cache[key_pre]
                eval_variant(Xe, 'auto', nt, sev, level, detail=mode)
        

    # Save grid results
    results_csv = os.path.join(out_dir, 'results_grid.csv')
    df = save_results_csv(rows, results_csv)

    # diagnostics
    print(f"[Diag] rows={len(df)}  noises={sorted(df['noise_type'].unique())}")
    print(f"[Diag] severities={sorted(df['severity'].unique())}")
    print(f"[Diag] preproc labels = {sorted(df['preproc'].unique())}")

    # ---- per-noise/per-run plots are SKIPPED when only-global-plots is set ----
    if args.only_global_plots:
        print(f"Done. CSV: {results_csv} ; (only-global-plots) no per-noise/per-run images saved.")
        return

    # ===== (OPTIONAL) Per-run mega-plots (if you ever need them again) =====
    df = pd.read_csv(results_csv)

    def make_mega_plot(df_all, model_key, preproc_label, out_png):
        levels_union = sorted(df_all['severity'].unique().tolist())
        ys_dict = {}
        for nt in sorted(df_all['noise_type'].unique().tolist()):
            sub_nt = df_all[df_all['noise_type'] == nt].sort_values('severity')
            m = {lv: v for lv, v in zip(sub_nt['severity'].tolist(), sub_nt[model_key].tolist())}
            ys = [m.get(lv, np.nan) for lv in levels_union]
            ys_dict[nt] = ys
        plot_curve_allseries(
            xs=levels_union,
            ys_dict=ys_dict,
            title=f'{model_key.upper()} — Accuracy vs Noise Severity (all noise types; preproc={preproc_label})',
            xlabel='Noise Severity (0–1)',
            ylabel='Accuracy',
            out_png=out_png
        )

    # (this block isn’t used when only_global_plots is True)
    pre_states = sorted(df['preproc'].unique().tolist()) if 'preproc' in df.columns else ['none']
    pre_states = [p for p in pre_states if p in ('none', 'auto')]
    for pstate in pre_states:
        sub_p = df[df['preproc'] == pstate].copy()
        if len(sub_p) == 0:
            continue
        pre_tag = pstate
        make_mega_plot(sub_p[['noise_type','severity','snn_acc']].rename(columns={'snn_acc':'SNN'}), 'SNN', pstate,
                       os.path.join(out_dir, f'curve_all_snn_{pre_tag}.png'))
        make_mega_plot(sub_p[['noise_type','severity','fuzzy_acc']].rename(columns={'fuzzy_acc':'Fuzzy'}), 'Fuzzy', pstate,
                       os.path.join(out_dir, f'curve_all_fuzzy_{pre_tag}.png'))

    print(f"Done. CSV: {results_csv} ; plots saved under {out_dir}/")

# -------------------------- multi-run orchestrator (Spyder-friendly) --------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--pca-dims', type=int, default=40)
    ap.add_argument('--snn-space', type=str, choices=['raw','pca'], default='pca')

    # IMPORTANT: interpret these as severities in [0,1]
    ap.add_argument('--noise-types', nargs='+', default=['gaussian','saltpepper','dropout'])
    ap.add_argument('--noise-levels', nargs='+', type=float, default=[0.0, 0.1, 0.3, 0.5])

    # Preproc controls
    ap.add_argument('--auto-preproc', action='store_true',
        help='Use heuristic preproc per noise type (from AUTO_MAP).')

    # Speed / outputs
    ap.add_argument('--eval-limit', type=int, default=0,
        help='Evaluate on first N test samples for speed (0 = full test).')
    ap.add_argument('--only-global-plots', action='store_true',
        help='Skip per-noise and per-run plots; only write consolidated global mega-plots.')
    ap.add_argument('--use-global-grid', action='store_true',
        help='Force all runs to use a shared severity grid [0, .25, .5, .75, 1.0].')

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='outputs')

    # NEW: gate for preprocessing (default 0.5)
    ap.add_argument('--preproc-severity-gate', type=float, default=0.5,
        help='Only apply preprocessing when severity >= this value.')

    # single consolidated run producing one results_all and four global plots
    common = [
        '--auto-preproc',
        '--only-global-plots',
        '--eval-limit','0',
        '--use-global-grid'
    ]
    sev = ['--noise-levels','0.0','0.25','0.50','0.75','1.00']

    runs = [
        ('float_noises',
         common + ['--noise-types','gaussian','saltpepper','dropout','speckle','poisson','uniform','shot','anisotropic','motionblur','jpeg','quantization'] + sev)
    ]

    out_root = 'outputs'
    os.makedirs(out_root, exist_ok=True)
    out_paths = []
    for name, arglist in runs:
        args = ap.parse_args(arglist + ['--output', os.path.join(out_root, name)])
        print(f'\n=== Running {name} ===')
        run(args)
        out_paths.append(os.path.join(out_root, name, 'results_grid.csv'))

    # Consolidate across ALL runs and create GLOBAL mega-plots that include every noise type
    dfs = []
    for p in out_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['run'] = os.path.basename(os.path.dirname(p))
            dfs.append(df)
    if len(dfs):
        df_all = pd.concat(dfs, ignore_index=True)
        all_csv = os.path.join(out_root, 'results_all.csv')
        df_all.to_csv(all_csv, index=False)
        print(f'\nConsolidated: {all_csv}')

    # Global mega-plots (ALL noise types together), saved in outputs/
    # Global mega-plots (ALL noise types together), saved in outputs/
    def make_global_mega(df_all, model_key, preproc_label, out_png, use_global_grid=False, gate=None):
        import numpy as _np
    
        levels_union = GLOBAL_SEVERITY_GRID if use_global_grid else sorted(df_all['severity'].unique().tolist())
        levels_union = [float(x) for x in levels_union]  # be safe
    
        # If plotting 'auto', optionally back-fill < gate with the baseline 'none'
        do_backfill = (preproc_label == 'auto') and (gate is not None)
    
        # Build series per noise
        ys_dict = {}
        noises = sorted(df_all['noise_type'].unique().tolist())
        for nt in noises:
            # values for requested label
            sub_lab = df_all[(df_all['noise_type'] == nt) & (df_all['preproc'] == preproc_label)]
            m_lab = {float(lv): v for lv, v in zip(sub_lab['severity'].tolist(), sub_lab[model_key].tolist())}
    
            if do_backfill:
                # baseline values for 'none'
                sub_none = df_all[(df_all['noise_type'] == nt) & (df_all['preproc'] == 'none')]
                m_none = {float(lv): v for lv, v in zip(sub_none['severity'].tolist(), sub_none[model_key].tolist())}
            else:
                m_none = {}
    
            ys = []
            for lv in levels_union:
                if do_backfill and lv < float(gate):
                    ys.append(m_none.get(lv, _np.nan))
                else:
                    ys.append(m_lab.get(lv, _np.nan))
            ys_dict[nt] = ys
    
        # Skip saving if there is no data at all (prevents 0-byte files)
        any_data = any(_np.any(~_np.isnan(_np.asarray(y, float))) for y in ys_dict.values())
        if not any_data:
            return
    
        plot_curve_allseries(
            xs=levels_union,
            ys_dict=ys_dict,
            title=f'GLOBAL {model_key.upper()} — Accuracy vs Noise Severity (all noise types; preproc={preproc_label})',
            xlabel='Noise Severity (0–1)',
            ylabel='Accuracy',
            out_png=out_png
        )
    
    # only plot for the two labels we care about
    if len(dfs):
        pre_states_all = sorted(df_all['preproc'].unique().tolist()) if 'preproc' in df_all.columns else ['none']
        pre_states = [p for p in pre_states_all if p in ('none', 'auto')]
    
        for pstate in pre_states:
            sub_p = df_all[df_all['preproc'] == pstate] if 'preproc' in df_all.columns else df_all
            if len(sub_p) == 0:
                continue
            pre_tag = pstate  # filenames end with _none or _auto
    
            make_global_mega(
                df_all, 'snn_acc', pstate,
                os.path.join(out_root, f'curve_all_GLOBAL_snn_{pre_tag}.png'),
                use_global_grid=True,
                gate=args.preproc_severity_gate
            )
            make_global_mega(
                df_all, 'fuzzy_acc', pstate,
                os.path.join(out_root, f'curve_all_GLOBAL_fuzzy_{pre_tag}.png'),
                use_global_grid=True,
                gate=args.preproc_severity_gate
            )


    print('Global mega-plots saved in outputs/: curve_all_GLOBAL_snn_*.png, curve_all_GLOBAL_fuzzy_*.png')
