"""
Evaluation of max_n_signal as a classifier: precision/recall/F1/AUC-ROC/
AUC-PR/pAUC, per campaign, per setting.

Consumes the per-campaign output of evaluation/max_n_signal.py
(max_n_signal_{campaign}.pkl.gz: accountid, data_type, plus the two
columns max_n_signal_fdr and max_n_signal_no_fdr.

For each campaign and each of the two settings:
    1. Sweep every integer threshold t present in that column.
       Classifier rule: predict IO if max_n_signal >= t.
    2. At each threshold, compute precision, recall, F1, fpr, tpr
       (positive class = io).
    3. Across the full score range, compute AUC-ROC and AUC-PR using
       max_n_signal directly as a continuous score, plus pAUC (partial
       AUC, McClish-corrected): AUC restricted to the low-false-positive-
       rate region [0, --pauc-max-fpr] -- "how good is this classifier
       when we're only willing to tolerate a small false-positive rate",
       often the more relevant question for coordination detection than
       full AUC-ROC.
    4. Identify the threshold with the best F1 (full range), and
       separately the best F1 achievable while staying within the same
       low-FPR region pAUC is measuring over ("at pAUC").
    5. Report baseline_auc_pr (= the positive rate, i.e. AUC-PR's actual
       no-skill expectation under class imbalance -- NOT 0.5, unlike
       AUC-ROC) and auc_pr_lift (= auc_pr / baseline_auc_pr), so "how much
       better than random" is explicit and comparable across campaigns/
       settings with different class balance, rather than requiring the
       reader to compute it by hand from AUC-PR alone.

Usage:
    python evaluation/max_n_signal_eval.py
    python evaluation/max_n_signal_eval.py --campaigns spain_082019_1
    python evaluation/max_n_signal_eval.py --pauc-max-fpr 0.2
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


# ---------------------------------------------------------------------------
# config -- match evaluation/max_n_signal.py's actual output location/naming
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(REPO_ROOT, 'results', 'max_n_signal')       # evaluation/max_n_signal.py output (input to this script)
RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'max_n_signal_eval')  # this script's own output

# Max false-positive rate defining the pAUC region [0, PAUC_MAX_FPR] and
# the "at pAUC" best-F1 search region. 0.1 (10% FPR) is a common default
# for partial-AUC analyses; change this if a different FPR tolerance is
# more meaningful for this use case.
PAUC_MAX_FPR = 0.1

SETTINGS = [
    'max_n_signal_fdr',
    'max_n_signal_no_fdr',
]

POSITIVE_LABEL = 'io'  # the label treated as the positive class for precision/recall/F1/AUC

# the combined multi-campaign file evaluation/max_n_signal.py also writes
# into --data-dir -- not a per-campaign file, so discover_campaigns must
# skip it rather than treating it as a campaign named 'fdr_all_campaigns'
_COMBINED_FILE_STEM = 'fdr_all_campaigns'


def discover_campaigns(data_dir: str) -> list:
    """Auto-detect campaigns from per-campaign max_n_signal files present
    in data_dir, i.e. every 'max_n_signal_<campaign>.pkl.gz' found there
    (excluding the combined max_n_signal_fdr_all_campaigns.pkl.gz)."""
    pattern = os.path.join(data_dir, "max_n_signal_*.pkl.gz")
    campaigns = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        stem = name[len("max_n_signal_"): -len(".pkl.gz")]
        if stem == _COMBINED_FILE_STEM:
            continue
        campaigns.append(stem)
    return campaigns


# ---------------------------------------------------------------------------
# threshold sweep + metrics for one (campaign, setting)
# ---------------------------------------------------------------------------

def evaluate_one_setting(df_campaign: pd.DataFrame, setting_col: str,
                         positive_label: str = POSITIVE_LABEL,
                         pauc_max_fpr: float = PAUC_MAX_FPR) -> tuple:
    """
    Sweep every integer threshold present in df_campaign[setting_col],
    compute precision/recall/F1/confusion counts/fpr/tpr at each threshold
    using the rule "predict positive if score >= threshold", and separately
    compute AUC-ROC, AUC-PR, and pAUC using the raw score (no threshold).

    Returns (df_sweep, auc_roc, auc_pr, pauc, baseline_auc_pr, auc_pr_lift)
    where df_sweep has one row per threshold (including 'fpr'/'tpr'
    columns). auc_roc/auc_pr/pauc/baseline_auc_pr/auc_pr_lift are np.nan if
    there's only one class present (undefined in that case) or fewer than
    2 users.

    baseline_auc_pr / auc_pr_lift: AUC-ROC's 0.5-chance baseline is
    misleading under class imbalance (a no-skill classifier still racks up
    a large true-negative count from the majority class, so AUC-ROC alone
    can look deceptively strong). AUC-PR's actual no-skill baseline is the
    positive rate n_pos/(n_pos+n_neg) -- a random/no-skill score has
    expected AUC-PR equal to that, not 0.5. auc_pr_lift = auc_pr /
    baseline_auc_pr makes "how much better than random" explicit and
    comparable across campaigns/settings with different class balance,
    where the raw AUC-PR number alone is not directly comparable.
    """
    y_true = (df_campaign['data_type'] == positive_label).astype(int).values
    y_score = df_campaign[setting_col].values

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0 or len(y_true) < 2:
        print(f"    [skip] {setting_col}: only one class present "
              f"(n_pos={n_pos}, n_neg={n_neg}) -- metrics undefined.")
        return pd.DataFrame(), np.nan, np.nan, np.nan, np.nan, np.nan

    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    # Standardized (McClish-corrected) partial AUC over FPR in
    # [0, pauc_max_fpr] -- sklearn returns this already rescaled to the
    # [0.5, 1.0] range (same interpretation as full AUC-ROC: 0.5 = chance,
    # 1.0 = perfect), NOT the raw (smaller) area under just that slice of
    # the curve. This is the standard way pAUC is reported in the
    # diagnostic-accuracy literature, so it stays comparable to auc_roc.
    pauc = roc_auc_score(y_true, y_score, max_fpr=pauc_max_fpr)

    baseline_auc_pr = n_pos / (n_pos + n_neg)
    auc_pr_lift = auc_pr / baseline_auc_pr if baseline_auc_pr > 0 else np.nan

    thresholds = sorted(df_campaign[setting_col].unique())
    print('threshold', thresholds)
    sweep_rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

        # fpr/tpr at this threshold -- tpr is identical to recall (both
        # tp/(tp+fn)) by definition; included explicitly under its ROC name
        # so pAUC-region filtering below (and anyone reading the sweep
        # output directly) doesn't have to rederive it from recall.
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        sweep_rows.append({
            'setting':   setting_col,
            'threshold': t,
            'precision': round(float(precision), 4),
            'recall':    round(float(recall), 4),
            'f1':        round(float(f1), 4),
            'fpr':       round(float(fpr), 4) if pd.notna(fpr) else np.nan,
            'tpr':       round(float(tpr), 4) if pd.notna(tpr) else np.nan,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })

    df_sweep = pd.DataFrame(sweep_rows)
    return df_sweep, auc_roc, auc_pr, pauc, baseline_auc_pr, auc_pr_lift


def best_f1_within_fpr(df_sweep: pd.DataFrame, max_fpr: float) -> dict:
    """
    Among thresholds whose FPR <= max_fpr (the same low-false-positive-rate
    region pAUC is computed over), find the one with the best F1.

    Returns a dict with best_threshold_at_pauc / best_f1_at_pauc /
    best_precision_at_pauc / best_recall_at_pauc, all np.nan if no
    threshold in the sweep achieves FPR <= max_fpr (can happen with very
    few negative examples -- every achievable FPR jumps straight past the
    cutoff).
    """
    candidates = df_sweep.loc[df_sweep['fpr'] <= max_fpr]
    if candidates.empty:
        return {
            'best_threshold_at_pauc':  np.nan,
            'best_f1_at_pauc':         np.nan,
            'best_precision_at_pauc':  np.nan,
            'best_recall_at_pauc':     np.nan,
        }

    best_row = candidates.loc[candidates['f1'].idxmax()]
    return {
        'best_threshold_at_pauc':  best_row['threshold'],
        'best_f1_at_pauc':         best_row['f1'],
        'best_precision_at_pauc':  best_row['precision'],
        'best_recall_at_pauc':     best_row['recall'],
    }


def evaluate_campaign(campaign: str, data_dir: str = DATA_DIR,
                      results_dir: str = RESULTS_DIR,
                      settings: list = SETTINGS,
                      pauc_max_fpr: float = PAUC_MAX_FPR) -> pd.DataFrame:
    """
    Run the full threshold sweep + best-F1 summary (full-range AND
    pAUC-region-restricted) for one campaign across all four settings, and
    save the combined result to results_dir/eval_pauc_{campaign}.pkl.gz.
    """
    print(f"\n{'='*60}")
    print(f"Campaign: {campaign}")
    print(f"{'='*60}")

    path = os.path.join(data_dir, f'max_n_signal_{campaign}.pkl.gz')
    if not os.path.exists(path):
        print(f"  [missing] {path} -- run evaluation/max_n_signal.py first.")
        return pd.DataFrame()

    df_campaign = pd.read_pickle(path)
    print(f"  Loaded {len(df_campaign)} users "
          f"({(df_campaign['data_type'] == POSITIVE_LABEL).sum()} {POSITIVE_LABEL}, "
          f"{(df_campaign['data_type'] != POSITIVE_LABEL).sum()} other)")

    all_sweeps = []
    best_f1_rows = []

    for setting_col in settings:
        if setting_col not in df_campaign.columns:
            print(f"  [skip] {setting_col}: column not found in {path}")
            continue

        print(f"  Evaluating {setting_col}...")
        df_sweep, auc_roc, auc_pr, pauc, baseline_auc_pr, auc_pr_lift = evaluate_one_setting(
            df_campaign, setting_col, pauc_max_fpr=pauc_max_fpr
        )

        if df_sweep.empty:
            best_f1_rows.append({
                'campaign': campaign, 'setting': setting_col,
                'best_threshold': np.nan, 'best_f1': np.nan,
                'best_precision': np.nan, 'best_recall': np.nan,
                'auc_roc': auc_roc, 'auc_pr': auc_pr,
                'baseline_auc_pr': baseline_auc_pr, 'auc_pr_lift': auc_pr_lift,
                'pauc': pauc, 'pauc_max_fpr': pauc_max_fpr,
                'best_threshold_at_pauc': np.nan, 'best_f1_at_pauc': np.nan,
                'best_precision_at_pauc': np.nan, 'best_recall_at_pauc': np.nan,
            })
            continue

        all_sweeps.append(df_sweep)

        best_row = df_sweep.loc[df_sweep['f1'].idxmax()]
        pauc_best = best_f1_within_fpr(df_sweep, pauc_max_fpr)

        best_f1_rows.append({
            'campaign':        campaign,
            'setting':         setting_col,
            'best_threshold':  best_row['threshold'],
            'best_f1':         best_row['f1'],
            'best_precision':  best_row['precision'],
            'best_recall':     best_row['recall'],
            'auc_roc':         round(float(auc_roc), 4),
            'auc_pr':          round(float(auc_pr), 4),
            'baseline_auc_pr': round(float(baseline_auc_pr), 4),
            'auc_pr_lift':     round(float(auc_pr_lift), 4),
            'pauc':            round(float(pauc), 4),
            'pauc_max_fpr':    pauc_max_fpr,
            **pauc_best,
        })

        print(f"    best F1={best_row['f1']:.4f} at threshold={best_row['threshold']} "
              f"(precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f}); "
              f"AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f} "
              f"(baseline={baseline_auc_pr:.4f}, lift={auc_pr_lift:.2f}x), "
              f"pAUC(FPR<={pauc_max_fpr})={pauc:.4f}")
        if pd.notna(pauc_best['best_f1_at_pauc']):
            print(f"    best F1 AT pAUC region: {pauc_best['best_f1_at_pauc']:.4f} "
                  f"at threshold={pauc_best['best_threshold_at_pauc']} "
                  f"(precision={pauc_best['best_precision_at_pauc']:.4f}, "
                  f"recall={pauc_best['best_recall_at_pauc']:.4f})")
        else:
            print(f"    [note] no threshold achieved FPR <= {pauc_max_fpr} -- "
                  f"'at pAUC' metrics are undefined for this setting/campaign.")

    if not all_sweeps and not best_f1_rows:
        print(f"  No evaluable settings for {campaign}.")
        return pd.DataFrame()

    df_sweep_all = pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame()
    if not df_sweep_all.empty:
        df_sweep_all['campaign'] = campaign
        df_sweep_all['row_type'] = 'threshold_sweep'

    df_best = pd.DataFrame(best_f1_rows)
    df_best['row_type'] = 'best_f1_summary'

    # combine sweep + summary into one file, distinguished by row_type;
    # summary-only columns (best_f1, pauc, best_f1_at_pauc, ...) are NaN on
    # sweep rows, and sweep-only columns (threshold, precision, recall, f1,
    # fpr, tpr, tp/fp/fn/tn) are NaN on summary rows.
    df_out = pd.concat([df_sweep_all, df_best], ignore_index=True)

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'eval_pauc_{campaign}.pkl.gz')
    df_out.to_pickle(out_path)
    print(f"\n  Saved: {out_path}")

    return df_out


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate max_n_signal as a classifier (precision/recall/F1/"
                    "AUC-ROC/AUC-PR/pAUC), per campaign, per setting."
    )
    parser.add_argument(
        '--campaigns', nargs='+', default=None,
        help="Campaign names to process. Default: auto-detect every "
             "'max_n_signal_<campaign>.pkl.gz' found under --data-dir.",
    )
    parser.add_argument(
        '--settings', nargs='+', default=None, choices=SETTINGS,
        help="Subset of max_n_signal settings to evaluate (default: all four).",
    )
    parser.add_argument('--data-dir', default=DATA_DIR,
                         help=f"Directory holding evaluation/max_n_signal.py's per-campaign "
                              f"max_n_signal_<campaign>.pkl.gz files (default: {DATA_DIR})")
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                         help=f"Directory to write evaluation outputs to (default: {RESULTS_DIR})")
    parser.add_argument('--pauc-max-fpr', type=float, default=PAUC_MAX_FPR,
                         help=f"Max false-positive rate defining the pAUC region and the "
                              f"'best F1 at pAUC' search region (default: {PAUC_MAX_FPR})")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.results_dir, exist_ok=True)

    campaigns = args.campaigns or discover_campaigns(args.data_dir)
    if not campaigns:
        print(f"No 'max_n_signal_<campaign>.pkl.gz' files found under {args.data_dir}, "
              f"and none given via --campaigns. Run evaluation/max_n_signal.py first.")
        return
    print(f"Campaigns: {campaigns}")

    settings = args.settings or SETTINGS
    print(f"Settings: {settings}")

    all_campaign_results = []
    for campaign in campaigns:
        df_out = evaluate_campaign(
            campaign, data_dir=args.data_dir, results_dir=args.results_dir,
            settings=settings, pauc_max_fpr=args.pauc_max_fpr,
        )
        if not df_out.empty:
            all_campaign_results.append(df_out)

    if not all_campaign_results:
        print("\nNo evaluation results produced for any campaign.")
        return

    print("\n" + "="*60)
    print("All campaigns evaluated.")
    print("="*60)


if __name__ == "__main__":
    main()
