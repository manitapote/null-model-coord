"""
AUC evaluation using per-user MINIMUM p-value as a continuous score,
instead of max_n_signal's discrete significant-indicator count.

Usage:
    python evaluation/min_pvalue_eval.py
    python evaluation/min_pvalue_eval.py --campaigns spain_082019_1
    python evaluation/min_pvalue_eval.py --pauc-max-fpr 0.2
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
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# config -- match null_model/pipeline.py's and evaluation/max_n_signal.py's
# actual output location/naming
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = os.path.join(REPO_ROOT, 'results', 'null_model')       # null_model output (input to this script)
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'data', 'filtered')            # raw filtered tweet data, for the full user universe / labels
RESULTS_DIR  = os.path.join(REPO_ROOT, 'results', 'min_pvalue_eval')  # this script's own output
ALPHA        = 0.05  # only used as a reference line in output, not to filter -- see module docstring
FILE_SUFFIX  = 'tpa0_tpu10_acc0-0'

# must match null_model/pipeline.py's INDICATOR_CONFIG keys exactly
INDICATORS = ['hashtag', 'retweet_userid', 'retweet_tweetid', 'url', 'sync']

# Max false-positive rate defining the pAUC region [0, PAUC_MAX_FPR] --
# same meaning as in max_n_signal_eval.py.
PAUC_MAX_FPR = 0.1

POSITIVE_LABEL = 'io'

# A p-value of exactly 0 can't be log-transformed; floor it. Doesn't change
# rank order since it only affects the smallest possible value uniformly.
_PVALUE_FLOOR = 1e-12

SETTINGS = ['min_pvalue_fdr', 'min_pvalue_no_fdr']


def discover_campaigns(raw_data_dir: str, file_suffix: str) -> list:
    """Auto-detect campaigns from raw filtered files present in raw_data_dir,
    i.e. every '<campaign>_<file_suffix>.pkl.gz' found there. Mirrors
    null_model/pipeline.py's discover_campaigns()."""
    pattern = os.path.join(raw_data_dir, f"*_{file_suffix}.pkl.gz")
    campaigns = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        suffix = f"_{file_suffix}.pkl.gz"
        if name.endswith(suffix):
            campaigns.append(name[: -len(suffix)])
    return campaigns


def load_full_user_universe(campaign: str, raw_data_dir: str = RAW_DATA_DIR,
                            file_suffix: str = FILE_SUFFIX) -> pd.DataFrame:
    """
    Load every userid that appears anywhere in the campaign's raw tweet
    file -- this is the FULL universe of users, independent of whether they
    ended up in any edge at all. Also pulls each user's data_type
    ('io'/'control') directly from the raw file's confirmed 'data_type'
    column. Same logic as evaluation/max_n_signal.py's helper of the same
    name (duplicated here rather than imported, matching how each script
    in this repo stays self-contained).
    """
    path = f"{raw_data_dir}/{campaign}_{file_suffix}.pkl.gz"
    if not os.path.exists(path):
        print(f"  [missing] {path} -- cannot determine full user universe, "
              f"falling back to only users present in edges.")
        return pd.DataFrame(columns=['accountid', 'data_type'])

    df_raw = pd.read_pickle(path)
    df_raw['userid'] = df_raw['userid'].astype(str)

    if 'data_type' not in df_raw.columns:
        print(f"  [warning] 'data_type' column not found in {path} -- "
              f"users with no edges will have no label and will be "
              f"dropped later.")
        universe = pd.DataFrame({'accountid': df_raw['userid'].unique()})
        universe['data_type'] = np.nan
        return universe

    universe = (
        df_raw[['userid', 'data_type']]
        .drop_duplicates(subset='userid')
        .rename(columns={'userid': 'accountid'})
    )
    return universe


# ---------------------------------------------------------------------------
# per-indicator loading + FDR correction 
# ---------------------------------------------------------------------------

def load_indicator_pvalues(campaign: str, indicator: str, data_dir: str = DATA_DIR,
                           alpha: float = ALPHA) -> pd.DataFrame:
    """
    Load one indicator's null_model p-value file and attach a
    Benjamini-Hochberg FDR-corrected column, computed within THIS
    indicator's full set of pairs (its own family of simultaneous tests) --
    same convention as auc_roc.py / max_n_signal.py.

    Returns columns [user_i, user_j, pvalue, pvalue_fdr], or None if the
    file doesn't exist (e.g. that indicator was skipped upstream).
    """
    path = f'{data_dir}/{campaign}_{indicator}_cosine_similarity_pvalue_index.pkl.gz'
    if not os.path.exists(path):
        print(f"    [missing] {path}")
        return None

    df = pd.read_pickle(path)
    print(f"    [read] {path} -> shape={df.shape}")
    if df.empty:
        return None

    expected_cols = {'user_i', 'user_j', 'pvalue'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")

    df = df[['user_i', 'user_j', 'pvalue']].copy()
    df['user_i'] = df['user_i'].astype(str)
    df['user_j'] = df['user_j'].astype(str)
    _, pvals_fdr, _, _ = multipletests(df['pvalue'].to_numpy(), alpha=alpha, method='fdr_bh')
    df['pvalue_fdr'] = pvals_fdr
    return df


# ---------------------------------------------------------------------------
# per-user minimum p-value across all indicators/edges (vectorized)
# ---------------------------------------------------------------------------

def compute_min_pvalue_per_user(campaign: str, data_dir: str = DATA_DIR,
                                indicators: list = INDICATORS,
                                alpha: float = ALPHA) -> pd.DataFrame:
    """
    For every user appearing in ANY indicator's edge list, take the single
    lowest p-value across ALL their edges, across ALL indicators -- their
    strongest available piece of coordination evidence. Computed for both
    the raw and FDR-corrected p-value.

    Returns columns ['accountid', 'min_pvalue_fdr', 'min_pvalue_no_fdr'].
    Users that never appear in any edge are NOT included here -- see
    fill_missing_users() for the zero-evidence fill step.
    """
    frames = []
    for ind in indicators:
        df = load_indicator_pvalues(campaign, ind, data_dir=data_dir, alpha=alpha)
        if df is None:
            continue
        # stack both endpoints into one long (accountid, pvalue, pvalue_fdr)
        # series so a single groupby-min covers every edge each user is in,
        # regardless of whether they were user_i or user_j.
        frames.append(df[['user_i', 'pvalue', 'pvalue_fdr']].rename(columns={'user_i': 'accountid'}))
        frames.append(df[['user_j', 'pvalue', 'pvalue_fdr']].rename(columns={'user_j': 'accountid'}))

    if not frames:
        return pd.DataFrame(columns=['accountid', 'min_pvalue_fdr', 'min_pvalue_no_fdr'])

    long = pd.concat(frames, ignore_index=True)
    out = (
        long.groupby('accountid')
        .agg(min_pvalue_no_fdr=('pvalue', 'min'), min_pvalue_fdr=('pvalue_fdr', 'min'))
        .reset_index()
    )
    print('Minimum pvalue ', out['min_pvalue_fdr'].min())
    return out[['accountid', 'min_pvalue_fdr', 'min_pvalue_no_fdr']]


def fill_missing_users(df_min: pd.DataFrame, full_universe: pd.DataFrame) -> pd.DataFrame:
    """
    Users present in the raw campaign data but absent from every edge have
    NO evidence of coordination at all -- fill their min p-value with 1.0
    (the least significant a p-value can be). 
    Without this, users with zero coordination signal
    would be silently dropped rather than correctly counted as strong
    evidence AGAINST being IO.
    """
    if full_universe.empty:
        return df_min

    merged = full_universe.merge(df_min, on='accountid', how='left')
    n_filled = merged['min_pvalue_fdr'].isna().sum()
    for col in ['min_pvalue_fdr', 'min_pvalue_no_fdr']:
        merged[col] = merged[col].fillna(1.0)

    print(f"    Filled {n_filled} users with min_pvalue=1.0 "
          f"(present in raw data, absent from every edge)")
    return merged[['accountid', 'data_type', 'min_pvalue_fdr', 'min_pvalue_no_fdr']]


# ---------------------------------------------------------------------------
# threshold sweep + AUC metrics for one (campaign, setting)
# ---------------------------------------------------------------------------

def evaluate_one_setting(df_users: pd.DataFrame, setting_col: str,
                         positive_label: str = POSITIVE_LABEL,
                         pauc_max_fpr: float = PAUC_MAX_FPR) -> tuple:
    """
    Sweep every p-value threshold t actually present in df_users[setting_col],
    classifier rule "predict IO if min_pvalue <= t" (LOWER p-value = MORE
    significant = more evidence of coordination, opposite direction from
    max_n_signal_eval.py's ">= t" rule on a count).

    AUC-ROC/AUC-PR/pAUC are computed on -log10(min_pvalue) as a continuous
    score (higher = more significant), same convention as auc_roc.py.

    Returns (df_sweep, auc_roc, auc_pr, pauc, baseline_auc_pr, auc_pr_lift)
    -- see max_n_signal_eval.py's evaluate_one_setting for what
    baseline_auc_pr/auc_pr_lift mean (AUC-PR's actual no-skill expectation
    under class imbalance, and the lift over it).
    """
    y_true = (df_users['data_type'] == positive_label).astype(int).values
    min_pvalue = df_users[setting_col].to_numpy(dtype=np.float64)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0 or len(y_true) < 2:
        print(f"    [skip] {setting_col}: only one class present "
              f"(n_pos={n_pos}, n_neg={n_neg}) -- metrics undefined.")
        return pd.DataFrame(), np.nan, np.nan, np.nan, np.nan, np.nan

    score = -np.log10(np.clip(min_pvalue, _PVALUE_FLOOR, 1.0))

    auc_roc = roc_auc_score(y_true, score)
    auc_pr = average_precision_score(y_true, score)
    pauc = roc_auc_score(y_true, score, max_fpr=pauc_max_fpr)
    baseline_auc_pr = n_pos / (n_pos + n_neg)
    auc_pr_lift = auc_pr / baseline_auc_pr if baseline_auc_pr > 0 else np.nan

    thresholds = sorted(np.unique(min_pvalue))
    sweep_rows = []
    for t in thresholds:
        y_pred = (min_pvalue <= t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

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
    """Among thresholds whose FPR <= max_fpr, find the one with the best F1.
    Same logic as max_n_signal_eval.py's helper of the same name."""
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


# ---------------------------------------------------------------------------
# per-campaign pipeline
# ---------------------------------------------------------------------------

def evaluate_campaign(campaign: str, data_dir: str = DATA_DIR,
                      raw_data_dir: str = RAW_DATA_DIR,
                      results_dir: str = RESULTS_DIR,
                      file_suffix: str = FILE_SUFFIX,
                      indicators: list = INDICATORS,
                      alpha: float = ALPHA,
                      pauc_max_fpr: float = PAUC_MAX_FPR) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Campaign: {campaign}")
    print(f"{'='*60}")

    print("  Computing per-user minimum p-value across all indicators...")
    df_min = compute_min_pvalue_per_user(campaign, data_dir=data_dir,
                                          indicators=indicators, alpha=alpha)

    print("  Loading full user universe from raw tweet data...")
    full_universe = load_full_user_universe(campaign, raw_data_dir=raw_data_dir,
                                             file_suffix=file_suffix)
    if full_universe.empty:
        print(f"  [warning] {campaign}: no full user universe available -- "
              f"users with zero edges won't be fillable, and labels will "
              f"only cover users who appear in at least one edge.")
        df_users = df_min
        df_users['data_type'] = np.nan
    else:
        df_users = fill_missing_users(df_min, full_universe)

    n_before = len(df_users)
    df_users = df_users.dropna(subset=['data_type'])
    print(f"  Labeled users: {len(df_users)} / {n_before} "
          f"(dropped {n_before - len(df_users)} unlabeled)")

    if df_users.empty:
        print(f"  No labeled users for {campaign}.")
        return pd.DataFrame()

    os.makedirs(results_dir, exist_ok=True)
    users_path = os.path.join(results_dir, f'min_pvalue_{campaign}.pkl.gz')
    df_users.to_pickle(users_path)
    print(f"  Saved per-user min p-values: {users_path}")

    all_sweeps = []
    best_f1_rows = []

    for setting_col in SETTINGS:
        print(f"  Evaluating {setting_col}...")
        df_sweep, auc_roc, auc_pr, pauc, baseline_auc_pr, auc_pr_lift = evaluate_one_setting(
            df_users, setting_col, pauc_max_fpr=pauc_max_fpr
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

        print(f"    {len(df_sweep)} achievable thresholds (vs. max_n_signal_eval.py's "
              f"handful) -- best F1={best_row['f1']:.4f} at min_pvalue<={best_row['threshold']:.4g} "
              f"(precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f}); "
              f"AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f} "
              f"(baseline={baseline_auc_pr:.4f}, lift={auc_pr_lift:.2f}x), "
              f"pAUC(FPR<={pauc_max_fpr})={pauc:.4f}")
        if pd.notna(pauc_best['best_f1_at_pauc']):
            print(f"    best F1 AT pAUC region: {pauc_best['best_f1_at_pauc']:.4f} "
                  f"at min_pvalue<={pauc_best['best_threshold_at_pauc']:.4g} "
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

    df_out = pd.concat([df_sweep_all, df_best], ignore_index=True)
    out_path = os.path.join(results_dir, f'eval_min_pvalue_{campaign}.pkl.gz')
    df_out.to_pickle(out_path)
    print(f"\n  Saved: {out_path}")

    return df_out


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate per-user minimum p-value as a continuous IO/"
                    "control classifier score (AUC-ROC/AUC-PR/pAUC + a "
                    "p-value threshold sweep), per campaign."
    )
    parser.add_argument(
        '--campaigns', nargs='+', default=None,
        help="Campaign names to process. Default: auto-detect every "
             "'<campaign>_<file-suffix>.pkl.gz' found under --raw-data-dir.",
    )
    parser.add_argument('--data-dir', default=DATA_DIR,
                         help=f"Directory holding null_model/pipeline.py's per-indicator "
                              f"p-value files (default: {DATA_DIR})")
    parser.add_argument('--raw-data-dir', default=RAW_DATA_DIR,
                         help=f"Directory with raw filtered tweet pkl.gz files, used for "
                              f"the full user universe / labels (default: {RAW_DATA_DIR})")
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                         help=f"Directory to write outputs to (default: {RESULTS_DIR})")
    parser.add_argument('--file-suffix', default=FILE_SUFFIX,
                         help=f"Filename suffix identifying the filtering run (default: {FILE_SUFFIX})")
    parser.add_argument('--alpha', type=float, default=ALPHA,
                         help=f"Alpha passed to the per-indicator Benjamini-Hochberg FDR "
                              f"correction (default: {ALPHA}); NOT used to filter p-values here.")
    parser.add_argument('--pauc-max-fpr', type=float, default=PAUC_MAX_FPR,
                         help=f"Max false-positive rate defining the pAUC region and the "
                              f"'best F1 at pAUC' search region (default: {PAUC_MAX_FPR})")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.results_dir, exist_ok=True)

    campaigns = args.campaigns or discover_campaigns(args.raw_data_dir, args.file_suffix)
    if not campaigns:
        print(f"No campaigns found under {args.raw_data_dir} matching "
              f"'*_{args.file_suffix}.pkl.gz', and none given via --campaigns. Nothing to do.")
        return
    print(f"Campaigns: {campaigns}")

    for campaign in campaigns:
        evaluate_campaign(
            campaign,
            data_dir=args.data_dir,
            raw_data_dir=args.raw_data_dir,
            results_dir=args.results_dir,
            file_suffix=args.file_suffix,
            alpha=args.alpha,
            pauc_max_fpr=args.pauc_max_fpr,
        )

    print("\n" + "="*60)
    print("All campaigns evaluated.")
    print("="*60)


if __name__ == "__main__":
    main()
