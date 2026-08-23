"""
max_n_signal for IO vs. control users

    1. Load per-indicator cosine similarity p-value files
       (output of null_model/pipeline.py: user_i, user_j, cosine_similarity,
       pvalue, data_type_i, data_type_j)
    2. Apply Benjamini-Hochberg FDR correction
    3. Build significant edges per indicator
       (hashtag, retweet_userid, retweet_tweetid, url, sync -- 'client'/
       tweet_client_name is intentionally excluded from detection, see
       INDICATORS below)
    4. Compute n_signal per edge (count of indicators with a significant edge)
    5. Compute max_n_signal per user (max across all of that user's edges)
    5b. Load the FULL user universe from the raw tweet file
        ({campaign}_{FILE_SUFFIX}.pkl.gz) and assign max_n_signal=0 to any
        user present in raw data but absent from every significant edge --
        otherwise users with no detected coordination signal at all would
        be silently excluded rather than correctly counted as a "0" data
        point in the IO-vs-control comparison.
    6. Labels are ALREADY attached by null_model/pipeline.py (data_type_i /
       data_type_j columns) and also available directly from the raw
       file's 'data_type' column (used for the zero-filled users).
    7. Merge max_n_signal with labels per campaign, under two settings
       (FDR-corrected vs. raw p-value)
    8. Run Mann-Whitney U test per campaign, per setting (H1: IO > control)


Usage:
    python evaluation/max_n_signal.py
    python evaluation/max_n_signal.py --campaigns spain_082019_1
    python evaluation/max_n_signal.py --data-dir results/null_model --raw-data-dir data/filtered
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = os.path.join(REPO_ROOT, 'results', 'null_model')          # null_model output (input to this script)
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'data', 'filtered')
RESULTS_DIR  = os.path.join(REPO_ROOT, 'results', 'max_n_signal')        # this script's own output, kept separate from null_model's
ALPHA        = 0.05

# raw tweet file naming convention (same FILE_SUFFIX used by null_model/pipeline.py):
#   {RAW_DATA_DIR}/{campaign}_{FILE_SUFFIX}.pkl.gz
FILE_SUFFIX = 'tpa0_tpu10_acc0-0'

# must match null_model/pipeline.py's INDICATOR_CONFIG keys exactly --
# retweet is split into two indicators. 'client' (tweet_client_name) is
# intentionally excluded from detection -- it isn't a coordination signal
# in the same sense as the others (huge numbers of unrelated accounts
# legitimately share the same client app), so it's dropped here rather
# than compared with/without as before.
INDICATORS = ['hashtag',
              'retweet_userid', 'retweet_tweetid',
              'url', 'sync'
              ]


def discover_campaigns(raw_data_dir: str, file_suffix: str) -> list:
    """Auto-detect campaigns from raw filtered files present in raw_data_dir,
    i.e. every '<campaign>_<file_suffix>.pkl.gz' found there. Mirrors
    null_model/pipeline.py's discover_campaigns()."""
    import glob
    pattern = os.path.join(raw_data_dir, f"*_{file_suffix}.pkl.gz")
    campaigns = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        suffix = f"_{file_suffix}.pkl.gz"
        if name.endswith(suffix):
            campaigns.append(name[: -len(suffix)])
    return campaigns


# ---------------------------------------------------------------------------
# FDR correction + significant edges per indicator
# ---------------------------------------------------------------------------

def load_and_fdr_correct(path: str, alpha: float = ALPHA, use_fdr: bool = True) -> pd.DataFrame:
    """Load a cosine-similarity p-value file (null_model/pipeline.py output)
    and flag significant edges.

    use_fdr=True  -- apply Benjamini-Hochberg FDR correction, flag where the
                      corrected p-value (pvalue_fdr) < alpha.
    use_fdr=False -- skip correction entirely, flag directly where the raw
                      pvalue < alpha (pvalue_fdr is just set equal to the
                      raw pvalue for a consistent column name downstream).

    Renames user_i/user_j to source/target for a consistent internal
    interface regardless of which indicator's file naming you're reading.
    """
    if not os.path.exists(path):
        print(f"    [missing] {path}")
        return pd.DataFrame(columns=['source', 'target', 'pvalue', 'pvalue_fdr', 'significant'])

    df = pd.read_pickle(path)
    print(f"    [read] {path} -> shape={df.shape}")
    if df.empty:
        print(f"    [warning] {path} loaded as EMPTY -- check this is the file you expect.")
        return df.assign(pvalue_fdr=np.nan, significant=False)

    expected_cols = {'user_i', 'user_j', 'pvalue'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns in {path}: {df.columns.tolist()}. "
            f"Expected at least {expected_cols} -- this should be direct "
            f"output from null_model/pipeline.py's run_campaign_indicator()."
        )

    df = df.copy()
    if use_fdr:
        reject, pvals_fdr, _, _ = multipletests(df['pvalue'].values, alpha=alpha, method='fdr_bh')
        df['pvalue_fdr']  = pvals_fdr
        df['significant'] = reject
    else:
        df['pvalue_fdr']  = df['pvalue']
        df['significant'] = df['pvalue'] < alpha

    # standardize to source/target for the rest of this pipeline
    df = df.rename(columns={'user_i': 'source', 'user_j': 'target'})

    return df


def get_significant_edges(campaign: str, data_dir: str = DATA_DIR, alpha: float = ALPHA,
                          indicators: list = None, use_fdr: bool = True) -> dict:
    """
    Return dict of indicator -> filtered significant edge DataFrame.

    File naming matches null_model/pipeline.py's actual save path EXACTLY:
        {data_dir}/{campaign}_{indicator}_cosine_similarity_pvalue_index.pkl.gz

    indicators: which indicator keys to load (defaults to the module-level
                INDICATORS list).
    use_fdr:    True applies BH-FDR correction; False thresholds raw
                pvalue < alpha directly (see load_and_fdr_correct).
    """
    if indicators is None:
        indicators = INDICATORS

    sig_edges = {}

    for ind in indicators:
        path = f'{data_dir}/{campaign}_{ind}_cosine_similarity_pvalue_index.pkl.gz'
        df = load_and_fdr_correct(path, alpha=alpha, use_fdr=use_fdr)

        if not df.empty:
            sig_edges[ind] = df.loc[df['significant'] == True, ['source', 'target']]
        else:
            sig_edges[ind] = pd.DataFrame(columns=['source', 'target'])

        fdr_label = 'FDR' if use_fdr else 'raw'
        print(f"    {ind:16s}: {len(sig_edges[ind])} significant edges ({fdr_label}<{alpha})")

    return sig_edges


# ---------------------------------------------------------------------------
# n_signal per edge -> max_n_signal per user (vectorized: groupby
# instead of iterrows(), per max_n_signal_pipeline_mpi.py)
# ---------------------------------------------------------------------------

def compute_max_n_signal(sig_edges: dict) -> pd.DataFrame:
    """
    For each user pair, count how many indicators have a significant edge.
    Then take max across all neighbors per user.

    Always returns a DataFrame with columns ['accountid', 'max_n_signal'],
    even if sig_edges is entirely empty.
    """
    frames = []
    for ind, df in sig_edges.items():
        if df.empty:
            continue
        src = df['source'].astype(str).to_numpy()
        tgt = df['target'].astype(str).to_numpy()
        # normalize pair order (unordered pair) exactly like
        # tuple(sorted([...])) -- np.minimum/np.maximum on object arrays of
        # strings applies elementwise Python string comparison, so this is
        # equivalent to sorting each pair.
        lo = np.minimum(src, tgt)
        hi = np.maximum(src, tgt)
        frames.append(pd.DataFrame({'u': lo, 'v': hi}))

    if not frames:
        return pd.DataFrame(columns=['accountid', 'max_n_signal'])

    combined = pd.concat(frames, ignore_index=True)
    # count(*) per unordered pair, across ALL indicators combined
    edge_counts = combined.groupby(['u', 'v']).size().reset_index(name='n_signal')

    # max n_signal per user across all of that user's edges -- stack both
    # endpoints into one long (accountid, n_signal) series, then groupby-max.
    long = pd.concat([
        edge_counts[['u', 'n_signal']].rename(columns={'u': 'accountid'}),
        edge_counts[['v', 'n_signal']].rename(columns={'v': 'accountid'}),
    ], ignore_index=True)

    df_max = (
        long.groupby('accountid')['n_signal']
        .max()
        .reset_index()
        .rename(columns={'n_signal': 'max_n_signal'})
    )
    return df_max


def load_full_user_universe(campaign: str, raw_data_dir: str = RAW_DATA_DIR,
                            file_suffix: str = FILE_SUFFIX) -> pd.DataFrame:
    """
    Load every userid that appears anywhere in the campaign's raw tweet
    file -- this is the FULL universe of users, independent of whether they
    ended up in any significant similarity edge. Also pulls each user's
    data_type ('io'/'control') directly from the raw file's confirmed
    'data_type' column, so labels are available even for users who never
    appear in compute_max_n_signal()'s output.

    Path: {raw_data_dir}/{campaign}_{file_suffix}.pkl.gz
    """
    path = f"{raw_data_dir}/{campaign}_{file_suffix}.pkl.gz"
    if not os.path.exists(path):
        print(f"  [missing] {path} -- cannot determine full user universe, "
              f"falling back to only users present in similarity edges.")
        return pd.DataFrame(columns=['accountid', 'data_type'])

    df_raw = pd.read_pickle(path)
    df_raw['userid'] = df_raw['userid'].astype(str)

    if 'data_type' not in df_raw.columns:
        print(f"  [warning] 'data_type' column not found in {path} -- "
              f"users filled in as zero will have no label and will be "
              f"dropped later by dropna(subset=['data_type']).")
        universe = pd.DataFrame({'accountid': df_raw['userid'].unique()})
        universe['data_type'] = np.nan
        return universe

    # one row per user with their label (use first occurrence; data_type
    # should be consistent per user per the upstream pipeline's design)
    universe = (
        df_raw[['userid', 'data_type']]
        .drop_duplicates(subset='userid')
        .rename(columns={'userid': 'accountid'})
    )
    return universe


def fill_missing_users_with_zero(df_max: pd.DataFrame, full_universe: pd.DataFrame) -> pd.DataFrame:
    """
    Given df_max (accountid, max_n_signal) covering only users who appeared
    in at least one significant edge, and full_universe (accountid,
    data_type) covering every user in the raw campaign data, return a
    combined DataFrame where every user in full_universe is present --
    users not in df_max get max_n_signal = 0.
    """
    if full_universe.empty:
        return df_max

    merged = full_universe.merge(df_max[['accountid', 'max_n_signal']], on='accountid', how='left')
    n_filled = merged['max_n_signal'].isna().sum()
    merged['max_n_signal'] = merged['max_n_signal'].fillna(0).astype(int)

    print(f"    Filled {n_filled} users with max_n_signal=0 "
          f"(present in raw data, absent from all significant edges)")

    return merged[['accountid', 'max_n_signal', 'data_type']]


# ---------------------------------------------------------------------------
# 6. Labels -- already attached by null_model/pipeline.py, reuse directly
# (vectorized: dict(zip(...)) instead of iterrows(), per
# max_n_signal_pipeline_mpi.py -- same left-to-right overwrite order)
# ---------------------------------------------------------------------------

def get_labels_from_null_model_output(campaign: str, data_dir: str = DATA_DIR,
                                       indicators: list = INDICATORS) -> dict:
    """
    null_model/pipeline.py already attaches data_type_i / data_type_j to
    every saved edge file (derived from the raw tweet data's confirmed
    'data_type' column). Rather than re-deriving labels from a separate
    IO/control folder structure that may not even exist for this dataset,
    pull labels directly out of whichever edge file(s) are available --
    they all carry the same underlying per-user labels.

    Returns dict: accountid (str) -> 'io' or 'control'
    """
    labels = {}

    for ind in indicators:
        path = f'{data_dir}/{campaign}_{ind}_cosine_similarity_pvalue_index.pkl.gz'
        if not os.path.exists(path):
            continue

        df = pd.read_pickle(path)
        if df.empty or not {'user_i', 'user_j', 'data_type_i', 'data_type_j'}.issubset(df.columns):
            continue

        s_i = df[['user_i', 'data_type_i']].dropna()
        s_j = df[['user_j', 'data_type_j']].dropna()

        labels.update(dict(zip(s_i['user_i'].astype(str), s_i['data_type_i'])))
        labels.update(dict(zip(s_j['user_j'].astype(str), s_j['data_type_j'])))

    n_io      = sum(v == 'io' for v in labels.values())
    n_control = sum(v == 'control' for v in labels.values())
    print(f"    Labels recovered from null model output: {n_io} io, {n_control} control")

    return labels


def compute_one_setting(campaign: str, data_dir: str, full_universe: pd.DataFrame,
                        indicators: list, use_fdr: bool, alpha: float,
                        column_name: str) -> pd.DataFrame:
    """
    Run the full significant-edges -> max_n_signal -> zero-fill pipeline for
    ONE (indicators, use_fdr) setting, returning a 2-column DataFrame
    (accountid, column_name) ready to be merged with other settings.
    """
    fdr_label = 'FDR-corrected' if use_fdr else 'raw p-value'
    print(f"  -- Setting: {fdr_label} --")

    sig_edges = get_significant_edges(
        campaign, data_dir=data_dir, alpha=alpha,
        indicators=indicators, use_fdr=use_fdr,
    )
    df_max = compute_max_n_signal(sig_edges)
    if df_max.empty:
        df_max = pd.DataFrame(columns=['accountid', 'max_n_signal'])

    if not full_universe.empty:
        df_max = fill_missing_users_with_zero(df_max, full_universe)
        df_max = df_max[['accountid', 'max_n_signal']]
    # if full_universe is empty, df_max only has whatever users appeared
    # in significant edges -- no zero-fill possible without the raw file

    df_max = df_max.rename(columns={'max_n_signal': column_name})
    return df_max


# ---------------------------------------------------------------------------
# 7. Full pipeline per campaign -- two settings (FDR-corrected / raw p-value)
# ---------------------------------------------------------------------------

def run_max_n_signal_pipeline(campaigns: list,
                               data_dir: str = DATA_DIR,
                               raw_data_dir: str = RAW_DATA_DIR,
                               file_suffix: str = FILE_SUFFIX,
                               results_dir: str = RESULTS_DIR,
                               alpha: float = ALPHA) -> pd.DataFrame:
    """
    For each campaign, computes max_n_signal under two settings (both over
    INDICATORS, which excludes 'client'/tweet_client_name):
        max_n_signal_fdr     -- FDR-corrected
        max_n_signal_no_fdr  -- raw pvalue < alpha

    Saves one file per campaign (max_n_signal_{campaign}.pkl.gz) in
    results_dir with both max_n_signal columns. Returns the combined
    DataFrame across campaigns.
    """
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for campaign in campaigns:
        print(f"\n{'='*60}")
        print(f"Campaign: {campaign}")
        print(f"{'='*60}")

        print("  Loading full user universe from raw tweet data...")
        full_universe = load_full_user_universe(
            campaign, raw_data_dir=raw_data_dir, file_suffix=file_suffix
        )
        if full_universe.empty:
            print(f"  [warning] {campaign}: no full user universe available -- "
                  f"zero-fill will be skipped for both settings, and any "
                  f"users absent from a given setting's significant edges will "
                  f"simply be missing from that setting's column after merge.")

        # compute both settings
        df_fdr = compute_one_setting(
            campaign, data_dir, full_universe,
            indicators=INDICATORS, use_fdr=True, alpha=alpha,
            column_name='max_n_signal_fdr',
        )
        df_no_fdr = compute_one_setting(
            campaign, data_dir, full_universe,
            indicators=INDICATORS, use_fdr=False, alpha=alpha,
            column_name='max_n_signal_no_fdr',
        )

        # merge both settings into one per-user table on accountid. outer
        # join so a user appearing in only one setting (e.g. when zero-fill
        # wasn't possible) still shows up -- missing values become NaN
        # rather than silently dropping the user.
        df_campaign = df_fdr.merge(df_no_fdr, on='accountid', how='outer')

        # attach labels: prefer full_universe's data_type (covers everyone,
        # including users only present via zero-fill); fall back to labels
        # recovered from null model output for any user full_universe
        # didn't cover (e.g. full_universe was empty/unavailable)
        if not full_universe.empty:
            df_campaign = df_campaign.merge(
                full_universe[['accountid', 'data_type']], on='accountid', how='left'
            )
        else:
            df_campaign['data_type'] = np.nan

        missing_label_mask = df_campaign['data_type'].isna()
        if missing_label_mask.any():
            fallback_labels = get_labels_from_null_model_output(
                campaign, data_dir=data_dir, indicators=INDICATORS
            )
            df_campaign.loc[missing_label_mask, 'data_type'] = (
                df_campaign.loc[missing_label_mask, 'accountid'].map(fallback_labels)
            )

        n_before = len(df_campaign)
        df_campaign = df_campaign.dropna(subset=['data_type'])
        print(f"  Labeled users: {len(df_campaign)} / {n_before} "
              f"(dropped {n_before - len(df_campaign)} unlabeled)")

        # fill any remaining NaN max_n_signal (e.g. a user present in one
        # setting's merge but not the other, when zero-fill wasn't possible)
        # with 0, and cast to int for cleanliness
        signal_cols = ['max_n_signal_fdr', 'max_n_signal_no_fdr']
        for col in signal_cols:
            df_campaign[col] = df_campaign[col].fillna(0).astype(int)

        df_campaign['campaign'] = campaign

        out_cols = ['campaign', 'accountid', 'data_type'] + signal_cols
        df_campaign = df_campaign[out_cols]

        per_campaign_path = os.path.join(results_dir, f'max_n_signal_{campaign}.pkl.gz')
        df_campaign.to_pickle(per_campaign_path)
        print(f"  Saved: {per_campaign_path}")

        all_results.append(df_campaign)

    if not all_results:
        print("\nNo results produced across any campaign.")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# 8. Significance test per campaign: IO vs control max_n_signal
# ---------------------------------------------------------------------------

def test_io_vs_control_per_campaign(df_result: pd.DataFrame,
                                    signal_col: str = 'max_n_signal_fdr') -> pd.DataFrame:
    """
    Mann-Whitney U test per campaign: H1 = IO[signal_col] > control[signal_col].

    signal_col: which of the two max_n_signal columns to test --
        'max_n_signal_fdr' (FDR-corrected) or 'max_n_signal_no_fdr' (raw p-value)
    """
    test_results = []

    for campaign in df_result['campaign'].unique():
        df_c = df_result[df_result['campaign'] == campaign]

        io_vals = df_c.loc[df_c['data_type'] == 'io', signal_col].values
        co_vals = df_c.loc[df_c['data_type'] == 'control', signal_col].values

        if len(io_vals) < 2 or len(co_vals) < 2:
            print(f"  [skip] {campaign}: insufficient samples "
                  f"(n_io={len(io_vals)}, n_control={len(co_vals)})")
            continue

        stat, pval = stats.mannwhitneyu(io_vals, co_vals, alternative='greater')

        test_results.append({
            'campaign':       campaign,
            'signal_col':     signal_col,
            'n_io':           len(io_vals),
            'n_control':      len(co_vals),
            'median_io':      round(float(np.median(io_vals)), 4),
            'median_control': round(float(np.median(co_vals)), 4),
            'mean_io':        round(float(np.mean(io_vals)), 4),
            'mean_control':   round(float(np.mean(co_vals)), 4),
            'U_stat':         round(stat, 4),
            'p_value':        round(pval, 6),
            'significant':    pval < 0.05,
        })

    # pd.DataFrame([]) has no columns at all, so .sort_values('p_value')
    # would raise KeyError rather than just producing an empty result --
    # guard explicitly instead (hit in practice whenever every campaign in
    # a run has fewer than 2 io or 2 control users, e.g. a control-less
    # campaign).
    if not test_results:
        print("\nNo campaigns had sufficient data for testing.")
        return pd.DataFrame(columns=[
            'campaign', 'signal_col', 'n_io', 'n_control', 'median_io',
            'median_control', 'mean_io', 'mean_control', 'U_stat',
            'p_value', 'significant',
        ])

    df_test = pd.DataFrame(test_results).sort_values('p_value')

    if not df_test.empty:
        n_sig = df_test['significant'].sum()
        print(f"\nMann-Whitney U test results for {signal_col} (H1: IO > Control):")
        print(df_test.to_string(index=False))
        print(f"\n{n_sig}/{len(df_test)} campaigns significant at p<0.05")
    else:
        print("\nNo campaigns had sufficient data for testing.")

    return df_test


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute max_n_signal per user (IO vs control) from "
                    "null_model/pipeline.py's output, and test IO > control "
                    "per campaign."
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
                         help=f"Directory to write max_n_signal outputs to (default: {RESULTS_DIR})")
    parser.add_argument('--file-suffix', default=FILE_SUFFIX,
                         help=f"Filename suffix identifying the filtering run (default: {FILE_SUFFIX})")
    parser.add_argument('--alpha', type=float, default=ALPHA,
                         help=f"Significance threshold for FDR/raw p-value filtering "
                              f"and the Mann-Whitney tests (default: {ALPHA})")
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

    print("STEP 1-7: Computing max_n_signal per user per campaign "
          "(2 settings: FDR-corrected / raw p-value)")
    df_result = run_max_n_signal_pipeline(
        campaigns,
        data_dir=args.data_dir,
        raw_data_dir=args.raw_data_dir,
        file_suffix=args.file_suffix,
        results_dir=args.results_dir,
        alpha=args.alpha,
    )

    if df_result.empty:
        print("\nPipeline produced no results. Check that null_model/pipeline.py "
              "has actually been run for these campaigns and that --data-dir points "
              "to its --results-dir.")
        return

    print(f"\nFinal combined shape: {df_result.shape}")

    signal_cols = ['max_n_signal_fdr', 'max_n_signal_no_fdr']
    for col in signal_cols:
        print(f"\nSummary by campaign and data_type -- {col}:")
        print(df_result.groupby(['campaign', 'data_type'])[col].describe())

    out_path = os.path.join(args.results_dir, 'max_n_signal_fdr_all_campaigns.pkl.gz')
    df_result.to_pickle(out_path)
    print(f"\nSaved combined per-user results to {out_path}")

    print("\n" + "="*60)
    print("STEP 8: IO vs control Mann-Whitney testing per campaign, per setting")
    print("="*60)
    all_tests = []
    for col in signal_cols:
        df_test = test_io_vs_control_per_campaign(df_result, signal_col=col)
        all_tests.append(df_test)

    df_test_all = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    test_path = os.path.join(args.results_dir, 'max_n_signal_significance_per_campaign.csv')
    df_test_all.to_csv(test_path, index=False)
    print(f"\nSaved significance test results to {test_path}")


if __name__ == "__main__":
    main()
