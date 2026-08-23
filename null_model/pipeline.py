"""
Null model pipeline.

Usage:
    python null_model/pipeline.py
    python null_model/pipeline.py --campaigns spain_082019_1 --num-permutations 200
    python null_model/pipeline.py --data-dir data/filtered --similarity-dir results
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gc
import glob
import time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_OF_TIMES = 1000
DATA_DIR       = os.path.join(REPO_ROOT, 'data', 'filtered')            # raw filtered tweet data (input)
SIMILARITY_DIR = os.path.join(REPO_ROOT, 'results')                     # pre-computed cosine-similarity edge lists (input, from the filtering pipeline)
RESULTS_DIR    = os.path.join(REPO_ROOT, 'results', 'null_model')       # this script's own output (p-value edge lists)
FILE_SUFFIX  = 'tpa0_tpu10_acc0-0'

MIN_USERS_FOR_NULL_MODEL = 5

INDICATOR_CONFIG = {
   
    'hashtag':         {'suffix': 'similarity_hashtags_all',
                        'column': 'hashtags',          
                        'explode': True,  
                        'filter_retweet': False, 
                        'use_tfidf_method': False
                        },
    'retweet_userid':  {'suffix': 'similarity_retweet_userid_all',  
                        'column': 'retweet_userid',    
                        'explode': False, 
                        'filter_retweet': True,  
                        'use_tfidf_method': False
                        },
    'retweet_tweetid': {'suffix': 'similarity_retweet_tweetid_all', 
                        'column': 'retweet_tweetid',   
                        'explode': False, 
                        'filter_retweet': True,  
                        'use_tfidf_method': False
                        },
    'url':             {'suffix': 'similarity_urls_all',            
                        'column': 'urls',              
                        'explode': True,  
                        'filter_retweet': False, 
                        'use_tfidf_method': False
                        },
    'sync':            {'suffix': 'similarity_temporal_30min_all',  
                        'column': 'time_30min',        
                        'explode': False, 
                        'filter_retweet': False, 
                        'use_tfidf_method': True
                        },
}


def _detect_available_cpus() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            # SLURM_JOB_CPUS_PER_NODE can look like "50" or "50(x2)"
            digits = ''.join(ch for ch in val.split('(')[0] if ch.isdigit())
            if digits:
                return int(digits)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


_RAW_CPU_COUNT = os.cpu_count() or 4
_AVAILABLE_CPUS = _detect_available_cpus()
DEFAULT_INNER_WORKERS = max(1, _AVAILABLE_CPUS - 2)


def discover_campaigns(data_dir: str, file_suffix: str) -> list:
    """Auto-detect campaigns from raw filtered files present in data_dir,
    i.e. every '<campaign>_<file_suffix>.pkl.gz' found there."""
    pattern = os.path.join(data_dir, f"*_{file_suffix}.pkl.gz")
    campaigns = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        suffix = f"_{file_suffix}.pkl.gz"
        if name.endswith(suffix):
            campaigns.append(name[: -len(suffix)])
    return campaigns


# ---------------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------------

def load_raw_tweet_data(campaign: str, data_dir: str, file_suffix: str) -> pd.DataFrame:
    path = f"{data_dir}/{campaign}_{file_suffix}.pkl.gz"
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return pd.DataFrame()
    df = pd.read_pickle(path)
    df['userid'] = df['userid'].astype(str)
    return df


def get_user_labels(df_raw: pd.DataFrame) -> dict:
    if 'data_type' not in df_raw.columns:
        print("  [warning] 'data_type' column not found -- no labels available.")
        return {}
    label_counts = (
        df_raw.groupby(['userid', 'data_type']).size().reset_index(name='n')
    )
    labels = {}
    for userid, group in label_counts.groupby('userid'):
        labels[userid] = 'io' if 'io' in group['data_type'].values else group['data_type'].iloc[0]
    n_io = sum(v == 'io' for v in labels.values())
    n_control = sum(v == 'control' for v in labels.values())
    print(f"  Labels extracted from raw data: {n_io} io, {n_control} control")
    return labels


def load_original_similarity(campaign: str, indicator: str, similarity_dir: str, file_suffix: str):
    cfg = INDICATOR_CONFIG[indicator]
    path = f"{similarity_dir}/{campaign}_{file_suffix}_{cfg['suffix']}.pkl.gz"
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    df_sim = pd.read_pickle(path)
    expected_cols = {'user_i', 'user_j', 'cosine_similarity'}
    if not expected_cols.issubset(df_sim.columns):
        raise ValueError(f"Unexpected columns in {path}: {df_sim.columns.tolist()}")
    df_sim['user_i'] = df_sim['user_i'].astype(str)
    df_sim['user_j'] = df_sim['user_j'].astype(str)
    return df_sim


def _to_hashable(x):
    if isinstance(x, dict):
        return str(
            x.get('text') or x.get('url') or x.get('expanded_url')
            or x.get('display_url') or next(iter(x.values()), x)
        )
    return str(x)


def filter_min_user_freq(df: pd.DataFrame, column: str,
                          min_user: int = 2, min_freq: int = 10) -> pd.DataFrame:
    stats = (
        df.groupby(column)
        .agg(n_users=('userid', 'nunique'), n_freq=('userid', 'size'))
        .reset_index()
    )
    valid_values = stats.loc[
        (stats['n_users'] >= min_user) & (stats['n_freq'] >= min_freq), column
    ]
    return df[df[column].isin(valid_values)]


def prepare_indicator_df(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    cfg = INDICATOR_CONFIG[indicator]
    df_need = df.copy()

    if indicator == 'sync' and 'time_30min' not in df_need.columns:
        df_need['tweet_time_dt'] = pd.to_datetime(
            df_need['tweet_time'], utc=True, errors='coerce'
        )
        n_before = len(df_need)
        df_need = df_need.dropna(subset=['tweet_time_dt'])
        if len(df_need) != n_before:
            print(f"    [sync] dropped {n_before - len(df_need)} rows with invalid timestamps")
        df_need['time_bin'] = df_need['tweet_time_dt'].dt.floor('30min')
        df_need['time_30min'] = df_need['time_bin'].dt.strftime('%Y-%m-%d_%H:%M')

    if cfg['filter_retweet']:
        if 'is_retweet' in df_need.columns:
            df_need = df_need.loc[df_need['is_retweet'] == True]
        df_need = df_need.loc[~df_need[cfg['column']].isnull()]
        df_need[cfg['column']] = df_need[cfg['column']].astype(str)

    if cfg['explode']:
        df_need = df_need.loc[~df_need[cfg['column']].isnull()]
        df_need = df_need[df_need[cfg['column']].apply(
            lambda x: isinstance(x, list) and len(x) > 1
        )]
        df_need = df_need.explode(cfg['column'])
        df_need = df_need.loc[~df_need[cfg['column']].isnull()]
        n_dicts = df_need[cfg['column']].apply(lambda x: isinstance(x, dict)).sum()
        if n_dicts > 0:
            print(f"    [{indicator}] normalizing {n_dicts} dict-valued entries "
                  f"in '{cfg['column']}' to strings")
        df_need[cfg['column']] = df_need[cfg['column']].apply(_to_hashable)

    df_need = filter_min_user_freq(df_need, cfg['column'], min_user=2, min_freq=10)
    if len(df_need) == 0:
        print('No enough data')
    print(f"    {indicator}: {len(df_need)} rows, {df_need['userid'].nunique()} users")
    return df_need


# ---------------------------------------------------------------------------
# FAST permutation engine
# ---------------------------------------------------------------------------

def _build_fixed_structures(df_need: pd.DataFrame, df_sim: pd.DataFrame, shuffle_column: str):
    """
    Compute everything that does NOT change across permutations, exactly
    once. This is the whole trick: user/value identity is invariant under
    shuffling, only the row->value assignment is permuted.
    """
    users_sorted = sorted(df_need['userid'].unique())
    values_sorted = sorted(df_need[shuffle_column].unique())
    user_idx = {u: i for i, u in enumerate(users_sorted)}
    value_idx = {v: i for i, v in enumerate(values_sorted)}
    n_users = len(users_sorted)
    n_values = len(values_sorted)

    row_user_codes = df_need['userid'].map(user_idx).to_numpy(dtype=np.int32)
    row_value_codes = df_need[shuffle_column].map(value_idx).to_numpy(dtype=np.int32)

    
    ui_codes = df_sim['user_i'].map(user_idx)
    uj_codes = df_sim['user_j'].map(user_idx)
    valid_mask = ui_codes.notna().to_numpy() & uj_codes.notna().to_numpy()

    idx_i_all = ui_codes.fillna(-1).to_numpy(dtype=np.int64)
    idx_j_all = uj_codes.fillna(-1).to_numpy(dtype=np.int64)
    orig_sim_all = df_sim['cosine_similarity'].to_numpy(dtype=np.float64)

    n_missing = int((~valid_mask).sum())
    if n_missing:
        print(f"    [info] {n_missing}/{len(df_sim)} pairs reference users outside "
              f"the active universe -- their similarity is 0.0 on every permutation "
              f"by construction, so their contribution is computed once, not "
              f"{NUM_OF_TIMES} times.")

    return {
        'n_users': n_users,
        'n_values': n_values,
        'row_user_codes': row_user_codes,
        'row_value_codes': row_value_codes,
        'idx_i_valid': idx_i_all[valid_mask],
        'idx_j_valid': idx_j_all[valid_mask],
        'orig_sim_valid': orig_sim_all[valid_mask],
        'valid_mask': valid_mask,
        'orig_sim_all': orig_sim_all,
    }


def _similarity_for_pairs(row_user_codes, value_codes, n_users, n_values, idx_i, idx_j):
    """
    One permutation's similarity, computed ONLY for the needed pairs.

    Uses sparse-sparse matmul + sparse PAIRED indexing (mat[idx_i, idx_j]),
    not row-gathering, to avoid materializing a dense n_users x n_users
    matrix or duplicating rows per pair.
    """
    n_rows = len(row_user_codes)
    mat = coo_matrix(
        (np.ones(n_rows, dtype=np.float64), (row_user_codes, value_codes)),
        shape=(n_users, n_values),
    ).tocsr()  # duplicate (row, col) entries are summed automatically here

    users_per_value = np.asarray((mat > 0).sum(axis=0)).flatten()
    idf_weights = np.log(n_users / np.maximum(users_per_value, 1))
    mat = mat.multiply(idf_weights).tocsr()

    row_norms = np.sqrt(np.asarray(mat.power(2).sum(axis=1)).flatten())
    row_norms[row_norms == 0] = 1.0
    normalized = mat.multiply(1.0 / row_norms[:, None]).tocsr()

    sim_full = (normalized @ normalized.T).tocsr()
    sim_vals = np.asarray(sim_full[idx_i, idx_j]).flatten()

    del mat, normalized, sim_full
    return sim_vals


_W = {}


def _init_worker(row_user_codes, n_users, n_values, idx_i, idx_j, orig_sim):
    _W['row_user_codes'] = row_user_codes
    _W['n_users'] = n_users
    _W['n_values'] = n_values
    _W['idx_i'] = idx_i
    _W['idx_j'] = idx_j
    _W['orig_sim'] = orig_sim


try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")

    def _release_memory_to_os():
        gc.collect()
        _libc.malloc_trim(0)
except (OSError, AttributeError):
    def _release_memory_to_os():
        gc.collect()


def _permutation_batch(seed_range):
    """
    Run a BATCH of permutations inside one worker and return a single
    summed int array -- not one dict per permutation.
    """
    seed_start, seed_end = seed_range
    counts = np.zeros(len(_W['orig_sim']), dtype=np.int64)
    row_user_codes = _W['row_user_codes']
    n_users = _W['n_users']
    n_values = _W['n_values']
    idx_i = _W['idx_i']
    idx_j = _W['idx_j']
    orig_sim = _W['orig_sim']

    for seed in range(seed_start, seed_end):
        rng = np.random.default_rng(seed)
        shuffled_values = rng.permutation(_W.get('row_value_codes_template'))
        sim_vals = _similarity_for_pairs(
            row_user_codes, shuffled_values, n_users, n_values, idx_i, idx_j
        )
        counts += (sim_vals >= orig_sim).astype(np.int64)

        del sim_vals, shuffled_values, rng
        _release_memory_to_os()

    return counts


def _init_worker_full(row_user_codes, row_value_codes_template, n_users, n_values, idx_i, idx_j, orig_sim):
    _init_worker(row_user_codes, n_users, n_values, idx_i, idx_j, orig_sim)
    _W['row_value_codes_template'] = row_value_codes_template


def _chunk_ranges(total: int, n_chunks: int):
    """Split [0, total) into n_chunks roughly-equal contiguous ranges."""
    n_chunks = max(1, min(n_chunks, total))
    base = total // n_chunks
    rem = total % n_chunks
    ranges = []
    start = 0
    for k in range(n_chunks):
        size = base + (1 if k < rem else 0)
        if size == 0:
            continue
        ranges.append((start, start + size))
        start += size
    return ranges


def run_null_model(df_need: pd.DataFrame,
                    df_sim: pd.DataFrame,
                    shuffle_column: str,
                    num_of_times: int = NUM_OF_TIMES,
                    max_workers: int = DEFAULT_INNER_WORKERS,
                    use_tfidf_method: bool = False) -> pd.DataFrame:
    """
    Same statistical result and output schema as the original
    run_null_model. Internally: arrays instead of dicts, batched workers
    instead of one task per permutation, sparse similarity instead of
    dense.
    """
    if use_tfidf_method:
        raise NotImplementedError(
            "The TF-IDF / 'sync' path isn't vectorized in this rewrite (it's "
            "currently disabled in INDICATOR_CONFIG). It needs the same "
            "treatment -- batching + array-based aggregation -- before it's "
            "fast at scale. Ask if you re-enable 'sync' and want that done too."
        )

    fx = _build_fixed_structures(df_need, df_sim, shuffle_column)

    n_valid = len(fx['orig_sim_valid'])
    print(f"    Pairs needing per-permutation computation: {n_valid} "
          f"(of {len(df_sim)} total)")

    # Benchmark ONE permutation before launching the full run.
    if n_valid > 0:
        t0 = time.perf_counter()
        _ = _similarity_for_pairs(
            fx['row_user_codes'],
            np.random.default_rng(0).permutation(fx['row_value_codes']),
            fx['n_users'], fx['n_values'], fx['idx_i_valid'], fx['idx_j_valid'],
        )
        per_perm = time.perf_counter() - t0
        est_serial = per_perm * num_of_times
        est_parallel = est_serial / max(1, max_workers)
        print(f"    Benchmark: one permutation took {per_perm:.2f}s on this core -> "
              f"~{est_serial:.0f}s if run serially, ~{est_parallel:.0f}s estimated "
              f"across {max_workers} workers (plus pool startup/IPC overhead).")

    chunks = _chunk_ranges(num_of_times, max_workers * 4)
    print(f"    Running {num_of_times} permutations across {len(chunks)} batches "
          f"on {max_workers} workers (~{num_of_times // max(1, len(chunks))} perms/batch)")

    total_counts = np.zeros(n_valid, dtype=np.int64)

    if n_valid > 0:
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker_full,
                initargs=(
                    fx['row_user_codes'],
                    fx['row_value_codes'],
                    fx['n_users'],
                    fx['n_values'],
                    fx['idx_i_valid'],
                    fx['idx_j_valid'],
                    fx['orig_sim_valid'],
                ),
            ) as executor:
                futures = {executor.submit(_permutation_batch, r): r for r in chunks}
                for fut in tqdm(as_completed(futures), total=len(futures),
                                 desc=f"Shuffling ({shuffle_column})"):
                    total_counts += fut.result()
        except KeyboardInterrupt:
            print("\n  [interrupted] Stopped by user. Check the benchmark line above "
                  "before assuming it was frozen -- with many pairs this can "
                  "legitimately take a while per batch.")
            raise
        except BrokenProcessPool:
            print("\n  [error] A worker died unexpectedly (often an out-of-memory kill "
                  "on HPC nodes, or a propagated Ctrl-C). Check your job's memory limit "
                  "vs (inner_workers x peak per-worker memory) -- reducing inner_workers "
                  "or the chunk size is the usual fix.")
            raise

    invalid_contrib = np.where(fx['orig_sim_all'] <= 0, num_of_times, 0).astype(np.int64)

    full_counts = invalid_contrib.copy()
    full_counts[fx['valid_mask']] = total_counts

    df_sim = df_sim.copy()
    df_sim['user_i'] = df_sim['user_i'].astype(str)
    df_sim['user_j'] = df_sim['user_j'].astype(str)
    df_sim['pvalue'] = (full_counts + 1) / (num_of_times + 1)

    return df_sim


# ---------------------------------------------------------------------------
# Per-campaign pipeline / job runner
# ---------------------------------------------------------------------------

def run_campaign_indicator(campaign: str, indicator: str,
                            data_dir: str, similarity_dir: str, results_dir: str, file_suffix: str,
                            num_of_times: int = NUM_OF_TIMES,
                            inner_workers: int = DEFAULT_INNER_WORKERS):
    print(f"\n  -- {campaign} / {indicator} --")

    df_sim = load_original_similarity(campaign, indicator, similarity_dir, file_suffix)
    if df_sim is None:
        print(f"  Skipping {campaign}/{indicator}: original similarity not found.")
        return

    df_raw = load_raw_tweet_data(campaign, data_dir, file_suffix)
    if df_raw.empty:
        print(f"  Skipping {campaign}/{indicator}: no raw tweet data found.")
        return

    cfg = INDICATOR_CONFIG[indicator]
    df_need = prepare_indicator_df(df_raw, indicator)

    if df_need.empty or df_need['userid'].nunique() < 2:
        print(f"  Skipping {campaign}/{indicator}: insufficient data after filtering.")
        return

    n_users_avail = df_need['userid'].nunique()
    if n_users_avail < MIN_USERS_FOR_NULL_MODEL:
        print(f"  [warning] {campaign}/{indicator}: only {n_users_avail} users -- skipping.")
        return

    print(f"    Loaded original similarity edge list: {len(df_sim)} pairs")
    print(f"    Running {num_of_times} permutations for p-value estimation...")
    try:
        df_sim_pval = run_null_model(
            df_need, df_sim, cfg['column'],
            num_of_times=num_of_times,
            max_workers=inner_workers,
            use_tfidf_method=cfg['use_tfidf_method'],
        )
    except NotImplementedError as exc:
        print(f"  [skipped] {campaign}/{indicator}: {exc}")
        return

    labels = get_user_labels(df_raw)
    df_sim_pval['data_type_i'] = df_sim_pval['user_i'].map(labels)
    df_sim_pval['data_type_j'] = df_sim_pval['user_j'].map(labels)

    os.makedirs(results_dir, exist_ok=True)
    edge_path = f"{results_dir}/{campaign}_{indicator}_cosine_similarity_pvalue_index.pkl.gz"
    df_sim_pval.to_pickle(edge_path)

    print(f"    Saved: {edge_path}")
    print(f"    Significant edges (p<0.05): {(df_sim_pval['pvalue'] < 0.05).sum()} / {len(df_sim_pval)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the vectorized null-model permutation pipeline "
    )
    parser.add_argument(
        '--campaigns', nargs='+', default=None,
        help="Campaign names to process. Default: auto-detect every "
             "'<campaign>_<file-suffix>.pkl.gz' found under --data-dir.",
    )
    parser.add_argument(
        '--indicators', nargs='+', default=None, choices=list(INDICATOR_CONFIG),
        help="Subset of indicators to run (default: all of %s)." % list(INDICATOR_CONFIG),
    )
    parser.add_argument('--data-dir', default=DATA_DIR,
                         help=f"Directory with raw filtered tweet pkl.gz files (default: {DATA_DIR})")
    parser.add_argument('--similarity-dir', default=SIMILARITY_DIR,
                         help=f"Directory to read pre-computed cosine-similarity edge lists from "
                              f"(default: {SIMILARITY_DIR})")
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                         help=f"Directory to write this script's own p-value results to "
                              f"(default: {RESULTS_DIR})")
    parser.add_argument('--file-suffix', default=FILE_SUFFIX,
                         help=f"Filename suffix identifying the filtering run (default: {FILE_SUFFIX})")
    parser.add_argument('--num-permutations', type=int, default=NUM_OF_TIMES,
                         help=f"Number of shuffles for the null model (default: {NUM_OF_TIMES})")
    parser.add_argument('--workers', type=int, default=DEFAULT_INNER_WORKERS,
                         help=f"Worker processes per campaign/indicator (default: {DEFAULT_INNER_WORKERS}, "
                              f"detected from available CPUs)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Detected {_AVAILABLE_CPUS} CPUs -> using workers={args.workers}")

    campaigns = args.campaigns or discover_campaigns(args.data_dir, args.file_suffix)
    if not campaigns:
        print(f"No campaigns found under {args.data_dir} matching "
              f"'*_{args.file_suffix}.pkl.gz', and none given via --campaigns. Nothing to do.")
        return
    print(f"Campaigns: {campaigns}")

    indicators = args.indicators or list(INDICATOR_CONFIG)
    print(f"Indicators: {indicators}")

    for campaign in campaigns:
        for indicator in indicators:
            run_campaign_indicator(
                campaign, indicator,
                data_dir=args.data_dir,
                similarity_dir=args.similarity_dir,
                results_dir=args.results_dir,
                file_suffix=args.file_suffix,
                num_of_times=args.num_permutations,
                inner_workers=args.workers,
            )

    print(f"\nAll campaigns processed. Results saved directly under: {args.results_dir}/")


if __name__ == '__main__':
    main()
