"""
Coordination Detection Pipeline (load -> filter -> similarity)
==============================================================
One command that, for every campaign found in the source IO/control data:

  1. LOAD    = read the campaign's IO + control tweet files
  2. FILTER  = drop low-activity accounts/apps, label rows 'io'/'control',
               and save one combined file per campaign to ./data/filtered
  3. SIMILARITY = compute pairwise user-user coordination signals:
               tweet_client_name, retweet_tweetid, retweet_userid,
               hashtags, urls (TF-IDF cosine), plus temporal co-occurrence
               (Pacheco et al. 2021), saving each signal's edge list to
               ./results/similarity

Usage:
    python filtering/pipeline.py
    python filtering/pipeline.py --campaigns ira IRA_202012
    python filtering/pipeline.py --data-dir ./data/raw \
        --out-dir ./data/filtered --similarity-dir ./results/similarity --bin-minutes 30
"""

import argparse
import ast
import gc
import glob
import logging
import os
import re
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "./data/raw"
DEFAULT_OUT_DIR = "./data/filtered"
DEFAULT_SIMILARITY_DIR = "./results/similarity"

REQUIRED_COLS = [
    "tweet_client_name", "tweetid", "userid",
    "tweet_text", "urls", "hashtags", "retweet_tweetid",
    "retweet_userid", "is_retweet", "tweet_time",
]

CONTENT_SIGNALS = [
    "client_name",
    "retweet_tweetid",
    "retweet_userid",
    "hashtags",
    "urls"
    ]


# =============================================================================
# STEP 1: LOAD
# =============================================================================

def read_ops_control_data(ops_file_path, 
                          control_file_path, 
                          includes=('ops', 'control')
                          ) -> dict:
    """Read influence-operation (ops) and control tweet files."""
    data = {}

    if 'ops' in includes:
        data['ops'] = pd.read_pickle(ops_file_path)

    if 'control' in includes:
        if not os.path.isfile(control_file_path):
            data['control'] = None
        else:
            data['control'] = pd.read_pickle(control_file_path)

    return data


def extract_campaign_name(file_path: str) -> str:
    """Derive the campaign name from either an IO or control file path."""
    name = os.path.basename(file_path)
    for suffix in ("_tweets_io.pkl.gz", "_tweets_control.pkl.gz"):
        name = name.replace(suffix, "")
    return name


def load_single_campaign(
    io_path: str,
    control_path: Optional[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load IO and control data for a single campaign. 
    Either may come back None.
    """
    includes = ["ops", "control"] if control_path else ["ops"]
    result = read_ops_control_data(io_path, control_file_path=control_path, includes=includes)

    df_ops = result.get("ops")
    if df_ops is None or df_ops.empty or "tweet_client_name" not in df_ops.columns:
        logger.debug("Skipping IO '%s' — missing or invalid.", io_path)
        df_ops = None
    else:
        df_ops = df_ops[REQUIRED_COLS].copy()

    df_control = result.get("control")
    if df_control is None or df_control.empty or "tweet_client_name" not in df_control.columns:
        logger.debug("Skipping control for '%s' — missing or invalid.", io_path)
        df_control = None
    else:
        df_control["retweet_userid"] = df_control["retweeted_user_id"]
        control_cols = [c for c in REQUIRED_COLS if c in df_control.columns]
        df_control = df_control[control_cols].copy()

    return df_ops, df_control


def discover_campaigns(data_dir: str, campaigns: Optional[list] = None) -> list:
    """Find IO file paths for the requested campaigns, or all campaigns if none given."""
    if campaigns:
        io_files = []
        for camp in campaigns:
            matches = glob.glob(f"{data_dir}/io/{camp}_tweets_io.pkl.gz")
            if not matches:
                logger.warning("No IO file found for campaign '%s'.", camp)
            io_files.extend(matches)
        return sorted(io_files)

    return sorted(glob.glob(f"{data_dir}/io/*_tweets_io.pkl.gz"))


# =============================================================================
# STEP 2: FILTER
# =============================================================================

def filter_by_accounts_per_app(df: pd.DataFrame, min_accounts: int, max_accounts: int) -> pd.DataFrame:
    """Keep only apps whose unique user count falls within [min_accounts, max_accounts]."""
    if min_accounts == 0 and max_accounts == 0:
        return df

    counts = df.groupby("tweet_client_name")["userid"].nunique().rename("count").reset_index()
    mask = counts["count"] >= min_accounts
    if max_accounts > 0:
        mask &= counts["count"] <= max_accounts

    valid_apps = counts.loc[mask, "tweet_client_name"]
    return df[df["tweet_client_name"].isin(valid_apps)]


def filter_by_tweets_per_app(df: pd.DataFrame, min_tweets: int) -> pd.DataFrame:
    """Keep only apps with at least `min_tweets` unique tweets."""
    if min_tweets == 0:
        return df
    counts = df.groupby("tweet_client_name")["tweetid"].nunique().rename("count").reset_index()
    valid_apps = counts.loc[counts["count"] >= min_tweets, "tweet_client_name"]
    return df[df["tweet_client_name"].isin(valid_apps)]


def filter_by_tweets_per_account(df: pd.DataFrame, min_tweets: int) -> pd.DataFrame:
    """Keep only accounts with at least `min_tweets` unique tweets."""
    if min_tweets == 0:
        return df
    counts = df.groupby("userid")["tweetid"].nunique().rename("count").reset_index()
    valid_accounts = counts.loc[counts["count"] >= min_tweets, "userid"]
    return df[df["userid"].isin(valid_accounts)]


def apply_filters(
    df: pd.DataFrame,
    min_tweets_app: int,
    min_tweets_account: int,
    min_accounts: int,
    max_accounts: int,
) -> pd.DataFrame:
    """
    Apply filters in order:
      1. Tweets per account  (remove low-activity users first)
      2. Accounts per app    (structural filter on remaining active users)
      3. Tweets per app      (volume filter after valid users/apps are known)
    """
    if df.empty:
        return df

    df_f = filter_by_tweets_per_account(df, min_tweets_account)
    df_f = filter_by_accounts_per_app(df_f, min_accounts, max_accounts)
    df_f = filter_by_tweets_per_app(df_f, min_tweets_app)
    return df_f


def build_filter_suffix(t_app: int, t_acc: int, min_acc: int, max_acc: int) -> str:
    """Descriptive suffix tag for a filter-threshold combination (no campaign name)."""
    return f"tpa{t_app}_tpu{t_acc}_acc{min_acc}-{max_acc}"


def filter_and_save_campaign(
    campaign: str,
    df_ops: Optional[pd.DataFrame],
    df_control: Optional[pd.DataFrame],
    out_dir: str,
    min_tweets_app: int,
    min_tweets_account: int,
    min_accounts: int,
    max_accounts: int,
) -> Optional[str]:
    """
    Filter IO and control independently, label rows 'io'/'control', concat,
    and save one combined file for this campaign. Returns the filter-threshold
    suffix tag (no campaign name) used for all downstream outputs, or
    None if nothing survived filtering.
    """
    parts = []

    for label, df in [("io", df_ops), ("control", df_control)]:
        if df is None or df.empty:
            logger.warning("[%s] '%s' data is empty — skipping.", campaign, label)
            continue

        df_f = apply_filters(df, min_tweets_app, min_tweets_account, min_accounts, max_accounts)

        if df_f.empty:
            logger.warning(
                "[%s] '%s' is empty after filtering (tpa=%d, tpu=%d, acc=%d-%d).",
                campaign, label, min_tweets_app, min_tweets_account, min_accounts, max_accounts,
            )
            continue

        df_f = df_f.copy()
        df_f["data_type"] = label
        parts.append(df_f)

        logger.info(
            "[%s] '%s' after filter → %d rows, %d unique users.",
            campaign, label, len(df_f), df_f["userid"].nunique(),
        )

    if not parts:
        logger.warning("[%s] No data survived filtering — nothing saved.", campaign)
        return None

    combined = pd.concat(parts, ignore_index=True)
    col_order = ["campaign", "data_type"] + [c for c in combined.columns if c not in ("campaign", "data_type")]
    combined = combined[col_order]

    file_suffix = build_filter_suffix(min_tweets_app, min_tweets_account, min_accounts, max_accounts)
    out_path = os.path.join(out_dir, f"{campaign}_{file_suffix}.pkl.gz")
    combined.to_pickle(out_path)

    logger.info(
        "[%s] Saved '%s' → %d total rows (%s).",
        campaign, out_path, len(combined),
        ", ".join(f"{p['data_type'].iloc[0]}: {len(p)}" for p in parts),
    )

    del combined, parts
    gc.collect()
    return file_suffix


# =============================================================================
# STEP 3 — SIMILARITY (content-based TF-IDF signals)
# =============================================================================

def tfidf_similarity(df: pd.DataFrame, camp: str, column: str, threshold: float = 0.0) -> pd.DataFrame:
    """Compute pairwise cosine similarity and return upper-triangle pairs DataFrame."""
    embeddings = df['embedding'].iloc[0]
    matrix = vstack(df['embedding'].tolist()) if issparse(embeddings) else np.vstack(df['embedding'].tolist())

    similarities = cosine_similarity(matrix)
    idx_i, idx_j = np.triu_indices(similarities.shape[0], k=1)
    sim_values = similarities[idx_i, idx_j]

    if threshold > 0.0:
        mask = sim_values >= threshold
        idx_i, idx_j, sim_values = idx_i[mask], idx_j[mask], sim_values[mask]

    labels = df[column].to_numpy()
    return pd.DataFrame({
        'user_i': labels[idx_i],
        'user_j': labels[idx_j],
        'cosine_similarity': sim_values,
        'campaign': camp,
    })


def similarity_graph(df, group_by_column, 
                     list_column, camp, 
                     idf=True, 
                     threshold=0.0, 
                     save_path=None
                     ):
    """Group rows by user, TF-IDF vectorize their joined tokens, and compute pairwise similarity."""
    df = df.copy()
    df[list_column] = df[list_column].astype(str)

    delim = '|||'
    df_grp = df.groupby(group_by_column)[list_column].apply(lambda x: delim.join(x)).reset_index()

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(delim),
        lowercase=False,
        use_idf=idf,
        token_pattern=None,
    )
    sparse_matrix = vectorizer.fit_transform(df_grp[list_column])
    df_grp['embedding'] = [sparse_matrix[i] for i in range(sparse_matrix.shape[0])]

    pairs_df = tfidf_similarity(df_grp, camp, column=group_by_column, threshold=threshold)
    pairs_df['idf'] = idf
    pairs_df['signal'] = list_column
    pairs_df = pairs_df[pairs_df['cosine_similarity'] > 0]

    if save_path:
        pairs_df.to_pickle(save_path)
    return pairs_df


def filter_count(df, column, min_user=2, min_freq=10):
    """
    Keep only values of `column` shared by >= min_user users,
    appearing >= min_freq times.
    """
    df_user_grp = (df.groupby([column])['userid'].nunique().to_frame('count').reset_index()
                   .query(f'count >= {min_user}'))
    df_size_grp = (df.groupby([column])['tweetid'].nunique().to_frame('count').reset_index()
                   .query(f'count >= {min_freq}'))
    df_bipartite = df_user_grp[[column]].merge(df_size_grp[[column]], on=column)
    return df.loc[df[column].isin(df_bipartite[column])]


def co_retweet(df, column, min_user=2, min_freq=10):
    df_retweet = df.loc[df['is_retweet'] == True]
    return filter_count(df_retweet, column, min_user=min_user, min_freq=min_freq)


def co_hashtag(df, hashtag_column='hashtags', min_user=2, min_freq=10):
    df_explode = df.explode(hashtag_column)
    df_user_grp = (df_explode.groupby([hashtag_column])['userid'].nunique().to_frame('count')
                   .reset_index().query(f'count >= {min_user}'))
    df_size_grp = (df_explode.groupby([hashtag_column])['tweetid'].nunique().to_frame('count')
                   .reset_index().query(f'count >= {min_freq}'))
    df_bipartite = df_user_grp[[hashtag_column]].merge(df_size_grp[[hashtag_column]], on=hashtag_column)
    return df_explode.loc[df_explode[hashtag_column].isin(df_bipartite[hashtag_column])]


def co_url(df, url_column='urls', min_user=2, min_freq=10):
    df = df.copy()
    df['url_len'] = df[url_column].apply(len)
    df_urls = df.loc[df['url_len'] > 0].explode(url_column)
    if len(df_urls) == 0:
        return df_urls
    return filter_count(df_urls, url_column, min_user=min_user, min_freq=min_freq)


def extract_urls(val) -> list:
    """Normalize the url column — handles dict, list of dicts, list of str, str, NaN."""
    if isinstance(val, dict):
        return [val.get('expanded_url') or val.get('url')]
    if isinstance(val, list):
        return [(v.get('expanded_url') or v.get('url')) if isinstance(v, dict) else v for v in val]
    if isinstance(val, str):
        return [val]
    return []


def extract_hashtags(val) -> list:
    """Normalize the hashtags column — handles all observed formats."""
    if val is None or isinstance(val, float):
        return []
    if isinstance(val, list):
        if all(isinstance(v, dict) for v in val if v):
            return [v['text'] for v in val if isinstance(v, dict) and 'text' in v]
        return [str(v) for v in val if v]
    if isinstance(val, dict):
        return [val['text']] if 'text' in val else []
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [v['text'] if isinstance(v, dict) and 'text' in v else str(v) for v in parsed if v]
            if isinstance(parsed, dict):
                return [parsed['text']] if 'text' in parsed else []
        except (ValueError, SyntaxError):
            pass
        found = re.findall(r"'text':\s*'([^']*)'", val)
        if found:
            return found
        return [val.strip()] if val.strip() else []
    return []


def run_content_signal(
    signal: str,
    df: pd.DataFrame,
    similarity_dir: str,
    campaign: str,
    file_suffix: str,
    idf: bool,
) -> pd.DataFrame:
    """
    Dispatch to the right pre-filter/grouping
    for one of the 5 content-based signals, compute pairwise similarity,
    and save the edge list. Returns the (possibly empty) pairs DataFrame.
    """
    logger.info("=== Running content signal: %s ===", signal)
    save_path = f"{similarity_dir}/{campaign}_{file_suffix}_similarity_{signal}_all.pkl.gz"

    if signal == "client_name":
        app_users = df.groupby('tweet_client_name')['userid'].nunique()
        app_activity = df.groupby('tweet_client_name')['tweet_client_name'].count()
        valid_apps = app_users[(app_users >= 2) & (app_activity >= 100)].index
        df_sig = df[df['tweet_client_name'].isin(valid_apps)]
        group_col, list_col = 'userid', 'tweet_client_name'

    elif signal == "retweet_tweetid":
        df_sig = df[df['is_retweet'] == True].copy()
        df_sig['retweet_tweetid'] = pd.to_numeric(df_sig['retweet_tweetid'], errors='coerce')
        df_sig['retweet_tweetid'] = df_sig['retweet_tweetid'].astype('Int64').astype(str)
        df_sig = co_retweet(df_sig, 'retweet_tweetid', min_user=2, min_freq=10)
        group_col, list_col = 'userid', 'retweet_tweetid'

    elif signal == "retweet_userid":
        df_sig = co_retweet(df, 'retweet_userid', min_user=2, min_freq=10).copy()
        df_sig['retweet_userid'] = df_sig['retweet_userid'].astype(str)
        group_col, list_col = 'userid', 'retweet_userid'

    elif signal == "hashtags":
        df_sig = df.copy()
        df_sig['hashtags'] = df_sig['hashtags'].apply(extract_hashtags)
        df_sig = df_sig.explode('hashtags')
        df_sig['hashtags'] = df_sig['hashtags'].apply(lambda x: str(x) if not isinstance(x, str) else x)
        df_sig = df_sig[df_sig['hashtags'].notna() & ~df_sig['hashtags'].isin(('', 'None', 'nan'))]
        df_sig = co_hashtag(df_sig, hashtag_column='hashtags', min_user=2, min_freq=10)
        group_col, list_col = 'userid', 'hashtags'

    elif signal == "urls":
        df_sig = df.copy()
        df_sig['urls'] = df_sig['urls'].apply(extract_urls)
        df_sig = df_sig[df_sig['urls'].apply(len) > 0]
        df_sig = co_url(df_sig, url_column='urls', min_user=2, min_freq=10)
        group_col, list_col = 'userid', 'urls'

    else:
        raise ValueError(f"Unknown content signal: {signal}")

    if df_sig is None or df_sig.empty:
        logger.warning("[%s] No data left to compute similarity — skipping.", signal)
        return pd.DataFrame()

    pairs = similarity_graph(df_sig, group_col, list_col, campaign, idf=idf, save_path=save_path)
    if pairs.empty:
        logger.warning("[%s] No nonzero similarity pairs.", signal)
    else:
        logger.info("[%s] Saved %d similarity pairs → '%s'", signal, len(pairs), save_path)

    gc.collect()
    return pairs


# =============================================================================
# STEP 3b — SIMILARITY (temporal co-occurrence, Pacheco et al. 2021)
# =============================================================================

def bin_tweets(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    """Floor each tweet's timestamp to a `bin_minutes` window."""
    df = df.copy()
    df['tweet_time_dt'] = pd.to_datetime(df['tweet_time'], utc=True, errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['tweet_time_dt'])
    if n_before != len(df):
        logger.warning("  Dropped %d rows with invalid timestamps", n_before - len(df))

    df['time_bin'] = df['tweet_time_dt'].dt.floor(f'{bin_minutes}min')
    df['time_bin_str'] = df['time_bin'].dt.strftime('%Y-%m-%d_%H:%M')
    logger.info("  Binned %d tweets into %d time windows (%d-min intervals)",
                len(df), df['time_bin_str'].nunique(), bin_minutes)
    return df


def filter_users_by_activity(df: pd.DataFrame, min_tweets: int) -> pd.DataFrame:
    """Keep only users with at least `min_tweets` tweets (for the temporal signal)."""
    user_counts = df.groupby('userid').size()
    active_users = user_counts[user_counts >= min_tweets].index
    df_filtered = df[df['userid'].isin(active_users)]
    logger.info("  Temporal filter: users %d → %d (min_tweets=%d)",
                df['userid'].nunique(), df_filtered['userid'].nunique(), min_tweets)
    return df_filtered


def build_temporal_tfidf_matrix(df: pd.DataFrame):
    """Build a user × time-bin count matrix, then TF-IDF weight it."""
    df_counts = df.groupby(['userid', 'time_bin_str']).size().reset_index(name='count')

    users = sorted(df_counts['userid'].unique())
    bins = sorted(df_counts['time_bin_str'].unique())
    user_idx = {u: i for i, u in enumerate(users)}
    bin_idx = {b: i for i, b in enumerate(bins)}

    rows = df_counts['userid'].map(user_idx).values
    cols = df_counts['time_bin_str'].map(bin_idx).values
    vals = df_counts['count'].values

    raw_matrix = csr_matrix((vals, (rows, cols)), shape=(len(users), len(bins)))
    tfidf_matrix = TfidfTransformer().fit_transform(raw_matrix)

    logger.info("  Temporal TF-IDF matrix: %d users × %d time bins", len(users), len(bins))
    return tfidf_matrix, users, bins


def run_temporal_signal(
    df: pd.DataFrame,
    similarity_dir: str,
    campaign: str,
    file_suffix: str,
    bin_minutes: int,
    min_tweets: int,
) -> pd.DataFrame:
    """
    Temporal co-occurrence signal: users who tweet in the same time windows.
    Computes pairwise similarity and saves the edge list. Returns the
    (possibly empty) pairs DataFrame.
    """
    logger.info("=== Running temporal signal (bin=%dmin, min_tweets=%d) ===", bin_minutes, min_tweets)

    df_t = bin_tweets(df, bin_minutes)
    df_t = filter_users_by_activity(df_t, min_tweets)

    if df_t['userid'].nunique() < 2:
        logger.warning("[temporal] Not enough users after filtering — skipping.")
        return pd.DataFrame()

    tfidf_matrix, users, bins = build_temporal_tfidf_matrix(df_t)

    t0 = time.time()
    sim_matrix = cosine_similarity(tfidf_matrix)
    rows_i, cols_j = np.triu_indices(len(users), k=1)
    sims = sim_matrix[rows_i, cols_j]
    mask = sims > 0
    pairs = pd.DataFrame({
        'user_i': np.array(users)[rows_i[mask]],
        'user_j': np.array(users)[cols_j[mask]],
        'cosine_similarity': sims[mask],
        'signal': 'temporal',
        'campaign': campaign,
    })
    logger.info("  Temporal similarity: %d nonzero pairs (%.1fs)", len(pairs), time.time() - t0)

    save_path = f"{similarity_dir}/{campaign}_{file_suffix}_similarity_temporal_{bin_minutes}min_all.pkl.gz"
    pairs.to_pickle(save_path)
    logger.info("[temporal] Saved %d similarity pairs → '%s'", len(pairs), save_path)

    gc.collect()
    return pairs


# =============================================================================
# Per-campaign orchestration
# =============================================================================

def process_campaign(io_path: str, args) -> bool:
    """
    Run load → filter → similarity for a single campaign.
    Returns True if similarity results were produced, False otherwise.
    """
    campaign = extract_campaign_name(io_path)
    logger.info("--- Processing campaign: '%s' ---", campaign)

    control_path = os.path.join(args.data_dir, 'control', f"{campaign}_tweets_control.pkl.gz")
    if not os.path.exists(control_path):
        logger.warning("[%s] No control file found.", campaign)
        control_path = None

    df_ops, df_control = load_single_campaign(io_path, control_path)
    if df_ops is not None:
        df_ops["campaign"] = campaign
    if df_control is not None:
        df_control["campaign"] = campaign

    file_suffix = filter_and_save_campaign(
        campaign, df_ops, df_control, args.out_dir,
        min_tweets_app=args.min_tweets_app,
        min_tweets_account=args.min_tweets_account,
        min_accounts=args.min_accounts,
        max_accounts=args.max_accounts,
    )
    del df_ops, df_control
    gc.collect()

    if file_suffix is None:
        return False

    df = pd.read_pickle(os.path.join(args.out_dir, f"{campaign}_{file_suffix}.pkl.gz"))
    df['userid'] = df['userid'].astype(str)
    total_io = df[df['data_type'] == 'io']['userid'].nunique()
    total_control = df[df['data_type'] == 'control']['userid'].nunique()
    logger.info("[%s] Filtered data: %d rows | IO users: %d | Control users: %d",
                campaign, len(df), total_io, total_control)

    produced_any = False

    if not args.skip_content:
        for signal in CONTENT_SIGNALS:
            pairs = run_content_signal(
                signal, df, args.similarity_dir, campaign, file_suffix, idf=args.idf,
            )
            produced_any = produced_any or not pairs.empty

    if not args.skip_temporal:
        pairs = run_temporal_signal(
            df, args.similarity_dir, campaign, file_suffix,
            bin_minutes=args.bin_minutes, min_tweets=args.min_tweets_temporal,
        )
        produced_any = produced_any or not pairs.empty

    del df
    gc.collect()
    return produced_any


# =============================================================================
# CLI / main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load, filter, and compute coordination-similarity signals for all campaigns in one run.",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing io/ and control/ subfolders of raw campaign data.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Directory to write filtered data (default: ./data/filtered).")
    parser.add_argument("--similarity-dir", type=str, default=DEFAULT_SIMILARITY_DIR,
                        help="Directory to write similarity results (default: ./results/similarity).")
    parser.add_argument("--campaigns", type=str, nargs="*", default=None,
                        help="Specific campaign names to run. Default: all campaigns found in --data-dir/io.")

    # Filter thresholds (defaults match the previously hardcoded filter grid)
    parser.add_argument("--min-tweets-app", type=int, default=0, help="Min unique tweets per app.")
    parser.add_argument("--min-tweets-account", type=int, default=10, help="Min unique tweets per account.")
    parser.add_argument("--min-accounts", type=int, default=0, help="Min unique accounts per app.")
    parser.add_argument("--max-accounts", type=int, default=0, help="Max unique accounts per app (0 = no cap).")

    # Similarity options
    parser.add_argument("--idf", type=bool, default=True, help="Apply IDF weighting in TF-IDF.")
    parser.add_argument("--bin-minutes", type=int, default=30, help="Temporal signal: time window size in minutes.")
    parser.add_argument("--min-tweets-temporal", type=int, default=8,
                        help="Temporal signal: min tweets per user to include.")
    parser.add_argument("--skip-content", action="store_true", help="Skip the 5 content-based TF-IDF signals.")
    parser.add_argument("--skip-temporal", action="store_true", help="Skip the temporal co-occurrence signal.")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.similarity_dir, exist_ok=True)

    io_files = discover_campaigns(args.data_dir, args.campaigns)
    logger.info("Found %d campaign(s) to process in '%s'.", len(io_files), args.data_dir)

    n_ok = 0
    for io_path in io_files:
        if process_campaign(io_path, args):
            n_ok += 1

    logger.info("Done. %d/%d campaign(s) produced similarity results in '%s'.",
                n_ok, len(io_files), args.similarity_dir)


if __name__ == "__main__":
    main()
