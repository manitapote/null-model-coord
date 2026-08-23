"""
Single entry point: filtering -> null_model -> min_pvalue_eval, per campaign.

Usage:
    python run_pipeline.py
    python run_pipeline.py --campaigns spain_082019_1 uae_082019_1
    python run_pipeline.py --skip-filtering                 # data/filtered + similarity already exist
    python run_pipeline.py --num-permutations 200 --workers 8
"""

import argparse
import glob
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

FILTERING_SCRIPT      = os.path.join(REPO_ROOT, 'filtering', 'pipeline.py')
NULL_MODEL_SCRIPT     = os.path.join(REPO_ROOT, 'null_model', 'pipeline.py')
MIN_PVALUE_EVAL_SCRIPT = os.path.join(REPO_ROOT, 'evaluation', 'min_pvalue_eval.py')

RAW_DATA_DIR       = os.path.join(REPO_ROOT, 'data', 'raw')             # filtering/pipeline.py's input: {raw}/io/, {raw}/control/
FILTERED_DIR       = os.path.join(REPO_ROOT, 'data', 'filtered')        # filtering output / null_model + eval input
SIMILARITY_DIR     = os.path.join(REPO_ROOT, 'results')                 # filtering output / null_model input -- see path note above
NULL_MODEL_DIR      = os.path.join(REPO_ROOT, 'results', 'null_model')       # null_model output / eval input
MIN_PVALUE_EVAL_DIR = os.path.join(REPO_ROOT, 'results', 'min_pvalue_eval')  # final output

FILE_SUFFIX = 'tpa0_tpu10_acc0-0'  # must match filtering/pipeline.py's default thresholds (tpa0, tpu10, acc0-0)


def discover_campaigns(raw_data_dir: str) -> list:
    """Auto-detect campaigns from data/raw/io/*_tweets_io.pkl.gz -- the
    earliest-stage source of truth, so this works even before filtering
    has ever been run (unlike each stage's own discover_campaigns, which
    looks at that stage's own INPUT and so can only see campaigns already
    processed by the stage before it)."""
    pattern = os.path.join(raw_data_dir, 'io', '*_tweets_io.pkl.gz')
    campaigns = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        campaigns.append(name[: -len('_tweets_io.pkl.gz')])
    return campaigns


def run_stage(label: str, script_path: str, args: list) -> bool:
    """Run one stage script as a subprocess. Returns True iff it exited 0.
    stdout/stderr are inherited (streamed live), not captured -- so you
    see each stage's own progress output in real time, same as running it
    directly."""
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"# {sys.executable} {script_path} {' '.join(args)}")
    print(f"{'#'*70}")

    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script_path] + args, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"# {label}: {status} ({elapsed:.1f}s)")
    return ok


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run filtering -> null_model -> min_pvalue_eval end-to-end, per campaign."
    )
    parser.add_argument(
        '--campaigns', nargs='+', default=None,
        help="Campaign names to process. Default: auto-detect every "
             "'<campaign>_tweets_io.pkl.gz' found under --raw-data-dir/io/.",
    )
    parser.add_argument('--raw-data-dir', default=RAW_DATA_DIR,
                         help=f"Directory with io/ and control/ subfolders of raw campaign "
                              f"data (default: {RAW_DATA_DIR})")
    parser.add_argument('--filtered-dir', default=FILTERED_DIR,
                         help=f"Directory for filtered per-campaign data (default: {FILTERED_DIR})")
    parser.add_argument('--similarity-dir', default=SIMILARITY_DIR,
                         help=f"Directory for pre-null-model similarity edge lists -- shared "
                              f"between the filtering and null_model stages, see the path note "
                              f"in this script's module docstring (default: {SIMILARITY_DIR})")
    parser.add_argument('--null-model-dir', default=NULL_MODEL_DIR,
                         help=f"Directory for null_model's p-value output (default: {NULL_MODEL_DIR})")
    parser.add_argument('--eval-dir', default=MIN_PVALUE_EVAL_DIR,
                         help=f"Directory for min_pvalue_eval's output (default: {MIN_PVALUE_EVAL_DIR})")
    parser.add_argument('--file-suffix', default=FILE_SUFFIX,
                         help=f"Filename suffix identifying the filtering run (default: {FILE_SUFFIX})")

    parser.add_argument('--skip-filtering', action='store_true',
                         help="Skip stage 1 -- use if data/filtered/ + --similarity-dir already exist.")
    parser.add_argument('--skip-null-model', action='store_true',
                         help="Skip stage 2 -- use if --null-model-dir already has this campaign's output.")
    parser.add_argument('--skip-eval', action='store_true',
                         help="Skip stage 3.")

    parser.add_argument('--num-permutations', type=int, default=None,
                         help="Passed through to null_model/pipeline.py's --num-permutations "
                              "(default: that script's own default, 1000).")
    parser.add_argument('--workers', type=int, default=None,
                         help="Passed through to null_model/pipeline.py's --workers.")
    parser.add_argument('--alpha', type=float, default=None,
                         help="Passed through to null_model/pipeline.py's --alpha correction "
                              "and min_pvalue_eval.py's --alpha (kept in sync across both).")
    parser.add_argument('--pauc-max-fpr', type=float, default=None,
                         help="Passed through to min_pvalue_eval.py's --pauc-max-fpr.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    campaigns = args.campaigns or discover_campaigns(args.raw_data_dir)
    if not campaigns:
        print(f"No campaigns found under {args.raw_data_dir}/io/, and none given via --campaigns. Nothing to do.")
        return
    print(f"Campaigns: {campaigns}")

    results = {}  # (campaign, stage) -> bool

    for campaign in campaigns:
        print(f"\n{'='*70}")
        print(f"CAMPAIGN: {campaign}")
        print(f"{'='*70}")

        if args.skip_filtering:
            results[(campaign, 'filtering')] = None
        else:
            filtering_args = [
                '--campaigns', campaign,
                '--data-dir', args.raw_data_dir,
                '--out-dir', args.filtered_dir,
                '--similarity-dir', args.similarity_dir,
            ]
            results[(campaign, 'filtering')] = run_stage(
                f"[{campaign}] Stage 1/3: filtering", FILTERING_SCRIPT, filtering_args
            )

        if args.skip_null_model:
            results[(campaign, 'null_model')] = None
        else:
            null_model_args = [
                '--campaigns', campaign,
                '--data-dir', args.filtered_dir,
                '--similarity-dir', args.similarity_dir,
                '--results-dir', args.null_model_dir,
                '--file-suffix', args.file_suffix,
            ]
            if args.num_permutations is not None:
                null_model_args += ['--num-permutations', str(args.num_permutations)]
            if args.workers is not None:
                null_model_args += ['--workers', str(args.workers)]
            results[(campaign, 'null_model')] = run_stage(
                f"[{campaign}] Stage 2/3: null_model", NULL_MODEL_SCRIPT, null_model_args
            )

        if args.skip_eval:
            results[(campaign, 'min_pvalue_eval')] = None
        else:
            eval_args = [
                '--campaigns', campaign,
                '--data-dir', args.null_model_dir,
                '--raw-data-dir', args.filtered_dir,
                '--results-dir', args.eval_dir,
                '--file-suffix', args.file_suffix,
            ]
            if args.alpha is not None:
                eval_args += ['--alpha', str(args.alpha)]
            if args.pauc_max_fpr is not None:
                eval_args += ['--pauc-max-fpr', str(args.pauc_max_fpr)]
            results[(campaign, 'min_pvalue_eval')] = run_stage(
                f"[{campaign}] Stage 3/3: min_pvalue_eval", MIN_PVALUE_EVAL_SCRIPT, eval_args
            )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    stages = ['filtering', 'null_model', 'min_pvalue_eval']
    header = f"{'campaign':30s} " + " ".join(f"{s:16s}" for s in stages)
    print(header)
    for campaign in campaigns:
        row = f"{campaign:30s} "
        for stage in stages:
            v = results.get((campaign, stage))
            label = "skipped" if v is None else ("OK" if v else "FAILED")
            row += f"{label:16s} "
        print(row)

    n_failed = sum(1 for v in results.values() if v is False)
    if n_failed:
        print(f"\n{n_failed} stage(s) failed -- see the output above for details.")
    else:
        print("\nAll stages completed (or were skipped) without error.")


if __name__ == '__main__':
    main()
