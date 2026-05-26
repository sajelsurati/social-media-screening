"""End-to-end pipeline: TwitterAAE -> NSFW filter -> AAVE→SAE translation.

Three stages, each checkpoints to disk so you can resume without re-running
prior work:

    1. load     -> data/01_aave.parquet
    2. detect   -> data/02_aave_nsfw.parquet  (subset with nsfw_score >= threshold)
    3. translate-> data/03_paired.parquet     (AAVE + SAE columns)

The final output is also written as CSV at data/03_paired.csv for handoff
to whatever external moderation tool is being audited.

Usage:
    python run_pipeline.py --input path/to/twitteraae_all.tsv
    python run_pipeline.py --input ... --sample 5000 --skip-existing
    python run_pipeline.py --stages detect translate  # resume from checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.loader import LoaderConfig, load_twitteraae
from pipeline.nsfw_classifier import ClassifierConfig, NSFWClassifier
from pipeline.translator import OllamaTranslator, TranslatorConfig

STAGES = ("load", "detect", "translate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="Path to TwitterAAE TSV (required for the 'load' stage)")
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--skip-existing", action="store_true", help="Skip a stage if its output already exists")

    # Load stage
    p.add_argument("--aae-threshold", type=float, default=0.8)
    p.add_argument("--sample", type=int, default=None, help="Random subsample after AAE filtering")
    p.add_argument("--seed", type=int, default=42)

    # Detect stage
    p.add_argument("--nsfw-model", default="unitary/unbiased-toxic-roberta")
    p.add_argument("--nsfw-threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)

    # Translate stage
    p.add_argument(
        "--ollama-model",
        default="llama3.3:70b",
        help="Ollama model tag. Default is llama3.3:70b (best quality, ~80GB GPU). "
        "Use llama3.1:8b for laptops or smaller GPUs.",
    )
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--translate-limit", type=int, default=None, help="Cap rows to translate (for smoke tests)")

    return p.parse_args()


def stage_load(args: argparse.Namespace, out_path: Path) -> pd.DataFrame:
    if not args.input:
        raise SystemExit("--input is required for the 'load' stage")
    cfg = LoaderConfig(
        path=args.input,
        aae_threshold=args.aae_threshold,
        sample_size=args.sample,
        seed=args.seed,
    )
    df = load_twitteraae(cfg)
    df.to_parquet(out_path, index=False)
    print(f"[load] {len(df):,} AAVE rows -> {out_path}")
    return df


def stage_detect(args: argparse.Namespace, in_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(in_path)
    clf = NSFWClassifier(
        ClassifierConfig(
            model_name=args.nsfw_model,
            batch_size=args.batch_size,
            nsfw_threshold=args.nsfw_threshold,
        )
    )
    scored = clf.classify_dataframe(df, text_column="message")
    nsfw = scored[scored["is_nsfw"]].reset_index(drop=True)
    nsfw.to_parquet(out_path, index=False)
    print(
        f"[detect] {len(nsfw):,} / {len(scored):,} rows flagged NSFW "
        f"(threshold={args.nsfw_threshold}) -> {out_path}"
    )
    return nsfw


def stage_translate(args: argparse.Namespace, in_path: Path, out_parquet: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_parquet(in_path)
    if args.translate_limit is not None:
        df = df.head(args.translate_limit).reset_index(drop=True)
    translator = OllamaTranslator(
        TranslatorConfig(model=args.ollama_model, base_url=args.ollama_url)
    )
    paired = translator.translate_dataframe(
        df, text_column="message", out_column="sae_message"
    )
    paired = paired.rename(columns={"message": "aave_message"})
    paired = paired[["tweet_id", "aae_prob", "nsfw_score", "aave_message", "sae_message"]]
    paired.to_parquet(out_parquet, index=False)
    paired.to_csv(out_csv, index=False)
    print(f"[translate] {len(paired):,} pairs -> {out_parquet} and {out_csv}")
    return paired


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p_load = args.output_dir / "01_aave.parquet"
    p_detect = args.output_dir / "02_aave_nsfw.parquet"
    p_pair = args.output_dir / "03_paired.parquet"
    p_pair_csv = args.output_dir / "03_paired.csv"

    if "load" in args.stages:
        if args.skip_existing and p_load.exists():
            print(f"[load] skipping; {p_load} exists")
        else:
            stage_load(args, p_load)

    if "detect" in args.stages:
        if args.skip_existing and p_detect.exists():
            print(f"[detect] skipping; {p_detect} exists")
        else:
            stage_detect(args, p_load, p_detect)

    if "translate" in args.stages:
        if args.skip_existing and p_pair.exists():
            print(f"[translate] skipping; {p_pair} exists")
        else:
            stage_translate(args, p_detect, p_pair, p_pair_csv)


if __name__ == "__main__":
    main()
