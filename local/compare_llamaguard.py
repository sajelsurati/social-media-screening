"""Score AAVE and SAE samples with Llama Guard 3 and report dialect-bias deltas.

Scores both line-aligned files with Llama Guard 3 via Ollama, writes a
per-line comparison CSV, and prints per-category flag-rate deltas — the
dialect-bias signal (e.g., how much S10/Hate drops when AAVE is rewritten
as SAE).

For each line the comparison records:
    line_index, flag_pattern, aave_categories, sae_categories,
    aave_text, sae_text

flag_pattern is one of {both, aave_only, sae_only, neither}, where
"flagged" means Llama Guard returned 'unsafe' for at least one MLCommons
category. Categories columns hold a comma-separated list (empty when safe).

Requires Ollama running locally with Llama Guard 3 pulled — see
`screen_pairs_llamaguard.py` for setup details.

Usage:
    python local/compare_llamaguard.py                          # 8B, score both sides
    python local/compare_llamaguard.py --reuse-aave             # 8B, score SAE only; reuse cached AAVE results
    python local/compare_llamaguard.py --model llama-guard3:1b  # 1B, score both sides
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from screen_pairs_llamaguard import (  # noqa: E402
    AAVE_PATH,
    DEFAULT_MODEL,
    DEFAULT_URL,
    SAE_PATH,
    classify,
    read_lines,
)

CACHED_AAVE_PATH = Path(__file__).parent / "results_llamaguard_any.csv"
OUT_PATH = Path(__file__).parent / "results_llamaguard_compare.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument(
        "--reuse-aave",
        action="store_true",
        help=f"Read cached AAVE flags from {CACHED_AAVE_PATH.name} instead of "
        "re-scoring. The cached file must have been produced with the same "
        "model on the same input.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the per-category AAVE vs SAE counts; skip the CSV "
        "and the per-line flag-pattern breakdown.",
    )
    return p.parse_args()


def score_all(texts: list[str], model: str, url: str, label: str) -> list[list[str]]:
    out: list[list[str]] = []
    for t in tqdm(texts, desc=f"Llama Guard 3 ({label})"):
        unsafe, cats = classify(t, model, url)
        out.append(cats if unsafe else [])
    return out


def load_cached_aave(n_lines: int) -> list[list[str]]:
    """Load AAVE flagged categories from the screen_pairs_llamaguard output.

    The cached file only contains flagged rows; missing line_index entries
    are treated as safe (empty category list).
    """
    cats_per_line: list[list[str]] = [[] for _ in range(n_lines)]
    with CACHED_AAVE_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["line_index"])
            cats_per_line[idx] = [c.strip() for c in row["categories"].split(",") if c.strip()]
    return cats_per_line


def flag_pattern(aave_cats: list[str], sae_cats: list[str]) -> str:
    a, s = bool(aave_cats), bool(sae_cats)
    if a and s:
        return "both"
    if a:
        return "aave_only"
    if s:
        return "sae_only"
    return "neither"


def main() -> None:
    args = parse_args()
    aave = read_lines(AAVE_PATH)
    sae = read_lines(SAE_PATH)
    if len(aave) != len(sae):
        raise SystemExit(
            f"Line count mismatch: {AAVE_PATH.name}={len(aave)} vs {SAE_PATH.name}={len(sae)}"
        )

    if args.reuse_aave:
        if not CACHED_AAVE_PATH.exists():
            raise SystemExit(
                f"--reuse-aave requested but {CACHED_AAVE_PATH} not found. "
                "Run screen_pairs_llamaguard.py first."
            )
        aave_cats = load_cached_aave(len(aave))
        print(f"Loaded cached AAVE results from {CACHED_AAVE_PATH.name}")
    else:
        aave_cats = score_all(aave, args.model, args.url, "AAVE")

    sae_cats = score_all(sae, args.model, args.url, "SAE")

    if not args.summary_only:
        with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "line_index", "flag_pattern",
                "aave_categories", "sae_categories",
                "aave_text", "sae_text",
            ])
            for i, (ac, sc, a, s) in enumerate(zip(aave_cats, sae_cats, aave, sae)):
                writer.writerow([i, flag_pattern(ac, sc), ",".join(ac), ",".join(sc), a, s])

        patterns = Counter(flag_pattern(ac, sc) for ac, sc in zip(aave_cats, sae_cats))
        print(f"\n=== Overall flag pattern (n={len(aave)}) ===")
        for p in ("both", "aave_only", "sae_only", "neither"):
            print(f"  {p:12s}: {patterns.get(p, 0):4d}")

    aave_cat_counts = Counter(c for cats in aave_cats for c in cats)
    sae_cat_counts = Counter(c for cats in sae_cats for c in cats)
    all_cats = sorted(set(aave_cat_counts) | set(sae_cat_counts))

    print(f"\n=== Per-category flag counts (AAVE vs SAE) ===")
    n = len(aave)
    print(f"  {'cat':>5s}  {'AAVE':>10s}  {'SAE':>10s}  {'delta':>7s}")
    for cat in all_cats:
        a = aave_cat_counts.get(cat, 0)
        s = sae_cat_counts.get(cat, 0)
        a_str = f"{a:4d} ({a / n * 100:4.1f}%)"
        s_str = f"{s:4d} ({s / n * 100:4.1f}%)"
        delta = f"{(s - a) / a * 100:+.0f}%" if a else "n/a"
        print(f"  {cat:>5s}  {a_str:>10s}  {s_str:>10s}  {delta:>7s}")

    if not args.summary_only:
        print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
