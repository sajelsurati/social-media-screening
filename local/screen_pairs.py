"""Score line-aligned AAVE samples for a Detoxify label and emit pairs.

Reads two line-aligned text files (AAVE + SAE rephrasings), classifies each
AAVE line with the project's moderation classifier, and writes the flagged
rows together with their SAE counterparts to a label-specific CSV.

Usage:
    python local/screen_pairs.py                    # → results_sexual_explicit.csv
    python local/screen_pairs.py --label obscene    # → results_obscene.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.nsfw_classifier import ClassifierConfig, NSFWClassifier  # noqa: E402

AAVE_PATH = Path(__file__).parent / "aave_samples.txt"
SAE_PATH = Path(__file__).parent / "sae_samples.txt"
DEFAULT_LABEL = "sexual_explicit"
DEFAULT_THRESHOLD = 0.5


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Detoxify label to filter on (default: {DEFAULT_LABEL}). "
        "Available: toxicity, severe_toxicity, obscene, identity_attack, "
        "insult, threat, sexual_explicit.",
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(__file__).parent / f"results_{args.label}.csv"

    aave = read_lines(AAVE_PATH)
    sae = read_lines(SAE_PATH)
    if len(aave) != len(sae):
        raise SystemExit(
            f"Line count mismatch: {AAVE_PATH.name}={len(aave)} vs {SAE_PATH.name}={len(sae)}"
        )

    clf = NSFWClassifier(
        ClassifierConfig(target_label=args.label, nsfw_threshold=args.threshold)
    )
    scores = clf.score(aave)

    flagged = [
        (i, score, a, s)
        for i, (score, a, s) in enumerate(zip(scores, aave, sae))
        if score >= args.threshold
    ]
    flagged.sort(key=lambda row: row[1], reverse=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["line_index", f"{args.label}_score", "aave_text", "sae_text"])
        for i, score, a, s in flagged:
            writer.writerow([i, f"{score:.4f}", a, s])

    print(
        f"Scored {len(aave):,} lines; kept {len(flagged):,} "
        f"({args.label} >= {args.threshold}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
