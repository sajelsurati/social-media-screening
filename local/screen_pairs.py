"""Score line-aligned AAVE samples for sexually explicit content and emit pairs.

Reads two line-aligned text files (AAVE + SAE rephrasings), classifies each
AAVE line with the project's NSFW classifier, and writes the flagged rows
together with their SAE counterparts to a CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.nsfw_classifier import ClassifierConfig, NSFWClassifier  # noqa: E402

AAVE_PATH = Path(__file__).parent / "aave_samples.txt"
SAE_PATH = Path(__file__).parent / "sae_samples.txt"
OUT_PATH = Path(__file__).parent / "results.csv"
NSFW_THRESHOLD = 0.5


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    aave = read_lines(AAVE_PATH)
    sae = read_lines(SAE_PATH)
    if len(aave) != len(sae):
        raise SystemExit(
            f"Line count mismatch: {AAVE_PATH.name}={len(aave)} vs {SAE_PATH.name}={len(sae)}"
        )

    clf = NSFWClassifier(ClassifierConfig(nsfw_threshold=NSFW_THRESHOLD))
    scores = clf.score(aave)

    kept = 0
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["line_index", "nsfw_score", "aave_text", "sae_text"])
        for i, (score, a, s) in enumerate(zip(scores, aave, sae)):
            if score >= NSFW_THRESHOLD:
                writer.writerow([i, f"{score:.4f}", a, s])
                kept += 1

    print(f"Scored {len(aave):,} lines; kept {kept:,} (score >= {NSFW_THRESHOLD}) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
