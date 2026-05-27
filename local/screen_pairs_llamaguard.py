"""Score line-aligned AAVE samples with Llama Guard 3 via Ollama.

Sends each AAVE line to Llama Guard 3, parses the safe/unsafe verdict
plus the list of MLCommons hazard category codes (S1–S13), and writes
flagged rows together with their SAE counterparts to a CSV.

Unlike the Detoxify script, Llama Guard returns a binary verdict per
category rather than a continuous score, so the CSV records the full
list of flagged categories instead of a probability.

Requires Ollama running locally with Llama Guard 3 pulled:

    ollama serve &
    ollama pull llama-guard3        # 8B (default)
    # or, for ~4-8x faster but lower-quality classification:
    ollama pull llama-guard3:1b

Usage:
    python local/screen_pairs_llamaguard.py                  # → results_llamaguard_any.csv (any unsafe)
    python local/screen_pairs_llamaguard.py --category S12   # → results_llamaguard_S12.csv
    python local/screen_pairs_llamaguard.py --model llama-guard3:1b
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests
from tqdm.auto import tqdm

AAVE_PATH = Path(__file__).parent / "aave_samples.txt"
SAE_PATH = Path(__file__).parent / "sae_samples.txt"

DEFAULT_CATEGORY = "any"  # any unsafe row; pass --category S12 (etc.) to filter
DEFAULT_MODEL = "llama-guard3"
DEFAULT_URL = "http://localhost:11434"


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help=f"MLCommons hazard code to filter on, or 'any' for all unsafe rows "
        f"(default: {DEFAULT_CATEGORY}). "
        "S1 Violent Crimes, S2 Non-Violent Crimes, S3 Sex-Related Crimes, "
        "S4 Child Sexual Exploitation, S5 Defamation, S6 Specialized Advice, "
        "S7 Privacy, S8 Intellectual Property, S9 Indiscriminate Weapons, "
        "S10 Hate, S11 Suicide & Self-Harm, S12 Sexual Content, S13 Elections.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--url", default=DEFAULT_URL)
    return p.parse_args()


def classify(text: str, model: str, url: str) -> tuple[bool, list[str]]:
    """Return (is_unsafe, list of MLCommons category codes)."""
    resp = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines or lines[0].lower() != "unsafe":
        return False, []
    cats: list[str] = []
    if len(lines) > 1:
        cats = [c.strip().upper() for c in lines[1].split(",") if c.strip()]
    return True, cats


def main() -> None:
    args = parse_args()
    raw = args.category.strip()
    target = "any" if raw.lower() == "any" else raw.upper()
    out_path = Path(__file__).parent / f"results_llamaguard_{target}.csv"

    aave = read_lines(AAVE_PATH)
    sae = read_lines(SAE_PATH)
    if len(aave) != len(sae):
        raise SystemExit(
            f"Line count mismatch: {AAVE_PATH.name}={len(aave)} vs {SAE_PATH.name}={len(sae)}"
        )

    flagged: list[tuple[int, str, str, str]] = []
    for i, (a, s) in enumerate(tqdm(list(zip(aave, sae)), desc="Llama Guard 3")):
        unsafe, cats = classify(a, args.model, args.url)
        if not unsafe:
            continue
        if target == "any" or target in cats:
            flagged.append((i, ",".join(cats), a, s))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["line_index", "categories", "aave_text", "sae_text"])
        for row in flagged:
            writer.writerow(row)

    print(
        f"Scored {len(aave):,} lines; kept {len(flagged):,} "
        f"(category {target}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
