"""Load and filter the TwitterAAE corpus (Blodgett et al. 2016).

The public release is a tab-separated file with columns:
    tweet_id, time, user_id, lat, lon, fips, AAE, Hispanic, Other, White, message

`AAE`, `Hispanic`, `Other`, `White` are posterior probabilities (0-1) over
the demographic topic model. `message` is the tweet text. Some releases use
slightly different column names or counts; this loader is lenient about both.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

EXPECTED_COLUMNS = [
    "tweet_id",
    "time",
    "user_id",
    "lat",
    "lon",
    "fips",
    "aae_prob",
    "hispanic_prob",
    "other_prob",
    "white_prob",
    "message",
]


@dataclass
class LoaderConfig:
    path: Path
    aae_threshold: float = 0.8
    sample_size: int | None = None
    seed: int = 42


def load_twitteraae(config: LoaderConfig) -> pd.DataFrame:
    """Load the TwitterAAE TSV, filtered to rows whose AAE posterior >= threshold.

    Returns a DataFrame with at least `tweet_id`, `aae_prob`, `message` columns.
    """
    df = pd.read_csv(
        config.path,
        sep="\t",
        header=None,
        names=EXPECTED_COLUMNS,
        quoting=csv.QUOTE_NONE,
        engine="python",
        on_bad_lines="skip",
        dtype={"tweet_id": str, "user_id": str, "fips": str, "message": str},
    )

    df["aae_prob"] = pd.to_numeric(df["aae_prob"], errors="coerce")
    df = df.dropna(subset=["aae_prob", "message"])
    df = df[df["message"].str.strip().str.len() > 0]
    df = df[df["aae_prob"] >= config.aae_threshold].reset_index(drop=True)

    if config.sample_size is not None and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.seed).reset_index(drop=True)

    return df[["tweet_id", "aae_prob", "message"]]


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]
