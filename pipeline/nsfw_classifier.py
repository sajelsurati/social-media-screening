"""Sexually-suggestive content classifier.

Wraps a HuggingFace text classification model (default:
`michellejieli/NSFW_text_classifier`) with batched inference and
Apple Silicon (MPS) / CUDA / CPU device selection.

The default model is a DistilBERT fine-tune that emits two labels:
`NSFW` and `SFW`. We score `NSFW` probability per row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "michellejieli/NSFW_text_classifier"


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ClassifierConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    max_length: int = 256
    device: str | None = None
    nsfw_threshold: float = 0.5


class NSFWClassifier:
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.device = self.config.device or _select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._nsfw_idx = self._resolve_nsfw_label_index()

    def _resolve_nsfw_label_index(self) -> int:
        id2label = self.model.config.id2label
        for idx, label in id2label.items():
            if str(label).upper().startswith("NSFW"):
                return int(idx)
        # Fall back to the last label if naming differs.
        return len(id2label) - 1

    @torch.no_grad()
    def score(self, texts: Iterable[str]) -> list[float]:
        texts = list(texts)
        scores: list[float] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, self._nsfw_idx]
            scores.extend(probs.detach().cpu().tolist())
        return scores

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "message",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Add `nsfw_score` and `is_nsfw` columns to `df`."""
        texts = df[text_column].fillna("").astype(str).tolist()
        all_scores: list[float] = []
        iterator = range(0, len(texts), self.config.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="NSFW scoring", unit="batch")
        for start in iterator:
            batch = texts[start : start + self.config.batch_size]
            all_scores.extend(self.score(batch))
        out = df.copy()
        out["nsfw_score"] = all_scores
        out["is_nsfw"] = out["nsfw_score"] >= self.config.nsfw_threshold
        return out
