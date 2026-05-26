"""Sexually-suggestive content classifier.

Wraps a HuggingFace multi-label moderation model (default:
`unitary/unbiased-toxic-roberta`, the Detoxify "unbiased" checkpoint)
with batched inference and Apple Silicon (MPS) / CUDA / CPU device
selection.

The default model emits independent sigmoid probabilities for seven
labels (`toxicity`, `severe_toxicity`, `obscene`, `identity_attack`,
`insult`, `threat`, `sexual_explicit`). We score the `sexual_explicit`
label per row so that profanity / slurs alone do not flag a row —
important when the input distribution is AAVE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "unitary/unbiased-toxic-roberta"
DEFAULT_TARGET_LABEL = "sexual_explicit"


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ClassifierConfig:
    model_name: str = DEFAULT_MODEL
    target_label: str = DEFAULT_TARGET_LABEL
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
        self._target_idx = self._resolve_target_label_index()

    def _resolve_target_label_index(self) -> int:
        id2label = self.model.config.id2label
        target = self.config.target_label.lower()
        for idx, label in id2label.items():
            if str(label).lower() == target:
                return int(idx)
        raise ValueError(
            f"Label {self.config.target_label!r} not found in model "
            f"{self.config.model_name!r}. Available labels: {list(id2label.values())}"
        )

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
            probs = torch.sigmoid(logits)[:, self._target_idx]
            scores.extend(probs.detach().cpu().tolist())
        return scores
