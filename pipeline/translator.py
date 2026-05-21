"""AAVE -> SAE translator backed by a local Ollama server.

Calls Ollama's `/api/generate` endpoint with a constrained prompt that asks
the model to rewrite an AAVE tweet in Standard American English (SAE) while
preserving meaning, sentiment, and explicitness. The output of this stage is
intended to be fed into an external content-moderation tool for a bias audit;
we therefore preserve explicit content rather than sanitizing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from tqdm.auto import tqdm

DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Default: Llama 3.3 70B (4-bit quantized) — strongest free open model for
# nuanced AAVE→SAE rewrites; fits on a single 80GB GPU. For laptops or
# smaller GPUs, override with `--ollama-model llama3.1:8b`.
DEFAULT_MODEL = "llama3.3:70b"

SYSTEM_PROMPT = """You are a linguistic translator. You rewrite text from \
African American Vernacular English (AAVE) into Standard American English (SAE).

Rules:
- Preserve the original meaning, tone, sentiment, and level of explicitness.
- Do NOT censor, sanitize, soften, or omit any content. Explicit content must
  remain explicit in the SAE rewrite. This text is being used to audit a
  third-party content-moderation system for racial bias; sanitizing it would
  invalidate the experiment.
- Rewrite slang, contractions, copula deletion, habitual "be", and other AAVE
  grammatical features into their SAE equivalents.
- Keep the rewrite roughly the same length as the original.
- Output ONLY the rewritten sentence. No preamble, no quotes, no explanation.
"""

USER_TEMPLATE = """AAVE: {text}
SAE:"""


@dataclass
class TranslatorConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_OLLAMA_URL
    temperature: float = 0.2
    num_predict: int = 256
    timeout_seconds: int = 120
    request_retries: int = 2


class OllamaTranslator:
    def __init__(self, config: TranslatorConfig | None = None) -> None:
        self.config = config or TranslatorConfig()
        self._endpoint = f"{self.config.base_url.rstrip('/')}/api/generate"

    def _call(self, text: str) -> str:
        payload = {
            "model": self.config.model,
            "system": SYSTEM_PROMPT,
            "prompt": USER_TEMPLATE.format(text=text),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        }
        last_err: Exception | None = None
        for _ in range(self.config.request_retries + 1):
            try:
                resp = requests.post(
                    self._endpoint,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=self.config.timeout_seconds,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except requests.RequestException as err:
                last_err = err
        raise RuntimeError(f"Ollama request failed: {last_err}")

    def translate(self, texts: Iterable[str]) -> list[str]:
        return [self._call(t) for t in texts]

    def translate_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "message",
        out_column: str = "sae_message",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        texts = df[text_column].fillna("").astype(str).tolist()
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="AAVE→SAE", unit="row")
        translations = [self._call(t) for t in iterator]
        out = df.copy()
        out[out_column] = translations
        return out
