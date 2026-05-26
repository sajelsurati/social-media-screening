# social-media-screening

An experiment that screens line-aligned AAVE/SAE samples for sexually
explicit content, producing a paired CSV intended as input to an
external content-moderation tool whose racial bias you want to audit.

## What it does

`local/screen_pairs.py`:

1. Reads two line-aligned text files (`local/aave_samples.txt` and
   `local/sae_samples.txt`) of AAVE tweets and their SAE rewrites.
2. Runs a local multi-label moderation model
   ([`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta),
   the Detoxify "unbiased" checkpoint) over the AAVE lines.
3. Filters on the model's `sexual_explicit` label — not a combined NSFW
   label — so that profanity, slurs, or AAVE features alone do not flag
   a row.
4. Writes flagged rows to `local/results.csv` alongside their SAE
   counterparts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the Detoxify model (~500 MB) from HuggingFace
into the local cache. Inference uses CUDA, Apple Silicon (MPS), or CPU,
auto-selected in that order.

## Run

```bash
python local/screen_pairs.py
```

The script prints how many lines were scored and how many were kept.

## Output

`local/results.csv` columns:

| column | description |
|---|---|
| `line_index` | 0-based line index in the input files |
| `nsfw_score` | `sexual_explicit` probability from the moderation model (0–1) |
| `aave_text` | Original AAVE line |
| `sae_text` | SAE rewrite |

Feed `aave_text` and `sae_text` into the external moderation tool you're
auditing and compare flagging rates.

## Configuration

The threshold is set at the top of `local/screen_pairs.py`:

```python
NSFW_THRESHOLD = 0.5
```

The classifier model and target label are configured in
`pipeline/nsfw_classifier.py` via `ClassifierConfig`:

```python
ClassifierConfig(
    model_name="unitary/unbiased-toxic-roberta",
    target_label="sexual_explicit",
    nsfw_threshold=0.5,
)
```

To use a different multi-label moderation model, pass a model name
whose `id2label` contains the desired `target_label`.

## Notes

- The moderation model inherits biases from its training data (Jigsaw
  "Unintended Bias in Toxicity Classification"). Spot-check flagged
  rows before drawing conclusions about the audit dataset.
- For background on dialect bias in toxicity / hate-speech classifiers,
  see Sap et al. 2019 (*The Risk of Racial Bias in Hate Speech
  Detection*) and Hofmann et al. 2024 (*Dialect prejudice predicts AI
  decisions about people's character, employability, and criminality*).
